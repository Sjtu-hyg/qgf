from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
from agents.qgf import QGFAgent


class BPTTAgent(QGFAgent):
    """Back-Propagation Through Time (BPTT) gradient guidance (inspired by DQL, Wang et al. 2022).

    At each denoising step t, runs the full BC denoising process from a_t to get
    a_clean = ODE(a_t), then backpropagates the Q-gradient through
    the entire denoising chain: g_t = ∇_{a_t} Q(s, ODE(a_t)). Gradients flow through
    all intermediate denoising steps. Expensive due to full BPTT and can be unstable
    in practice because of high variance in the gradient signal over long chains.
    """

    @partial(jax.jit, static_argnames=["rejection_sampling"])
    def sample_actions(
        self,
        observations: jnp.ndarray,
        *,
        seed: Any,
        guidance_weight: float = 1.0,
        rejection_sampling: int = 1,
    ) -> jnp.ndarray:
        has_batch_dim = observations.ndim == 2
        observations = observations if has_batch_dim else observations[None]

        batch_size = observations.shape[0]
        if rejection_sampling > 1:
            observations = jnp.repeat(observations, rejection_sampling, axis=0)

        H = self.config.get("horizon_length", 1)
        ad = self.config["action_dim"]
        full_action_dim = ad * (H if self.config.get("action_chunking", False) else 1)
        a = jax.random.normal(seed, (observations.shape[0], full_action_dim))
        dt = 1.0 / self.config["denoise_steps"]

        def step(a, t_idx):
            ti = jnp.ones((a.shape[0],)) * (t_idx / self.config["denoise_steps"])
            v_bc = self.policy(observations, a, ti)

            def q_of_at(a_curr):
                # get the clean action ODE(a_t)
                a_approx = self._bc_flow_from(
                    observations,
                    a_curr,
                    jnp.full((a_curr.shape[0],), t_idx, dtype=jnp.int32),
                )
                a_approx = jnp.clip(a_approx, -1, 1)
                return self._aggregate_q(
                    self.target_critic(observations, a_approx)
                ).sum()

            # backpropagate the Q-gradient through the entire denoising chain
            # backpropagation is handled automatically by jax.grad through q_of_at
            qgrad = jax.grad(q_of_at)(a)

            return a + (v_bc + guidance_weight * qgrad) * dt, None

        actions, _ = jax.lax.scan(
            step,
            a,
            jnp.arange(self.config["denoise_steps"]),
            length=self.config["denoise_steps"],
        )
        actions = jnp.clip(actions, -1, 1)

        if rejection_sampling > 1:
            q = self._aggregate_q(self.target_critic(observations, actions))
            q = q.reshape((batch_size, rejection_sampling))
            actions = actions.reshape(
                (batch_size, rejection_sampling, *actions.shape[1:])
            )
            actions = actions[jnp.arange(batch_size), jnp.argmax(q, axis=1)]

        if not has_batch_dim:
            actions = actions[0]

        return actions


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="bptt",
            # Common hyperparameters.
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            bc_lr=3e-4,
            critic_lr=3e-4,
            value_lr=3e-4,
            use_layer_norm=1,
            activation="gelu",
            # n-step returns & action chunking.
            horizon_length=1,
            action_chunking=False,
            # RL hyperparameters.
            num_qs=2,
            value_network_class="MLP",
            value_network_kwargs=dict(
                hidden_dims=(512, 512, 512, 512),
                layer_norm=True,
            ),
            q_aggregation="min",  # "min" or "mean".
            discount=0.99,
            expectile=0.9,
            tau=0.005,
            denoise_steps=10,  # Number of flow-matching denoising steps.
        )
    )
    return config
