from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from agents.qgf import QGFAgent
from agents.qgf import get_config as get_qgf_config


class GradStepAgent(QGFAgent):
    """GradStep: post-denoising gradient ascent (inspired by PA-RL, Mark et al. 2024).

    Samples a clean action by running the full BC denoising process, then iteratively
    improves it with L gradient-ascent steps directly in clean action space:
    a^(l) <- a^(l-1) + alpha * ∇_a Q(s, a^(l-1)).
    The Q-gradient is applied only to the fully denoised action, not during denoising.
    """

    support_guidance = False

    @partial(jax.jit, static_argnames=["rejection_sampling"])
    def sample_actions(
        self,
        observations: jnp.ndarray,
        *,
        seed: Any,
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
        dt = 1.0 / self.config["denoise_steps"]

        def _bc_denoise(obs, noise):
            # denoise a_t to a_1 by integrating the BC flow
            def bc_step(x, t_idx):
                ti = jnp.ones((x.shape[0],)) * (t_idx / self.config["denoise_steps"])
                return x + self.policy(obs, x, ti) * dt, None

            out, _ = jax.lax.scan(
                bc_step,
                noise,
                jnp.arange(self.config["denoise_steps"]),
                length=self.config["denoise_steps"],
            )
            return jnp.clip(out, -1, 1)

        def _qgrad_refine(obs, actions):
            # refine the denoised action a_1 with gradient ascent through the Q-function
            step_size = self.config.get("qgrad_step_size", 0.1)

            def q_fn(a):
                return self._aggregate_q(self.target_critic(obs, a)).sum()

            def grad_step(a, _):
                g = jax.grad(q_fn)(a)
                if self.config.get("use_sign_gradient", False):
                    g = jnp.sign(g)
                return jnp.clip(a + step_size * g, -1, 1), None

            out, _ = jax.lax.scan(
                grad_step,
                actions,
                None,
                length=self.config.get("qgrad_steps", 1),
            )
            return out

        noise = jax.random.normal(seed, (observations.shape[0], full_action_dim))
        actions = _bc_denoise(observations, noise)
        actions = _qgrad_refine(observations, actions)

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
    config = get_qgf_config()
    config.agent_name = "grad_step"
    config.qgrad_step_size = 0.1
    config.qgrad_steps = 1
    return config
