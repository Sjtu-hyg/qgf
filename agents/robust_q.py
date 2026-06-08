from functools import partial
from typing import Any, Optional

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.qgf import QGFAgent
from agents.qgf import get_config as get_qgf_config
from utils.activation import get_activation
from utils.flax_utils import TrainState, target_update
from utils.networks import ActorFlowField, Value, timestep_embedding


class RobustQAgent(QGFAgent):
    """RobustQ: test-time guidance with a noise-conditioned critic (inspired by robust image classifiers).

    Trains a time-conditioned Q_robust(s, a_t, t) to regress onto the clean IQL Q(s, a):
    L_robust = (Q_robust(s, a_t, t) - Q(s, a))^2. Since Q_robust is trained on noisy
    actions at every timestep, the gradient ∇_{a_t} Q_robust(s, a_t, t) is
    in-distribution at each denoising step and can be used directly as guidance.

    Training: BC flow + IQL as in the base QGF training, plus regression of Q_robust.
    Inference: guidance via ∇_{x_t} Q_robust(s, x_t, t) — no Jacobian correction needed.
    """

    robust_critic: TrainState

    def robust_critic_loss(self, batch, robust_critic_params=None, rng=None):
        if rng is None:
            rng = self.rng
        eps_rng, time_rng = jax.random.split(rng, 2)

        if self.config.get("action_chunking", False):
            actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            actions = batch["actions"][..., 0, :]
        observations = batch["observations"]

        # Noise the clean dataset actions along the flow path
        x0 = jax.random.normal(eps_rng, actions.shape)
        t = jax.random.uniform(time_rng, (actions.shape[0],))
        tv = t[..., None]
        a_t = x0 * (1 - tv) + actions * tv

        t_emb = timestep_embedding(t, emb_size=self.config["robust_critic_t_emb_size"])
        a_t_with_t = jnp.concatenate([a_t, t_emb], axis=-1)

        # Regression target: Q(s, a) evaluated at the clean action
        target_qs = self.target_critic(observations, actions)
        target_q = jax.lax.stop_gradient(self._aggregate_q(target_qs))

        robust_q = self.robust_critic(
            observations, a_t_with_t, params=robust_critic_params
        )
        loss = jnp.mean((robust_q - target_q) ** 2)
        return loss, {
            "robust_critic_loss": loss,
            "robust_q_mean": robust_q.mean(),
            "robust_q_target_mean": target_q.mean(),
        }

    @jax.jit
    def update(self, batch):
        new_rng, policy_rng, critic_rng, robust_rng = jax.random.split(self.rng, 4)

        new_policy, policy_info = self.policy.apply_loss_fn(
            loss_fn=lambda p: self.policy_loss(batch, p, rng=policy_rng)
        )
        new_critic, critic_info = self.critic.apply_loss_fn(
            loss_fn=lambda p: self.critic_loss(batch, p, rng=critic_rng)
        )
        new_target_critic = target_update(
            self.critic, self.target_critic, self.config["tau"]
        )
        new_value, value_info = self.value.apply_loss_fn(
            loss_fn=lambda p: self.value_loss(batch, p)
        )
        new_robust_critic, robust_critic_info = self.robust_critic.apply_loss_fn(
            loss_fn=lambda p: self.robust_critic_loss(batch, p, rng=robust_rng)
        )

        return self.replace(
            rng=new_rng,
            policy=new_policy,
            critic=new_critic,
            target_critic=new_target_critic,
            value=new_value,
            robust_critic=new_robust_critic,
        ), {**policy_info, **critic_info, **value_info, **robust_critic_info}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, additional_agents=None):
        _ = additional_agents
        if rng is None:
            rng = self.rng
        rng, robust_rng = jax.random.split(rng)

        bc_loss, policy_info = self.policy_loss(batch, rng=rng)
        critic_loss, critic_info = self.critic_loss(batch, rng=rng)
        value_loss, value_info = self.value_loss(batch)
        robust_loss, robust_info = self.robust_critic_loss(batch, rng=robust_rng)

        return bc_loss + critic_loss + value_loss + robust_loss, {
            **policy_info,
            **critic_info,
            **value_info,
            **robust_info,
        }

    @partial(jax.jit, static_argnames=["rejection_sampling"])
    def sample_actions(
        self,
        observations: jnp.ndarray,
        *,
        seed: Any,
        rejection_sampling: int = 1,
    ) -> jnp.ndarray:
        """Denoise with per-step guidance: v = v_bc + cfg * ∇_{x_t} Q_robust(s, x_t, t)."""
        has_batch_dim = observations.ndim == 2
        observations = observations if has_batch_dim else observations[None]

        batch_size = observations.shape[0]
        if rejection_sampling > 1:
            observations = jnp.repeat(observations, rejection_sampling, axis=0)

        H = self.config.get("horizon_length", 1)
        ad = self.config["action_dim"]
        full_action_dim = ad * (H if self.config.get("action_chunking", False) else 1)
        x = jax.random.normal(seed, (observations.shape[0], full_action_dim))
        dt = 1.0 / self.config["denoise_steps"]

        def step(x, t_idx):
            ti = jnp.ones((x.shape[0],)) * (t_idx / self.config["denoise_steps"])
            v_bc = self.policy(observations, x, ti)

            def robust_qval(a):
                t_emb = timestep_embedding(
                    ti, emb_size=self.config["robust_critic_t_emb_size"]
                )
                return self.robust_critic(
                    observations, jnp.concatenate([a, t_emb], axis=-1)
                ).mean()

            qgrad = jax.grad(robust_qval)(x)

            return x + (v_bc + cfg * qgrad) * dt, None

        actions, _ = jax.lax.scan(
            step,
            x,
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

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, policy_key, critic_key, value_key, robust_critic_key = jax.random.split(
            rng, 5
        )

        action_dim = ex_actions.shape[-1]
        H = config.get("horizon_length", 1)
        if config.get("action_chunking", False):
            ex_full_actions = jnp.concatenate([ex_actions] * H, axis=-1)
        else:
            ex_full_actions = ex_actions
        full_action_dim = ex_full_actions.shape[-1]

        config = dict(config)
        config["action_dim"] = action_dim

        activation_fn = get_activation(config["activation"])
        mlp_kwargs = dict(activation=activation_fn, layer_norm=config["use_layer_norm"])
        ex_t = jnp.zeros(ex_actions.shape[0])

        policy_def = ActorFlowField(
            config["actor_hidden_dims"], full_action_dim, mlp_kwargs=mlp_kwargs
        )
        policy_params = policy_def.init(
            policy_key, ex_observations, ex_full_actions, ex_t
        )["params"]
        policy = TrainState.create(
            policy_def, policy_params, tx=optax.adam(learning_rate=config["bc_lr"])
        )

        critic_def = Value(
            network_class=config["value_network_class"],
            network_kwargs={
                **config["value_network_kwargs"],
                "activation": activation_fn,
            },
            num_ensembles=config["num_qs"],
        )
        critic_params = critic_def.init(critic_key, ex_observations, ex_full_actions)[
            "params"
        ]
        critic = TrainState.create(
            critic_def, critic_params, tx=optax.adam(learning_rate=config["critic_lr"])
        )
        target_critic = TrainState.create(critic_def, critic_params)

        value_def = Value(
            network_class=config["value_network_class"],
            network_kwargs={
                **config["value_network_kwargs"],
                "activation": activation_fn,
            },
            num_ensembles=1,
        )
        value_params = value_def.init(value_key, ex_observations)["params"]
        value = TrainState.create(
            value_def, value_params, tx=optax.adam(learning_rate=config["value_lr"])
        )

        # Robust critic: Q_robust(s, a, t) with input [obs; action; t_embedding]
        emb_size = config["robust_critic_t_emb_size"]
        ex_t_emb = timestep_embedding(ex_t, emb_size=emb_size)
        ex_actions_with_t = jnp.concatenate([ex_full_actions, ex_t_emb], axis=-1)
        robust_critic_def = Value(
            network_class=config["value_network_class"],
            network_kwargs={
                **config["value_network_kwargs"],
                "activation": activation_fn,
            },
            num_ensembles=1,
        )
        robust_critic_params = robust_critic_def.init(
            robust_critic_key, ex_observations, ex_actions_with_t
        )["params"]
        robust_critic = TrainState.create(
            robust_critic_def,
            robust_critic_params,
            tx=optax.adam(learning_rate=config["robust_critic_lr"]),
        )

        config_dict = flax.core.FrozenDict(**config)
        return cls(
            rng,
            policy=policy,
            critic=critic,
            target_critic=target_critic,
            value=value,
            robust_critic=robust_critic,
            config=config_dict,
        )


def get_config():
    config = get_qgf_config()
    config["agent_name"] = "robust_q"
    # RobustQ-specific hyperparameters.
    config[
        "robust_critic_lr"
    ] = 3e-4  # Learning rate for the noise-conditioned robust critic.
    config[
        "robust_critic_t_emb_size"
    ] = 16  # Timestep embedding size for the robust critic.
    return config
