"""
Actor-critic algorithm for flow policies.
Critic: train with IQL
Actor: trained to maximize Q(s, a_t) at each denoising step t plus a weighted flow matching loss.
Use the one-step approximation trick to compute Q(s, a_t) = Q(s, a_t + (1-t)*v_t) and acatually maximize this target.
"""

from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.common import aggregate_q, get_flat_batch
from utils.activation import get_activation
from utils.flax_utils import TrainState, expectile_loss, target_update
from utils.networks import ActorFlowField, Value


class EDPAgent(flax.struct.PyTreeNode):
    """EDP: Efficient Diffusion Policy (Kang et al. 2023).

    Avoids BPTT by approximating the clean action with a single Euler step:
    a_eval = clip(a_t + (1-t)*v_theta(s, a_t, t), -1, 1). The actor is trained to
    directly maximize Q(s, a_eval) with a BC regularization term:
    L_actor = -Q(s, a_eval) + bc_weight * ||v_pred - v_target||^2.
    Q-maximization is built into the actor training; inference is standard flow denoising.
    """

    support_guidance = False

    rng: Any
    policy: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def _aggregate_q(self, qs):
        return aggregate_q(qs, self.config)

    def _get_flat_batch(self, batch):
        return get_flat_batch(batch, self.config)

    def policy_loss(self, batch, policy_params=None, rng=None):
        """Flow matching BC loss + direct Q maximization via one-step approx."""
        if rng is None:
            rng = self.rng
        if policy_params is None:
            policy_params = self.policy.params

        eps_rng, time_rng = jax.random.split(rng, 2)
        if self.config.get("action_chunking", False):
            x1 = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            x1 = batch["actions"][..., 0, :]
        observations = batch["observations"]
        x0 = jax.random.normal(eps_rng, x1.shape)
        t = (
            jax.random.randint(
                time_rng, (x1.shape[0],), 0, self.config["denoise_steps"] + 1
            ).astype(jnp.float32)
            / self.config["denoise_steps"]
        )
        tv = t[..., None]
        x_t = x0 * (1 - tv) + x1 * tv
        vel = x1 - x0

        v_pred = self.policy(observations, x_t, t, params=policy_params)
        bc_loss = jnp.mean((vel - v_pred) ** 2)

        # One-step denoised action — gradient flows through v_pred to policy_params
        a_eval = jnp.clip(x_t + (1 - tv) * v_pred, -1, 1)
        qs = self.target_critic(observations, a_eval)
        q = self._aggregate_q(qs)
        q_loss = -q.mean()

        actor_loss = q_loss + self.config["bc_weight"] * bc_loss
        return actor_loss, {
            "bc_loss": bc_loss,
            "q_loss": q_loss,
            "actor_q": q.mean(),
        }

    def critic_loss(self, batch, critic_params=None):
        """IQL critic (Q) loss."""
        H = self.config.get("horizon_length", 1)
        batch_actions, next_obs, rewards, masks, valid_w = self._get_flat_batch(batch)
        next_v = self.value(next_obs)
        target_q = rewards + (self.config["discount"] ** H) * masks * next_v
        qs = self.critic(batch["observations"], batch_actions, params=critic_params)
        critic_loss = (((qs - target_q[None]) ** 2) * valid_w).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q": qs[0].mean(),
        }

    def value_loss(self, batch, value_params=None):
        """IQL value (V) loss with expectile regression."""
        batch_actions, _, _, _, valid_w = self._get_flat_batch(batch)
        qs = self.target_critic(batch["observations"], batch_actions)
        q = self._aggregate_q(qs)
        v = self.value(batch["observations"], params=value_params)
        value_loss = (expectile_loss(q - v, self.config["expectile"]) * valid_w).mean()
        return value_loss, {
            "value_loss": value_loss,
            "v": v.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, additional_agents=None):
        _ = additional_agents
        if rng is None:
            rng = self.rng
        policy_params = grad_params if grad_params is not None else self.policy.params
        actor_loss, policy_info = self.policy_loss(
            batch, policy_params=policy_params, rng=rng
        )
        critic_loss, critic_info = self.critic_loss(batch)
        value_loss, value_info = self.value_loss(batch)
        info = {}
        for k, v in policy_info.items():
            info[f"policy/{k}"] = v
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v
        for k, v in value_info.items():
            info[f"value/{k}"] = v
        return actor_loss + critic_loss + value_loss, info

    @jax.jit
    def update(self, batch):
        new_rng, policy_rng = jax.random.split(self.rng, 2)

        new_policy, policy_info = self.policy.apply_loss_fn(
            loss_fn=lambda p: self.policy_loss(batch, p, rng=policy_rng)
        )
        new_critic, critic_info = self.critic.apply_loss_fn(
            loss_fn=lambda p: self.critic_loss(batch, p)
        )
        new_target_critic = target_update(
            self.critic, self.target_critic, self.config["tau"]
        )
        new_value, value_info = self.value.apply_loss_fn(
            loss_fn=lambda p: self.value_loss(batch, p)
        )

        return self.replace(
            rng=new_rng,
            policy=new_policy,
            critic=new_critic,
            target_critic=new_target_critic,
            value=new_value,
        ), {**policy_info, **critic_info, **value_info}

    @partial(jax.jit, static_argnames=["rejection_sampling"])
    def sample_actions(
        self,
        observations: jnp.ndarray,
        *,
        seed: Any,
        rejection_sampling: int = 1,
    ) -> jnp.ndarray:
        """Standard flow matching denoising (Q maximization is baked in during training)."""

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
            v = self.policy(observations, x, ti)
            return x + v * dt, None

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
        rng, policy_key, critic_key, value_key = jax.random.split(rng, 4)

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
            policy_def, policy_params, tx=optax.adam(learning_rate=config["actor_lr"])
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

        config_dict = flax.core.FrozenDict(**config)
        return cls(
            rng,
            policy=policy,
            critic=critic,
            target_critic=target_critic,
            value=value,
            config=config_dict,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="edp",
            # Common hyperparameters.
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_lr=3e-4,
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
            denoise_steps=10,
            # EDP-specific hyperparameters.
            bc_weight=1.0,  # Weight on the BC flow matching loss relative to Q maximization.
        )
    )
    return config
