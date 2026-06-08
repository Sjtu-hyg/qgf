from functools import partial
from typing import Any, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from agents.common import aggregate_q, get_flat_batch
from utils.activation import get_activation
from utils.flax_utils import TrainState, expectile_loss, supply_rng, target_update
from utils.networks import MLP, ConditionalFlowField, Value


class CFGRLAgent(flax.struct.PyTreeNode):
    """CFGRL: Classifier-Free Guidance RL (Frans et al. 2025).

    Trains a flow policy conditioned on an optimality variable o ∈ {0, 1}, learning
    both a base BC velocity field and an optimality-conditioned velocity field.
    At test time, interpolates between the two via a guidance weight alpha:
    v(s, a_t, t) = v_BC(s, a_t, t) + guidance_weight * v_conditioned(s, a_t, t).
    """

    support_guidance = True

    rng: Any
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    actor: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def _aggregate_q(self, qs):
        return aggregate_q(qs, self.config)

    def _get_flat_batch(self, batch):
        return get_flat_batch(batch, self.config)

    def critic_loss(self, batch, critic_params=None):
        """Compute the IQL critic loss."""
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
        """Compute the IQL value loss."""
        batch_actions, _, _, _, valid_w = self._get_flat_batch(batch)
        qs = self.target_critic(batch["observations"], batch_actions)
        q = self._aggregate_q(qs)
        v = self.value(batch["observations"], params=value_params)
        value_loss = (expectile_loss(q - v, self.config["expectile"]) * valid_w).mean()
        return value_loss, {
            "value_loss": value_loss,
            "v": v.mean(),
            "v_min": v.min(),
            "v_max": v.max(),
        }

    def actor_loss(self, batch, actor_params=None, rng=None, additional_agents={}):
        """Compute the actor loss."""
        if rng is None:
            rng = self.rng
        eps_rng, time_rng = jax.random.split(rng, 2)

        if self.config.get("action_chunking", False):
            B = batch["actions"].shape[0]
            actions = jnp.reshape(batch["actions"], (B, -1))
        else:
            actions = batch["actions"][..., 0, :]
        observations = batch["observations"]

        # Compute q and v
        if self.config.get("critic_loss_type", "iql") == "ddpg":
            sampled_action = self.sample_actions(observations, seed=rng)
            v = self.target_critic(observations, sampled_action)
            v = self._aggregate_q(v)
        else:
            v = self.value(observations)
        if self.config["target_extraction"]:
            qs = self.target_critic(observations, actions)
        else:
            qs = self.critic(observations, actions)
        q = self._aggregate_q(qs)
        adv = q - v

        loss_weight = (adv > self.config["adv_threshold"]).astype(jnp.float32)

        # Compute noised actions
        x1 = actions
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

        # Positive samples
        idx_positive = jnp.ones((x1.shape[0],), dtype=jnp.int32)
        pred_vel_positive = self.actor(
            observations, idx_positive, x_t, t, params=actor_params
        )
        vel_loss_positive = (
            jnp.mean(((vel - pred_vel_positive) ** 2), axis=-1) * loss_weight
        )

        # Unconditional samples
        idx_uncond = jnp.zeros((x1.shape[0],), dtype=jnp.int32)
        pred_vel_uncond = self.actor(
            observations, idx_uncond, x_t, t, params=actor_params
        )
        vel_loss_uncond = jnp.mean(((vel - pred_vel_uncond) ** 2), axis=-1)

        actor_loss = jnp.mean(vel_loss_positive + vel_loss_uncond * 0.1)

        # Compute additional metrics
        effective_batch_size = jnp.sum(loss_weight) ** 2 / (
            jnp.sum(loss_weight**2) + 1e-8
        )
        sorted_weights = jnp.sort(loss_weight)
        total_mass = jnp.sum(sorted_weights)
        top_10_mass = sorted_weights[-int(0.1 * len(sorted_weights)) :].sum() / (
            total_mass + 1e-8
        )

        v_cond_norms = jnp.linalg.norm(pred_vel_positive, axis=-1)
        v_uncond_norms = jnp.linalg.norm(pred_vel_uncond, axis=-1)
        dot_products = jnp.sum(pred_vel_positive * pred_vel_uncond, axis=-1)
        cosine_similarities = dot_products / (v_cond_norms * v_uncond_norms + 1e-8)

        return actor_loss, {
            "adv": adv,
            "adv_mean": adv.mean(),
            "adv_min": adv.min(),
            "adv_max": adv.max(),
            "actor_q": q.mean(),
            "actor_loss": actor_loss,
            "actor_positive_ratio": ((q - v) > self.config["adv_threshold"]).mean(),
            "actor_loss_positive": vel_loss_positive.mean(),
            "actor_loss_uncond": vel_loss_uncond.mean(),
            "actor_losses": vel_loss_positive + vel_loss_uncond * 0.1,
            "effective_batch_size": effective_batch_size,
            "loss_weights_top_10_mass": top_10_mass,
            "loss_weights_max": jnp.max(loss_weight),
            "loss_weights": loss_weight,
            "v_cond_v_uncond_cosine_sim": cosine_similarities.mean(),
            "v_cond_v_uncond_cosine_sim_min": cosine_similarities.min(),
            "v_cond_v_uncond_cosine_sim_max": cosine_similarities.max(),
            "v_cond_norm_mean": v_cond_norms.mean(),
            "v_uncond_norm_mean": v_uncond_norms.mean(),
        }

    @jax.jit
    def update(agent, batch, additional_agents={}):
        new_rng, actor_rng = jax.random.split(agent.rng, 2)

        def critic_loss_fn(critic_params):
            return agent.critic_loss(batch, critic_params)

        def value_loss_fn(value_params):
            return agent.value_loss(batch, value_params)

        def actor_loss_fn(actor_params):
            return agent.actor_loss(
                batch,
                actor_params=actor_params,
                rng=actor_rng,
                additional_agents=additional_agents,
            )

        new_critic, critic_info = agent.critic.apply_loss_fn(loss_fn=critic_loss_fn)
        new_target_critic = target_update(
            agent.critic, agent.target_critic, agent.config["tau"]
        )
        new_value, value_info = agent.value.apply_loss_fn(loss_fn=value_loss_fn)
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn)

        return agent.replace(
            rng=new_rng,
            critic=new_critic,
            target_critic=new_target_critic,
            value=new_value,
            actor=new_actor,
        ), {**critic_info, **value_info, **actor_info}

    @partial(jax.jit, static_argnames=["rejection_sampling"])
    def sample_actions(
        agent,
        observations: np.ndarray,
        *,
        seed: Any,
        guidance_weight=1.0,
        rejection_sampling=1,
    ) -> jnp.ndarray:
        has_batch_dim = observations.ndim == 2
        observations = observations if has_batch_dim else observations[None]

        batch_size = observations.shape[0]
        if rejection_sampling > 1:
            observations = jnp.repeat(observations, rejection_sampling, axis=0)

        ad = agent.config["action_dim"]
        H = agent.config.get("horizon_length", 1)
        full_action_dim = ad * (H if agent.config.get("action_chunking", False) else 1)
        x = jax.random.normal(seed, (observations.shape[0], full_action_dim))
        dt = 1.0 / agent.config["denoise_steps"]
        idx_positive = jnp.ones((x.shape[0],), dtype=jnp.int32)
        idx_uncond = jnp.zeros((x.shape[0],), dtype=jnp.int32)

        def step(x, t):
            ti = jnp.ones((x.shape[0],)) * (t / agent.config["denoise_steps"])
            v_positive = agent.actor(observations, idx_positive, x, ti)
            v_uncond = agent.actor(observations, idx_uncond, x, ti)
            v = v_uncond + guidance_weight * (v_positive - v_uncond)
            x = x + v * dt
            return x, None

        actions, _ = jax.lax.scan(
            step,
            x,
            jnp.arange(agent.config["denoise_steps"]),
            length=agent.config["denoise_steps"],
        )
        actions = jnp.clip(actions, -1, 1)

        if rejection_sampling > 1:
            q = agent.critic(observations, actions)
            q = agent._aggregate_q(q)
            q = q.reshape((batch_size, rejection_sampling))
            actions = actions.reshape(
                (batch_size, rejection_sampling, *actions.shape[1:])
            )
            actions = actions[jnp.arange(batch_size), jnp.argmax(q, axis=1)]

        if not has_batch_dim:
            actions = actions[0]

        return actions

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, additional_agents={}):
        """Compute the total loss for compatibility with main.py evaluation."""
        if rng is None:
            rng = self.rng

        critic_loss, critic_info = self.critic_loss(batch)
        value_loss, value_info = self.value_loss(batch)
        actor_loss, actor_info = self.actor_loss(
            batch, rng=rng, additional_agents=additional_agents
        )

        total_loss = critic_loss + value_loss + actor_loss
        info = {**critic_info, **value_info, **actor_info}

        return total_loss, info

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config,
    ):
        """Create a CFGRLAgent."""
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = ex_actions.shape[-1]
        H = config.get("horizon_length", 1)
        if config.get("action_chunking", False):
            ex_full_actions = jnp.concatenate([ex_actions] * H, axis=-1)
        else:
            ex_full_actions = ex_actions
        full_action_dim = ex_full_actions.shape[-1]
        config = dict(config)  # Make a mutable copy
        config["action_dim"] = action_dim
        # Get activation function
        activation_fn = get_activation(config["activation"])
        print(f"Using activation function: {activation_fn}")

        actor_def = ConditionalFlowField(
            config["actor_hidden_dims"],
            full_action_dim,
            mlp_kwargs=dict(
                activation=activation_fn, layer_norm=config["actor_layer_norm"]
            ),
        )

        # create actor
        actor_tx = optax.adam(learning_rate=config["actor_lr"])
        ex_idx = jnp.ones((ex_actions.shape[0],), dtype=jnp.int32)
        ex_t = jnp.zeros(
            ex_actions.shape[0],
        )
        actor_params = actor_def.init(
            actor_key,
            ex_observations,
            ex_idx,
            ex_full_actions,
            ex_t,
        )["params"]
        actor = TrainState.create(actor_def, actor_params, tx=actor_tx)

        # create critic
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
            critic=critic,
            target_critic=target_critic,
            value=value,
            actor=actor,
            config=config_dict,
        )


def get_config(variant: Optional[str] = None):
    config = ml_collections.ConfigDict(
        dict(
            agent_name="cfgrl",
            # Common hyperparameters.
            batch_size=256,
            actor_lr=3e-4,
            value_lr=3e-4,
            critic_lr=3e-4,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=True,
            value_network_class="MLP",
            value_network_kwargs=dict(
                hidden_dims=(512, 512, 512, 512),
                layer_norm=True,
            ),
            activation="gelu",
            # n-step returns & action chunking.
            horizon_length=1,
            action_chunking=False,
            # RL hyperparameters.
            num_qs=2,
            q_aggregation="min",  # "min" or "mean".
            discount=0.99,
            expectile=0.9,
            tau=0.005,
            denoise_steps=10,
            # CFGRL-specific hyperparameters.
            target_extraction=1,  # Use target critic (1) or online critic (0) for advantage computation.
            o_embedding="linear",  # Optimality embedding type: "linear" or "sinusoidal".
            dataset_action_clip_eps=None,  # Clip dataset actions to [-1+eps, 1-eps]. None disables clipping.
            adv_threshold=0.0,  # Minimum advantage threshold for filtering training samples.
        )
    )
    return config
