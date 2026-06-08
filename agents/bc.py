import math
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import TrainState
from utils.networks import MLP


def timestep_embedding(t, emb_size=16, max_period=10000):
    """t is between [0, 1]."""
    t = jax.lax.convert_element_type(t, jnp.float32)
    t = t * max_period
    dim = emb_size
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
    )
    args = t[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return embedding


class FlowMatchingPolicy(nn.Module):
    """Flow matching policy for behavior cloning."""

    hidden_dims: Any
    action_dim: int
    mlp_kwargs: dict = flax.struct.field(pytree_node=False)

    @nn.compact
    def __call__(self, obs, noised_action, t):
        t_embedding = timestep_embedding(t)
        concat_input = jnp.concatenate([obs, noised_action, t_embedding], axis=-1)
        outputs = MLP(self.hidden_dims, activate_final=True, **self.mlp_kwargs)(
            concat_input
        )
        v = nn.Dense(self.action_dim)(outputs)
        return v


class BCAgent(flax.struct.PyTreeNode):
    """Flow matching behavior cloning (BC) agent.

    Trains a flow policy via the flow matching objective to imitate the dataset
    distribution.
    """

    rng: Any
    policy: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    def policy_loss(self, batch, policy_params=None, rng=None):
        """Compute flow matching behavior cloning loss."""
        if rng is None:
            rng = self.rng
        if policy_params is None:
            policy_params = self.policy.params

        eps_rng, time_rng = jax.random.split(rng, 2)

        if self.config["action_chunking"]:
            actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            actions = batch["actions"][..., 0, :]
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

        # Predict velocity
        pred_vel = self.policy(batch["observations"], x_t, t, params=policy_params)
        bc_loss = jnp.mean((vel - pred_vel) ** 2)

        return bc_loss, {
            "bc_loss": bc_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, additional_agents={}):
        """Compute the total loss for compatibility with main.py evaluation."""
        if rng is None:
            rng = self.rng
        policy_params = grad_params if grad_params is not None else self.policy.params
        loss, info = self.policy_loss(batch, policy_params=policy_params, rng=rng)
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the policy using flow matching behavior cloning."""
        new_rng, policy_rng = jax.random.split(self.rng)

        def policy_loss_fn(policy_params):
            return self.policy_loss(batch, policy_params, rng=policy_rng)

        new_policy, policy_info = self.policy.apply_loss_fn(loss_fn=policy_loss_fn)

        return self.replace(rng=new_rng, policy=new_policy), policy_info

    @jax.jit
    def sample_actions(self, observations: jnp.ndarray, *, seed: Any) -> jnp.ndarray:
        """Sample actions from the flow matching policy."""
        observations = observations[None] if observations.ndim == 1 else observations

        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        x = jax.random.normal(seed, (observations.shape[0], full_action_dim))
        dt = 1.0 / self.config["denoise_steps"]

        def step(x, t):
            ti = jnp.ones((x.shape[0],)) * (t / self.config["denoise_steps"])
            v = self.policy(observations, x, ti)
            x = x + v * dt
            return x, None

        actions, _ = jax.lax.scan(
            step,
            x,
            jnp.arange(self.config["denoise_steps"]),
            length=self.config["denoise_steps"],
        )

        actions = actions[0] if observations.shape[0] == 1 else actions
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config,
    ):
        """Create a new flow matching BC agent."""
        rng = jax.random.PRNGKey(seed)
        rng, policy_key = jax.random.split(rng)

        if config["horizon_length"] > 1:
            assert config["action_chunking"]

        action_dim = ex_actions.shape[-1]
        config = dict(config)  # Make a mutable copy
        config["action_dim"] = action_dim
        full_action_dim = (
            action_dim * config["horizon_length"]
            if config["action_chunking"]
            else action_dim
        )

        # Get activation function
        try:
            activation_fn = getattr(nn, config["activation"])
        except:
            activation_fn = nn.gelu

        # Create flow matching policy network
        policy_def = FlowMatchingPolicy(
            config["hidden_dims"],
            full_action_dim,
            mlp_kwargs=dict(
                activation=activation_fn, layer_norm=config["use_layer_norm"]
            ),
        )
        ex_full_actions = jnp.zeros((ex_actions.shape[0], full_action_dim))
        policy_params = policy_def.init(
            policy_key, ex_observations, ex_full_actions, jnp.zeros(ex_actions.shape[0])
        )["params"]
        policy = TrainState.create(
            policy_def, policy_params, tx=optax.adam(learning_rate=config["bc_lr"])
        )

        config_dict = flax.core.FrozenDict(**config)
        return cls(rng, policy=policy, config=config_dict)


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="bc",
            # Common hyperparameters.
            batch_size=256,
            hidden_dims=(512, 512, 512, 512),
            bc_lr=3e-4,
            use_layer_norm=1,
            activation="gelu",
            # Action chunking: set horizon_length > 1 and action_chunking=True together.
            horizon_length=1,
            action_chunking=False,
            discount=1.0,  # Placeholder; BC has no discounting but main.py uses this field.
            # BC-specific hyperparameters.
            denoise_steps=10,  # Number of flow-matching denoising steps.
        )
    )
    return config
