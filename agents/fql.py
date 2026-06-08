import copy
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.activation import get_activation
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, expectile_loss, nonpytree_field
from utils.networks import ActorFlowField, Value


class FQLAgent(flax.struct.PyTreeNode):
    """FQL: Flow Q-Learning (Park et al. 2025).

    Distills a multi-step flow BC policy into a one-step policy Omega(s, z), then
    trains Omega to maximize Q. The one-step policy loss balances proximity to the BC
    flow distribution (2-Wasserstein regularization) against Q-value maximization:
    L(omega) = E_z [alpha * ||Omega(s,z) - ODE(f_theta, z)||^2 - Q(s, Omega(s,z))].
    Avoids backpropagating through the multi-step denoising process entirely.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _get_batch_actions(self, batch):
        if self.config["action_chunking"]:
            return jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        return batch["actions"][..., 0, :]

    def ddpg_critic_loss(self, batch, grad_params, rng):
        """DDPG-style critic loss: Q bootstraps from next action samples."""
        H = self.config["horizon_length"]
        batch_actions = self._get_batch_actions(batch)
        next_obs = batch["next_observations"][..., -1, :]
        rewards = batch["rewards"][..., -1]
        masks = batch["masks"][..., -1]
        valid_w = batch["valid"][..., -1]

        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(next_obs, seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select("target_critic")(next_obs, actions=next_actions)
        if self.config["q_agg"] == "min":
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = rewards + (self.config["discount"] ** H) * masks * next_q

        q = self.network.select("critic")(
            batch["observations"], actions=batch_actions, params=grad_params
        )
        critic_loss = (jnp.square(q - target_q) * valid_w).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def iql_critic_loss(self, batch, grad_params, rng):
        """IQL-style critic loss: Q regresses onto r + gamma^H * V(s')."""
        H = self.config["horizon_length"]
        batch_actions = self._get_batch_actions(batch)

        next_v = self.network.select("value")(batch["next_observations"][..., -1, :])
        target_q = (
            batch["rewards"][..., -1]
            + (self.config["discount"] ** H) * batch["masks"][..., -1] * next_v
        )
        valid_w = batch["valid"][..., -1]

        q = self.network.select("critic")(
            batch["observations"], actions=batch_actions, params=grad_params
        )
        critic_loss = (jnp.square(q - target_q) * valid_w).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def value_loss(self, batch, grad_params, rng):
        """Expectile regression of V onto Q (IQL mode only)."""
        valid_w = batch["valid"][..., -1]
        batch_actions = self._get_batch_actions(batch)
        qs = self.network.select("target_critic")(
            batch["observations"], actions=batch_actions
        )
        q = qs.min(axis=0)
        v = self.network.select("value")(batch["observations"], params=grad_params)
        val_loss = (expectile_loss(q - v, self.config["expectile"]) * valid_w).mean()
        return val_loss, {
            "value_loss": val_loss,
            "v_mean": v.mean(),
            "v_min": v.min(),
            "v_max": v.max(),
        }

    def critic_loss(self, batch, grad_params, rng):
        """Dispatch to DDPG or IQL critic loss."""
        if self.config["critic_loss_type"] == "iql":
            return self.iql_critic_loss(batch, grad_params, rng)
        return self.ddpg_critic_loss(batch, grad_params, rng)

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_actions = self._get_batch_actions(batch)
        valid_w = batch["valid"][..., -1]

        batch_size = batch_actions.shape[0]
        full_action_dim = batch_actions.shape[-1]
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, full_action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_bc_flow")(
            batch["observations"], x_t, t, params=grad_params
        )
        bc_flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * valid_w)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, full_action_dim))
        target_flow_actions = self.compute_flow_actions(
            batch["observations"], noises=noises
        )
        actor_actions = self.network.select("actor_onestep_flow")(
            batch["observations"], noises, params=grad_params
        )
        distill_loss = jnp.mean(
            jnp.square(actor_actions - target_flow_actions).mean(axis=-1) * valid_w
        )

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select("critic")(batch["observations"], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -(q * valid_w).mean()
        if self.config["normalize_q_loss"]:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q * valid_w).mean())
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = bc_flow_loss + self.config["alpha"] * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch["observations"], seed=rng)
        mse = jnp.mean((actions - batch_actions) ** 2)

        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "q": q.mean(),
            "mse": mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        val_loss = 0.0
        if self.config["critic_loss_type"] == "iql":
            val_loss, value_info = self.value_loss(batch, grad_params, value_rng)
            for k, v in value_info.items():
                info[f"value/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + val_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        **kwargs,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                full_action_dim,
            ),
        )
        actions = self.network.select("actor_onestep_flow")(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config["encoder"] is not None:
            observations = self.network.select("actor_bc_flow_encoder")(observations)
        actions = noises
        # Euler method.
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            vels = self.network.select("actor_bc_flow")(
                observations, actions, t, is_encoded=True
            )
            actions = actions + vels / self.config["flow_steps"]
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        H = config["horizon_length"]
        if config["action_chunking"]:
            ex_full_actions = jnp.concatenate([ex_actions] * H, axis=-1)
        else:
            ex_full_actions = ex_actions
        full_action_dim = ex_full_actions.shape[-1]
        ex_full_times = ex_full_actions[..., :1]

        # Define encoders.
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor_bc_flow"] = encoder_module()
            encoders["actor_onestep_flow"] = encoder_module()
            if config.get("critic_loss_type", "ddpg") == "iql":
                encoders["value"] = encoder_module()

        activation_fn = get_activation(config["activation"])
        print(f"Using activation function: {activation_fn}")

        # Define networks.
        critic_def = Value(
            network_class=config["value_network_class"],
            network_kwargs={
                **config["value_network_kwargs"],
                "activation": activation_fn,
            },
            num_ensembles=2,
            encoder=encoders.get("critic"),
        )
        actor_mlp_kwargs = dict(
            activation=activation_fn,
            layer_norm=config["actor_layer_norm"],
        )
        actor_bc_flow_def = ActorFlowField(
            config["actor_hidden_dims"],
            full_action_dim,
            mlp_kwargs=actor_mlp_kwargs,
            encoder=encoders.get("actor_bc_flow"),
            time_embedding="raw",
        )
        actor_onestep_flow_def = ActorFlowField(
            config["actor_hidden_dims"],
            full_action_dim,
            mlp_kwargs=actor_mlp_kwargs,
            encoder=encoders.get("actor_onestep_flow"),
            time_embedding="raw",
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_full_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_full_actions),
            ),
            actor_bc_flow=(
                actor_bc_flow_def,
                (ex_observations, ex_full_actions, ex_full_times),
            ),
            actor_onestep_flow=(
                actor_onestep_flow_def,
                (ex_observations, ex_full_actions),
            ),
        )
        if encoders.get("actor_bc_flow") is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info["actor_bc_flow_encoder"] = (
                encoders.get("actor_bc_flow"),
                (ex_observations,),
            )
        if config.get("critic_loss_type", "ddpg") == "iql":
            value_def = Value(
                network_class=config["value_network_class"],
                network_kwargs={
                    **config["value_network_kwargs"],
                    "activation": activation_fn,
                },
                num_ensembles=1,
                encoder=encoders.get("value"),
            )
            network_info["value"] = (value_def, (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config(variant=None):
    config = ml_collections.ConfigDict(
        dict(
            agent_name="fql",
            ob_dims=ml_collections.config_dict.placeholder(list),  # Set automatically.
            action_dim=ml_collections.config_dict.placeholder(
                int
            ),  # Set automatically.
            # Common hyperparameters.
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_network_class="MLP",
            value_network_kwargs=dict(
                hidden_dims=(512, 512, 512, 512),
                layer_norm=True,
            ),
            activation="gelu",
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            # n-step returns & action chunking.
            horizon_length=1,
            action_chunking=False,
            # RL hyperparameters.
            discount=0.99,
            tau=0.005,
            q_agg="mean",  # "mean" or "min".
            flow_steps=10,
            dataset_action_clip_eps=None,  # Clip dataset actions to [-1+eps, 1-eps]. None disables clipping.
            # Critic loss type.
            critic_loss_type="ddpg",  # "ddpg" (Q-bootstrap) or "iql" (value-bootstrap).
            expectile=0.9,  # Expectile for value regression (used when critic_loss_type="iql").
            # FQL-specific hyperparameters.
            alpha=300.0,  # BC coefficient (needs tuning per environment).
            normalize_q_loss=False,  # Normalize the Q loss by its absolute mean.
        )
    )
    return config
