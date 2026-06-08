import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorFlowField, Value


class FBRACAgent(flax.struct.PyTreeNode):
    """Flow-matching BRAC (FBRAC).

    Combines a flow-matching BC policy with direct Q-maximization via
    backpropagation through the Euler flow integration (BPTT).

    Total actor loss = BC flow loss + Q-maximization loss / alpha.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        H = self.config["horizon_length"]
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )
        else:
            batch_actions = batch["actions"][..., 0, :]
        next_obs = batch["next_observations"][..., -1, :]
        rewards = batch["rewards"][..., -1]
        masks = batch["masks"][..., -1]
        valid_w = batch["valid"][..., -1]

        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(next_obs, seed=sample_rng)

        next_qs = self.network.select("target_critic")(next_obs, next_actions)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        target_q = rewards + (self.config["discount"] ** H) * masks * next_q

        q = self.network.select("critic")(
            batch["observations"], batch_actions, params=grad_params
        )
        critic_loss = (jnp.square(q - target_q) * valid_w).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )
        else:
            batch_actions = batch["actions"][..., 0, :]
        valid_w = batch["valid"][..., -1]

        batch_size, full_action_dim = batch_actions.shape
        rng, x_rng, t_rng, noise_rng = jax.random.split(rng, 4)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, full_action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_flow")(
            batch["observations"], x_t, t, params=grad_params
        )
        bc_flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * valid_w)

        # Q-maximization loss via BPTT through the Euler flow solver.
        noises = jax.random.normal(noise_rng, (batch_size, full_action_dim))
        actor_actions = self.compute_flow_actions(
            batch["observations"], noises=noises, params=grad_params
        )
        qs = self.network.select("critic")(batch["observations"], actor_actions)
        q_loss = -jnp.mean(qs, axis=0).mean()

        actor_loss = bc_flow_loss + q_loss / self.config["alpha"]

        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss,
            "q_loss": q_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, *, seed, **kwargs):
        if observations.ndim == 1:
            observations = observations[None, :]

        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        noises = jax.random.normal(
            seed,
            (observations.shape[0], self.config["best_of_n"], full_action_dim),
        )
        observations_rep = jnp.repeat(
            observations[..., None, :], self.config["best_of_n"], axis=-2
        )
        actions = self.compute_flow_actions(observations_rep, noises)

        # Best-of-n selection.
        q = self.network.select("critic")(observations_rep, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bsize = indices.shape[0]
        actions = jnp.reshape(
            actions, (bsize, self.config["best_of_n"], full_action_dim)
        )[jnp.arange(bsize), indices, :]

        if actions.shape[0] == 1:
            actions = actions.squeeze(axis=0)
        return actions

    @jax.jit
    def compute_flow_actions(self, observations, noises, params=None):
        """Euler flow integration.  Pass ``params`` to enable BPTT."""
        actions = noises
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            if params is not None:
                vels = self.network.select("actor_flow")(
                    observations, actions, t, is_encoded=True, params=params
                )
            else:
                vels = self.network.select("actor_flow")(
                    observations, actions, t, is_encoded=True
                )
            actions = actions + vels / self.config["flow_steps"]
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        H = config["horizon_length"]
        if config["action_chunking"]:
            ex_full_actions = jnp.concatenate([ex_actions] * H, axis=-1)
        else:
            ex_full_actions = ex_actions
        full_action_dim = ex_full_actions.shape[-1]

        critic_def = Value(
            network_class="MLP",
            network_kwargs=dict(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["value_layer_norm"],
            ),
            num_ensembles=config["num_qs"],
        )
        actor_flow_def = ActorFlowField(
            config["actor_hidden_dims"],
            full_action_dim,
            mlp_kwargs=dict(layer_norm=config["actor_layer_norm"]),
            time_embedding="raw",
        )

        network_info = dict(
            actor_flow=(actor_flow_def, (ex_observations, ex_full_actions, ex_times)),
            critic=(critic_def, (ex_observations, ex_full_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_full_actions),
            ),
        )
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


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="fbrac",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            # Common hyperparameters.
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_hidden_dims=(512, 512, 512, 512),
            value_layer_norm=True,
            # n-step returns & action chunking.
            horizon_length=1,
            action_chunking=False,
            # RL hyperparameters.
            num_qs=10,
            rho=0.5,
            discount=0.99,
            tau=0.005,
            flow_steps=10,
            best_of_n=1,
            # FBRAC-specific hyperparameter.
            # alpha scales the BC loss relative to Q-maximization (larger = more BC).
            alpha=100.0,
        )
    )
    return config
