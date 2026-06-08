import copy

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.iql import IQLAgent
from typing_extensions import override
from utils.activation import get_activation
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import ActorFlowField, Value


class IQLDiffusionAgent(IQLAgent):
    """
    IQL agent with diffusion-based actor.

    For the actor loss, grad_log_pi is approximiated with the diffusion loss,
    which is the ELBO of the likelihood.
    """

    @override
    def actor_loss(self, batch, grad_params, rng=None, additional_agents={}):
        if rng is None:
            rng = self.rng
        rng, eps_rng, time_rng = jax.random.split(rng, 3)

        v = self.network.select("value")(batch["observations"])
        q1, q2 = self.network.select(
            "target_critic" if self.config["target_extraction"] else "critic"
        )(batch["observations"], actions=batch["actions"])
        q = jnp.minimum(q1, q2)
        adv = q - v
        exp_a = jnp.exp(adv * self.config["alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)

        # compute the diffusion loss
        x1 = batch["actions"]
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

        pred_vel = self.network.select("actor")(
            batch["observations"], x_t, t, params=grad_params
        )
        diffusion_loss = (vel - pred_vel) ** 2

        # weight the diffusion loss
        loss = (diffusion_loss.mean(axis=-1) * exp_a).mean()

        return loss, {
            "actor_loss": loss,
            "diffusion_loss": diffusion_loss.mean(),
            "q_mean": q.mean(),
            "q_abs_mean": jnp.abs(q).mean(),
            "adv": adv.mean(),
            "exp_a": exp_a.mean(),
        }

    @override
    def sample_actions(
        self,
        observations,
        seed=None,
        rejection_sampling=1,
    ):
        assert rejection_sampling == 1

        x = jax.random.normal(seed, (observations.shape[0], self.config["action_dim"]))
        dt = 1.0 / self.config["denoise_steps"]

        def step(x, t):
            ti = jnp.ones((x.shape[0],)) * (t / self.config["denoise_steps"])
            v = self.network.select("actor")(
                observations, x, ti, params=self.network.params
            )
            x = x + v * dt
            return x, None

        actions, _ = jax.lax.scan(
            step,
            x,
            jnp.arange(self.config["denoise_steps"]),
            length=self.config["denoise_steps"],
        )
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

        action_dim = ex_actions.shape[-1]
        config["action_dim"] = action_dim

        # Define encoders.
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["value"] = encoder_module()
            encoders["critic"] = encoder_module()
            encoders["actor"] = encoder_module()

        activation_fn = get_activation(config["activation"])
        print(f"Using activation function: {activation_fn}")

        # Define networks.
        value_def = Value(
            network_class=config["value_network_class"],
            network_kwargs={
                **config["value_network_kwargs"],
                "activation": activation_fn,
            },
            num_ensembles=1,
            encoder=encoders.get("value"),
        )
        critic_def = Value(
            network_class=config["value_network_class"],
            network_kwargs={
                **config["value_network_kwargs"],
                "activation": activation_fn,
            },
            num_ensembles=2,
            encoder=encoders.get("critic"),
        )
        actor_def = ActorFlowField(
            config["actor_hidden_dims"],
            action_dim,
            mlp_kwargs=dict(
                activation=activation_fn, layer_norm=config["actor_layer_norm"]
            ),
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(
                actor_def,
                (ex_observations, ex_actions, jnp.zeros(ex_actions.shape[0])),
            ),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_critic"] = params["modules_critic"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config(variant=None):
    config = ml_collections.ConfigDict(
        dict(
            agent_name="iql_diffusion",
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
            # RL hyperparameters.
            discount=0.99,
            tau=0.005,
            expectile=0.9,  # IQL expectile.
            denoise_steps=10,  # Number of denoising steps for the diffusion actor.
            # IQL-diffusion-specific hyperparameters.
            actor_loss="awr",  # Actor loss type: "awr" or "ddpgbc".
            alpha=1.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
            target_extraction=True,  # Use target critic (True) or online critic (False) for AWR advantages.
        )
    )
    return config
