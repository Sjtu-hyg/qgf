import copy
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.diffusion import (
    DDPM,
    FourierFeatures,
    cosine_beta_schedule,
    vp_beta_schedule,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, Value


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


class DCGQLAgent(flax.struct.PyTreeNode):
    """Diffusion Classifier-Guidance Q-Learning (DCGQL).

    Unifies two prior works under a single agent:
    - QSM (Q-Score Matching): actor loss matches Q-function gradient to predicted noise.
    - DAC (Diffusion Actor-Critic): actor loss is a weighted dot product between
        Q-gradient and predicted noise, scaled by the noise level.

    Select the variant via ``config.actor_loss_type`` ("qsm" or "dac").

    References:
    - QSM: https://github.com/escontra/score_matching_rl
    - DAC: https://github.com/Fang-Lin93/DAC
    - Original unified implementation: ~/repos/qam/agents/dcgql.py
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    betas: Any
    alphas: Any
    alpha_hats: Any

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )
        else:
            batch_actions = batch["actions"][..., 0, :]
        valid_w = batch["valid"][..., -1]

        qs = self.network.select("target_critic")(batch["observations"], batch_actions)
        q = qs.min(axis=0)
        v = self.network.select("value")(batch["observations"], params=grad_params)
        value_loss = (
            self.expectile_loss(q - v, q - v, self.config["expectile"]) * valid_w
        ).mean()
        return value_loss, {"value_loss": value_loss, "v_mean": v.mean()}

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

        if self.config["critic_loss_type"] == "iql":
            next_q = self.network.select("value")(next_obs)
        else:
            next_actions = self.sample_actions(next_obs, seed=rng)
            next_qs = self.network.select("target_critic")(next_obs, next_actions)
            next_q = next_qs.mean(axis=0) - next_qs.std(axis=0) * self.config["rho"]
        target_q = rewards + (self.config["discount"] ** H) * masks * next_q

        q = self.network.select("critic")(
            batch["observations"], batch_actions, params=grad_params
        )
        critic_loss = (jnp.square(q - target_q) * valid_w).mean()
        return critic_loss, {"critic_loss": critic_loss, "q_mean": q.mean()}

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )
        else:
            batch_actions = batch["actions"][..., 0, :]
        valid_w = batch["valid"][..., -1]
        valid_w_2d = valid_w[..., None]

        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        t = jax.random.randint(
            t_rng, batch_actions.shape[:-1], 1, self.config["diffusion_steps"] + 1
        )
        noise_sample = jax.random.normal(noise_rng, batch_actions.shape)

        alpha_hats = self.alpha_hats[t]
        t_expanded = jnp.expand_dims(t, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch_actions + alpha_2 * noise_sample

        if self.config["use_target_critic_grad"]:
            q_grad_fn = jax.grad(
                lambda a: self.network.select("target_critic")(batch["observations"], a)
                .mean(axis=0)
                .sum()
            )
        else:
            q_grad_fn = jax.grad(
                lambda a: self.network.select("critic")(batch["observations"], a)
                .mean(axis=0)
                .sum()
            )

        eps_pred = self.network.select("actor")(
            batch["observations"], noisy_actions, t_expanded, params=grad_params
        )

        bc_loss = (jnp.square(noise_sample - eps_pred).mean(axis=-1) * valid_w).mean()

        if self.config["actor_loss_type"] == "qsm":
            actor_loss = (
                jnp.square(
                    -self.config["inv_temp"] * q_grad_fn(noisy_actions) - eps_pred
                ).mean(axis=-1)
                * valid_w
            ).mean()
        elif self.config["actor_loss_type"] == "dac":
            actor_loss = (
                alpha_2 * q_grad_fn(noisy_actions) * eps_pred * valid_w_2d
            ).mean()
        else:
            raise ValueError(
                f"Unknown actor_loss_type: {self.config['actor_loss_type']}"
            )

        total_loss = bc_loss * self.config["alpha"] + actor_loss
        return total_loss, {"actor_loss": actor_loss, "bc_loss": bc_loss}

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

        if self.config["critic_loss_type"] == "iql":
            value_loss, value_info = self.value_loss(batch, grad_params)
            for k, v in value_info.items():
                info[f"value/{k}"] = v
        else:
            value_loss = 0.0

        return critic_loss + actor_loss + value_loss, info

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

    def ddpm_sampler(self, rng, observations, noise):
        batch_size = observations.shape[0]
        input_time_proto = jnp.ones((*noise.shape[:-1], 1))

        def fn(input_tuple, t):
            current_x, rng_ = input_tuple
            input_time = input_time_proto * t

            eps_pred = self.network.select("actor")(observations, current_x, input_time)

            x0_hat = (
                1
                / jnp.sqrt(self.alpha_hats[t])
                * (current_x - jnp.sqrt(1 - self.alpha_hats[t]) * eps_pred)
            )
            if self.config["clip_sampler_before"]:
                x0_hat = jnp.clip(x0_hat, -1, 1)
                current_x = (
                    1
                    / (1 - self.alpha_hats[t])
                    * (
                        jnp.sqrt(self.alpha_hats[t - 1]) * (1 - self.alphas[t]) * x0_hat
                        + jnp.sqrt(self.alphas[t])
                        * (1 - self.alpha_hats[t - 1])
                        * current_x
                    )
                )
            else:
                current_x = x0_hat

            rng_, key_ = jax.random.split(rng_, 2)
            z = jax.random.normal(key_, shape=(batch_size,) + current_x.shape[1:])
            sigmas_t = jnp.sqrt(1 - self.alphas[t])
            current_x = current_x + (t > 1) * (sigmas_t * z)

            if self.config["clip_sampler_after"]:
                current_x = jnp.clip(current_x, -1.0, 1.0)
            return (current_x, rng_), ()

        rng, denoise_key = jax.random.split(rng, 2)
        output_tuple, () = jax.lax.scan(
            fn,
            (noise, denoise_key),
            jnp.arange(self.config["diffusion_steps"], 0, -1),
            unroll=self.config["diffusion_steps"],
        )
        return output_tuple[0]

    @jax.jit
    def sample_actions(self, observations, *, seed, **kwargs):
        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )

        if observations.ndim == 1:
            observations = observations[None, :]

        noise_key, sampler_key = jax.random.split(seed)
        noise = jax.random.normal(noise_key, (observations.shape[0], full_action_dim))
        actions = self.ddpm_sampler(sampler_key, observations, noise)

        if actions.shape[0] == 1:
            actions = actions.squeeze(axis=0)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        action_dim = ex_actions.shape[-1]
        H = config["horizon_length"]
        if config["action_chunking"]:
            ex_full_actions = jnp.concatenate([ex_actions] * H, axis=-1)
        else:
            ex_full_actions = ex_actions
        full_action_dim = ex_full_actions.shape[-1]

        preprocess_time_cls = partial(
            FourierFeatures, output_size=config["time_dim"], learnable=True
        )
        cond_model_cls = partial(
            MLP,
            hidden_dims=config["actor_hidden_dims"],
            activation=mish,
            activate_final=False,
        )
        base_model_cls = partial(
            MLP,
            hidden_dims=tuple(list(config["actor_hidden_dims"]) + [full_action_dim]),
            activation=mish,
            layer_norm=config["actor_layer_norm"],
            activate_final=False,
        )

        actor_def = DDPM(
            time_preprocess_cls=preprocess_time_cls,
            cond_encoder_cls=cond_model_cls,
            reverse_encoder_cls=base_model_cls,
        )

        ex_times = jnp.zeros((ex_observations.shape[0], 1))
        critic_def = Value(
            network_class="MLP",
            network_kwargs=dict(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["value_layer_norm"],
            ),
            num_ensembles=config["num_qs"],
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_full_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_full_actions),
            ),
            actor=(actor_def, (ex_observations, ex_full_actions, ex_times)),
        )
        if config["critic_loss_type"] == "iql":
            value_def = Value(
                network_class="MLP",
                network_kwargs=dict(
                    hidden_dims=config["value_hidden_dims"],
                    layer_norm=config["value_layer_norm"],
                ),
                num_ensembles=1,
            )
            network_info["value"] = (value_def, (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.chain(
            optax.clip_by_global_norm(max_norm=config["clip_grad_norm"]),
            optax.adam(learning_rate=config["lr"]),
        )
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]

        beta_schedule = config["beta_schedule"]
        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(config["diffusion_steps"]))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, config["diffusion_steps"])
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(config["diffusion_steps"]))
        else:
            raise ValueError(f"Invalid beta schedule: {beta_schedule}")

        betas = jnp.concatenate([jnp.zeros((1,)), betas])
        alphas = 1 - betas
        alpha_hats = jnp.cumprod(alphas)

        config["action_dim"] = action_dim

        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            alphas=alphas,
            alpha_hats=alpha_hats,
            betas=betas,
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="dcgql",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            # Common hyperparameters.
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            actor_cond_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            value_layer_norm=True,
            # n-step returns.
            horizon_length=1,
            action_chunking=False,
            # RL hyperparameters.
            num_qs=10,
            rho=0.5,
            discount=0.99,
            tau=0.005,
            diffusion_steps=10,
            time_dim=64,
            beta_schedule="vp",
            best_of_n=1,
            # DCGQL-specific hyperparameters.
            # critic_loss_type: "ddpg" (Q-bootstrap) or "iql" (value-bootstrap).
            critic_loss_type="ddpg",
            # IQL expectile for value regression (used when critic_loss_type="iql").
            expectile=0.9,
            # actor_loss_type: "qsm" or "dac".
            actor_loss_type="qsm",
            # DAC: map to noise-free space, clip, then map back before each step.
            clip_sampler_before=False,
            # QSM: clip intermediate noisy actions after each step.
            clip_sampler_after=False,
            inv_temp=1.0,
            # Weight for BC loss (used by QSM).
            alpha=0.0,
            use_target_critic_grad=True,
            clip_grad_norm=1.0,
        )
    )
    return config
