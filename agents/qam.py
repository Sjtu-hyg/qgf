import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, expectile_loss, nonpytree_field
from utils.networks import MLP, ActorFlowField, LogParam, Value


class QAMAgent(flax.struct.PyTreeNode):
    """QAM: Q-learning with Adjoint Matching (Li et al. 2026).

    Incorporates the critic's action gradient into flow policy training without BPTT.
    Uses adjoint matching to target the KL-regularized optimal policy
    pi*(a|s) ∝ pi_BC(a|s) * exp(tau * Q(s,a)). The adjoint state g_t is propagated
    backwards through the *base* BC flow (not the evolving trained policy), avoiding
    the ill-conditioning that arises from backpropagating through the optimized policy.
    The actor loss matches the learned flow to the BC flow shifted by a scaled adjoint term.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _flat_batch_for_critic(self, batch):
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
        return batch_actions, next_obs, rewards, masks, valid_w

    def ddpg_critic_loss(self, batch, grad_params, rng):
        """DDPG-style TD critic loss."""
        H = self.config["horizon_length"]
        batch_actions, next_obs, rewards, masks, valid_w = self._flat_batch_for_critic(
            batch
        )

        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(next_obs, seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)
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

    def iql_critic_loss(self, batch, grad_params, rng):
        """IQL-style critic loss: Q regresses onto r + gamma * V(s')."""
        H = self.config["horizon_length"]
        batch_actions, next_obs, rewards, masks, valid_w = self._flat_batch_for_critic(
            batch
        )
        next_v = self.network.select("value")(next_obs)
        target_q = rewards + (self.config["discount"] ** H) * masks * next_v

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

    def value_loss(self, batch, grad_params, rng):
        """Expectile regression of V onto Q (IQL mode only)."""
        batch_actions, _, _, _, valid_w = self._flat_batch_for_critic(batch)
        qs = self.network.select("target_critic")(batch["observations"], batch_actions)
        q = qs.min(axis=0)
        v = self.network.select("value")(batch["observations"], params=grad_params)
        value_loss = (expectile_loss(q - v, self.config["expectile"]) * valid_w).mean()

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_min": v.min(),
            "v_max": v.max(),
        }

    def critic_loss(self, batch, grad_params, rng):
        """Dispatch to DDPG or IQL critic loss."""
        if self.config["critic_loss_type"] == "iql":
            return self.iql_critic_loss(batch, grad_params, rng)
        return self.ddpg_critic_loss(batch, grad_params, rng)

    @partial(jax.jit, static_argnames=("flow_steps"))
    def adj_matching(self, obs, rng, flow_steps=None):
        flow_steps = self.config["flow_steps"] if flow_steps is None else flow_steps

        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        x = jax.random.normal(rng, shape=obs.shape[:-1] + (action_dim,))

        actor_slow = self.network.select(
            "target_actor_slow" if self.config["target_actor"] else "actor_slow"
        )

        h = 1 / flow_steps
        xs = [x]
        ts = []
        for i, key in zip(range(flow_steps), jax.random.split(rng, flow_steps)):
            t = i / flow_steps * jnp.ones_like(x[..., 0:1])
            sigma = jnp.sqrt(2 * (1 - t + h) / (t + h))
            noise = jax.random.normal(key, x.shape)
            if i != flow_steps - 1:
                if self.config["residual"]:
                    v = self.network.select("actor_fast")(obs, x, t) + actor_slow(
                        obs, x, t
                    )
                else:
                    v = self.network.select("actor_fast")(obs, x, t)
                x = x + h * (2 * v - x / (t + h)) + jnp.sqrt(h) * sigma * noise
            else:  # use ODE integration for the last step following the adjoint-matching paper
                x = x + h * actor_slow(obs, x, t)

            xs.append(x)
            ts.append(t)

        # Compute the critic's action gradient as the adjoint state initialization
        critic_network = "target_critic" if self.config["use_target_grad"] else "critic"
        if self.config["clip_adj"]:
            grad_fn = jax.grad(
                lambda x, y: self.network.select(critic_network)(
                    x, jnp.clip(y, -1.0, 1.0)
                )
                .mean(axis=0)
                .sum(),
                1,
            )
        else:
            grad_fn = jax.grad(
                lambda x, y: self.network.select(critic_network)(x, y)
                .mean(axis=0)
                .sum(),
                1,
            )

        adj = -grad_fn(obs, xs[-1]) * self.config["inv_temp"]
        pre_adj_info = {
            "adj_max": jnp.abs(adj).max(),
            "adj_std": jnp.abs(adj).std(),
            "adj_mean": jnp.abs(adj).mean(),
        }
        adjs = []
        for i in reversed(range(flow_steps)):
            t = (i / flow_steps) * jnp.ones_like(x[..., 0:1])

            def fn(xi):
                return 2 * actor_slow(obs, xi, t + h) - xi / (t + h)

            vjp = jax.vjp(fn, xs[i])[1](adj)[0]
            adj = adj + h * vjp

            adjs.append(adj)
        return (
            jnp.stack(xs[:-1], axis=0),
            jnp.stack(list(reversed(adjs)), axis=0),
            jnp.stack(ts, axis=0),
            pre_adj_info,
        )

    def actor_loss(self, batch, grad_params, rng):
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(
                batch["actions"], (batch["actions"].shape[0], -1)
            )
        else:
            batch_actions = batch["actions"][..., 0, :]
        valid_w = batch["valid"][..., -1]

        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng, adj_rng, edit_rng = jax.random.split(rng, 5)

        ## BC flow-matching loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_slow")(
            batch["observations"], x_t, t, params=grad_params
        )
        flow_loss = jnp.mean(jnp.square(pred - vel).mean(axis=-1) * valid_w)
        actor_loss = flow_loss

        info = {}
        total_fast_loss = 0
        actor_slow = self.network.select(
            "target_actor_slow" if self.config["target_actor"] else "actor_slow"
        )

        ## Adjoint-matching
        # Compute the adjoint states
        xs, adjs, ts, pre_adj_info = self.adj_matching(batch["observations"], adj_rng)
        h = 1 / self.config["flow_steps"]
        sigmas = jnp.sqrt(2 * (1 - ts + h) / (ts + h))

        observations = jnp.repeat(
            batch["observations"][None], self.config["flow_steps"], axis=0
        )
        vf_fine = self.network.select("actor_fast")(
            observations, xs, ts, params=grad_params
        )

        vf_base = actor_slow(observations, xs, ts)

        # Compute the adjoint matching loss
        if self.config["residual"]:
            adj_loss = jnp.sum(
                jnp.square(vf_fine * 2 / sigmas + sigmas * adjs), axis=-1
            )
        else:
            adj_loss = jnp.sum(
                jnp.square((vf_fine - vf_base) * 2 / sigmas + sigmas * adjs), axis=-1
            )

        adj_loss = jnp.mean(jnp.sum(adj_loss, axis=0))

        info["adj_loss"] = adj_loss
        info.update(pre_adj_info)
        total_fast_loss += adj_loss

        if self.config["fql_alpha"] > 0.0:
            edit_base_rng, edit_rng = jax.random.split(edit_rng, 2)
            fql_noises = jax.random.normal(edit_base_rng, (batch_size, action_dim))
            flow_actions = self.compute_flow_actions(
                batch["observations"],
                fql_noises,
                model="slow,fast" if self.config["residual"] else "fast",
            )

            os_actions = self.network.select("one_step_actor")(
                batch["observations"], fql_noises, params=grad_params
            )
            fql_distill_loss = jnp.mean((flow_actions - os_actions) ** 2)

            # FQL loss.
            os_actions = jnp.clip(os_actions, -1, 1)
            fql_qs = self.network.select(f"critic")(
                batch["observations"], actions=os_actions
            )
            fql_q = jnp.mean(fql_qs, axis=0)
            fql_q_loss = -fql_q.mean()

            info["fql_distill_loss"] = fql_distill_loss
            info["fql_q_loss"] = fql_q_loss

            actor_loss += fql_q_loss + fql_distill_loss * self.config["fql_alpha"]

        if self.config["edit_scale"] > 0.0:
            edit_base_rng, edit_rng = jax.random.split(edit_rng, 2)
            flow_actions = self.compute_flow_actions(
                batch["observations"],
                jax.random.normal(edit_base_rng, (batch_size, action_dim)),
                model="slow,fast" if self.config["residual"] else "fast",
            )

            edit_dist = self.network.select("edit_actor")(
                jnp.concatenate((batch["observations"], flow_actions), axis=-1),
                params=grad_params,
            )
            edit = edit_dist.sample(seed=edit_rng)
            edit_log_probs = edit_dist.log_prob(edit)

            edited_actions = flow_actions + edit * self.config["edit_scale"]

            # Edit policy loss.
            edited_actions = jnp.clip(edited_actions, -1, 1)
            qs = self.network.select(f"critic")(
                batch["observations"], actions=edited_actions
            )
            q = jnp.mean(qs, axis=0)
            edit_q_loss = -q.mean()

            edit_entropy_loss = (
                edit_log_probs * self.network.select("edit_alpha")()
            ).mean()

            alpha = self.network.select("edit_alpha")(params=grad_params)
            entropy = -jax.lax.stop_gradient(edit_log_probs).mean()
            edit_alpha_loss = (
                alpha * (entropy - self.config["edit_target_entropy"])
            ).mean()

            actor_loss += edit_q_loss + edit_entropy_loss + edit_alpha_loss

            info["edit_q_loss"] = edit_q_loss
            info["edit_entropy_loss"] = edit_entropy_loss
            info["edit_alpha_loss"] = edit_alpha_loss
            info["edit_entropy"] = entropy
            info["edit_alpha"] = alpha

        return actor_loss + total_fast_loss, {
            "flow_loss": flow_loss,
            "fast_loss": total_fast_loss,
            **info,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, additional_agents={}):
        """Compute the total loss for compatibility with main.py evaluation."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        total_loss = critic_loss

        if self.config["critic_loss_type"] == "iql":
            val_loss, value_info = self.value_loss(batch, grad_params, value_rng)
            for k, v in value_info.items():
                info[f"value/{k}"] = v
            total_loss = total_loss + val_loss

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        return total_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        agent.target_update(new_network, "actor_slow")

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(
        self,
        observations,
        *,
        seed,
    ):
        """Sample actions from the QAM policy.

        Args:
            observations: Batch of observations.
            seed: Random seed for action sampling.
        """
        rng, edit_rng = jax.random.split(seed)

        if observations.ndim == 1:
            observations = observations[None, :]

        action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        noises = jax.random.normal(
            rng,
            (observations.shape[0], self.config["best_of_n"], action_dim),  # batch_size
        )
        observations = jnp.repeat(
            observations[..., None, :], self.config["best_of_n"], axis=-2
        )

        if self.config["fql_alpha"] > 0.0:  # if fql_alpha > 0, use the one-step policy
            actions = self.network.select("one_step_actor")(observations, noises)
            actions = jnp.clip(actions, -1, 1)
        else:  # otherwise use the flow policy
            if self.config["inv_temp"] == 0.0:
                actions = self.compute_flow_actions(observations, noises, model="slow")
            else:
                actions = self.compute_flow_actions(
                    observations,
                    noises,
                    model="slow,fast" if self.config["residual"] else "fast",
                )
            if (
                self.config["edit_scale"] > 0.0
            ):  # if there is an edit policy, refine the action further
                edit_dist = self.network.select("edit_actor")(
                    jnp.concatenate((observations, actions), axis=-1)
                )
                actions = (
                    actions
                    + edit_dist.sample(seed=edit_rng) * self.config["edit_scale"]
                )
            actions = jnp.clip(actions, -1, 1)

        # best-of-n sampling
        q = self.network.select("critic")(observations, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config["best_of_n"], action_dim))[
            jnp.arange(bsize), indices, :
        ].reshape(bshape + (action_dim,))

        if actions.shape[0] == 1:
            actions = actions.squeeze(axis=0)
        return actions

    @partial(jax.jit, static_argnames="model")
    def compute_flow_actions(
        self,
        observations,
        noises,
        model="slow",
    ):
        actions = noises
        networks = [self.network.select(f"actor_{m}") for m in model.split(",")]

        def step(x, t):
            ti = jnp.full((*observations.shape[:-1], 1), t / self.config["flow_steps"])
            vels = sum([network(observations, x, ti) for network in networks])
            x = x + vels / self.config["flow_steps"]
            return x, None

        actions, _ = jax.lax.scan(
            step,
            actions,
            jnp.arange(self.config["flow_steps"]),
            length=self.config["flow_steps"],
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
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate(
                [ex_actions] * config["horizon_length"], axis=-1
            )
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        if config["edit_target_entropy"] is None:
            config["edit_target_entropy"] = (
                -config["edit_target_entropy_multiplier"] * full_action_dim
            )

        critic_def = Value(
            network_class="MLP",
            network_kwargs=dict(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["value_layer_norm"],
            ),
            num_ensembles=config["num_qs"],
        )
        actor_def = ActorFlowField(
            config["actor_hidden_dims"],
            full_action_dim,
            mlp_kwargs=dict(layer_norm=config["actor_layer_norm"]),
            time_embedding="raw",
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_fast=(
                copy.deepcopy(actor_def),
                (ex_observations, full_actions, ex_times),
            ),
            actor_slow=(
                copy.deepcopy(actor_def),
                (ex_observations, full_actions, ex_times),
            ),
            target_actor_slow=(
                copy.deepcopy(actor_def),
                (ex_observations, full_actions, ex_times),
            ),
        )

        assert (
            config["fql_alpha"] * config["edit_scale"] == 0.0
        ), "Only one of fql_alpha and edit_scale can be non-zero."

        if config.get("critic_loss_type", "ddpg") == "iql":
            value_def = Value(
                network_class="MLP",
                network_kwargs=dict(
                    hidden_dims=config["value_hidden_dims"],
                    layer_norm=config["value_layer_norm"],
                ),
                num_ensembles=1,
            )
            network_info["value"] = (value_def, (ex_observations,))

        if config["fql_alpha"] > 0.0:
            network_info.update(
                dict(
                    one_step_actor=(
                        copy.deepcopy(actor_def),
                        (ex_observations, full_actions, None),
                    ),
                )
            )

        # if config["edit_scale"] > 0.:
        #     edit_actor_base_cls = partial(MLP, hidden_dims=config["actor_hidden_dims"], activate_final=True)
        #     edit_actor_def = TanhNormal(edit_actor_base_cls, full_action_dim)
        #     alpha_def = LogParam()

        #     network_info.update(dict(
        #         edit_actor=(edit_actor_def, jnp.concatenate((ex_observations, full_actions), axis=-1)),
        #         edit_alpha=(alpha_def, ()),
        #     ))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)["params"]

        actor_lr = config["actor_lr"]
        critic_lr = config["critic_lr"]

        def _make_tx(lr):
            if config["clip_grad"] > 0.0:
                return optax.chain(
                    optax.clip_by_global_norm(max_norm=config["clip_grad"]),
                    optax.adam(learning_rate=lr),
                )
            return optax.adam(learning_rate=lr)

        actor_modules = {"actor_fast", "actor_slow", "target_actor_slow"}
        if "one_step_actor" in network_info:
            actor_modules.add("one_step_actor")
        if "edit_actor" in network_info:
            actor_modules.add("edit_actor")
        if "edit_alpha" in network_info:
            actor_modules.add("edit_alpha")

        param_labels = {}
        for key in network_params:
            module_name = key.removeprefix("modules_")
            if module_name in actor_modules:
                param_labels[key] = "actor"
            else:
                param_labels[key] = "critic"

        network_tx = optax.multi_transform(
            {"actor": _make_tx(actor_lr), "critic": _make_tx(critic_lr)},
            param_labels,
        )
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]
        params["modules_target_actor_slow"] = params["modules_actor_slow"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="qam",
            ob_dims=ml_collections.config_dict.placeholder(list),  # Set automatically.
            action_dim=ml_collections.config_dict.placeholder(
                int
            ),  # Set automatically.
            # Common hyperparameters.
            actor_lr=3e-4,
            critic_lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=False,
            value_hidden_dims=(512, 512, 512, 512),
            value_layer_norm=True,
            # n-step returns.
            horizon_length=1,
            action_chunking=False,
            # RL hyperparameters.
            num_qs=2,
            rho=0.0,  # Pessimistic backup coefficient.
            discount=0.99,
            tau=0.005,
            flow_steps=10,
            best_of_n=1,  # Best-of-n samples for Q-target and action selection.
            # QAM-specific hyperparameters.
            inv_temp=0.3,  # Inverse temperature controlling Q-gradient influence.
            # If > 0, train a one-step policy distilled from the QAM flow policy while maximizing Q.
            fql_alpha=0.0,
            # If > 0, train an edit policy that refines the QAM flow output to maximize Q.
            edit_scale=0.0,
            # Other design variants.
            target_actor=True,  # Use the target actor for flow guidance.
            residual=False,  # Add Q-gradient as a residual to the BC velocity field.
            clip_adj=True,  # Clip the Q-gradient adjustment.
            clip_grad=1.0,  # Global gradient clipping norm.
            use_target_grad=True,  # Use the target critic for Q-gradient computation.
            edit_target_entropy=ml_collections.config_dict.placeholder(
                float
            ),  # Target entropy for edit policy (None for automatic tuning).
            edit_target_entropy_multiplier=0.5,  # Multiplier to dim(A) for edit policy target entropy.
            # Critic loss type.
            critic_loss_type="ddpg",  # "ddpg" (Q-bootstrap) or "iql" (value-bootstrap).
            expectile=0.9,  # Expectile for value regression (used when critic_loss_type="iql").
        )
    )
    return config
