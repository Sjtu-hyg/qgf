import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


def target_update(model, target_model, tau):
    """Interpolate from model to target_model. Tau = ratio of current model to target model"""
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wraps a function to supply jax rng. It will remember the rng state for that function.
    """

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f"When `name` is not specified, kwargs must contain the arguments for each module. "
                    f"Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}"
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {"params": params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate(
            [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0
        )
        grad_min_flat = jnp.concatenate(
            [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0
        )
        grad_norm_flat = jnp.concatenate(
            [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0
        )

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update(
            {
                "grad/max": final_grad_max,
                "grad/min": final_grad_min,
                "grad/norm": final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info


def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f"params_{epoch}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f)

    print(f"Saved to {save_path}")


def _resolve_checkpoint_dir_from_glob(restore_path_glob: str) -> str:
    """Resolve a checkpoint directory; may be a glob pattern.

    If multiple directories match, picks the first after lexicographic sort so
    choice is stable across machines and runs.
    """
    candidates = glob.glob(restore_path_glob)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint directory matched pattern: {restore_path_glob!r}"
        )
    if len(candidates) > 1:
        chosen = sorted(candidates)[0]
        print(
            f"Warning: {len(candidates)} dirs matched {restore_path_glob!r}; "
            f"using first after sort: {chosen}"
        )
        return chosen
    return candidates[0]


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    restore_path = (
        _resolve_checkpoint_dir_from_glob(restore_path) + f"/params_{restore_epoch}.pkl"
    )

    with open(restore_path, "rb") as f:
        load_dict = pickle.load(f)

    agent_state = load_dict["agent"]
    agent = flax.serialization.from_state_dict(agent, agent_state)

    print(f"Restored from {restore_path}")

    return agent


def restore_params(agent, restore_path, restore_epoch, module_strs):
    """Restore the module from a file.

    Args:
        module: Module.
        restore_path: Path to the directory containing the saved module.
        restore_epoch: Epoch number.
        module_strs: List of module strings to be restored.
    """
    restore_path = (
        _resolve_checkpoint_dir_from_glob(restore_path) + f"/params_{restore_epoch}.pkl"
    )

    with open(restore_path, "rb") as f:
        load_dict = pickle.load(f)

    # Helper: find source params for a given module, regardless of checkpoint layout
    def _get_source_params(d, module_name):
        root = d.get("agent", d)
        key = f"modules_{module_name}"
        # Try network-based checkpoint layout
        net = root.get("network") if isinstance(root, dict) else None
        if isinstance(net, dict):
            net_params = net.get("params")
            if isinstance(net_params, dict) and key in net_params:
                return net_params[key]
        # Try top-level TrainState layout
        top = root.get(module_name) if isinstance(root, dict) else None
        if isinstance(top, dict) and "params" in top:
            return top["params"]
        raise KeyError(f"Could not find params for module {module_name} in checkpoint")

    for module_str in module_strs:
        source_params = _get_source_params(load_dict, module_str)

        # Apply to target agent depending on its structure
        applied = False
        if hasattr(agent, "network") and isinstance(
            getattr(agent, "network"), TrainState
        ):
            current_params = agent.network.params
            key = f"modules_{module_str}"
            # Only apply to network if that module key exists in target params
            try:
                target_has_key = key in current_params
            except Exception:
                target_has_key = False
            if target_has_key:
                if isinstance(current_params, flax.core.FrozenDict):
                    new_params = flax.core.unfreeze(current_params)
                else:
                    new_params = dict(current_params)
                new_params[key] = source_params
                if isinstance(current_params, flax.core.FrozenDict):
                    new_params = flax.core.freeze(new_params)
                agent = agent.replace(network=agent.network.replace(params=new_params))
                applied = True

        if not applied and hasattr(agent, module_str):
            field_value = getattr(agent, module_str)
            if isinstance(field_value, TrainState):
                agent = agent.replace(
                    **{module_str: field_value.replace(params=source_params)}
                )
                applied = True

        if not applied:
            raise AttributeError(
                f"Cannot apply params for {module_str}: target agent has neither matching network key "
                f"'modules_{module_str}' nor a TrainState attribute '{module_str}'."
            )

    return agent
