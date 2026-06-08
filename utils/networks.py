import math
from dataclasses import field
from typing import Any, Callable, Dict, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={"params": 0, "intermediates": 0},
        split_rngs={"params": True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activation: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activation: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            # no layer norm for the last layer
            if i + 1 < len(self.hidden_dims) and self.layer_norm:
                x = nn.LayerNorm()(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activation(x)
            if i == len(self.hidden_dims) - 2:
                self.sow("intermediates", "feature", x)
        return x


class BroNet(nn.Module):
    """
    BroNet for critic learning: https://arxiv.org/pdf/2405.16158
    It's one architecture (among others) we can use to scale up critic networks.

    It's basically residual connections + layer normalization.
    """

    num_blocks: int
    hidden_dim: int
    kernel_init: Callable[[jnp.ndarray], jnp.ndarray] = default_init()
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)
        for _ in range(self.num_blocks):
            res = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
            res = nn.LayerNorm()(res)
            res = self.activation(res)
            res = nn.Dense(self.hidden_dim, kernel_init=default_init())(res)
            res = nn.LayerNorm()(res)
            x += res
        x = nn.Dense(1, kernel_init=default_init())(x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param(
            "log_value", init_fn=lambda key: jnp.full((), jnp.log(self.init_value))
        )
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


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


def embed_time(t, time_embedding: str):
    """Embed or pass through flow/diffusion time inputs."""
    if time_embedding == "sinusoidal":
        return timestep_embedding(t)
    if time_embedding == "raw":
        t = jax.lax.convert_element_type(t, jnp.float32)
        if t.ndim == 1:
            return t[..., None]
        return t
    raise ValueError(
        f"Unknown time_embedding {time_embedding!r}; expected 'sinusoidal' or 'raw'."
    )


class ActorFlowField(nn.Module):
    """Flow / diffusion policy.

    Attributes:
        hidden_dims: Hidden layer dimensions for the trunk MLP.
        action_dim: Action dimension.
        mlp_kwargs: Keyword arguments forwarded to the trunk MLP.
        encoder: Optional encoder for observations.
        time_embedding: ``'sinusoidal'`` (default) or ``'raw'`` concatenation.
    """

    hidden_dims: Any
    action_dim: int
    mlp_kwargs: dict = flax.struct.field(pytree_node=False)
    encoder: nn.Module = None
    time_embedding: str = "sinusoidal"

    @nn.compact
    def __call__(self, obs, noised_action, t=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            obs = self.encoder(obs)

        parts = [obs, noised_action]
        if t is not None:
            parts.append(embed_time(t, self.time_embedding))

        concat_input = jnp.concatenate(parts, axis=-1)
        outputs = MLP(self.hidden_dims, activate_final=True, **self.mlp_kwargs)(
            concat_input
        )
        v = nn.Dense(self.action_dim)(outputs)
        return v


class ConditionalFlowField(nn.Module):
    """
    Input is a noised action, observation, and t. is_positive is a binary mask.
    Used for CFGRL.
    """

    hidden_dims: Any
    action_dim: int
    mlp_kwargs: dict = flax.struct.field(pytree_node=False)

    @nn.compact
    def __call__(self, obs, is_positive, noised_action, t):
        t_embedding = timestep_embedding(t)
        is_positive_embedding = nn.Embed(2, 32)(is_positive)
        concat_input = jnp.concatenate(
            [obs, noised_action, t_embedding, is_positive_embedding], axis=-1
        )
        outputs = MLP(self.hidden_dims, activate_final=True, **self.mlp_kwargs)(
            concat_input
        )
        v = nn.Dense(self.action_dim)(outputs)
        return v


class GaussianActor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        activation: Activation function (default: nn.gelu).
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    activation: Any = nn.gelu
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm,
            activation=self.activation,
        )
        self.mean_net = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )
        else:
            if not self.const_std:
                self.log_stds = self.param(
                    "log_stds", nn.initializers.zeros, (self.action_dim,)
                )

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    num_ensembles: int = 2
    encoder: nn.Module = None
    network_class: str = "MLP"  # 'MLP' or 'BroNet'
    network_kwargs: Dict[str, Any] = field(default_factory=dict)

    def setup(self):
        if self.network_class == "MLP":
            network = MLP
            network_args = {
                "hidden_dims": (*self.network_kwargs["hidden_dims"], 1),
                "activate_final": False,
                "layer_norm": self.network_kwargs["layer_norm"],
            }
            if "activation" in self.network_kwargs:
                network_args["activation"] = self.network_kwargs["activation"]
        elif self.network_class == "BroNet":
            network = BroNet
            network_args = {
                "num_blocks": self.network_kwargs["num_blocks"],
                "hidden_dim": self.network_kwargs["hidden_dim"],
            }
        else:
            raise ValueError(f"Invalid network class: {self.network_class}")

        if self.num_ensembles > 1:
            network = ensemblize(network, self.num_ensembles)
        value_net = network(**network_args)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v
