import flax.linen as nn
import jax.numpy as jnp


def mish(x):
    """Mish activation function: x * tanh(softplus(x))."""
    return x * jnp.tanh(nn.softplus(x))


def get_activation(activation):
    if activation == "mish":
        return mish
    else:
        return getattr(nn, activation, nn.gelu)
