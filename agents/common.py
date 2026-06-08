import jax.numpy as jnp


def get_flat_batch(batch, config):
    """Return (batch_actions, next_obs, rewards, masks, valid_w) with H handled."""
    if config.get("action_chunking", False):
        B = batch["actions"].shape[0]
        batch_actions = jnp.reshape(batch["actions"], (B, -1))
    else:
        batch_actions = batch["actions"][..., 0, :]
    next_obs = batch["next_observations"][..., -1, :]
    rewards = batch["rewards"][..., -1]
    masks = batch["masks"][..., -1]
    valid_w = batch["valid"][..., -1]
    return batch_actions, next_obs, rewards, masks, valid_w


def aggregate_q(qs, config):
    aggregation_fn = getattr(jnp, config.get("q_aggregation", "min"))
    return aggregation_fn(qs, axis=0)
