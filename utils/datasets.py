import dataclasses
import os
import pickle
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=("padding",))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=("padding",))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert "observations" in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = (
            False  # Whether to additionally return next actions; set outside the class.
        )

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self["terminals"] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[
                np.searchsorted(self.initial_locs, idxs, side="right") - 1
            ]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(
                    jax.tree_util.tree_map(
                        lambda arr: arr[cur_idxs], self["observations"]
                    )
                )
                if i != self.frame_stack - 1:
                    next_obs.append(
                        jax.tree_util.tree_map(
                            lambda arr: arr[cur_idxs], self["observations"]
                        )
                    )
            next_obs.append(
                jax.tree_util.tree_map(lambda arr: arr[idxs], self["next_observations"])
            )

            batch["observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *obs
            )
            batch["next_observations"] = jax.tree_util.tree_map(
                lambda *args: np.concatenate(args, axis=-1), *next_obs
            )
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ["observations", "next_observations"])
        return batch

    def sample_sequence(self, batch_size, sequence_length, discount):
        """Sample a batch of sequences for n-step returns / action chunking.

        Args:
            batch_size: Number of sequences to sample.
            sequence_length: Length of each sequence (horizon H).
            discount: Discount factor for cumulative reward computation.

        Returns:
            dict with:
                observations:      (batch_size, obs_dim)  — initial obs at t
                actions:           (batch_size, H, action_dim)
                next_observations: (batch_size, H, obs_dim)  — freezes at episode end
                rewards:           (batch_size, H)  — cumulative discounted rewards
                masks:             (batch_size, H)
                terminals:         (batch_size, H)
                valid:             (batch_size, H)  — 0 after episode terminates
        """
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)

        data = {k: v[idxs] for k, v in self.items()}

        rewards = np.zeros(data["rewards"].shape + (sequence_length,), dtype=float)
        masks = np.ones(data["masks"].shape + (sequence_length,), dtype=float)
        valid = np.ones(data["masks"].shape + (sequence_length,), dtype=float)
        observations = np.zeros(
            data["observations"].shape[:-1]
            + (sequence_length, data["observations"].shape[-1]),
            dtype=float,
        )
        next_observations = np.zeros(
            data["observations"].shape[:-1]
            + (sequence_length, data["observations"].shape[-1]),
            dtype=float,
        )
        actions = np.zeros(
            data["actions"].shape[:-1] + (sequence_length, data["actions"].shape[-1]),
            dtype=float,
        )
        terminals = np.zeros(data["terminals"].shape + (sequence_length,), dtype=float)

        for i in range(sequence_length):
            cur_idxs = idxs + i

            if i == 0:
                rewards[..., 0] = self["rewards"][cur_idxs]
                masks[..., 0] = self["masks"][cur_idxs]
                terminals[..., 0] = self["terminals"][cur_idxs]
            else:
                valid[..., i] = 1.0 - terminals[..., i - 1]
                rewards[..., i] = rewards[..., i - 1] + (
                    self["rewards"][cur_idxs] * (discount**i) * valid[..., i]
                )
                masks[..., i] = np.minimum(
                    masks[..., i - 1], self["masks"][cur_idxs]
                ) * valid[..., i] + masks[..., i - 1] * (1.0 - valid[..., i])
                terminals[..., i] = np.maximum(
                    terminals[..., i - 1], self["terminals"][cur_idxs]
                )

            actions[..., i, :] = self["actions"][cur_idxs]
            next_observations[..., i, :] = self["next_observations"][cur_idxs] * valid[
                ..., i : i + 1
            ] + next_observations[..., i - 1, :] * (1.0 - valid[..., i : i + 1])
            observations[..., i, :] = self["observations"][cur_idxs]

        return dict(
            observations=data["observations"].copy(),
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
        )

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result["next_actions"] = self._dict["actions"][
                np.minimum(idxs + 1, self.size - 1)
            ]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1
        )
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding))
                if len(arr.shape) == 4
                else arr,
                batch[key],
            )

    def save(self, save_dir, epoch, prefix="dataset"):
        """Save the dataset to a file.

        Args:
            save_dir: Directory to save the dataset.
            epoch: Epoch number.
            prefix: Prefix for the filename (default: "dataset").
        """
        # Save only the valid portion of the dataset (up to size)
        valid_size = self.size
        dataset_data = {}
        for key, arr in self._dict.items():
            # Only save the valid portion
            dataset_data[key] = arr[:valid_size].copy()

        save_dict = dict(
            dataset_data=dataset_data,
            size=self.size,
            frame_stack=self.frame_stack,
            p_aug=self.p_aug,
            return_next_actions=self.return_next_actions,
            class_name=self.__class__.__name__,
        )
        save_path = os.path.join(save_dir, f"{prefix}_{epoch}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

        print(f"Saved {self.__class__.__name__} to {save_path} (size: {valid_size})")

    @classmethod
    def load(cls, save_path):
        """Load a dataset from a file.

        Args:
            save_path: Path to the saved dataset file.

        Returns:
            Loaded Dataset or ReplayBuffer instance.
        """
        with open(save_path, "rb") as f:
            load_dict = pickle.load(f)

        # Determine the class to use
        class_name = load_dict.get("class_name", "Dataset")
        if class_name == "ReplayBuffer":
            dataset_class = ReplayBuffer
        else:
            dataset_class = cls

        # Get saved size and ensure data is truncated to this size
        saved_size = load_dict["size"]
        dataset_data = load_dict["dataset_data"]

        # Truncate arrays to saved size if they're longer (safety check)
        # This handles cases where saved arrays might be longer than the saved size
        truncated_data = {}
        for key, arr in dataset_data.items():
            if len(arr) > saved_size:
                truncated_data[key] = arr[:saved_size].copy()
            else:
                truncated_data[key] = arr

        # Create the dataset with truncated data
        dataset = dataset_class(truncated_data)

        # Restore attributes - override size with saved size
        dataset.size = saved_size
        dataset.frame_stack = load_dict.get("frame_stack")
        dataset.p_aug = load_dict.get("p_aug")
        dataset.return_next_actions = load_dict.get("return_next_actions", False)

        # Recompute terminal and initial locations
        dataset.terminal_locs = np.nonzero(dataset["terminals"] > 0)[0]
        dataset.initial_locs = np.concatenate([[0], dataset.terminal_locs[:-1] + 1])

        # Restore ReplayBuffer-specific attributes if applicable
        if isinstance(dataset, ReplayBuffer):
            dataset.max_size = load_dict.get("max_size", saved_size)
            dataset.pointer = load_dict.get("pointer", saved_size)
            # Ensure size is correct (override any size computed from array length)
            dataset.size = saved_size

        print(f"Loaded {class_name} from {save_path} (size: {dataset.size})")
        return dataset


def sparsify_dataset(dataset):
    """Sparsify rewards: ``-1.0`` where dense reward was non-zero, else ``0.0``."""
    rewards = np.asarray(dataset["rewards"])
    sparse_rewards = np.where(rewards != 0.0, -1.0, 0.0).astype(rewards.dtype)
    return dataset.copy(add_or_replace=dict(rewards=sparse_rewards))


def load_replay_buffer(save_dir, epoch):
    """Load replay buffer from saved file.

    Args:
        save_dir: Directory containing the saved replay buffer.
        epoch: Epoch number.

    Returns:
        Loaded ReplayBuffer instance.
    """
    replay_buffer_path = os.path.join(save_dir, f"replay_buffer_{epoch}.pkl")
    replay_buffer = Dataset.load(replay_buffer_path)
    return replay_buffer


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0

    def save(self, save_dir, epoch, prefix="replay_buffer"):
        """Save the replay buffer to a file.

        Args:
            save_dir: Directory to save the buffer.
            epoch: Epoch number.
            prefix: Prefix for the filename (default: "replay_buffer").
        """
        # Extract only the valid portion of the buffer (up to size)
        valid_size = self.size
        buffer_data = {}
        for key, arr in self._dict.items():
            # Only save the valid portion
            buffer_data[key] = arr[:valid_size].copy()

        save_dict = dict(
            dataset_data=buffer_data,  # Use same key as Dataset for compatibility
            size=self.size,
            pointer=self.pointer,
            max_size=self.max_size,
            frame_stack=self.frame_stack,
            p_aug=self.p_aug,
            return_next_actions=getattr(self, "return_next_actions", False),
            class_name=self.__class__.__name__,
        )
        save_path = os.path.join(save_dir, f"{prefix}_{epoch}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

        print(f"Saved replay buffer to {save_path} (size: {valid_size})")
