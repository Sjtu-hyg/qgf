import collections
import re
import time

import gymnasium
import numpy as np
import ogbench
from gymnasium.spaces import Box
from utils.datasets import Dataset, sparsify_dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if terminated or truncated:
            info["episode"] = {}
            info["episode"]["final_reward"] = reward
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self.unwrapped, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.unwrapped.get_normalized_score(info["episode"]["return"])
                    * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class RewardWrapper(gymnasium.RewardWrapper):
    def __init__(self, env, reward_scale=1.0, reward_bias=0.0):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias

    def reward(self, reward):
        return self.reward_scale * reward + self.reward_bias


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if "goal" in info:
            info["goal"] = np.concatenate([info["goal"]] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def transform_dataset_rewards(
    dataset,
    *,
    sparse=False,
    reward_scale=1.0,
    reward_bias=0.0,
):
    """Apply dataset reward transforms in sparse -> scale/bias order."""
    if dataset is None:
        return None
    if sparse:
        dataset = sparsify_dataset(dataset)
    if reward_scale != 1.0 or reward_bias != 0.0:
        rewards = dataset["rewards"] * reward_scale + reward_bias
        dataset = dataset.copy(add_or_replace=dict(rewards=rewards))
    return dataset


def make_env_and_datasets(
    env_name,
    frame_stack=None,
    action_clip_eps=1e-5,
    reward_scale=1.0,
    reward_bias=0.0,
    sparse=False,
    eval_env_only=False,
):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.
        reward_scale: Scale for reward.
        reward_bias: Bias for reward.
        sparse: Whether to sparsify offline train rewards before scale/bias.
    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if "singletask" in env_name:
        # OGBench.
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=[".*privileged.*", ".*proprio.*"])
        eval_env = EpisodeMonitor(
            eval_env, filter_regexes=[".*privileged.*", ".*proprio.*"]
        )
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    elif "antmaze" in env_name and (
        "diverse" in env_name or "play" in env_name or "umaze" in env_name
    ):
        # D4RL AntMaze.
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif (
        "pen" in env_name
        or "hammer" in env_name
        or "relocate" in env_name
        or "door" in env_name
    ):
        # D4RL Adroit.
        import d4rl.hand_manipulation_suite  # noqa
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif "low_dim" in env_name:
        # Robomimic.
        from envs import robomimic_utils

        if not robomimic_utils.is_robomimic_env(env_name):
            raise ValueError(f"Unsupported robomimic environment: {env_name}")

        env = robomimic_utils.make_env(env_name)
        eval_env = robomimic_utils.make_env(env_name)
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
        dataset = robomimic_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif "exorl" in env_name:
        import os

        os.environ["DISPLAY"] = ":0"
        import envs.exorl.dmc as dmc

        _, domain_name, task_name = env_name.split("_", 2)

        def make_env(env_name, task_name):
            # No Action Repeat, No Frame Stack.
            env = dmc.make(
                f"{env_name}_{task_name}",
                obs_type="states",
                frame_stack=1,
                action_repeat=1,
                seed=0,
            )
            frame_skip = 1  # TODO: maybe add frame skip
            env = dmc.DMCWrapper(
                env, 0, from_pixels=False, frame_skip=frame_skip, width=64, height=64
            )
            return env

        env = make_env(domain_name, task_name)
        eval_env = make_env(domain_name, task_name)

        from envs.exorl.exorl_utils import get_dataset

        env_name_short = env_name.split("_", 1)[1]
        dataset = get_dataset(env, env_name_short)
        train_dataset, val_dataset = dataset, None  # TODO: maybe do train_val split?
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    if reward_scale != 1.0 or reward_bias != 0.0:
        env = RewardWrapper(env, reward_scale, reward_bias)
        eval_env = RewardWrapper(eval_env, reward_scale, reward_bias)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(
                actions=np.clip(
                    train_dataset["actions"], -1 + action_clip_eps, 1 - action_clip_eps
                )
            )
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(
                    actions=np.clip(
                        val_dataset["actions"],
                        -1 + action_clip_eps,
                        1 - action_clip_eps,
                    )
                )
            )

    train_dataset = transform_dataset_rewards(
        train_dataset,
        sparse=sparse,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
    )
    val_dataset = transform_dataset_rewards(
        val_dataset,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
    )

    if eval_env_only:
        del env, train_dataset, val_dataset
        return eval_env

    return env, eval_env, train_dataset, val_dataset


def make_gc_env_and_datasets(dataset_name, frame_stack=None):
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the dataset.
        frame_stack: Number of frames to stack.

    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    # Use compact dataset to save memory.
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        dataset_name, compact_dataset=True
    )
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env.reset()

    return env, train_dataset, val_dataset
