# based on https://github.com/kvfrans/cfgrl/blob/main/rlbase/envs/exorl/exorl_utils.py

import glob
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import tqdm
from utils.datasets import Dataset

# Use environment variable if set, otherwise default to a persistent location
if "EXORL_DATA_PATH" in os.environ:
    data_path = os.environ["EXORL_DATA_PATH"]
else:
    # Use home directory to persist data across runs
    data_path = os.path.expanduser("~/.cache/exorl_data/")
print("Path to exorl data is", data_path)


def get_dataset(env, env_name, method="rnd", use_task_reward=True, **kwargs):

    print("Extra kwargs are", kwargs)

    domain_name, task_name = env_name.split("_", 1)

    path = os.path.join(data_path, domain_name, method)
    if not os.path.exists(path):
        print("Downloading exorl data.")
        os.makedirs(path)
        url = (
            "https://dl.fbaipublicfiles.com/exorl/"
            + domain_name
            + "/"
            + method
            + ".zip"
        )
        print("Downloading from", url)
        os.system("wget " + url + " -P " + path)
        os.system("unzip " + path + "/" + method + ".zip -d " + path)

    # process data into Dataset object.
    path = os.path.join(data_path, domain_name, method, "buffer")
    npzs = sorted(glob.glob(f"{path}/*.npz"))
    dataset_npy = os.path.join(data_path, domain_name, method, task_name + ".npy")
    if os.path.exists(dataset_npy):
        with open(dataset_npy, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("No path at {}. Creating dataset.".format(dataset_npy))
        print("Calculating exorl rewards. There are {} npz files.".format(len(npzs)))
        dataset = defaultdict(list)
        num_steps = 0
        for i, npz in tqdm.tqdm(enumerate(npzs)):
            traj_data = dict(np.load(npz))
            dataset["observations"].append(traj_data["observation"][:-1, :])
            dataset["next_observations"].append(traj_data["observation"][1:, :])
            dataset["actions"].append(traj_data["action"][1:, :])
            dataset["physics"].append(
                traj_data["physics"][1:, :]
            )  # Note that this corresponds to next_observations (i.e., r(s, a, s') = r(s') -- following the original DMC rewards)

            if use_task_reward:
                # TODO: make this faster and sanity check it
                rewards = []
                pixels = []
                reward_spec = env.reward_spec()
                states = traj_data["physics"]
                for j in range(states.shape[0]):
                    with env.physics.reset_context():
                        env.physics.set_state(states[j])
                    reward = env.task.get_reward(env.physics)
                    reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
                    rewards.append(reward)
                traj_data["reward"] = np.array(rewards, dtype=reward_spec.dtype)
                dataset["rewards"].append(traj_data["reward"][1:])
            else:
                dataset["rewards"].append(traj_data["reward"][1:, 0])

            terminals = np.full((len(traj_data["observation"]) - 1,), False)
            dataset["terminals"].append(terminals)
            num_steps += len(traj_data["observation"]) - 1
        print("Loaded {} steps".format(num_steps))
        for k, v in dataset.items():
            dataset[k] = np.concatenate(v, axis=0)
        # Use pickle protocol 4 to support datasets larger than 4GB
        print(f"Saving dataset to {dataset_npy}")
        with open(dataset_npy, "wb") as f:
            pickle.dump(dataset, f, protocol=4)

    # Processing - cast to float32
    terminals = dataset["terminals"].astype(np.float32)
    masks = (1.0 - terminals).astype(np.float32)

    return Dataset.create(
        observations=dataset["observations"].astype(np.float32),
        actions=dataset["actions"].astype(np.float32),
        next_observations=dataset["next_observations"].astype(np.float32),
        rewards=dataset["rewards"].astype(np.float32),
        masks=masks,
        terminals=terminals,
    )
