"""
Test-time eval — QGF-Jacobian (J+): one-step Euler approx guidance with full Jacobian correction.
Requires bc_iql checkpoint: TRAIN_RUN_GROUP=bc_iql python scripts/exp_qgf_jacobian_test_time_eval.py  →  sbatch/qgf_jacobian_test_time_eval.sh
"""
import glob
import itertools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate import SbatchGenerator

TRAIN_RUN_GROUP = os.environ.get("TRAIN_RUN_GROUP", "qgf")
RESTORE_EPOCH = 500_000
SAVE_DIR = "exp"
SAVE_WANDB_PROJECT_DIR = "qgf"

run_group = "qgf_jacobian_test_time_eval"

num_jobs_per_gpu = 4

ENV_NAMES = [
    "cube-triple",
    "cube-quadruple",
    "puzzle-4x4",
    "scene",
]
TASKS = [1, 2, 3, 4, 5]

GUIDANCE_WEIGHTS = "0.004, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,0.12"

OGBENCH_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/path/to/ogbench/data")


def env_dir_name(env_name):
    splits = env_name.split("-")
    if "singletask" not in splits:
        raise ValueError(f"Expected singletask env id, got {env_name!r}")
    pos = splits.index("singletask")
    prefix = "-".join(splits[:pos])
    ver = splits[-1]
    return f"{prefix}-100m-{ver}"


def find_qgf_checkpoint(env_name, seed):
    env_short = re.sub(r"-(singletask|v0|v1|v2)", "", env_name)
    pattern = os.path.join(
        SAVE_DIR,
        SAVE_WANDB_PROJECT_DIR,
        TRAIN_RUN_GROUP,
        f"{TRAIN_RUN_GROUP}_qgf_{env_short}_seed{seed:02d}_*",
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for env={env_name} seed={seed}. Searched: {pattern}"
        )
    return sorted(matches)[0]


for debug in [True, False]:
    gen = SbatchGenerator(
        j=num_jobs_per_gpu,
        prefix=("MUJOCO_GL=egl", "python main.py"),
        job_name=run_group,
        time="02:00:00",
    )
    if debug:
        gen.add_common_prefix(
            {
                "wandb_run_group": run_group + "_debug",
                "eval_episodes": 1,
                "offline_steps": 0,
                "online_steps": 0,
                "guidance_weights": "1.0,5.0",
            }
        )
    else:
        gen.add_common_prefix(
            {
                "wandb_run_group": run_group,
                "eval_episodes": 30,
                "offline_steps": 0,
                "online_steps": 0,
                "guidance_weights": GUIDANCE_WEIGHTS,
            }
        )

    for env_type, task in itertools.product(ENV_NAMES, TASKS):
        env_name = f"{env_type}-play-singletask-task{task}-v0"
        for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if debug and seed != 1:
                continue

            try:
                restore_path = find_qgf_checkpoint(env_name, seed)
            except FileNotFoundError as e:
                print(f"[exp_qgf_jacobian_test_time_eval] skip: {e}")
                continue

            kwargs = {
                "env_name": env_name,
                "seed": seed,
                "agent": "agents/qgf.py",
                "agent.denoise_steps": 10,
                "agent.denoised_action_approx": "one_euler_step_approx",
                "agent.apply_jacobian": True,
                "agent.expectile": 0.9,
                # larger batch size and network
                "agent.batch_size": 1024,
                "agent.value_network_kwargs.hidden_dims": "(1024,1024,1024,1024)",
                "agent.actor_hidden_dims": "(1024,1024,1024,1024)",
                "agent.discount": 0.999,
                # action chunking (H=5 horizon)
                "agent.action_chunking": True,
                "agent.horizon_length": 5,
                # 100M OGBench dataset
                "ogbench_dataset_dir": f"{OGBENCH_DATA_DIR}/{env_dir_name(env_name)}/",
                "restore_path": restore_path,
                "restore_epoch": RESTORE_EPOCH,
                "eval_only": True,
            }

            gen.add_run(kwargs)

    sbatch_str = gen.generate_str(print_commands=debug)

    if not debug:
        os.makedirs("sbatch", exist_ok=True)
        with open(f"sbatch/{run_group}.sh", "w") as f:
            f.write(sbatch_str)
