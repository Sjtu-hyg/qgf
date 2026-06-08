"""
Test-time eval — GradStep: post-denoising Q-gradient ascent steps on the clean action.
Requires bc_iql checkpoint: TRAIN_RUN_GROUP=bc_iql python scripts/exp_grad_step_test_time_eval.py  →  sbatch/grad_step.sh
"""
import glob
import os
import re
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate import SbatchGenerator

run_group = "grad_step"

TRAIN_RUN_GROUP = os.environ.get("TRAIN_RUN_GROUP", "qgf")
RESTORE_EPOCH = 500_000
SAVE_DIR = "exp"
SAVE_WANDB_PROJECT_DIR = "qgf"

num_jobs_per_gpu = 2

ENV_NAMES = [
    "cube-triple",
    "cube-quadruple",
    "puzzle-4x4",
    "scene",
]
TASKS = [1, 2, 3, 4, 5]

# Per-environment tuned GradStep hyperparameters
STEP_SIZE_STEPS = {
    "cube-triple": (0.01, 3),
    "cube-quadruple": (0.01, 3),
    "puzzle-4x4": (0.01, 3),
    "scene": (0.01, 5),
}

OGBENCH_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/path/to/ogbench/data")


def env_dir_name(env_name):
    splits = env_name.split("-")
    if "singletask" not in splits:
        raise ValueError(f"Expected singletask env id, got {env_name!r}")
    pos = splits.index("singletask")
    prefix = "-".join(splits[:pos])
    ver = splits[-1]
    return f"{prefix}-100m-{ver}"


def find_qgf_checkpoint(env_name, seed, run_group, save_dir, wandb_project, epoch):
    """Return a glob pattern that locates the QGF checkpoint for (env_name, seed)."""
    env_short = re.sub(r"-(singletask|v0|v1|v2)", "", env_name)
    pattern = os.path.join(
        save_dir,
        wandb_project,
        run_group,
        f"{run_group}_qgf_{env_short}_seed{seed:02d}_*",
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No QGF checkpoint found for env={env_name} seed={seed}.\n"
            f"Searched: {pattern}\n"
            f"Run exp_qgf.py first and ensure checkpoints exist under {save_dir}/."
        )
    # If multiple match (e.g. re-runs), pick the first after sorting for stability.
    return sorted(matches)[0]


for debug in [True, False]:
    gen = SbatchGenerator(
        j=num_jobs_per_gpu,
        prefix=("MUJOCO_GL=egl", "python main.py"),
        job_name=run_group,
        time="01:00:00",
    )
    if debug:
        gen.add_common_prefix(
            {
                "wandb_run_group": run_group + "_debug",
                "eval_episodes": 1,
                "offline_steps": 0,
                "online_steps": 0,
            }
        )
    else:
        gen.add_common_prefix(
            {
                "wandb_run_group": run_group,
                "eval_episodes": 30,
                "offline_steps": 0,
                "online_steps": 0,
            }
        )

    for env_type, task in itertools.product(ENV_NAMES, TASKS):
        env_name = f"{env_type}-play-singletask-task{task}-v0"
        grad_step_size, grad_steps = STEP_SIZE_STEPS[env_type]
        for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if debug and seed != 1:
                continue

            try:
                restore_path = find_qgf_checkpoint(
                    env_name,
                    seed,
                    run_group=TRAIN_RUN_GROUP,
                    save_dir=SAVE_DIR,
                    wandb_project=SAVE_WANDB_PROJECT_DIR,
                    epoch=RESTORE_EPOCH,
                )
            except FileNotFoundError as e:
                print(f"[exp_grad_step] skip: {e}")
                continue

            kwargs = {
                "env_name": env_name,
                "seed": seed,
                "agent": "agents/grad_step.py",
                "agent.qgrad_step_size": grad_step_size,
                "agent.qgrad_steps": grad_steps,
                "agent.denoise_steps": 10,
                # larger batch size and network
                "agent.batch_size": 1024,
                "agent.value_network_kwargs.hidden_dims": "(1024,1024,1024,1024)",
                "agent.actor_hidden_dims": "(1024,1024,1024,1024)",
                "agent.discount": 0.999,
                "agent.expectile": 0.9,
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
