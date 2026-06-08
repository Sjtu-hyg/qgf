"""
Train — QAM: Q-guided Action Matching with IQL critic.
python scripts/exp_qam.py  →  sbatch/qam.sh
"""
import os
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate import SbatchGenerator

run_group = "qam"

num_jobs_per_gpu = 2

OFFLINE_STEPS = 500_000

ENV_NAMES = [
    "cube-triple",
    "cube-quadruple",
    "puzzle-4x4",
    "scene",
]
TASKS = [1, 2, 3, 4, 5]

INV_TEMPS = {
    "cube-triple": 0.1,
    "cube-quadruple": 0.1,
    "puzzle-4x4": 0.1,
    "scene": 0.1,
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


for debug in [True, False]:
    gen = SbatchGenerator(
        j=num_jobs_per_gpu,
        prefix=("MUJOCO_GL=egl", "python main.py"),
        job_name=run_group,
        time="10:00:00",
    )
    if debug:
        gen.add_common_prefix(
            {
                "wandb_run_group": run_group + "_debug",
                "offline_steps": 10_000,
                "online_steps": 0,
                "eval_episodes": 1,
                "eval_interval": 5_000,
            }
        )
    else:
        gen.add_common_prefix(
            {
                "wandb_run_group": run_group,
                "online_steps": 0,
                "offline_steps": OFFLINE_STEPS,
                "eval_episodes": 30,
            }
        )

    for env_type, task in itertools.product(ENV_NAMES, TASKS):
        env_name = f"{env_type}-play-singletask-task{task}-v0"
        for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if debug and seed != 1:
                continue

            kwargs = {
                "env_name": env_name,
                "seed": seed,
                "agent": "agents/qam.py",
                "agent.critic_loss_type": "iql",
                "agent.num_qs": 2,
                "agent.inv_temp": INV_TEMPS[env_type],
                # larger batch size and network
                "agent.batch_size": 1024,
                "agent.value_hidden_dims": "(1024,1024,1024,1024)",
                "agent.actor_hidden_dims": "(1024,1024,1024,1024)",
                "agent.discount": 0.999,
                "agent.expectile": 0.9,
                # action chunking (H=5 horizon)
                "agent.action_chunking": True,
                "agent.horizon_length": 5,
                # 100M OGBench dataset
                "ogbench_dataset_dir": f"{OGBENCH_DATA_DIR}/{env_dir_name(env_name)}/",
            }

            gen.add_run(kwargs)

    sbatch_str = gen.generate_str(print_commands=debug)

    if not debug:
        os.makedirs("sbatch", exist_ok=True)
        with open(f"sbatch/{run_group}.sh", "w") as f:
            f.write(sbatch_str)
