import typing as tp

from . import cheetah, hopper, jaco, quadruped, walker


def make(
    domain,
    task,
    task_kwargs=None,
    environment_kwargs=None,
    visualize_reward: bool = False,
):

    if domain == "cheetah":
        return cheetah.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain == "walker":
        return walker.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain == "hopper":
        return hopper.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain == "quadruped":
        return quadruped.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain == "point_mass_maze":
        raise ValueError(f"{task} not supported")

    else:
        raise ValueError(f"{task} not found")


def make_jaco(task, obs_type, seed) -> tp.Any:
    return jaco.make(task, obs_type, seed)
