import copy
from collections import defaultdict
from functools import partial
from typing import List, Optional

import jax
import numpy as np
import tqdm


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def _is_test_time_guidance_agent(agent) -> bool:
    return getattr(agent, "support_guidance", False)


class SingleEnvBatchAdapter:
    """Adapter to present a single Gymnasium env as a batched (num_envs=1) env.

    Accepts batched actions with leading batch dimension 1 and always returns
    batched outputs, emulating vector env API sufficiently for evaluation.
    """

    def __init__(self, env):
        self._env = env
        self.num_envs = 1

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return np.expand_dims(obs, axis=0), info

    def step(self, actions):
        action = (
            actions[0]
            if isinstance(actions, np.ndarray)
            and actions.ndim >= 1
            and actions.shape[0] == 1
            else actions
        )
        obs, reward, terminated, truncated, info = self._env.step(action)
        return (
            np.expand_dims(obs, axis=0),
            np.array([reward], dtype=np.float32),
            np.array([terminated], dtype=np.bool_),
            np.array([truncated], dtype=np.bool_),
            info,
        )

    def render(self):
        return self._env.render()


def _maybe_concat_goal_to_obs(
    observations: np.ndarray,
    goals: List,
    num_envs: int,
    policy_obs_dim: Optional[int],
) -> np.ndarray:
    """If env returns state-only but training used state‖goal, match dataset observation width."""
    if policy_obs_dim is None:
        return observations
    obs = np.asarray(observations)
    if obs.ndim == 1:
        obs = obs[None, :]
    rows = []
    for i in range(num_envs):
        o = np.asarray(obs[i]).reshape(-1)
        g = goals[i] if i < len(goals) else None
        if g is not None:
            g = np.asarray(g).reshape(-1)
            if o.shape[0] == policy_obs_dim:
                rows.append(o)
            elif o.shape[0] + g.shape[0] == policy_obs_dim:
                rows.append(np.concatenate([o, g], axis=-1))
            else:
                rows.append(o)
        else:
            rows.append(o)
    return np.stack(rows, axis=0)


def _vector_infos_to_list(infos, num_envs):
    if isinstance(infos, dict) and any(k.startswith("_") for k in infos):
        per_env = [dict() for _ in range(num_envs)]
        for key, value in infos.items():
            if key.startswith("_"):
                continue
            mask = infos.get(f"_{key}")
            if mask is None:
                mask = np.ones(num_envs, dtype=bool)
            for idx in range(num_envs):
                if mask[idx]:
                    per_env[idx][key] = value[idx]
        return per_env
    if isinstance(infos, dict):
        return [infos for _ in range(num_envs)]
    if isinstance(infos, (list, tuple)):
        return list(infos)
    if infos is None:
        return [dict() for _ in range(num_envs)]
    return [{"info": infos} for _ in range(num_envs)]


def _prepare_actor(agent, guidance_weight, rejection_sampling):
    rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
    sample_actions = partial(supply_rng(agent.sample_actions, rng=rng))

    if _is_test_time_guidance_agent(agent):
        return partial(
            sample_actions,
            guidance_weight=guidance_weight,
            rejection_sampling=rejection_sampling,
        )

    # Standard action sampling for other agents
    assert (
        guidance_weight is None
    ), "guidance_weight is only supported for test time guidance agents"
    return partial(
        sample_actions,
        rejection_sampling=rejection_sampling,
    )


def run_episodes(
    agent,
    env,
    task_id=None,
    eval_gaussian=None,
    guidance_weight=None,
    goal_conditioned=False,
    should_render=False,
    video_frame_skip=3,
    rejection_sampling=1,
):
    """Shared rollout for sequential and vectorized environments (batch-first).

    Always treat inputs/outputs as batched. If `env` is single, use SingleEnvBatchAdapter.
    """

    if not hasattr(env, "num_envs"):
        env = SingleEnvBatchAdapter(env)
    if env.num_envs > 1 and should_render:
        raise ValueError("Rendering is only supported for single environments.")

    actor_fn = _prepare_actor(
        agent,
        guidance_weight=guidance_weight,
        rejection_sampling=rejection_sampling,
    )

    # Detect action chunking from agent config.
    horizon_length = 1
    action_dim = None
    policy_obs_dim = None
    if hasattr(agent, "config"):
        horizon_length = int(agent.config.get("horizon_length", 1))
        action_dim = agent.config.get("action_dim", None)
        if action_dim is not None:
            action_dim = int(action_dim)
        _pod = agent.config.get("policy_observation_dim")
        if _pod is not None:
            policy_obs_dim = int(_pod)

    observations, reset_infos = env.reset(
        options=dict(task_id=task_id, render_goal=should_render)
    )

    num_envs = env.num_envs

    # Per-env state
    active = np.ones(num_envs, dtype=bool)
    returns = np.zeros(num_envs, dtype=np.float32)
    lengths = np.zeros(num_envs, dtype=np.int32)
    trajectories = [defaultdict(list) for _ in range(num_envs)]
    renders = [[] for _ in range(num_envs)]

    # Goals
    infos_per_env = (
        _vector_infos_to_list(reset_infos, num_envs)
        if isinstance(reset_infos, dict)
        else ([reset_infos] * num_envs)
    )
    goals = [info.get("goal") for info in infos_per_env]
    goal_frames = [info.get("goal_rendered") for info in infos_per_env]

    observations = _maybe_concat_goal_to_obs(
        observations, goals, num_envs, policy_obs_dim
    )

    rng = np.random.default_rng()

    # Per-env action queues for action chunking (H > 1).
    action_queues = [[] for _ in range(num_envs)]

    while not np.all(~active):
        # Determine which envs need a new action chunk.
        need_chunk = [i for i in range(num_envs) if not action_queues[i]]
        if need_chunk:
            subset_obs = observations[need_chunk]
            raw = (
                actor_fn(observations=subset_obs, goals=[goals[i] for i in need_chunk])
                if goal_conditioned
                else actor_fn(observations=subset_obs)
            )
            raw = np.atleast_2d(np.array(raw))

            if horizon_length > 1:
                # raw: (len(need_chunk), H * action_dim) -> (len(need_chunk), H, action_dim)
                if action_dim is None:
                    if raw.shape[-1] % horizon_length != 0:
                        raise ValueError(
                            f"Cannot infer per-step action_dim: raw.shape[-1]={raw.shape[-1]}, "
                            f"horizon_length={horizon_length}"
                        )
                    action_dim = raw.shape[-1] // horizon_length
                chunks = raw.reshape(len(need_chunk), horizon_length, action_dim)
                for j, idx in enumerate(need_chunk):
                    action_queues[idx].extend(chunks[j])
            else:
                for j, idx in enumerate(need_chunk):
                    action_queues[idx].append(raw[j])

        # Pop one action per env from the queue.
        actions = np.array([action_queues[i].pop(0) for i in range(num_envs)])

        if eval_gaussian is not None:
            actions = rng.normal(loc=actions, scale=eval_gaussian)
        actions = np.clip(actions, -1, 1)

        next_observations, rewards, terminations, truncations, step_infos = env.step(
            actions
        )
        infos_per_step = _vector_infos_to_list(step_infos, num_envs)
        done_now = np.logical_or(terminations, truncations)

        for idx in range(num_envs):
            reward = rewards[idx]
            info = infos_per_step[idx]
            next_observation = next_observations[idx]

            if active[idx]:
                lengths[idx] += 1
                returns[idx] += reward

                if done_now[idx]:
                    if "final_observation" in info:
                        next_observation = info["final_observation"]
                    if "final_info" in info:
                        info = copy.deepcopy(info["final_info"])
                if "goal" in info:
                    goals[idx] = info["goal"]
                if should_render and "goal_rendered" in info:
                    goal_frames[idx] = info["goal_rendered"]
                transition = dict(
                    observation=observations[idx],
                    next_observation=next_observation,
                    action=actions[idx],
                    reward=reward,
                    done=done_now[idx],
                    info=info,
                )
                add_to(trajectories[idx], copy.deepcopy(transition))

                if should_render and (
                    lengths[idx] % video_frame_skip == 0 or done_now[idx]
                ):
                    frame = env.render().copy()
                    goal_frame = goal_frames[idx]
                    renders[idx].append(
                        np.concatenate([goal_frame, frame], axis=0)
                        if goal_frame is not None
                        else frame
                    )

                if done_now[idx]:
                    action_queues[idx] = []  # clear queue on episode end
                    active[idx] = False

            o_next = np.asarray(next_observation).reshape(-1)
            if (
                policy_obs_dim is not None
                and goals[idx] is not None
                and o_next.shape[0] < policy_obs_dim
            ):
                g = np.asarray(goals[idx]).reshape(-1)
                if o_next.shape[0] + g.shape[0] == policy_obs_dim:
                    o_next = np.concatenate([o_next, g], axis=-1)
            observations[idx] = o_next

    if should_render:
        renders = [np.array(r) for r in renders]

    return trajectories, renders, returns, lengths


def eval_standard(
    agent,
    env,
    vec_eval_env=None,
    task_id=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_gaussian=None,
    guidance_weight=None,
    goal_conditioned=False,
    rejection_sampling=1,
):
    """Evaluate the agent in the environment with optimized execution.

    Returns a tuple: (stats, trajectories, renders)
    """
    trajs = []
    stats = defaultdict(list)
    renders = []

    batch_size = vec_eval_env.num_envs if vec_eval_env is not None else 1
    assert num_eval_episodes % batch_size == 0
    total_batches = num_eval_episodes // batch_size

    for _ in range(total_batches):
        traj_batch, _, returns, lengths = run_episodes(
            agent,
            vec_eval_env if vec_eval_env is not None else env,
            task_id,
            eval_gaussian,
            guidance_weight,
            goal_conditioned,
            should_render=False,
            video_frame_skip=video_frame_skip,
            rejection_sampling=rejection_sampling,
        )
        for idx in range(batch_size):
            info_flat = {
                "episode.return": returns[idx],
                "episode.length": lengths[idx],
                **flatten(traj_batch[idx]["info"][-1]),
            }
            add_to(stats, info_flat)
        trajs.extend(traj_batch)

    for _ in range(num_video_episodes):
        _, render_batch, _, _ = run_episodes(
            agent,
            env,
            task_id,
            eval_gaussian,
            guidance_weight,
            goal_conditioned,
            should_render=True,
            video_frame_skip=video_frame_skip,
            rejection_sampling=rejection_sampling,
        )
        renders.append(render_batch[0])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def eval_with_test_time_guidance(
    agent,
    eval_env,
    vec_eval_env,
    *,
    num_eval_episodes: int,
    rejection_sampling: int,
    guidance_weights: List[str],
    num_video_episodes: int,
):
    """Rollout over multiple guidance weights."""
    renders = []
    eval_metrics = {}
    max_return = -np.inf
    guidance_weights = [float(x) for x in guidance_weights]

    w_results = {}
    for w in tqdm.tqdm(guidance_weights, desc="Evaluating various guidance weights"):
        eval_info, _, cur_renders = eval_standard(
            agent=agent,
            env=eval_env,
            vec_eval_env=vec_eval_env,
            num_eval_episodes=num_eval_episodes,
            num_video_episodes=num_video_episodes,
            guidance_weight=w,
            rejection_sampling=rejection_sampling,
        )
        renders.extend(cur_renders)
        w_results[w] = eval_info
        max_return = max(max_return, eval_info["episode.return"])

    eval_metrics["evaluation/episode.return"] = max_return
    if "success" in w_results[guidance_weights[0]]:
        eval_metrics["evaluation/success"] = max(
            w_results[w]["success"] for w in guidance_weights
        )
    eval_metrics["evaluation/episode_length"] = min(
        w_results[w]["episode.length"] for w in guidance_weights
    )

    best_w = None
    for w in guidance_weights:
        if w_results[w]["episode.return"] == max_return:
            best_w = w
            break

    eval_metrics["evaluation/best_guidance_weight"] = best_w

    for w in guidance_weights:
        result = w_results[w]
        prefix = f"evaluation_guidance_weight_{w}"
        eval_metrics[f"{prefix}/episode_return"] = result["episode.return"]
        if "success" in result:
            eval_metrics[f"{prefix}/success"] = result["success"]
        eval_metrics[f"{prefix}/episode_length"] = result["episode.length"]

    return eval_metrics, renders
