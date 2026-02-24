"""
Canonical evaluation API: 50 goals per task, same protocol for all approaches.
Uses core.env and core.tasks. Policy is a callable (e.g. torch module) that takes
input tensor and returns action (numpy or tensor).
"""
import numpy as np
from typing import List, Tuple, Optional

from core.tasks import get_tasks, MAX_EPISODE_STEPS
from core.env import make_env


def _one_hot_task(task_id: int, num_tasks: int) -> np.ndarray:
    out = np.zeros(num_tasks, dtype=np.float32)
    out[task_id] = 1.0
    return out


def _set_rng(seed: Optional[int]):
    if seed is None:
        return
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            try:
                xpu.manual_seed_all(seed)
            except Exception:
                pass
    except ImportError:
        pass


def run_50_goal_eval(
    policy,
    task_name: str = "reach-v3",
    clip_actions: bool = True,
    seed: Optional[int] = 42,
    device=None,
    max_steps: int = MAX_EPISODE_STEPS,
) -> Tuple[float, List[bool], List[int]]:
    """
    Run 50 episodes (1 per goal) for a single task.
    Returns (success_rate_pct, goal_success_list, failed_goal_indices).
    policy: callable that takes a single tensor input and returns action (numpy or tensor).
    """
    import torch
    if device is None:
        device = torch.device("cpu")
    _set_rng(seed)
    if hasattr(policy, "eval"):
        policy.eval()
    env, train_tasks = make_env(task_name)
    n_goals = min(50, len(train_tasks))
    goal_success = []
    try:
        for goal_idx in range(n_goals):
            task = train_tasks[goal_idx]
            env.set_task(task)
            if seed is not None:
                try:
                    out = env.reset(seed=seed + goal_idx)
                except TypeError:
                    out = env.reset()
            else:
                out = env.reset()
            obs = out[0] if isinstance(out, tuple) else out
            if isinstance(obs, tuple):
                obs = obs[0] if len(obs) > 0 else obs
            obs = np.asarray(obs).flatten()
            done = False
            steps = 0
            while not done and steps < max_steps:
                obs_t = torch.FloatTensor(obs).to(device)
                with torch.no_grad():
                    action = policy(obs_t).cpu().numpy()
                action = np.asarray(action).flatten().astype(np.float64)
                if clip_actions:
                    action = np.clip(action, -1.0, 1.0)
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, _, term, trunc, info = step_out
                else:
                    obs, _, done, info = step_out
                    term, trunc = done, False
                done = term or trunc
                obs = np.asarray(obs).flatten() if not done else obs
                steps += 1
            goal_success.append(bool(info.get("success", False)))
    finally:
        try:
            env.close()
        except Exception:
            pass
    failed_goals = [i for i, s in enumerate(goal_success) if not s]
    success_rate = sum(goal_success) / len(goal_success) * 100
    return success_rate, goal_success, failed_goals


def run_multitask_eval(
    policy,
    suite: str,
    clip_actions: bool = True,
    seed: Optional[int] = 42,
    device=None,
    max_steps: int = MAX_EPISODE_STEPS,
) -> Tuple[List[float], float]:
    """
    Run 50 episodes (1 per goal) for each task in the suite.
    Policy input = concat(obs, one_hot(task_id)).
    Returns (success_rate_per_task, success_rate_avg).
    """
    import torch
    if device is None:
        device = torch.device("cpu")
    _set_rng(seed)
    if hasattr(policy, "eval"):
        policy.eval()
    task_list = get_tasks(suite)
    n_tasks = len(task_list)
    success_rate_per_task = []
    for task_id, task_name in enumerate(task_list):
        env, train_tasks = make_env(task_name)
        n_goals = min(50, len(train_tasks))
        goal_success = []
        try:
            for goal_idx in range(n_goals):
                task = train_tasks[goal_idx]
                env.set_task(task)
                if seed is not None:
                    try:
                        out = env.reset(seed=seed + task_id * 1000 + goal_idx)
                    except TypeError:
                        out = env.reset()
                else:
                    out = env.reset()
                obs = out[0] if isinstance(out, tuple) else out
                if isinstance(obs, tuple):
                    obs = obs[0] if len(obs) > 0 else obs
                obs = np.asarray(obs).flatten()
                oh = _one_hot_task(task_id, n_tasks)
                done = False
                steps = 0
                while not done and steps < max_steps:
                    x = np.concatenate([obs, oh]).astype(np.float32)
                    obs_t = torch.FloatTensor(x).to(device)
                    with torch.no_grad():
                        action = policy(obs_t).cpu().numpy()
                    action = np.asarray(action).flatten().astype(np.float64)
                    if clip_actions:
                        action = np.clip(action, -1.0, 1.0)
                    step_out = env.step(action)
                    if len(step_out) == 5:
                        obs, _, term, trunc, info = step_out
                    else:
                        obs, _, done, info = step_out
                        term, trunc = done, False
                    done = term or trunc
                    obs = np.asarray(obs).flatten() if not done else obs
                    steps += 1
                goal_success.append(bool(info.get("success", False)))
        finally:
            try:
                env.close()
            except Exception:
                pass
        rate = sum(goal_success) / len(goal_success) * 100
        success_rate_per_task.append(rate)
    success_rate_avg = sum(success_rate_per_task) / len(success_rate_per_task)
    return success_rate_per_task, success_rate_avg
