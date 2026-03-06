"""
VAE data collection. Uses core for task list and env (get_tasks, make_env).
"""
import os
import sys
import importlib
import numpy as np

# Project root for core
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VAE_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_VAE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from core.tasks import get_tasks
from core.env import make_env

# Special-case mismatches between task names and policy class names in metaworld
SPECIAL_TASK_TO_POLICY = {
    "peg-insert-side-v3": "SawyerPegInsertionSideV3Policy",
}


def _to_camel(s: str) -> str:
    return "".join(p.capitalize() for p in s.split("_"))


def _candidate_policy_class_names(task_name: str):
    if task_name in SPECIAL_TASK_TO_POLICY:
        return [SPECIAL_TASK_TO_POLICY[task_name]]
    base = task_name.replace("-v3", "").replace("-", "_")
    camel = _to_camel(base)
    return [
        f"Sawyer{camel}V3Policy",
        f"Sawyer{camel}Policy",
    ]


def get_expert_policy(task_name: str):
    policies_mod = importlib.import_module("metaworld.policies")
    for cls_name in _candidate_policy_class_names(task_name):
        if hasattr(policies_mod, cls_name):
            return getattr(policies_mod, cls_name)()
    raise ValueError(
        f"Could not find an expert policy for task '{task_name}'. "
        f"Tried: {', '.join(_candidate_policy_class_names(task_name))}."
    )


def collect_expert_data(task_name: str = "reach-v3", num_episodes: int = 50):
    """Collect expert (obs, action) pairs for a single task. Uses core.make_env."""
    env, train_tasks = make_env(task_name)
    expert = get_expert_policy(task_name)
    n_goals = min(num_episodes, len(train_tasks))
    all_obs, all_actions = [], []
    try:
        for ep in range(n_goals):
            task = train_tasks[ep % len(train_tasks)]
            env.set_task(task)
            obs, _ = env.reset()
            obs = np.asarray(obs).flatten()
            done = False
            while not done:
                action = np.clip(expert.get_action(obs), -1.0, 1.0)
                all_obs.append(obs)
                all_actions.append(action)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                obs = np.asarray(obs).flatten() if not done else obs
    finally:
        try:
            env.close()
        except Exception:
            pass
    return np.asarray(all_obs, dtype=np.float32), np.asarray(all_actions, dtype=np.float32)


def collect_expert_data_mt(
    benchmark: str = "MT10",
    num_episodes_per_task: int = 100,
    use_all_train_task_variations: bool = True,
    seed: int = 0,
):
    """Collect expert data for multi-task (MT10). Uses core.get_tasks and core.make_env."""
    rng = np.random.default_rng(seed)
    if benchmark.upper() != "MT10":
        raise ValueError(f"Unknown benchmark '{benchmark}'. Supported: MT10.")
    task_names = get_tasks("mt10")

    all_obs, all_actions, all_task_ids = [], [], []
    task_names_used = []

    for task_name in task_names:
        try:
            expert = get_expert_policy(task_name)
        except ValueError:
            continue
        env, train_tasks = make_env(task_name)
        if not train_tasks:
            try:
                env.close()
            except Exception:
                pass
            continue

        task_id = len(task_names_used)
        task_names_used.append(task_name)

        try:
            for ep in range(num_episodes_per_task):
                task = (
                    train_tasks[ep % len(train_tasks)]
                    if use_all_train_task_variations
                    else train_tasks[int(rng.integers(0, len(train_tasks)))]
                )
                env.set_task(task)
                obs, _ = env.reset()
                obs = np.asarray(obs).flatten()
                done = False
                while not done:
                    action = np.clip(expert.get_action(obs), -1.0, 1.0)
                    all_obs.append(obs)
                    all_actions.append(action)
                    all_task_ids.append(task_id)
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    obs = np.asarray(obs).flatten() if not done else obs
        finally:
            try:
                env.close()
            except Exception:
                pass

    return (
        np.asarray(all_obs, dtype=np.float32),
        np.asarray(all_actions, dtype=np.float32),
        np.asarray(all_task_ids, dtype=np.int64),
        task_names_used,
    )
