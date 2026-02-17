"""
Shared task registry for Meta-World. Single source of truth for task lists.
Use get_tasks(suite) to get the list of task names; supports mt1, mt10, mt50.
"""
from __future__ import annotations

from typing import List, Optional

# MT-10: 10 tasks (same as Meta-World MT10 benchmark)
MT10_TASKS: List[str] = [
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "door-close-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-v3",
    "lever-pull-v3",
    "window-open-v3",
]

# MT-50: try to get from metaworld at import; else use extended list
_MT50_TASKS: Optional[List[str]] = None


def _load_mt50_tasks() -> List[str]:
    """Load MT-50 task list from metaworld if available."""
    try:
        import metaworld

        mt50 = metaworld.MT50()
        # MT50 exposes train_tasks; each task has env_name or we get unique env names
        if hasattr(mt50, "train_tasks"):
            tasks = []
            seen = set()
            for t in mt50.train_tasks:
                name = getattr(t, "env_name", None) or getattr(t, "task_name", None)
                if name and name not in seen:
                    seen.add(name)
                    tasks.append(name)
            if len(tasks) >= 50:
                return tasks
        # Fallback: use ALL_TASK_NAMES if the package exposes it
        if hasattr(metaworld, "ALL_TASK_NAMES"):
            return list(metaworld.ALL_TASK_NAMES)[:50]
    except Exception:
        pass
    # Hardcoded MT-50 (Meta-World 1.0 benchmark): MT10 + 40 more
    return MT10_TASKS + [
        "assembly-v3",
        "basketball-v3",
        "bin-picking-v3",
        "box-close-v3",
        "button-press-topdown-v3",
        "button-press-topdown-wall-v3",
        "coffee-button-v3",
        "coffee-pull-v3",
        "coffee-push-v3",
        "dial-turn-v3",
        "disassemble-v3",
        "door-lock-v3",
        "door-unlock-v3",
        "hand-insert-v3",
        "faucet-open-v3",
        "hammer-v3",
        "handle-press-side-v3",
        "handle-press-v3",
        "handle-pull-side-v3",
        "peg-insert-side-v3",
        "peg-unplug-side-v3",
        "pick-out-of-hole-v3",
        "pick-place-wall-v3",
        "push-back-v3",
        "push-wall-v3",
        "reach-wall-v3",
        "shelf-place-v3",
        "soccer-v3",
        "stick-push-v3",
        "stick-pull-v3",
        "sweep-into-v3",
        "sweep-v3",
    ]


def get_tasks(suite: str) -> List[str]:
    """
    Return the list of task names for the given suite.
    - mt1: single task (reach-v3), 1 task
    - mt10: 10 tasks
    - mt50: 50 tasks
    """
    suite = (suite or "mt1").lower().strip()
    if suite == "mt1":
        return ["reach-v3"]
    if suite == "mt10":
        return list(MT10_TASKS)
    if suite == "mt50":
        global _MT50_TASKS
        if _MT50_TASKS is None:
            _MT50_TASKS = _load_mt50_tasks()
        return list(_MT50_TASKS)
    raise ValueError(f"Unknown suite '{suite}'. Use mt1, mt10, or mt50.")


def num_tasks(suite: str) -> int:
    """Return the number of tasks for the given suite."""
    return len(get_tasks(suite))


def obs_dim() -> int:
    """Observation dimension for Meta-World (e.g. reach-v3)."""
    return 39


def policy_input_dim(suite: str) -> int:
    """Policy input dimension: obs_dim + num_tasks (one-hot) for multi-task, else obs_dim."""
    if suite == "mt1":
        return obs_dim()
    return obs_dim() + num_tasks(suite)
