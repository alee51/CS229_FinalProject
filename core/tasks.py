"""
Task registry for Meta-World. Single source of truth for task lists.
MT10/MT50 loaded from metaworld when available; fallback to hardcoded lists.
"""
from __future__ import annotations

import sys
from typing import List, Optional


def _warn_fallback_and_confirm(suite: str) -> None:
    """Warn that fallback task list is being used; prompt y/n when stdin is a TTY."""
    print(
        f"\nWarning: {suite.upper()} task list could not be loaded from the metaworld library. "
        "Using hardcoded fallback list (may differ from the library's benchmark).",
        file=sys.stderr,
    )
    if sys.stdin.isatty():
        try:
            reply = input("Proceed with fallback list? [y/N]: ").strip().lower()
            if reply not in ("y", "yes"):
                print("Aborted.", file=sys.stderr)
                sys.exit(1)
        except (EOFError, KeyboardInterrupt):
            print("Aborted.", file=sys.stderr)
            sys.exit(1)

# MT-10: try to load from metaworld first; fallback to canonical list
_MT10_TASKS: Optional[List[str]] = None

# Hardcoded MT-10 (canonical Meta-World MT10 benchmark) used when library doesn't expose it
MT10_TASKS_FALLBACK: List[str] = [
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


def _load_mt10_tasks() -> List[str]:
    """Load MT-10 task list from metaworld if available; else use fallback."""
    try:
        import metaworld
        if hasattr(metaworld, "MT10"):
            mt10 = metaworld.MT10()
            if hasattr(mt10, "train_tasks"):
                tasks = []
                seen = set()
                for t in mt10.train_tasks:
                    name = getattr(t, "env_name", None) or getattr(t, "task_name", None)
                    if name and name not in seen:
                        seen.add(name)
                        tasks.append(name)
                if len(tasks) >= 10:
                    return tasks
    except Exception:
        pass
    _warn_fallback_and_confirm("mt10")
    return list(MT10_TASKS_FALLBACK)


def _load_mt50_tasks() -> List[str]:
    """Load MT-50 task list from metaworld if available."""
    try:
        import metaworld
        mt50 = metaworld.MT50()
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
        if hasattr(metaworld, "ALL_TASK_NAMES"):
            return list(metaworld.ALL_TASK_NAMES)[:50]
    except Exception:
        pass
    _warn_fallback_and_confirm("mt50")
    return get_tasks("mt10") + [
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
    - mt10: 10 tasks (from library when available)
    - mt50: 50 tasks (from library when available)
    """
    global _MT10_TASKS, _MT50_TASKS
    suite = (suite or "mt1").lower().strip()
    if suite == "mt1":
        return ["reach-v3"]
    if suite == "mt10":
        if _MT10_TASKS is None:
            _MT10_TASKS = _load_mt10_tasks()
        return list(_MT10_TASKS)
    if suite == "mt50":
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


# Constants for eval/training
MAX_EPISODE_STEPS = 500
