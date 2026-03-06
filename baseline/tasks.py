"""
Thin re-export of core task registry. Single source of truth is core.tasks.
"""
from core.tasks import (
    get_tasks,
    num_tasks,
    obs_dim,
    policy_input_dim,
    MT10_TASKS_FALLBACK,
)
from core.tasks import get_tasks as _get_tasks

# Backward compatibility: MT10_TASKS for code that imports it (e.g. test.py)
MT10_TASKS = _get_tasks("mt10")

__all__ = [
    "get_tasks",
    "num_tasks",
    "obs_dim",
    "policy_input_dim",
    "MT10_TASKS",
    "MT10_TASKS_FALLBACK",
]
