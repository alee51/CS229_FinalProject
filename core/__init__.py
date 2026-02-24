"""
Shared layer for Meta-World: task lists (from library), env factory, eval API.
All approaches (baseline, vae, tce, dagger) use this for correct, consistent interaction.
"""
from core.tasks import (
    get_tasks,
    num_tasks,
    obs_dim,
    policy_input_dim,
)
from core.env import make_env
from core.eval import run_50_goal_eval, run_multitask_eval

__all__ = [
    "get_tasks",
    "num_tasks",
    "obs_dim",
    "policy_input_dim",
    "make_env",
    "run_50_goal_eval",
    "run_multitask_eval",
]
