"""
Visualize expert demonstrations for MetaWorld tasks.

Uses core for task list and env (library as source of truth). Supports any task
in the chosen suite (mt10 or mt50). Run from project root.

Usage (from project root):
  python scripts_all/visualize_expert_demo.py -n 3 --task reach-v3
  python scripts_all/visualize_expert_demo.py --task lever-pull-v3 --goals 14 43 47
  python scripts_all/visualize_expert_demo.py --suite mt50 --task peg-insert-side-v3 -n 2 --max-steps 750
"""
import argparse
import importlib
import os
import sys
import time

# Project root so we can import core
_script_dir = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_script_dir)
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
    """Return an expert policy instance for the given task name (derived from metaworld naming)."""
    policies_mod = importlib.import_module("metaworld.policies")
    for cls_name in _candidate_policy_class_names(task_name):
        if hasattr(policies_mod, cls_name):
            return getattr(policies_mod, cls_name)()
    raise ValueError(
        f"Could not find an expert policy for task '{task_name}'. "
        f"Tried: {', '.join(_candidate_policy_class_names(task_name))}."
    )


def visualize_expert_demos(
    task_name: str = "reach-v3",
    goal_indices: list = None,
    max_steps: int = 500,
) -> None:
    """Run and render expert episodes for the given task on the specified goal indices."""
    if goal_indices is None:
        goal_indices = [0]

    print(f"[Visualize Expert Demo] task={task_name}, goals={goal_indices}, max_steps={max_steps}")
    print("Close the render window or press Ctrl+C to stop.\n")

    env, train_tasks = make_env(task_name, render_mode="human")
    policy = get_expert_policy(task_name)
    num_goals = len(train_tasks)

    for gi in goal_indices:
        if gi < 0 or gi >= num_goals:
            raise ValueError(f"Goal index {gi} out of range [0, {num_goals - 1}]")

    n_show = len(goal_indices)
    for idx, goal_idx in enumerate(goal_indices):
        task = train_tasks[goal_idx]
        env.set_task(task)
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        done = False
        steps = 0

        print(f"  Goal index {goal_idx} ({idx + 1}/{n_show}) ... ", end="", flush=True)
        env.render()

        while not done and steps < max_steps:
            action = policy.get_action(obs)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, _, term, trunc, info = step_out
            else:
                obs, _, done, info = step_out
                term, trunc = done, False
            done = term or trunc
            env.render()
            steps += 1
            time.sleep(0.02)

        success = info.get("success", False)
        print("success!" if success else f"failed ({steps} steps).")
        time.sleep(0.5)

    print("\nDone.")
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize expert completing a MetaWorld task on one or more goals (task list from core/suite)."
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="mt10",
        choices=("mt10", "mt50"),
        help="Suite to validate --task against (default: mt10)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1,
        help="Number of goals when --goals is not set: run goals 0, 1, ..., n-1 (default: 1)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="reach-v3",
        help="Task name (e.g. reach-v3, lever-pull-v3). Must be in --suite.",
    )
    parser.add_argument(
        "--goals",
        type=int,
        nargs="+",
        default=None,
        help="Goal indices to run (e.g. 14 43 47). If set, overrides -n.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max steps per episode (default: 500)",
    )
    args = parser.parse_args()

    valid_tasks = get_tasks(args.suite)
    if args.task not in valid_tasks:
        parser.error(
            f"Task '{args.task}' is not in {args.suite.upper()}. "
            f"Valid tasks: {', '.join(valid_tasks)}"
        )

    if args.n < 1 and args.goals is None:
        parser.error("either -n >= 1 or --goals must be provided")

    env_check, train_tasks = make_env(args.task)
    try:
        num_goals = len(train_tasks)
    finally:
        try:
            env_check.close()
        except Exception:
            pass

    if args.goals is not None:
        goal_indices = sorted(set(args.goals))
        for gi in goal_indices:
            if gi < 0 or gi >= num_goals:
                parser.error(f"Goal index {gi} out of range [0, {num_goals - 1}] for task {args.task}")
    else:
        goal_indices = list(range(min(args.n, num_goals)))

    visualize_expert_demos(
        task_name=args.task,
        goal_indices=goal_indices,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
