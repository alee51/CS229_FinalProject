"""
Visualize expert demonstrations for MetaWorld tasks (MT-10 supported).

Usage:
  python visualize_expert_demo.py -n 3 --task reach-v3
  python visualize_expert_demo.py --task lever-pull-v3 --goals 14 43 47
  python visualize_expert_demo.py --task reach-v3 --goals 14 43 47 --max-steps 750
"""
import argparse
import importlib
import time

import metaworld

# MT-10 task -> (module_path, policy_class_name) for lazy import (same as collect_one_per_goal)
TASK_TO_POLICY = {
    'reach-v3': ('metaworld.policies.sawyer_reach_v3_policy', 'SawyerReachV3Policy'),
    'push-v3': ('metaworld.policies.sawyer_push_v3_policy', 'SawyerPushV3Policy'),
    'pick-place-v3': ('metaworld.policies.sawyer_pick_place_v3_policy', 'SawyerPickPlaceV3Policy'),
    'door-open-v3': ('metaworld.policies.sawyer_door_open_v3_policy', 'SawyerDoorOpenV3Policy'),
    'door-close-v3': ('metaworld.policies.sawyer_door_close_v3_policy', 'SawyerDoorCloseV3Policy'),
    'drawer-open-v3': ('metaworld.policies.sawyer_drawer_open_v3_policy', 'SawyerDrawerOpenV3Policy'),
    'drawer-close-v3': ('metaworld.policies.sawyer_drawer_close_v3_policy', 'SawyerDrawerCloseV3Policy'),
    'button-press-v3': ('metaworld.policies.sawyer_button_press_v3_policy', 'SawyerButtonPressV3Policy'),
    'lever-pull-v3': ('metaworld.policies.sawyer_lever_pull_v3_policy', 'SawyerLeverPullV3Policy'),
    'window-open-v3': ('metaworld.policies.sawyer_window_open_v3_policy', 'SawyerWindowOpenV3Policy'),
}


def get_expert_policy(task_name):
    """Return an expert policy instance for the given MT-10 task name."""
    if task_name not in TASK_TO_POLICY:
        raise ValueError(
            f"Unknown task '{task_name}'; supported: {list(TASK_TO_POLICY.keys())}"
        )
    mod_path, cls_name = TASK_TO_POLICY[task_name]
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)()


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

    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name](render_mode="human")
    policy = get_expert_policy(task_name)

    num_goals = len(mt1.train_tasks)
    for gi in goal_indices:
        if gi < 0 or gi >= num_goals:
            raise ValueError(f"Goal index {gi} out of range [0, {num_goals - 1}]")

    n_show = len(goal_indices)
    for idx, goal_idx in enumerate(goal_indices):
        task = mt1.train_tasks[goal_idx]
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
        description="Visualize expert completing a MetaWorld (MT-10) task on one or more goals."
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
        help="MT-10 task name (e.g. reach-v3, lever-pull-v3)",
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

    if args.n < 1 and args.goals is None:
        parser.error("either -n >= 1 or --goals must be provided")

    mt1 = metaworld.MT1(args.task)
    num_goals = len(mt1.train_tasks)

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
