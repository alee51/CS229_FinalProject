"""
Visualize expert demonstrations for MetaWorld tasks.

Usage:
  python visualize_expert_demo.py --n 1 --task reach-v3   # one goal, reach-v3
  python visualize_expert_demo.py -n 3 --task reach-v3   # three goals

Supported task: reach-v3 (uses SawyerReachV3Policy).
"""
import argparse
import time

import metaworld
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy


def visualize_expert_demos(n: int = 1, task_name: str = "reach-v3") -> None:
    """Run and render n expert episodes (one per goal) for the given task."""
    if task_name != "reach-v3":
        raise ValueError(
            f"Only task 'reach-v3' is supported for expert visualization. Got: {task_name}"
        )

    print(f"[Visualize Expert Demo] task={task_name}, n={n}")
    print("Close the render window or press Ctrl+C to stop.\n")

    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name](render_mode="human")
    policy = SawyerReachV3Policy()

    max_goals = len(mt1.train_tasks)
    to_show = min(n, max_goals)

    for i in range(to_show):
        task = mt1.train_tasks[i]
        env.set_task(task)
        obs, info = env.reset()
        done = False
        steps = 0

        print(f"  Goal {i + 1}/{to_show} (task index {i}) ... ", end="", flush=True)
        env.render()

        while not done and steps < 500:
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated
            steps += 1
            time.sleep(0.02)

        success = info.get("success", False)
        print("success!" if success else "failed.")
        time.sleep(0.5)

    print("\nDone.")
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize expert completing MetaWorld task (e.g. reach-v3) on one or more goals."
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1,
        help="Number of goals to visualize (default: 1)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="reach-v3",
        help="Task name (default: reach-v3)",
    )
    args = parser.parse_args()

    if args.n < 1:
        parser.error("n must be at least 1")

    visualize_expert_demos(n=args.n, task_name=args.task)


if __name__ == "__main__":
    main()
