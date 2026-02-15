"""
Collect exactly 1 expert trajectory per goal (50 total) for reach-v3.
Fast, minimal dataset: one deterministic expert rollout per of the 50 goals.
Saves to baseline/data/expert_data_reach-v3.npz (default for training).
"""
import metaworld
import numpy as np
import os
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy


def collect_one_per_goal(task_name='reach-v3', output_dir=None):
    """Collect one successful expert trajectory per goal (50 total)."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()
    policy = SawyerReachV3Policy()

    all_states = []
    all_actions = []
    failed = []

    for goal_idx, task in enumerate(mt1.train_tasks):
        env.set_task(task)
        obs, info = env.reset()
        done = False
        steps = 0
        episode_states = []
        episode_actions = []

        while not done and steps < 500:
            action = policy.get_action(obs)
            episode_states.append(obs)
            episode_actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if info.get('success', False):
                break

        if info.get('success', False):
            all_states.append(np.array(episode_states))
            all_actions.append(np.array(episode_actions))
            print(f"  Goal {goal_idx + 1:2d}/50: success ({len(episode_states)} steps)")
        else:
            failed.append(goal_idx)
            print(f"  Goal {goal_idx + 1:2d}/50: expert failed (skip)")

    if failed:
        print(f"\nWarning: expert failed on {len(failed)} goals: {failed}")
    if len(all_states) == 0:
        raise RuntimeError("No successful trajectories collected.")

    out_path = os.path.join(output_dir, f'expert_data_{task_name}.npz')
    np.savez(out_path, states=np.array(all_states, dtype=object), actions=np.array(all_actions, dtype=object))
    print(f"\nSaved {len(all_states)} trajectories to {out_path}")
    total_samples = sum(len(s) for s in all_states)
    print(f"Total (s,a) pairs: {total_samples}")
    return out_path


if __name__ == "__main__":
    collect_one_per_goal()
