import numpy as np
import metaworld
import random
import argparse
import os
import sys

from metaworld.policies import (
    SawyerReachV3Policy,
    SawyerPushV3Policy,
    SawyerPickPlaceV3Policy,
    SawyerDoorOpenV3Policy,
    SawyerDoorCloseV3Policy,
    SawyerDrawerOpenV3Policy,
    SawyerDrawerCloseV3Policy,
    SawyerButtonPressTopdownV3Policy,
    SawyerLeverPullV3Policy,
    SawyerWindowOpenV3Policy,
)

# =============================================================
# MT10_TASKS defines the canonical task order for one-hot encoding.
# Index 0 = reach-v3, index 1 = push-v3, ..., index 9 = window-open-v3.
# THIS ORDER MUST BE IDENTICAL in collect_data.py, merge_data.py,
# and evaluate.py — changing it will silently break task conditioning.
# =============================================================
MT10_TASKS = [
    'reach-v3',                   # 0
    'push-v3',                    # 1
    'pick-place-v3',              # 2
    'door-open-v3',               # 3
    'door-close-v3',              # 4
    'drawer-open-v3',             # 5
    'drawer-close-v3',            # 6
    'button-press-topdown-v3',    # 7
    'lever-pull-v3',              # 8
    'window-open-v3',             # 9
]

POLICY_MAP = {
    'reach-v3':                 SawyerReachV3Policy,
    'push-v3':                  SawyerPushV3Policy,
    'pick-place-v3':            SawyerPickPlaceV3Policy,
    'door-open-v3':             SawyerDoorOpenV3Policy,
    'door-close-v3':            SawyerDoorCloseV3Policy,
    'drawer-open-v3':           SawyerDrawerOpenV3Policy,
    'drawer-close-v3':          SawyerDrawerCloseV3Policy,
    'button-press-topdown-v3':  SawyerButtonPressTopdownV3Policy,
    'lever-pull-v3':            SawyerLeverPullV3Policy,
    'window-open-v3':           SawyerWindowOpenV3Policy,
}


def collect_expert_data(task_name, num_episodes=100):
    print(f"Task: {task_name} | Episodes: {num_episodes}")

    if task_name not in POLICY_MAP:
        print(f"Error: No expert policy found for '{task_name}'.")
        print(f"  Valid tasks: {MT10_TASKS}")
        sys.exit(1)

    # =========================================================
    # ONE-HOT CHANGE 1: Resolve this task's integer index.
    # Every transition collected here gets this label so the
    # dataset knows which one-hot vector to attach to the state.
    # =========================================================
    task_idx = MT10_TASKS.index(task_name)
    print(f"  Task index for one-hot: {task_idx} / {len(MT10_TASKS) - 1}")

    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()

    ExpertPolicy = POLICY_MAP[task_name]
    expert = ExpertPolicy()
    print(f"  Loaded Expert: {ExpertPolicy.__name__}")

    states, actions, next_states, rewards, traj_ids = [], [], [], [], []
    # =========================================================
    # ONE-HOT CHANGE 2: New list to store per-transition task index.
    # Stored as int64 scalar; contrastive_data.py converts to one-hot.
    # =========================================================
    task_labels = []
    total_steps = 0

    for episode_idx in range(num_episodes):
        current_task = random.choice(mt1.train_tasks)
        env.set_task(current_task)

        obs, _ = env.reset()
        done = False
        steps_in_ep = 0

        while not done and steps_in_ep < 500:
            action = expert.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            rewards.append(reward)
            traj_ids.append(episode_idx)
            task_labels.append(task_idx)   # same value for every step in this file

            obs = next_obs
            steps_in_ep += 1
            total_steps += 1

        if (episode_idx + 1) % 10 == 0:
            print(f"  Episode {episode_idx + 1}/{num_episodes} | "
                  f"steps so far: {total_steps}")

    print(f"Collection complete. Total transitions: {total_steps}")

    return (
        np.array(states,      dtype=np.float32),
        np.array(actions,     dtype=np.float32),
        np.array(next_states, dtype=np.float32),
        np.array(rewards,     dtype=np.float32).reshape(-1, 1),
        np.array(traj_ids,    dtype=np.int64),
        np.array(task_labels, dtype=np.int64),   # ONE-HOT CHANGE 3: returned + saved
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',     type=str, default='reach-v3',
                        help=f'One of: {MT10_TASKS}')
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)
    filename = f"data/expert_{args.task}.npz"

    s, a, s_next, r, ids, labels = collect_expert_data(args.task, args.episodes)

    # ONE-HOT CHANGE 4: Save task_labels alongside the other arrays
    np.savez(filename, states=s, actions=a, next_states=s_next,
             rewards=r, traj_ids=ids, task_labels=labels)
    print(f"Saved to {filename}  (includes task_labels)")