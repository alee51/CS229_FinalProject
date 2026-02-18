"""
merge_data.py — Combines per-task expert data into a single mt10 dataset.

Run this after collect_data.py has been run for all 10 tasks.
Output: data/expert_mt10.npz
"""
import numpy as np
import os
import sys

MT10_TASKS = [
    'reach-v3',
    'push-v3',
    'pick-place-v3',
    'door-open-v3',
    'door-close-v3',
    'drawer-open-v3',
    'drawer-close-v3',
    'button-press-topdown-v3',
    'lever-pull-v3',
    'window-open-v3',
]

all_states, all_actions, all_next_states, all_rewards, all_traj_ids = [], [], [], [], []
traj_offset = 0   # keeps trajectory IDs globally unique across tasks

missing = []
for task in MT10_TASKS:
    path = f"data/expert_{task}.npz"
    if not os.path.exists(path):
        missing.append(task)
        continue

    d = np.load(path)
    n = len(d['states'])

    all_states.append(d['states'])
    all_actions.append(d['actions'])
    all_next_states.append(d['next_states'])
    all_rewards.append(d['rewards'])
    # Offset so task 2's episode 0 gets a different ID than task 1's episode 0
    all_traj_ids.append(d['traj_ids'] + traj_offset)
    traj_offset += int(d['traj_ids'].max()) + 1

    print(f"  Loaded {task:<35} {n:>6} transitions | "
          f"traj IDs {traj_offset - int(d['traj_ids'].max()) - 1}"
          f"–{traj_offset - 1}")

if missing:
    print(f"\nMissing data for: {missing}")
    print("Run collect_data.py for each missing task before merging.")
    sys.exit(1)

merged = dict(
    states      = np.concatenate(all_states),
    actions     = np.concatenate(all_actions),
    next_states = np.concatenate(all_next_states),
    rewards     = np.concatenate(all_rewards),
    traj_ids    = np.concatenate(all_traj_ids),
)

os.makedirs('data', exist_ok=True)
out_path = 'data/expert_mt10.npz'
np.savez(out_path, **merged)

total = len(merged['states'])
n_trajs = len(np.unique(merged['traj_ids']))
print(f"\nSaved {out_path}")
print(f"  Total transitions : {total}")
print(f"  Total trajectories: {n_trajs}")
print(f"  Avg per task      : {total // len(MT10_TASKS)}")