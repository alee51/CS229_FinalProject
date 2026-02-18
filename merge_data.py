"""
merge_data.py — Combines per-task expert data into a single MT-10 dataset.

Run after collect_data.py has been run for all 10 tasks.
Output: data/expert_mt10.npz
"""
import numpy as np
import os
import sys

# =============================================================
# MUST match the order in collect_data.py and evaluate.py.
# The integer index of each task is its one-hot position.
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

all_states, all_actions, all_next_states = [], [], []
all_rewards, all_traj_ids, all_task_labels = [], [], []
traj_offset = 0

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
    all_traj_ids.append(d['traj_ids'] + traj_offset)
    traj_offset += int(d['traj_ids'].max()) + 1

    # =============================================================
    # ONE-HOT CHANGE 5: Collect task_labels from each per-task file.
    # These are absolute indices (0-9) — no offset needed unlike traj_ids.
    # =============================================================
    all_task_labels.append(d['task_labels'])

    print(f"  Loaded {task:<35} {n:>6} transitions  "
          f"(task_label={d['task_labels'][0]})")

if missing:
    print(f"\nMissing data for: {missing}")
    print("Run collect_data.py for each missing task first.")
    sys.exit(1)

merged = dict(
    states      = np.concatenate(all_states),
    actions     = np.concatenate(all_actions),
    next_states = np.concatenate(all_next_states),
    rewards     = np.concatenate(all_rewards),
    traj_ids    = np.concatenate(all_traj_ids),
    task_labels = np.concatenate(all_task_labels),   # ONE-HOT CHANGE 6: saved to merged
)

os.makedirs('data', exist_ok=True)
out_path = 'data/expert_mt10.npz'
np.savez(out_path, **merged)

total   = len(merged['states'])
n_trajs = len(np.unique(merged['traj_ids']))
labels  = np.unique(merged['task_labels'])
print(f"\nSaved {out_path}")
print(f"  Total transitions : {total}")
print(f"  Total trajectories: {n_trajs}")
print(f"  Task labels present: {labels}  (should be 0-9)")
print(f"  Avg per task      : {total // len(MT10_TASKS)}")