# Implementation Plan: MT-10 Data Collection in collect_one_per_goal.py

## Goal

Add MT-10 data collection as a **command-line option** to the existing `baseline/scripts/collect_one_per_goal.py`, so that:

- **Default (no new args)**: Behavior unchanged — collect one trajectory per goal for a single task (default `reach-v3`), save `expert_data_{task_name}.npz` with `states` and `actions` only.
- **With `--mt10`**: Collect for all 10 MT-10 tasks (one trajectory per goal per task), save a single combined `.npz` with `states`, `actions`, and `task_ids` (and optionally `goal_indices`, `task_names`) as in MT10_DATA_COLLECTION_PLAN.md.

No new script; all logic lives in `collect_one_per_goal.py`.

---

## CLI

Add argument parsing under `if __name__ == "__main__"`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--mt10` | False | If set, collect for all 10 MT-10 tasks (one per goal per task) and save combined npz with `task_ids`. |
| `--task` | `'reach-v3'` | Single-task mode: which task to collect (used only when `--mt10` is not set). |
| `--output` | None | Output path. If not set: single-task → `{output_dir}/expert_data_{task_name}.npz`; MT-10 → `{output_dir}/expert_data_mt10.npz`. |
| `--output-dir` | `../data` (relative to script) | Directory for output file when `--output` is not given. |

**Examples:**

```bash
# Current behavior (single task, reach, default path)
python collect_one_per_goal.py

# Single task, explicit task name
python collect_one_per_goal.py --task push-v3

# MT-10: all 10 tasks, one trajectory per goal each, combined npz
python collect_one_per_goal.py --mt10

# MT-10 with explicit output path
python collect_one_per_goal.py --mt10 --output ../data/expert_data_mt10.npz
```

---

## Code Structure

1. **Constants**
   - Define `MT10_TASKS = ['reach-v3', 'push-v3', 'pick-place-v3', 'door-open-v3', 'door-close-v3', 'drawer-open-v3', 'drawer-close-v3', 'button-press-v3', 'lever-pull-v3', 'window-open-v3']` at module level (or near the top).

2. **Expert policy lookup**
   - Add a function `get_expert_policy(task_name)` that returns the expert policy instance for a given task name. Options:
     - **A)** A dict mapping `task_name` → policy class, and instantiate when needed; or
     - **B)** Lazy imports: map task name to `(module_path, class_name)`, e.g. `'reach-v3'` → `('metaworld.policies.sawyer_reach_v3_policy', 'SawyerReachV3Policy')`, then `importlib.import_module(module).PolicyClass()`.
   - Use the mapping from MT10_DATA_COLLECTION_PLAN.md for all 10 tasks. On unknown task, raise or return None and handle in caller.

3. **Single-task collection (existing)**
   - Keep `collect_one_per_goal(task_name=..., output_dir=None)` as is for the single-task case: 50 goals, one trajectory each, save `states` and `actions` only. Optionally allow passing an explicit `output_path` if we add that to the function signature for CLI.

4. **MT-10 collection**
   - Add `collect_mt10_one_per_goal(output_dir=None, output_path=None)` (or a name like `collect_mt10`) that:
     - Loops over `task_id, task_name in enumerate(MT10_TASKS)`.
     - For each task: `mt1 = metaworld.MT1(task_name)`, create env, get policy via `get_expert_policy(task_name)`.
     - For each goal index in `range(len(mt1.train_tasks))`: set_task, reset, run expert until success or 500 steps; flatten `obs` to 1D before appending (e.g. `np.asarray(obs).flatten()`).
     - On success: append to `all_states`, `all_actions`, append `task_id` to `all_task_ids`, and optionally `goal_idx` to `all_goal_indices`, `task_name` to `all_task_names`.
     - On failure: log (e.g. print or list of failed (task_name, goal_idx)) and do not add a trajectory.
     - After all tasks: validate (see below), then save one npz with `states`, `actions`, `task_ids`, and optionally `goal_indices`, `task_names`. Output path: `output_path` if given, else `os.path.join(output_dir, 'expert_data_mt10.npz')`.

5. **Observation shape**
   - In both single-task and MT-10, ensure each stored state is 1D (e.g. `np.asarray(obs).flatten()`) so trajectories are `(T, 39)` and compatible with existing `train.py` / data plan.

6. **Validation (MT-10)**
   - Before saving in MT-10 mode: `len(states) == len(actions) == len(task_ids)`; all `task_ids` in 0..9; if `goal_indices` is saved, assert no duplicate `(task_ids[i], goal_indices[i])`; each `states[i].shape[1] == 39`, `actions[i].shape[1] == 4`. Optionally print a short summary (trajectories per task, total (s,a) pairs).

7. **Main**
   - Parse `--mt10`, `--task`, `--output`, `--output-dir`.
   - If `--mt10`: call `collect_mt10_one_per_goal(output_dir=..., output_path=...)`.
   - Else: call `collect_one_per_goal(task_name=..., output_dir=...)` and, if `--output` is set, after save we could copy/move to that path, or extend `collect_one_per_goal` to accept optional `output_path` and use it when provided.

---

## File Changes

- **Only file to edit**: `baseline/scripts/collect_one_per_goal.py`.
- Add: `MT10_TASKS`, `get_expert_policy()`, `collect_mt10_one_per_goal()`, and `argparse` in `__main__` with the flags above. Preserve existing `collect_one_per_goal()` behavior and docstring when called without MT-10.

---

## Optional Enhancements

- **`--tasks` for MT-10 subset**: e.g. `--mt10 --tasks reach-v3 push-v3` to collect only those two tasks (task_id would still be 0 and 1 in the saved npz). Can be added later if needed.
- **`goal_indices` and `task_names`**: Save them in the MT-10 npz by default for debugging and uniqueness checks; training only requires `states`, `actions`, `task_ids`.

---

## Summary

| Mode | Trigger | Output | Contents |
|------|---------|--------|----------|
| Single task | no `--mt10` | `expert_data_{task_name}.npz` (or `--output`) | `states`, `actions` |
| MT-10 | `--mt10` | `expert_data_mt10.npz` (or `--output`) | `states`, `actions`, `task_ids` [, `goal_indices`, `task_names`] |

Uniqueness in MT-10 mode: one trajectory per (task in MT10, goal for that task); no duplicate (task, goal) pairs.
