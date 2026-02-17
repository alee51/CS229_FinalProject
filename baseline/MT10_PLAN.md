# Plan: Train and Evaluate Baseline on MT-10 Task Suite

## Goal

- Train and evaluate the existing behavioral cloning baseline on **Meta-World MT-10** (10 tasks) using the **current train/test framework** with minimal new scripts.
- Use **one-hot task encoding** so the policy knows which task it is solving (Farama’s MT10 benchmark already appends one-hot task IDs to observations; we will match that convention when using single-task envs).
- **Long-term**: Baseline should do well on **reach** (and maybe a few other simple tasks) and poorly on the rest, so that encoder-based improvements can show gains on the harder tasks.

---
## Farama MT-10 Summary (from [Metaworld README](https://github.com/Farama-Foundation/Metaworld))

- **MT10**: Multi-task benchmark with **10 tasks**. Observations have **one-hot task IDs appended** (obs dim = 39 + 10 = **49**). Action dim = **4** (same as MT1).
- **API**: `gym.make_vec('Meta-World/MT10', vector_strategy='sync', seed=seed)` returns a vector env; `obs, info = envs.reset()` gives batched obs with one-hot already in the state.
- **Single-task alternative**: We can keep using `metaworld.MT1(task_name)` per task and **append one-hot(task_id)** ourselves so we don’t depend on the vector env for data collection and eval. Task index `task_id in 0..9` for the 10 tasks below.

**MT-10 task list (v3 env names):**

| Index | Task name      |
|-------|----------------|
| 0     | reach-v3       |
| 1     | push-v3        |
| 2     | pick-place-v3  |
| 3     | door-open-v3   |
| 4     | door-close-v3  |
| 5     | drawer-open-v3 |
| 6     | drawer-close-v3|
| 7     | button-press-v3|
| 8     | lever-pull-v3  |
| 9     | window-open-v3 |

---

## 1. Data Format for MT-10

- **Single-task (current)**: One `.npz` with `states` (list of trajectories), `actions` (list of trajectories). No task id → input = raw obs (39-dim for reach), same as today.
- **MT-10**: One or more `.npz` with **task identity** so we can append one-hot:
  - **Option A (recommended)**: One combined `.npz` with:
    - `states`: list of trajectory arrays (each shape `(T, 39)`)
    - `actions`: list of trajectory arrays (each shape `(T, 4)`)
    - `task_ids`: 1D array of length `num_trajectories`, `task_ids[i] in 0..9`
  - **Option B**: Multiple `.npz` files (e.g. one per task); train script accepts `--data path1 path2 ...` and assigns `task_id = 0, 1, ...` by order.

**One-hot convention**: For task index `k` in `0..9`, one-hot is a length-10 vector with 1 at index `k` and 0 elsewhere. Policy input = **concatenate(obs_39, one_hot_10)** → **49-dim**.

**Collected data and usage:** The MT-10 dataset is saved as `baseline/data/expert_data_mt10.npz` (keys: `states`, `actions`, `task_ids`, and optionally `goal_indices`, `task_names`). One trajectory per (task, goal); a few goals may be missing if the scripted expert fails within 500 steps. **Collect:** `cd baseline/scripts; python collect_one_per_goal.py --mt10` (see COMMANDS.md). **Train (once implemented):** `train.py --mt10 --data baseline/data/expert_data_mt10.npz`. **Eval:** `test.py --suite mt10` with a 49-dim policy.

---


## 2. Extend `train.py`

- **New CLI**:
  - `--mt10`: Enable MT-10 mode (multi-task, one-hot).
  - `--data`: Unchanged for single task (one path). For MT-10: either one path to a combined npz that has `task_ids`, or multiple paths (one npz per task, order = task index 0..9).
- **Data loading (MT-10)**:
  - If `task_ids` present in npz (or multiple files): for each trajectory, append one-hot(task_id) to every state → `X_train` has shape `(N, 49)`. Actions stay `(N, 4)`.
  - If single file and no `task_ids`: keep current behavior (39-dim, single-task).
- **Model**: `ClonePolicy(input_dim, output_dim)` with no structural change. For MT-10: `input_dim=49`, `output_dim=4`.
- **Eval after training**:
  - **Single-task (current)**: Keep `eval_50_goals(policy, task_name=..., ...)` for the one task (e.g. reach-v3).
  - **MT-10**: Add `eval_mt10(policy, clip_actions, eval_seed)`: for each task index `k` in 0..9, create MT1 env for `MT10_TASKS[k]`, run 50 episodes (1 per goal), feed **concat(obs, one_hot(k))** to the policy; record success rate per task and average. Optionally only run eval on a subset of tasks (e.g. reach first) to save time.
- **Run logging**: For MT-10 runs, log `success_rate_per_task` (list of 10 floats) and `success_rate_avg` in `training_runs.json` and summarize in `RUNS_SUMMARY.md` (e.g. one column for “reach” and “avg” or a small table).

**Constants**: Define `MT10_TASKS = ['reach-v3', 'push-v3', 'pick-place-v3', 'door-open-v3', 'door-close-v3', 'drawer-open-v3', 'drawer-close-v3', 'button-press-v3', 'lever-pull-v3', 'window-open-v3']` in `train.py` (or a small shared `mt10_config.py`).

---

## 3. Extend `test.py`

- **New CLI**:
  - `--suite mt10`: Evaluate on full MT-10 (all 10 tasks, 50 goals per task).
  - `--task`: Keep current behavior for single-task (default `reach-v3`). With `--suite mt10`, `--task` can be ignored or used to run only one task by name for debugging.
- **Model loading**:
  - Single-task: `ClonePolicy(39, 4)` (current).
  - MT-10: `ClonePolicy(49, 4)`.
  - Option: save in the checkpoint or in a small sidecar (e.g. `cloned_policy.mt10.json`) a flag `mt10: true` so test can infer 49-dim; or require `--suite mt10` to use 49-dim (simplest).
- **Evaluation (MT-10)**:
  - For each task in `MT10_TASKS`, create MT1 env, run 50 episodes (1 per goal), at each step pass **concat(obs, one_hot(task_idx))** to the policy; clip actions by default (same as train.py; use `--no-clip` to disable).
  - Report: per-task success rate (e.g. table or list), then **average success rate** across the 10 tasks (and optionally overall 500 episodes).
- **Backward compatibility**: Default remains single-task reach-v3, 39-dim, so existing commands and scripts keep working.

---

## 4. Data Collection for MT-10

- **Reuse pattern of `collect_one_per_goal.py`** but for all 10 tasks:
  - New script: `collect_mt10.py` (in `baseline/scripts/`).
  - For each task name in `MT10_TASKS`:
    - Get env and expert policy (Metaworld provides per-task expert policies, e.g. `SawyerReachV3Policy`, `SawyerPushV2Policy`, etc.; need to map task name → policy class).
    - Collect one trajectory per goal (50 goals per task) — same as current reach script.
    - Append task index `0..9` for each trajectory.
  - Save one npz: `states`, `actions`, `task_ids` (length = 10 * 50 = 500 trajectories if all succeed).
- **Fallback for missing experts**: If not all 10 tasks have a ready-made expert in Metaworld, we can either (a) implement only the tasks that do, or (b) use a script that collects only reach first and then add others as needed. Document which tasks have expert data.
- **Alternative (faster to implement)**: Start with **reach-only MT-10 data**: one npz with only reach trajectories and `task_ids = 0` for all. Then train with `--mt10` so the policy gets 49-dim input (39 + one-hot for reach). This checks the pipeline; then add more tasks and their experts to `collect_mt10.py`.

---

## 5. Implementation Order

1. **Constants and one-hot helper**  
   Add `MT10_TASKS` and `def one_hot_task(task_id, num_tasks=10)` (returns shape `(10,)`) in `train.py` (or shared module).

2. **train.py: MT-10 data loading**  
   - If `--mt10` and single `--data` with `task_ids` in npz: load states/actions/task_ids, append one-hot to states, build X (49-dim), Y (4-dim), same weighting/upsampling as now.  
   - If `--mt10` and multiple `--data` paths: load each npz, assign task_id by index, concatenate all trajectories with one-hot; then same as above.

3. **train.py: eval_mt10 and run record**  
   - Implement `eval_mt10(policy, clip_actions, eval_seed)` using MT1 per task and 50 goals per task, feeding concat(obs, one_hot(k)).  
   - When `--mt10`, after training call `eval_mt10` instead of `eval_50_goals`, and log per-task + average success in `training_runs.json` and summary.

4. **test.py: --suite mt10**  
   - If `--suite mt10`: load `ClonePolicy(49, 4)`, run 50 episodes per task (1 per goal) for each of 10 tasks, report per-task and average.

5. **Data collection: collect_mt10.py**  
   - Implement script that loops over `MT10_TASKS`, uses Metaworld experts where available, collects 50 trajectories per task, saves npz with `task_ids`. Start with reach-only if needed, then extend.

6. **Docs**  
   - Update `TRAINING_AND_TESTING.md` (or add a short section) with: MT-10 usage, `--mt10` and `--suite mt10`, data format, and expectation that baseline should do well on reach and a few tasks and poorly on others.

---

## 6. Expected Outcome

- **Reach (and maybe 1–2 others)**: With enough expert data and same tricks (end weighting, clip, etc.), baseline can achieve high success on reach and possibly push/pick-place.
- **Others (door, drawer, button, lever, window)**: Baseline will likely perform poorly without encoder or extra structure, establishing a clear gap for future encoder-based improvements.

---

## 7. File Summary

| File / change | Purpose |
|---------------|---------|
| `train.py` | Add `--mt10`, load task_ids / multi-file, append one-hot, `input_dim=49` for MT-10; add `eval_mt10()`; log per-task and avg success for MT-10. |
| `test.py` | Add `--suite mt10`, load 49-dim policy, run 50 goals × 10 tasks with one-hot, report per-task and avg. |
| `collect_mt10.py` (new) | Collect expert demos for all 10 tasks (or subset), save npz with `states`, `actions`, `task_ids`. |
| `MT10_PLAN.md` (this file) | Plan and reference for MT-10 integration. |

No new repository structure; everything stays under `baseline/scripts` and `baseline/data`, reusing the existing train/test framework and only adding MT-10 options and one-hot encoding so the policy “knows” which task it’s training on or solving.
