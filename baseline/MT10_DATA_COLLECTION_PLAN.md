# MT-10 Data Collection Plan

## Goal

Collect expert demonstration data for the MT-10 task suite in a **format ready for the training framework**, with **no duplicate (task, goal) trajectories**: exactly **one trajectory per goal per task**.

- The **task set** is the **MT-10 suite**: the 10 tasks (reach, push, pick-place, door-open, door-close, drawer-open, drawer-close, button-press, lever-pull, window-open). Uniqueness is defined over this set.
- For **each task in MT10**, the environment provides **50 distinct goals** for that task. We collect **one** expert trajectory per goal → at most 50 trajectories per task, **500 trajectories total** for MT-10.
- No extra episodes per goal (same goal + same expert → same trajectory; more episodes would be identical copies).

---

## Output Format (for `train.py --mt10`)

Single combined `.npz` file that the training script can load with `task_ids` and build 49-dim inputs (obs + one-hot).

| Key        | Shape / content | Description |
|-----------|------------------|-------------|
| `states`  | `(num_trajectories,)` array of arrays | `states[i]` is shape `(T_i, 39)` — one trajectory of raw env observations (no one-hot; train script will append one-hot from `task_ids`). |
| `actions` | `(num_trajectories,)` array of arrays | `actions[i]` is shape `(T_i, 4)` — expert actions for that trajectory. |
| `task_ids`| `(num_trajectories,)` int array | `task_ids[i] in 0..9` — task index for trajectory `i`. Same as MT10_PLAN.md. |

**Constraints:**

- Every trajectory corresponds to a **unique (task_id, goal_index)** pair. We iterate over tasks, then over the 50 goals for that task, and collect **one** successful rollout per (task, goal). If the expert fails on a goal, we either skip it (and record it) or retry once; we do **not** add multiple successful trajectories for the same goal.
- Trajectories can be in any order (e.g. all reach, then all push, …), as long as `task_ids[i]` matches the task for `states[i]` / `actions[i]`.

**Optional (for debugging / reproducibility):**

- `task_names`: `(num_trajectories,)` array of strings (e.g. `'reach-v3'`) so we can verify mapping.
- `goal_indices`: `(num_trajectories,)` int array, `goal_indices[i] in 0..49` — which goal within the task. Not required for training but useful to ensure we never have two trajectories with same (task_id, goal_index).

---

## Uniqueness: From MT10 — One Trajectory Per (Task, Goal)

Uniqueness is defined by **MT10**: for each task in the MT-10 suite, collect data from the expert doing **each goal once**.

- **Task set**: The 10 tasks are the **MT-10 task list** (reach-v3, push-v3, …, window-open-v3). We iterate over these 10 tasks; `task_id` is the index in this list (0..9).
- **Goals per task**: For each MT10 task we need that task’s 50 goals. We get them by creating an env for that task (e.g. via `metaworld.MT1(task_name)`), which gives us `train_tasks` — 50 goal variants for that task. So for each task in **MT10**, we collect one trajectory per goal from that task’s 50 goals.
- **Per goal**: For each goal index `g` in 0..49, call `env.set_task(mt1.train_tasks[g])`, `env.reset()`, run the expert until success or 500 steps. If success, append **one** trajectory and record `(task_id, g)`. If failure, do **not** add a trajectory for that (task, goal); optionally retry once or log and skip.
- **No second episode** for the same (task, goal): we never run a second rollout for the same task and goal, so we never get duplicate trajectories.

So: **from MT10** we take the 10 tasks; for each of those tasks we collect the expert doing each of that task’s goals once. Implementation uses `MT1(task_name)` only to obtain the env and the 50 goals *for* that MT10 task; the task set and the “one per goal” rule come from MT10.

---

## Task → Expert Policy Mapping

Metaworld provides one expert policy per v3 env. Use the following (module path → class name from repo):

| task_name (MT10 index) | Policy module | Policy class |
|------------------------|---------------|--------------|
| reach-v3 (0)           | metaworld.policies.sawyer_reach_v3_policy | SawyerReachV3Policy |
| push-v3 (1)            | metaworld.policies.sawyer_push_v3_policy | SawyerPushV3Policy |
| pick-place-v3 (2)      | metaworld.policies.sawyer_pick_place_v3_policy | SawyerPickPlaceV3Policy |
| door-open-v3 (3)       | metaworld.policies.sawyer_door_open_v3_policy | SawyerDoorOpenV3Policy |
| door-close-v3 (4)      | metaworld.policies.sawyer_door_close_v3_policy | SawyerDoorCloseV3Policy |
| drawer-open-v3 (5)     | metaworld.policies.sawyer_drawer_open_v3_policy | SawyerDrawerOpenV3Policy |
| drawer-close-v3 (6)    | metaworld.policies.sawyer_drawer_close_v3_policy | SawyerDrawerCloseV3Policy |
| button-press-v3 (7)    | metaworld.policies.sawyer_button_press_v3_policy | SawyerButtonPressV3Policy |
| lever-pull-v3 (8)      | metaworld.policies.sawyer_lever_pull_v3_policy | SawyerLeverPullV3Policy |
| window-open-v3 (9)     | metaworld.policies.sawyer_window_open_v3_policy | SawyerWindowOpenV3Policy |

Implementation can use a dict or list of `(task_name, policy_class)` and import policies lazily per task to avoid requiring all 10 policies if we only collect a subset of tasks.

---

## Collection Algorithm (Pseudocode)

We iterate over the **MT10 task list**; for each task in MT10, we collect one trajectory per goal (using that task’s 50 goals from the env).

```
MT10_TASKS = [reach-v3, push-v3, ...]  # the 10 MT-10 tasks (source of truth for task set)
all_states, all_actions, all_task_ids = [], [], []
optional: all_goal_indices = []

for task_id, task_name in enumerate(MT10_TASKS):   # each task in MT10
    # Get env and 50 goals for this MT10 task (MT1 is just the API to get goals per task)
    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()
    policy = get_expert_policy(task_name)  # from mapping above

    for goal_idx in range(len(mt1.train_tasks)):   # each of this task's 50 goals, once
        task = mt1.train_tasks[goal_idx]
        env.set_task(task)
        obs, info = env.reset()
        episode_states, episode_actions = [], []
        done, steps = False, 0

        while not done and steps < 500:
            action = policy.get_action(obs)
            episode_states.append(obs)   # raw 39-dim
            episode_actions.append(action)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
            if info.get('success'): break

        if info.get('success'):
            all_states.append(np.array(episode_states))
            all_actions.append(np.array(episode_actions))
            all_task_ids.append(task_id)
            optional: all_goal_indices.append(goal_idx)
        else:
            log or record: (task_name, goal_idx) failed

save npz: states, actions, task_ids [, goal_indices [, task_names]]
```

**Invariant**: For each task in MT10 and each of that task’s 50 goals we run at most one episode and append at most one trajectory. So the number of trajectories is at most 50 × 10 = 500, with no duplicate (task, goal) pairs. Uniqueness is over the MT10 suite: one trajectory per (MT10 task, goal for that task).

---

## File Naming and Location

- **Output path**: e.g. `baseline/data/expert_data_mt10.npz` (one file for the full MT-10 dataset).
- **Optional**: If collecting a subset of tasks (e.g. reach-only for testing), could save to `expert_data_mt10_reach_only.npz` or still use `expert_data_mt10.npz` with only `task_ids=0` trajectories; training script doesn’t care as long as `states`, `actions`, `task_ids` are present.

---

## Validation Before Saving

Before writing the `.npz`:

1. **Uniqueness (MT10)**: If `goal_indices` was collected, check that there are no duplicate `(task_ids[i], goal_indices[i])` pairs — i.e. each (task in MT10, goal index) appears at most once.
2. **Shapes**: Each `states[i].shape[1] == 39`, each `actions[i].shape[1] == 4`, `len(task_ids) == len(states) == len(actions)`.
3. **task_ids**: All in 0..9.

Optionally print a short summary: number of trajectories per task (e.g. 50 for reach, maybe less for others if some goals failed), total (s,a) pairs.

---

## Script to Implement: `collect_mt10.py`

- **Location**: `baseline/scripts/collect_mt10.py`.
- **CLI**: e.g. `--output baseline/data/expert_data_mt10.npz`, optional `--tasks reach-v3 push-v3 ...` to collect only a subset (default: all 10).
- **Logic**: Implement the algorithm above; one episode per (task, goal); save in the format above; no duplicate (task, goal) trajectories.
- **Robustness**: Flatten `obs` to 1D (e.g. `np.asarray(obs).flatten()`) before appending, so each state is (39,) and each trajectory is (T, 39). Match `collect_one_per_goal.py` and `train.py` expectations.

This plan keeps the data format aligned with the training framework and ensures we never collect multiple copies of the same expert doing the same task with the same goal. The task set and uniqueness are defined by **MT10**: for each task in MT10, we collect the expert doing each goal once.
