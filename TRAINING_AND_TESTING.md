# Training and Testing: The 50 Goals Explained

## What are the 50 "goals"?

- **One task**: `reach-v3` = "move the robot hand to a target position."
- **50 goal variations**: MetaWorld defines 50 different **target positions** (`mt1.train_tasks[0]` … `mt1.train_tasks[49]`). So the *task* is the same (reach), but the *goal location* is different each time.

So you have one task type and 50 different goal positions. The observation (39-dim) includes the current state **and** the goal (e.g. goal position in 3D). So the policy's input changes with the goal, and the expert's actions are **not** the same across goals—they depend on where the target is.

---

## Is the expert data "the exact same" for each goal?

**No.** What is the same:

- The **expert policy** (same `SawyerReachV3Policy`).
- The **data collection procedure** (reset, run expert, record (s,a)).

What is **different**:

- For each of the 50 goals, the **states** are different (different goal coordinates in the obs).
- So the **actions** are different (expert moves toward that goal).
- So the **(state, action) pairs** in the dataset are **not** identical across goals.

If you ever had only one goal (e.g. only `train_tasks[0]`) in your expert data, then all demos would be for that single goal and the data would be "the same" in the sense of same goal; that would hurt generalization to the other 49 goals. You need data from **all 50 goals** so the policy sees different (s,a) pairs.

---

## What is actually in the expert data file?

If you use **`collect_one_per_goal.py`** (the minimal dataset):

- **50 goals** → 50 expert trajectories (one per goal).
- The `.npz` has two arrays: `states` and `actions`, each of length 50. Each element is a **trajectory**: an array of (state or action) vectors, one per timestep.
- **Total (s,a) pairs** = sum of trajectory lengths. Example: if you have **2474 samples**, that’s 2474 ÷ 50 ≈ **49 steps per goal on average**. So yes: roughly 49 “frames” of the expert reaching toward the ball per goal.
- Trajectory length **varies per goal**: the expert runs until success (or 500 steps). So one goal might have 45 steps, another 52; the total is just the sum. Training concatenates all trajectories into one flat list of (s,a) pairs and does not use goal identity—the state already encodes the goal.

So: **50 goals, ~49 frames per goal on average (2474 total samples)** is the right picture.

---

## Expert data collection: do we need multiple episodes per goal?

**No.** Same logic as testing: same goal + deterministic reset → same initial state; expert policy is deterministic → same actions every time. So the expert solves each goal in **exactly the same way** every time. Multiple episodes per goal just record the **same trajectory** (same sequence of (s,a) pairs) over and over—they don't add new diversity.

So for **diversity** of expert behavior, **one trajectory per goal is enough** (50 trajectories total). That gives you 50 different “ways” the expert solves the task (one per goal). Collecting 40 episodes per goal gives you the same 50 trajectories repeated 40 times—redundant.

**In practice:** Your scripts (e.g. `collect_balanced_data.py`) may use 40 episodes × 50 tasks = 2000 trajectories. That’s 40 identical copies per goal. The *unique* (s,a) content is still just 50 trajectories. Training on 2000 trajectories means each transition is seen 40× more often; that can still affect optimization (e.g. more gradient steps on the same data). So you can either:
- Collect **1 episode per goal** (50 total) for a minimal, non-redundant dataset, or  
- Keep 40 per goal if you want a larger dataset size for training (same diversity, more repeated samples).

---

## Training

- You load one `.npz` with many **(state, action)** pairs from the expert.
- Those pairs come from **many episodes** and (if you use random or balanced collection) **many of the 50 goals**.
- Training ignores "which goal" explicitly; the model just learns **state → action**. Because the state includes the goal, the policy can (and should) learn to behave differently for different goals.
- So: one training run, one policy, trained on data that (ideally) spans all 50 goals.

---

## Testing

- You evaluate the **same** policy on **multiple episodes**.
- Each episode: you pick one of the 50 tasks with `env.set_task(task)`, reset, and run the policy until done; you record success or failure.
- In `test.py` you do:
  ```python
  task = mt1.train_tasks[i % len(mt1.train_tasks)]
  ```
  So with 100 episodes you cycle through all 50 goals twice. Because reset is deterministic per task, the second run on each goal gives the same outcome as the first—so 50 episodes is enough for the success rate.

**Same task multiple times = same outcome?**
**Yes.** For reach-v3, `env.reset()` is **deterministic** for a given task (verified: 3 resets on task 0 gave identical observations). Your policy is deterministic. So: same goal + same policy → same initial state → same trajectory → same outcome. Running 10 episodes on the same goal just repeats the same result 10 times—it does **not** add new information.

So you do **not** need multiple episodes per goal. **One episode per goal is enough** (50 episodes total). Running 100 or 500 only repeats the same 50 outcomes; the success rate is unchanged.

---

## Summary

| Question | Answer |
|----------|--------|
| Are the 50 goals the same? | Same **task** (reach), 50 **different goal positions**. |
| Is expert data identical for every goal? | No. Same expert, same procedure, but different (s,a) because state includes goal. |
| Do we need multiple expert trajectories per goal? | No. Expert is deterministic; same goal → same trajectory. 1 trajectory per goal (50 total) is enough for diversity. |
| Do multiple test episodes per goal add information? | No. Same goal + deterministic policy = same outcome. 50 episodes (1 per goal) is enough. |
| What does our test script do? | Cycles through the 50 tasks. 50 episodes = 1 per goal (sufficient). 100+ just repeats the same 50 outcomes. |

For a **per-goal** breakdown (which goals are hard?), use:

```bash
cd baseline/scripts && python per_task_eval.py
```

Per goal you get the same outcome every time (deterministic), so each goal's rate is either 0% or 100%.

---

## Bug checks and tips

- **Observation**: Must be flat 39-dim for reach-v3. Test script now flattens `obs` so 1D or 2D from env both work.
- **Actions**: MetaWorld expects actions in **[-1, 1]**. The expert can output larger values (env clips them). Training on raw expert actions is fine; at test time use `--clip` so model outputs are clipped to [-1,1] before `env.step()`.
- **reset() / step()**: Script handles both (obs, info) and older single-value reset; and 5-tuple vs 4-tuple step return.
- **Epochs**: With only 50 trajectories (~2.5k samples), training longer often helps. Default is now **100 epochs**; try more if loss is still high.
- **Visualize**: Run with `--visualize --episodes 1` (and optionally `--clip`) to watch one episode. For **3 success + 3 fail** in one run (labels match): `python test.py --approach baseline --model latest.pth --clip --visualize-success-fail 3`. See **COMMANDS.md** for full command reference.

---

## How is success evaluated?

**Per step, not “at any point in time”.** The environment checks every step whether the end-effector is within a small distance (epsilon) of the goal. When it is, the env sets `info['success'] = True` and typically **terminates the episode** on that step. So if the robot ever truly reached the goal, the episode would end immediately and we’d count success—you wouldn’t see more steps after that.

So when you see the arm **get to the ball and then drift away**, it means:

- The arm got **close** (looks “in grasp”) but **never** entered the success threshold.
- The episode keeps going.
- The policy then outputs actions that move the arm away, so it drifts.

So success is **not** “ball in grasp at any point”; it’s “end-effector within epsilon of goal on some step, and the episode ends there.” Our script records success from the **last step** of the episode; in practice that’s the step where the env set `terminated=True` (either from success or timeout).

---

## Why does the arm reach the ball then drift off?

Common reasons with behavioral cloning:

1. **Strict success threshold** – “Ball in grasp” visually can still be slightly beyond the env’s epsilon, so the episode doesn’t end and the policy keeps acting (often badly) near the goal.

2. **Distribution shift / poor “near goal” behavior** – Training data has many (s,a) pairs for “moving toward goal” and relatively few for “almost at goal, make small corrections.” The policy is good at approaching but not at holding or fine-tuning near the goal. When the state is “almost there” but not exactly like the expert’s trajectory, the policy can output an action that was meant for a different part of the path, so the arm moves away (e.g. 90° drift).

3. **Compounding errors** – Small errors earlier put the arm in a state the expert rarely visited. The policy wasn’t trained well for that state, so it outputs a wrong action; that can look like “reach then suddenly go the wrong way.”

4. **No explicit “stay at goal”** – The expert may reach and stop quickly. We only have a short tail of “at goal” (s,a) pairs. The clone doesn’t learn to “hold” the pose and can drift after getting close.

**Things to try:** more/better data near the goal, train longer, add action clipping, or use a loss that up-weights states near the goal (e.g. goal-weighted behavioral cloning).

**Implemented:** We use **end-of-trajectory weighting** by default: the last 30% of each expert trajectory is weighted 3× in the BC loss so the policy fits the final approach better. See PROJECT_STRUCTURE.md; this is tuned for reach and we do not know if it is desirable for other MT-10 tasks.

---

## Why did 50 epochs + ~94k expert data get ~50% success, but 100 epochs + 2.5k data gets 0%?

The difference is **how many gradient updates** the model gets, not how many *unique* behaviors it sees.

- **94k samples**: With batch_size=64, each epoch has ~1,470 batches. In 50 epochs that’s ~73,500 gradient updates. The model sees the same underlying 50 trajectories over and over (many copies), so it gets many chances to fit each (s,a) pair.
- **2.5k samples**: Each epoch has ~39 batches. In 100 epochs that’s ~3,900 gradient updates—about **19× fewer** than above. So the model gets far fewer chances to fit the same content.

So the old setup didn’t succeed because the data was “more diverse”; the expert trajectories were still the same 50 goals. It succeeded because **duplicate (s,a) pairs = more batches = more gradient steps = better fit** to the same expert behavior (including near the goal).

**Takeaway:** Even if copies are “all the same,” having more copies in the dataset helps by increasing updates per epoch. So either train for **many more epochs** on the small dataset, or **add duplicate copies** of the 50 trajectories so the dataset is larger and you get more batches per epoch.
