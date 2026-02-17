# Command Reference: Data, Training, Testing, and Visualization

Run all commands from the **project root** (`CS229_FinalProject/`) unless noted.

---

## 1. Data collection

**Requires:** `metaworld` installed (`pip install metaworld`). Run from `baseline/scripts` or project root.

### Single task (default: reach-v3)

Collect **one expert trajectory per goal** (50 total). Saves `baseline/data/expert_data_reach-v3.npz`. Run once before training.

```bash
cd baseline/scripts
python collect_one_per_goal.py
```

With another task or custom output:

```bash
cd baseline/scripts
python collect_one_per_goal.py --task push-v3
python collect_one_per_goal.py --task reach-v3 --output ../data/my_reach.npz
```

### MT-10 (all 10 tasks)

Collect **one trajectory per goal per task** for the MT-10 suite (at most 500 trajectories: 10 tasks × 50 goals). Saves `baseline/data/expert_data_mt10.npz` with `states`, `actions`, `task_ids` (and `goal_indices`, `task_names`). Use this file for training with `train.py --mt10` (when implemented).

```bash
cd baseline/scripts
python collect_one_per_goal.py --mt10
```

With custom output path or directory:

```bash
python collect_one_per_goal.py --mt10 --output ../data/expert_data_mt10.npz
python collect_one_per_goal.py --mt10 --output-dir /path/to/data
```

**CLI reference:**

| Option | Description |
|--------|-------------|
| `--mt10` | Collect for all 10 MT-10 tasks; save combined npz with `task_ids`. |
| `--task NAME` | Single-task mode: task name (default: `reach-v3`). Ignored if `--mt10`. |
| `--output PATH` | Output file path. Default: `../data/expert_data_{task}.npz` or `../data/expert_data_mt10.npz`. |
| `--output-dir DIR` | Output directory when `--output` is not set (default: `baseline/data`). |

From project root:

```bash
python baseline/scripts/collect_one_per_goal.py
python baseline/scripts/collect_one_per_goal.py --mt10
```

---

## 2. Training

**From project root** (unified entrypoint):

```bash
python train.py --approach baseline
```

- Defaults are loaded from **`baseline/train_config.yaml`** (single source of truth); override with CLI flags.
- **W&B:** Logging to Weights & Biases is **on by default**. Use `--no-wandb` to disable. Use `--wandb-tag name:alice` (repeatable) to tag runs. See **WANDB.md** for setup, sweeps, and config.
- Saves to `baseline/models/cloned_policy.pth` and logs to `baseline/training_runs.json`; each run also saved as `baseline/models/runs/run_YYYYMMDD_HHMMSS_*.pth`.

**Common options:**

```bash
# More epochs, custom LR
python train.py --approach baseline --epochs 500 --lr 0.0003

# No end-weighting (plain BC)
python train.py --approach baseline --end-weight 1.0

# Custom save name
python train.py --approach baseline --name latest.pth

# Custom data path
python train.py --approach baseline --data path/to/expert_data.npz

# More epochs (e.g. 750-1000) for best config
python train.py --approach baseline --end-inner-weight 5.0 --end-inner-fraction 0.12 --epochs 750

# Optional LR decay (e.g. multiply LR by 0.5 every 250 epochs)
python train.py --approach baseline --epochs 750 --lr-decay-epoch 250 --lr-decay-gamma 0.5

# Architecture ablation (e.g. larger first layer)
python train.py --approach baseline --hidden 512 256 128
```

**Direct baseline script** (full options):

```bash
cd baseline/scripts
python train.py --epochs 100
```

- Training uses **all** samples in the .npz (no 50k subsampling).

---

## 3. Testing

**Single entrypoint:** Use **`python test.py`** from the project root for all testing (baseline and other approaches). Baseline eval logic lives in `baseline/scripts/test.py`; root delegates to it when `--approach baseline`. You can still run `baseline/scripts/test.py` directly for quick baseline-only runs (e.g. from `baseline/scripts`); root is the recommended entrypoint.

### Single-task (MT1) — from project root

**50-goal eval** (1 episode per goal; deterministic):

```bash
python test.py --approach baseline --model cloned_policy.pth --episodes 50 --seed 42
```

- **Clip**: Actions are clipped to [-1, 1] by default (same as train.py). Use `--no-clip` to disable.
- `--seed 42`: reproducible eval (use same seed as training eval for comparable numbers).
- Success rate = fraction of the 50 goals the policy solved.
- Training and testing both default to clip; use `--no-clip` in either to disable.
- **Device:** Use `--device auto` (default), `cuda`, `xpu`, or `cpu` for baseline.

**Model path shortcuts:**

- `latest.pth` or `cloned_policy.pth`: under `baseline/models/`.
- Run checkpoints use descriptive names: `baseline/models/runs/run_YYYYMMDD_HHMMSS_end3_inner5x10_noclip.pth` (end weight, inner tier, clip/noclip).
- `latest-upsampled-end`: resolves to the **latest run with end_weight ≠ 1** in `baseline/training_runs.json`.

**Other test modes:**

```bash
# One episode with visualization (watch the robot)
python test.py --approach baseline --model latest.pth --episodes 1 --visualize

# N episodes in series in same window
python test.py --approach baseline --model latest.pth --visualize-series 5

# N envs in parallel (different goals, side by side)
python test.py --approach baseline --model latest.pth --visualize-parallel 5
```

### Testing an MT-10 model

From project root, use **`--suite mt10`** to run a **49-dim** multi-task policy over all 10 MT-10 tasks (50 episodes per task, 1 per goal):

```bash
python test.py --approach baseline --model runs/run_YYYYMMDD_HHMMSS_end1_clip_mt10.pth --suite mt10
```

**Examples:**

```bash
# Basic MT-10 eval (clip is default, same as training)
python test.py --approach baseline --model runs/run_20260216_185600_end1_clip_mt10.pth --suite mt10

# Reproducible eval with seed
python test.py --approach baseline --model runs/run_20260216_185600_end1_clip_mt10.pth --suite mt10 --seed 42

# With device selection
python test.py --approach baseline --model runs/run_20260216_185600_end1_clip_mt10.pth --suite mt10 --device cuda
```

**Output:** Per-task success rate (%) for each of the 10 tasks, then the average success rate across all tasks.

**Optional — direct baseline script:** You can also run the baseline test script directly (e.g. from `baseline/scripts`):

```bash
cd baseline/scripts
python test.py --model runs/run_20260216_185600_end1_clip_mt10.pth --suite mt10
```

**MT-10 / test options (root test.py):**

| Option | Description |
|--------|-------------|
| `--model` | **Required.** Model filename or path (under `baseline/models/` when relative). |
| `--suite mt10` | Use `mt10` to test multi-task (49-dim); default is `mt1` (single task, 39-dim). |
| `--no-clip` | Do not clip actions (default: clip to [-1, 1], same as train.py). |
| `--seed N` | Env seed for reproducibility (e.g. `--seed 42`). |
| `--verbose N` | Print progress every N episodes (0 = off). |
| `--device` | `auto` (default), `cuda`, `xpu`, or `cpu` (baseline only). |

Note: When `--suite mt10`, evaluation is fixed at 50 goals per task, 10 tasks; `--episodes` and `--task` are ignored.

---

## 4. View 3 success + 3 fail (visualize-success-fail)

**Single run:** eval 50 goals (no render), then show **3 success** and **3 fail** episodes **with** rendering in the **same** env so labels match.

Use **`--visualize-success-fail N`** and **`--task <name>`** together. The task name selects which task to run (for both MT1 and MT10). Invalid `--task` prints the list of valid tasks and exits with an error.

**MT1 (single-task, 39-dim policy):**

```bash
python test.py --approach baseline --model latest.pth --visualize-success-fail 3 --task reach-v3
python test.py --approach baseline --model latest.pth --visualize-success-fail 3 --task door-open-v3
```

**MT10 (multi-task, 49-dim policy):** Add `--suite mt10` and choose the task with `--task`:

```bash
python test.py --approach baseline --model <mt10_model>.pth --suite mt10 --visualize-success-fail 3 --task door-open-v3
```

- Uses first 50 episodes (no window), then N success + N fail with the MetaWorld window. Clip is on by default.
- Terminal prints: `>>> Episode j/6: Goal X — SUCCESS` or `FAIL`, then `-> Env result: success/fail`. Because it’s the same env, these match.
- With `latest-upsampled-end`: same flow but uses the latest upsampled-end run’s checkpoint.

```bash
python test.py --approach baseline --model latest-upsampled-end --visualize-success-fail 3 --task reach-v3
```

---

## 5. Diagnostics (when success rate is below target)

**Visualize failures** to see if failures are "almost there" (wrong final approach) vs completely wrong direction:

```bash
python test.py --approach baseline --model baseline/models/runs/run_YYYYMMDD_HHMMSS.pth --seed 42 --visualize-success-fail 3
```

- Uses the same 50-goal run; then shows 3 success + 3 fail episodes with rendering so you can inspect behavior.

**Per-goal analysis:** Each run in `baseline/training_runs.json` has a `failed_goals` list (goal indices 0–49). If the **same goals fail across runs**, those goals may have weak or bad coverage in the expert data; consider collecting more demos for those goals or checking data balance. Inspect `baseline/RUNS_SUMMARY.md` or the JSON for the list.

**Loss vs success:** If final loss is very low but success rate plateaus, the model is fitting the data but not generalizing to rollout (classic BC issue). Options: more/better data, or (beyond baseline) DAgger / online correction.

---

## 6. Comparing runs and inspecting data

**Compare runs:** Inspect `baseline/training_runs.json` and `baseline/RUNS_SUMMARY.md` for timestamps, end_inner settings, success_rate, and failed_goals. Per-run checkpoints are in `baseline/models/runs/run_YYYYMMDD_HHMMSS.pth`.

**Check expert data size** (optional):

```bash
cd baseline/scripts
python check_data_len.py   # if present: prints trajectory count and total samples
```

---

## Quick reference

| What              | Command |
|-------------------|--------|
| Collect expert (single task) | `cd baseline/scripts; python collect_one_per_goal.py` |
| Collect expert (MT-10) | `cd baseline/scripts; python collect_one_per_goal.py --mt10` |
| Train             | `python train.py --approach baseline` |
| Eval 50 goals     | `python test.py --approach baseline --model latest.pth --episodes 50 --seed 42` |
| **Eval MT-10 model** | `python test.py --approach baseline --model runs/run_YYYYMMDD_HHMMSS.pth --suite mt10` |
| View 3 success + 3 fail | `python test.py --approach baseline --model latest.pth --seed 42 --visualize-success-fail 3 --task reach-v3` |
| View 3/3 (MT-10, one task) | `python test.py --approach baseline --model <mt10>.pth --suite mt10 --visualize-success-fail 3 --task door-open-v3` |
| View 3/3 (upsampled-end model) | `python test.py --approach baseline --model latest-upsampled-end --visualize-success-fail 3 --task reach-v3` |
