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

- Default: 500 epochs, LR 0.0003, batch 64, clip actions; end-of-trajectory weight 3.0 (last 30% of each traj weighted 3×). Use `--no-clip` to disable clipping.
- Saves to `baseline/models/cloned_policy.pth` and logs to `baseline/training_runs.json`; each run also saved as `baseline/models/runs/run_YYYYMMDD_HHMMSS.pth`.

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

**50-goal eval** (1 episode per goal; deterministic):

```bash
python test.py --approach baseline --model cloned_policy.pth --episodes 50 --seed 42 --clip
```

- `--clip`: clip actions to [-1, 1] at test time (recommended for MetaWorld).
- `--seed 42`: reproducible eval (use same seed as training eval for comparable numbers).
- Success rate = fraction of the 50 goals the policy solved.
- Training defaults to clip; use `--no-clip` when training to disable.

**Model path shortcuts:**

- `latest.pth` or `cloned_policy.pth`: under `baseline/models/`.
- Run checkpoints use descriptive names: `baseline/models/runs/run_YYYYMMDD_HHMMSS_end3_inner5x10_noclip.pth` (end weight, inner tier, clip/noclip).
- `latest-upsampled-end`: resolves to the **latest run with end_weight ≠ 1** in `baseline/training_runs.json`.

**Other test modes:**

```bash
# One episode with visualization (watch the robot)
python test.py --approach baseline --model latest.pth --episodes 1 --clip --visualize

# N episodes in series in same window
python test.py --approach baseline --model latest.pth --clip --visualize-series 5

# N envs in parallel (different goals, side by side)
python test.py --approach baseline --model latest.pth --clip --visualize-parallel 5
```

---

## 4. View 3 success + 3 fail (visualize-success-fail)

**Single run:** eval 50 goals (no render), then show **3 success** and **3 fail** episodes **with** rendering in the **same** env so labels match.

```bash
python test.py --approach baseline --model latest.pth --clip --visualize-success-fail 3
```

- Uses **one** `test_policy` call: first 50 episodes (no window), then 6 episodes with the MetaWorld window (first 3 goals that succeeded, then 3 that failed).
- Terminal prints: `>>> Episode j/6: Goal X — SUCCESS` or `FAIL`, then `-> Env result: success/fail`. Because it’s the same env, these match.
- With `latest-upsampled-end`: same flow but uses the latest upsampled-end run’s checkpoint.

```bash
python test.py --approach baseline --model latest-upsampled-end --clip --visualize-success-fail 3
```

---

## 5. Diagnostics (when success rate is below target)

**Visualize failures** to see if failures are "almost there" (wrong final approach) vs completely wrong direction:

```bash
python test.py --approach baseline --model baseline/models/runs/run_YYYYMMDD_HHMMSS.pth --seed 42 --clip --visualize-success-fail 3
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
| Eval 50 goals     | `python test.py --approach baseline --model latest.pth --episodes 50 --seed 42 --clip` |
| View 3 success + 3 fail | `python test.py --approach baseline --model latest.pth --seed 42 --clip --visualize-success-fail 3` |
| View 3/3 (upsampled-end model) | `python test.py --approach baseline --model latest-upsampled-end --clip --visualize-success-fail 3` |
