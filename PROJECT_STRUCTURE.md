# CS229 Imitation Learning Project

## Goal

**Train a single policy that succeeds on as many of the 50 distinct reach-v3 goals as possible.** Evaluation: run 1 episode per goal (50 total); success rate = fraction of goals solved.

### How the baseline policy is trained (reach-v3)

- **End-of-trajectory weighting:** We up-weight the **last fraction** of each expert trajectory (default: last 30%, weight 3×) so the policy fits “approach + final reach” better and reduces “gets close then drifts” failures. Use `--end-weight 1.0` to disable; use `--end-fraction 0.2` or `--end-weight 5` to tune.
- **Note:** This weighting is tuned for reach (where the main failure was “almost there”). **We do not know whether it is desirable for other MT-10 tasks** (push, pick-place, etc.); it may help, hurt, or be neutral depending on the task. Disable with `--end-weight 1.0` when training or evaluating on other tasks if you want unweighted BC.

## Minimal pipeline (recommended)

1. **Collect expert data** (50 trajectories, 1 per goal; fast):
   ```bash
   cd baseline/scripts && python collect_one_per_goal.py
   ```
   Writes `baseline/data/expert_data_reach-v3.npz`.

2. **Train** (from repo root; uses end-of-trajectory weighting by default):
   ```bash
   python train.py --approach baseline
   ```
   Saves to `baseline/models/cloned_policy.pth`. To train without end-weighting: `--end-weight 1.0`.

3. **Evaluate** (50 episodes = 1 per goal):
   ```bash
   python test.py --approach baseline --model cloned_policy.pth --episodes 50
   ```
   Reported success rate = fraction of the 50 goals the policy solved.

## Project Structure

```
CS229_FinalProject/
├── baseline/              # Baseline behavioral cloning (reach-v3)
│   ├── models/           # Trained policy .pth files
│   ├── scripts/          # Training and utility scripts
│   └── data/             # Expert trajectories (.npz files)
│
├── vae/                  # VAE-based representation (TODO)
│   ├── models/
│   ├── scripts/
│   └── data/
│
├── tce/                  # Temporal Contrastive Encoding (TODO)
│   ├── models/
│   ├── scripts/
│   └── data/
│
├── hybrid/               # Hybrid VAE + TCE approach (TODO)
│   ├── models/
│   ├── scripts/
│   └── data/
│
├── train.py              # Unified training script
├── test.py               # Unified testing script
├── archive/              # Legacy scripts (see archive/README.md)
└── README.md
```

## Quick Start

### Training

Train a baseline policy with default hyperparameters:
```bash
python train.py --approach baseline
```

Train with custom hyperparameters:
```bash
python train.py --approach baseline --lr 0.001 --epochs 50 --batch 32 --name my_policy.pth
```

Train with action clipping:
```bash
python train.py --approach baseline --clip --epochs 50 --lr 0.001
```

### Testing

Test on all 50 goals (1 episode per goal; sufficient because env is deterministic):
```bash
python test.py --approach baseline --model cloned_policy.pth --episodes 50
```
Success rate = fraction of the 50 goals succeeded.

Test with action clipping:
```bash
python test.py --approach baseline --model my_policy.pth --clip
```

Test on a different task:
```bash
python test.py --approach baseline --model cloned_policy.pth --task push-v3
```

## Available Scripts

### baseline/scripts/collect_one_per_goal.py
Collects exactly 1 expert trajectory per goal (50 total). Saves to `baseline/data/expert_data_reach-v3.npz`. Run once before training.

### baseline/scripts/train.py
Training script for the baseline approach. Supports custom LR, epochs, batch size, hidden sizes, action clipping, end-weight/end-fraction, and `--data` path. By default each run is logged and models are kept in unique files:
- **baseline/training_runs.json** – full list of run records (timestamp, hyperparameters, final_loss, success_rate, goal_success, failed_goals, run_path). All runs are stored here for hyperparameter tuning.
- **baseline/models/runs/** – each run saved as `run_YYYYMMDD_HHMMSS.pth` (unique per run). Last `--keep-runs` copies kept (default 50; use `--keep-runs 0` to keep all). Latest run is also copied to `cloned_policy.pth`.
- **baseline/RUNS_SUMMARY.md** – table of the **last 10 runs** (timestamp, epochs, end_weight, final_loss, success_rate, failed_goals). Regenerated after each training run.
- Use `--no-save-run` to skip logging and run copies.

### baseline/scripts/compare_runs.py
Compare which goals different policies failed on. Reads `training_runs.json` and prints goals failed in all runs, goals failed in at least one run, and pairwise overlap. Usage: `python compare_runs.py` (last 10 runs) or `python compare_runs.py --last 5` or `python compare_runs.py --timestamps 20260214_222014 20260214_120000`.

### baseline/scripts/test.py
Testing script (rarely used directly—prefer root `test.py`).

## Next Steps

1. **Baseline** - Improve success rate on the 50 goals (data, architecture, training)
2. **VAE representation** - Smooth latent manifold encoding (TODO)
3. **TCE** - Temporal contrastive learning (TODO)
4. **Hybrid** - Combine VAE + TCE (TODO)
5. **MT-10** - Evaluate on all 10 MetaWorld tasks (optional)

## Dependencies

- PyTorch
- MetaWorld (v1+)
- NumPy
- (See requirements or environment setup docs)

## Notes

- All models are stored as PyTorch state dicts (.pth files)
- Expert data is in .npz format (numpy compressed)
- Scripts automatically handle path resolution
- Test episodes use canonical training tasks unless perturbed

