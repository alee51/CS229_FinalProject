# Tuning approach: one variable at a time

To avoid tweaking too many parameters at once and getting unclear results, use a single baseline and change **one** variable per run.

## Baseline (anchor)

- **Anchor epochs:** First anchor was 500 epochs; **current anchor is 1000 epochs** (saved as `models/anchor_1000epochs.pth`).
- **End weight 3.0** (last 30% of each traj at 3×)
- **End inner 5.0 @ 10%** (last 10% at 5×)
- **Clip:** default (yes)
- **Hidden:** [256, 256, 128]
- **LR:** 0.0003, **batch:** 64

Train once with this exact config and record success rate. Then, for each experiment, change **only one** of the following and compare to the baseline.

## One-variable experiments

1. **Epochs:** 1000 (baseline) → try 750 or 1500, keep everything else the same.
2. **End weight:** 3.0 (baseline) → try 2.0 or 4.0, keep inner 5.0@10% and rest same.
3. **End inner weight:** 5.0 (baseline) → try 4.0 or 6.0, keep fraction 10% and rest same.
4. **End inner fraction:** 10% (baseline) → try 5% or 12%, keep inner weight 5.0 and rest same.
5. **LR / batch / hidden:** only after the above are explored; change one at a time.

## Commands

**Baseline run:**
```bash
python train.py --approach baseline --end-inner-weight 5.0 --end-inner-fraction 0.1
```

**Example single change (e.g. more epochs):**
```bash
python train.py --approach baseline --end-inner-weight 5.0 --end-inner-fraction 0.1 --epochs 750
```

Compare success rates in `RUNS_SUMMARY.md` and `training_runs.json`; run names include config (e.g. `end3_inner5x10_noclip.pth`).
