# End upsampling tuning plan (approach baseline)

**Goal:** Improve success on goals where the policy is “almost there”—grasping slightly off center. Tuning focuses on upsampling the **very end** of trajectories (final approach) on top of the existing 3× last-30% that already helped.

**Idea:** Add a second tier: **5× (or higher) for the last 5–10%** of each trajectory, so the model sees more of the final approach/grasp.

---

## Four levels to run and test

| Level | Description | Train command | Notes |
|-------|-------------|---------------|--------|
| **0** | Baseline (current) | `python train.py --approach baseline` | 3× last 30% only. Use as reference (or skip if already run). |
| **1** | Inner 5%, 5× | `python train.py --approach baseline --end-inner-weight 5.0 --end-inner-fraction 0.05` | 3× last 30%; last 5% at 5×. |
| **2** | Inner 10%, 5× | `python train.py --approach baseline --end-inner-weight 5.0 --end-inner-fraction 0.1` | 3× last 30%; last 10% at 5×. |
| **3** | Inner 5%, 7× | `python train.py --approach baseline --end-inner-weight 7.0 --end-inner-fraction 0.05` | Stronger emphasis on very last 5%. |

Each run is logged to `baseline/training_runs.json` and a run copy is saved under `baseline/models/runs/run_YYYYMMDD_HHMMSS.pth`. The script runs a 50-goal eval after training and records success rate and failed goals.

---

## Workflow

1. **Train all four levels** (run the four commands above). Level 0 can be skipped if you already have a recent “3× last 30%” run to use as reference.
2. **Compare success rates** in `baseline/RUNS_SUMMARY.md` (and `baseline/training_runs.json` for full run details).
3. **Test the best run(s)** explicitly:
   ```bash
   python test.py --approach baseline --model baseline/models/runs/run_YYYYMMDD_HHMMSS.pth --episodes 50
   ```
4. **Optional: visualize failures** for the best config to see if failures are still “slightly off center” and decide next steps (e.g. 8× last 5%, or 5× last 7%):
   ```bash
   python test.py --approach baseline --model baseline/models/runs/run_YYYYMMDD_HHMMSS.pth --visualize-success-fail 3
   ```

---

## Summary

- **Level 0:** 3× last 30% (current baseline).
- **Levels 1–3:** Same 3× last 30%, plus an **inner tier** (last 5% or 10% at 5× or 7×).
- Train → compare success rates → test best run → optionally visualize 3 success + 3 fail to guide further tuning.
