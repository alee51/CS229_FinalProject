# Training run comparison (from session)

These three runs were done before run-logging was added. Recorded here so we can compare training error across them.

| Run | Epochs | End weighting | Final loss (train) | Test success (50 goals) |
|-----|--------|----------------|--------------------|-------------------------|
| **1** | 500 | No (`--end-weight` didn’t exist yet) | ~0.000005 | 54% |
| **2** | 1000 | No | ~0.000008 | 58% |
| **3** | 100 | Yes (last 30% × 3) | ~0.000029 | **58%** |
| **4** | 500 | Yes (last 30% × 3) | ~0.000009 | **54%** |

So:

- **Run 3 (upsampled ending)** has **higher** training loss (~0.000029) than runs 1 and 2 (~0.000005–0.000008) because the loss is **weighted**: the last 30% of each trajectory counts 3×, so the optimizer prioritizes fitting that part. The raw MSE on “all steps equally” would be lower; the weighted loss is what we optimized.
- Runs 1 and 2 had **lower** (unweighted) loss but **54% and 58%** success; run 3 is intended to improve success by fitting the end of the trajectory better, even if the weighted training error is higher.

- **Run 4** (500 epochs, same end-weighting): lower training loss (0.000009) but **54%** test success—slightly worse than run 3 (58%). So with end-weighting, 100 epochs did better than 500 on this eval (possible overfitting or variance).

From now on, every new train is logged in `baseline/training_runs.json` with final loss and hyperparameters.
