# Chat handoff — CS229 Final Project (baseline BC, reach-v3)

Use this to start a new chat and continue working on the project.

## Project goal

Train a single policy that succeeds on as many of the **50 reach-v3 goals** as possible. Eval: 1 episode per goal (50 total); success rate = fraction of goals solved.

## Current state (baseline)

- **Tail-end upsampling:** Two-tier weighting is implemented. Outer: last 30% of each traj at 3×. Inner (optional): last 5% or 10% at 5× or 7× via `--end-inner-weight` and `--end-inner-fraction`.
- **Best so far:** Level 2 (5× last 10%) gave the highest success rate. Level 1 (5× last 5%) and Level 3 (7× last 5%) were worse.
- **Default training:** 500 epochs. Eval at end of training uses seed 42 and matches `test.py --seed 42` (RNG is seeded in both).
- **Default training:** Clip actions (default). Use `--no-clip` to disable. Always test with `--seed 42 --clip` for comparable numbers.
- **Reproducible test:** `python test.py --approach baseline --model <path> --episodes 50 --seed 42 --clip`

## Commands (see COMMANDS.md)

- **Train (e.g. Level 2 again):**  
  `python train.py --approach baseline --end-inner-weight 5.0 --end-inner-fraction 0.1`
- **Test (reproducible):**  
  `python test.py --approach baseline --model baseline/models/runs/run_YYYYMMDD_HHMMSS.pth --episodes 50 --seed 42`
- **Visualize 3 success + 3 fail:**  
  `python test.py --approach baseline --model <path> --seed 42 --clip --visualize-success-fail 3`

## Next: tune one variable at a time

- **Approach:** Use a single baseline (500 epochs, end 3.0, inner 5.0@10%, clip) and change **one** variable per run. See **`baseline/TUNING_APPROACH.md`** for the anchor config and suggested one-at-a-time experiments.
- Compare with `test.py --seed 42 --clip` and use `baseline/RUNS_SUMMARY.md` and `baseline/training_runs.json` to track runs.

## Key files

- **Root:** `train.py`, `test.py`, `COMMANDS.md`
- **baseline:** `scripts/train.py`, `data/expert_data_reach-v3.npz`, `models/runs/*.pth`, `training_runs.json`, `END_UPSAMPLING_PLAN.md`, `RUNS_SUMMARY.md`
- **Venv:** run from project root; activate `venv` if needed.
