# Knob-by-knob tuning with visualize 3 success + 3 fail

## Current situation

- **Anchor config** (from [baseline/TUNING_APPROACH.md](TUNING_APPROACH.md)): First anchor was 500 epochs; **anchor is now 1000 epochs** (see `models/anchor_1000epochs.pth`). End weight 3.0, inner 5.0 @ 10%, clip, hidden [256,256,128], LR 0.0003, batch 64.
- **Best recent runs (with clip):** 60% (750 ep, 5.0@12%), 56% (500 ep, 5.0@12%). The exact anchor (5.0@10%, clip) in your table was 32% in one run; a fresh anchor run will give a clean reference.
- **Visualize workflow:** [test.py](../test.py) runs 50 goals (no render), then replays **3 success** then **3 fail** with a **rendered** MetaWorld window. For each of the 6 episodes it prints `Goal X — SUCCESS` or `Goal X — FAIL` and then `Env result: success/fail`. You watch the arm and can note failure mode (e.g. overshoot, wrong direction, drifts away, never gets close).

---

## Workflow per knob

For every experiment:

1. **Train** one run changing only that knob (commands below).
2. **Note** success rate from the script output (and from [baseline/RUNS_SUMMARY.md](RUNS_SUMMARY.md) after it updates).
3. **Run visualize** so you can see how the robot fails:
  ```bash
   python test.py --approach baseline --model baseline/models/runs/<run_file>.pth --seed 42 --visualize-success-fail 3
  ```
   Use the **exact run file** from the training output (e.g. `run_YYYYMMDD_HHMMSS_end3_inner5x10_clip.pth`).
4. **While watching the 3 failures**, note:
  - **Reach-specific:** Does the hand get close then miss (precision)? Wrong direction from the start? Drift near the end? Too slow / times out? Same failure pattern across the 3 or mixed?
5. **Decide next step:** If failures look like "almost there" → try more end emphasis (e.g. inner fraction 12% or inner weight 6). If wrong direction → might need more data or epochs. If mixed → try one other knob and compare.

---

## Suggested order of knobs and values

Stick to **one variable per run**. After each run, run visualize and compare both the **number** (success rate) and the **failure mode** you see.


| Order | Knob                               | Baseline value | Single-change experiments to try                  |
| ----- | ---------------------------------- | -------------- | ------------------------------------------------- |
| 1     | **Epochs**                         | 1000 (anchor)   | 750 to compare, or 1500 if you want to try more   |
| 2     | **End inner fraction**             | 0.1 (10%)      | 0.12 (12%), then 0.05 (5%) if you want to compare |
| 3     | **End inner weight**               | 5.0            | 6.0, then 4.0 if needed                           |
| 4     | **End weight**                     | 3.0            | 4.0 or 2.0 (outer tier only)                      |
| 5     | **LR / batch / LR decay / hidden** | —              | Only after 1–4; change one at a time              |


Rationale: epochs and end-weighting (fraction then weight) most directly affect "getting to the goal" and "final approach"; LR/batch/hidden are noisier and better tuned after the main knobs are set.

---

## Commands reference

**1. Establish baseline (run once and keep this run file as "anchor"):**

```bash
python train.py --approach baseline --end-inner-weight 5.0 --end-inner-fraction 0.1
```

Then:

```bash
python test.py --approach baseline --model baseline/models/runs/<anchor_run>.pth --seed 42 --visualize-success-fail 3
```

Watch the 3 failures and briefly write down the failure mode (e.g. "all 3: close then miss by a bit").

**2. One-knob experiments (repeat for each; replace `<RUN_FILE>` with the new run after each train):**

- Epochs 750 (all else baseline):
  ```bash
  python train.py --approach baseline --end-inner-weight 5.0 --end-inner-fraction 0.1 --epochs 750
  ```
- Inner fraction 12% (all else baseline):
  ```bash
  python train.py --approach baseline --end-inner-weight 5.0 --end-inner-fraction 0.12
  ```
- Inner weight 6 (all else baseline):
  ```bash
  python train.py --approach baseline --end-inner-weight 6.0 --end-inner-fraction 0.1
  ```
- End weight 4 (all else baseline):
  ```bash
  python train.py --approach baseline --end-weight 4.0 --end-inner-weight 5.0 --end-inner-fraction 0.1
  ```

After **each** train, run visualize with that run's `.pth` and compare failure behavior to the anchor.

**3. Visualize (same every time):**

```bash
python test.py --approach baseline --model baseline/models/runs/<RUN_FILE>.pth --seed 42 --visualize-success-fail 3
```

---

## What to look for when watching failures (reach)

When you watch the 3 failure episodes, you can loosely categorize:

- **"Almost there" / precision:** Hand gets near the target but doesn't reach the success threshold (slightly off, or drifts at the end). → Favor **more end emphasis**: e.g. inner fraction 12%, or inner weight 6, or more epochs.
- **Wrong direction / never gets close:** Hand moves the wrong way or to the wrong region. → Could need **more epochs** or **more/better data**; end-weighting alone may not fix it.
- **Timeout / too slow:** Motion is plausible but too slow and episode ends before success. → Less likely for reach; if you see it, consider that some goals may need different pacing (later-phase idea).
- **Consistent vs mixed:** Same failure type on all 3 → one kind of fix. Mixed → might need a mix of knobs or data.

You don't need to change code for this: just run visualize after each run and keep short notes (e.g. in a text file or in [baseline/RUNS_SUMMARY.md](RUNS_SUMMARY.md) as a comment) so you can compare "before/after" when you change a knob.

---

## Optional: short checklist in the repo

If you want a minimal "run log" next to the code, you could add a short section to [baseline/TUNING_APPROACH.md](TUNING_APPROACH.md) (or a small `TUNING_LOG.md`) with:

- Anchor run file and its success rate.
- For each experiment: run file, knob changed, success rate, and one line on failure mode from visualize (e.g. "3/3: close then miss").

That keeps "what I tried and what I saw" in one place without changing any code or scripts.

---

## Summary

1. **Run anchor once** (500 ep, 3.0 end, 5.0@10%, clip) and run visualize; note success rate and how the 3 failures look.
2. **For each knob in order** (epochs → inner fraction → inner weight → end weight): train one run with only that knob changed, then run visualize with that run's `.pth`, compare success rate and failure mode to the anchor.
3. **Use failure mode** to decide the next step (e.g. more end emphasis vs more epochs vs other knobs) and keep brief notes so you're not tweaking blindly.
