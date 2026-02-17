# Archive

Legacy scripts and old artifacts. Not needed for the current pipeline.

- **scripts_legacy/** – Old data collection and analysis scripts (replaced by `baseline/scripts/collect_one_per_goal.py`):
  - `collect_data.py` – Random mix of goals, many episodes
  - `collect_balanced_data.py` – 40 episodes × 50 goals
  - `collect_expert_data_split.py` – Train/test goal split
  - `analyze_expert_variation.py` – Trajectory variation analysis
  - `per_task_eval.py` – Per-goal eval (10 eps per goal; redundant with 1 per goal)
- **analyze_data.py** – Old data stats (hardcoded `attempt 1` path)
- **scripts/** – Other archived training/testing variants

Current pipeline: see PROJECT_STRUCTURE.md (collect_one_per_goal → train → test with 50 episodes).
