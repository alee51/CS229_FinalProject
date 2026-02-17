"""Regenerate RUNS_SUMMARY.md from training_runs.json (same logic as train.py)."""
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.join(script_dir, "..")
json_path = os.path.join(baseline_dir, "training_runs.json")
summary_path = os.path.join(baseline_dir, "RUNS_SUMMARY.md")

with open(json_path) as f:
    runs_list = json.load(f)

max_runs = 60
all_recent = runs_list[-max_runs:]
all_recent.reverse()
latest_batch_size = min(5, len(all_recent))
latest_batch = all_recent[:latest_batch_size]
older = all_recent[latest_batch_size:]

def row(r):
    run_path = r.get("run_path", "")
    run_file = os.path.basename(run_path) if run_path else "—"
    ep = r.get("epochs", "")
    ew = r.get("end_weight", "")
    ei_str = (f"{r.get('end_inner_weight')}@{r.get('end_inner_fraction', 0)*100:.0f}%"
              if (r.get("end_inner_weight") and r.get("end_inner_fraction") is not None) else "—")
    ca = r.get("clip_actions")
    clip_str = "yes" if ca is True else ("no" if ca is False else "—")
    fl = r.get("final_loss")
    fl_display = f"{fl * 1e6:.4f}" if fl is not None else "—"
    sr = r.get("success_rate")
    sr = f"{sr}%" if sr is not None else "—"
    fg = r.get("failed_goals", [])
    fg_str = ",".join(str(x) for x in fg[:15]) + ("..." if len(fg) > 15 else "")
    return f"| {run_file} | {ep} | {ew} | {ei_str} | {clip_str} | {fl_display} | {sr} | {fg_str} |"

lines = [
    "# Training runs (all recent)",
    "",
    "Full history in `training_runs.json`. Per-run models in `models/runs/` with descriptive names (end weight, inner tier, clip/noclip).",
    "",
    "**Latest batch** (most recent runs):",
    "",
    "| run_file | epochs | end_weight | end_inner | clip | final_loss (*10e6) | success_rate | failed_goals |",
    "|----------|--------|------------|-----------|------|-----------------|--------------|--------------|",
]
for r in latest_batch:
    lines.append(row(r))
lines.extend([
    "",
    "---",
    "",
    "**Older runs:**",
    "",
    "| run_file | epochs | end_weight | end_inner | clip | final_loss (*10e6) | success_rate | failed_goals |",
    "|----------|--------|------------|-----------|------|-----------------|--------------|--------------|",
])
for r in older:
    lines.append(row(r))

with open(summary_path, "w") as f:
    f.write("\n".join(lines))
print(f"Written {summary_path}")
