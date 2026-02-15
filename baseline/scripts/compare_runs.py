"""
Compare failed goals across training runs. Reads baseline/training_runs.json.
Usage:
  python compare_runs.py                    # overlap of failed goals across last N runs
  python compare_runs.py --last 5          # last 5 runs
  python compare_runs.py --timestamps 20260214_222014 20260214_120000  # specific runs
"""
import os
import json
import argparse

def load_runs(baseline_dir=None):
    if baseline_dir is None:
        baseline_dir = os.path.join(os.path.dirname(__file__), "..")
    path = os.path.join(baseline_dir, "training_runs.json")
    if not os.path.exists(path):
        print(f"No {path} found.")
        return []
    with open(path, "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Compare failed goals across runs")
    parser.add_argument("--last", type=int, default=10, help="Use last N runs (default: 10)")
    parser.add_argument("--timestamps", type=str, nargs="*", help="Use runs with these timestamps only")
    parser.add_argument("--baseline-dir", type=str, help="Path to baseline/ (default: script dir/..)")
    args = parser.parse_args()

    runs_list = load_runs(args.baseline_dir)
    if not runs_list:
        return

    if args.timestamps:
        runs = [r for r in runs_list if r.get("timestamp") in args.timestamps]
        if len(runs) != len(args.timestamps):
            print(f"Warning: only {len(runs)} of {len(args.timestamps)} timestamps found.")
    else:
        runs = runs_list[-args.last:]

    if not runs:
        print("No runs to compare.")
        return

    print(f"Comparing {len(runs)} run(s):")
    for r in runs:
        ts = r.get("timestamp", "?")
        sr = r.get("success_rate", "?")
        fg = r.get("failed_goals", [])
        print(f"  {ts}: success_rate={sr}%, failed_goals={fg}")
    print()

    # Goals that failed in every run
    all_failed = [set(r.get("failed_goals", [])) for r in runs]
    if all_failed:
        common = set.intersection(*all_failed)
        any_fail = set.union(*all_failed)
        print(f"Goals failed in ALL {len(runs)} runs: {sorted(common)}")
        print(f"Goals that failed in at least one run: {sorted(any_fail)}")
        print()
        # Pairwise overlap
        if len(runs) >= 2:
            print("Pairwise overlap (goals failed in both runs):")
            for i in range(len(runs)):
                for j in range(i + 1, len(runs)):
                    a, b = all_failed[i], all_failed[j]
                    overlap = sorted(a & b)
                    ts_a = runs[i].get("timestamp", "?")
                    ts_b = runs[j].get("timestamp", "?")
                    print(f"  {ts_a} vs {ts_b}: {overlap}")

if __name__ == "__main__":
    main()
