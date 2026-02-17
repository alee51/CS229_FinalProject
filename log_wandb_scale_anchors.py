#!/usr/bin/env python
"""
One-off script to log two W&B runs (scale-anchor-0 and scale-anchor-100) so the
parallel coordinates plot scales all MT10 success-rate axes from 0 to 100%.
Run once per project: python log_wandb_scale_anchors.py
"""
import argparse
import sys
import os

# Project root; allow import of baseline.tasks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline.tasks import get_tasks


def main():
    parser = argparse.ArgumentParser(description="Log W&B scale-anchor runs for MT10 parallel plot 0-100% scaling")
    parser.add_argument("--project", type=str, default="cs229-metaworld", help="W&B project name")
    args = parser.parse_args()

    task_list = get_tasks("mt10")
    tags = ["scale-anchor", "dummy", "mt10"]
    config_stub = {"source": "scale-anchor", "approach": "baseline", "suite": "mt10"}

    import wandb

    # Run 1: all zeros
    wandb.init(project=args.project, job_type="eval", name="scale-anchor-0", tags=tags, reinit=True)
    wandb.config.update(config_stub, allow_val_change=True)
    metrics_0 = {"eval/success_rate_avg": 0, **{f"eval/success_rate_{t}": 0 for t in task_list}}
    wandb.log(metrics_0)
    wandb.finish()

    # Run 2: all 100s
    wandb.init(project=args.project, job_type="eval", name="scale-anchor-100", tags=tags, reinit=True)
    wandb.config.update(config_stub, allow_val_change=True)
    metrics_100 = {"eval/success_rate_avg": 100, **{f"eval/success_rate_{t}": 100 for t in task_list}}
    wandb.log(metrics_100)
    wandb.finish()

    print("Logged scale-anchor-0 and scale-anchor-100. Include these runs in your parallel coordinates panel so axes scale 0-100%.")


if __name__ == "__main__":
    main()
