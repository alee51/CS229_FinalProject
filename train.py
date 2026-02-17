#!/usr/bin/env python
"""
Unified train script for CS229 project policies.
Defaults are loaded from baseline/train_config.yaml when --approach baseline.
"""

import sys
import os
import argparse


def load_baseline_config(config_path=None):
    """Load baseline train config YAML. Returns dict."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "baseline", "train_config.yaml")
    if not os.path.isfile(config_path):
        return {}
    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Train policies for CS229 project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --approach baseline
  python train.py --approach baseline --epochs 50 --lr 0.001 --name improved.pth
  python train.py --approach baseline --no-wandb --epochs 100
  python train.py --approach baseline --wandb-tag name:alice --suite mt10
        """,
    )
    parser.add_argument("--approach", type=str, default="baseline",
                        choices=["baseline", "vae", "tce", "hybrid"],
                        help="Which approach to train (default: baseline)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to train config YAML (baseline only; default: baseline/train_config.yaml)")
    cfg = load_baseline_config() if os.path.exists(os.path.join(os.path.dirname(__file__) or ".", "baseline", "train_config.yaml")) else {}
    parser.add_argument("--lr", type=float, default=cfg.get("lr", 0.0003), help="Learning rate")
    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 500), help="Number of epochs")
    parser.add_argument("--batch", type=int, default=cfg.get("batch_size", 64), help="Batch size")
    parser.add_argument("--hidden", type=int, nargs="+", default=cfg.get("hidden_sizes", [256, 256, 128]),
                        help="Hidden layer sizes")
    parser.add_argument("--name", type=str, default=cfg.get("save_name", "cloned_policy.pth"),
                        help="Model save name")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip actions (default: clip to [-1, 1])")
    parser.add_argument("--end-weight", type=float, default=cfg.get("end_weight", 3.0),
                        help="Weight for last fraction of each trajectory (1.0 = no weighting)")
    parser.add_argument("--end-fraction", type=float, default=cfg.get("end_fraction", 0.3),
                        help="Fraction of each trajectory to up-weight from end (e.g. 0.3 = last 30%%)")
    parser.add_argument("--end-inner-weight", type=float, default=cfg.get("end_inner_weight"),
                        help="Inner tier weight for last end-inner-fraction (e.g. 5.0); optional")
    parser.add_argument("--end-inner-fraction", type=float, default=cfg.get("end_inner_fraction", 0.05),
                        help="Fraction for inner tier (e.g. 0.05 = last 5%%, 0.1 = last 10%%)")
    parser.add_argument("--end-upsample", action="store_true", default=cfg.get("end_upsample", False),
                        help="Use end upsampling (duplicate last segments) instead of weighted MSE")
    parser.add_argument("--no-save-run", action="store_true",
                        help="Do not log run or copy model to baseline/models/runs/")
    parser.add_argument("--keep-runs", type=int, default=cfg.get("keep_runs", 50),
                        help="Max run copies to keep in baseline/models/runs/ (default: 50; 0 = keep all)")
    parser.add_argument("--eval-seed", type=int, default=cfg.get("eval_seed", 42),
                        help="Seed for post-training 50-goal eval; use test.py --seed N with same N to match")
    parser.add_argument("--lr-decay-epoch", type=int, default=cfg.get("lr_decay_epoch", 250),
                        help="Decay LR by --lr-decay-gamma every N epochs (default: 250)")
    parser.add_argument("--lr-decay-gamma", type=float, default=cfg.get("lr_decay_gamma", 0.5),
                        help="LR decay factor (default 0.5)")
    parser.add_argument("--no-lr-decay", action="store_true", help="Disable LR decay")
    parser.add_argument("--mt10", action="store_true", help="MT-10 mode (same as --suite mt10)")
    parser.add_argument("--suite", type=str, default=cfg.get("suite", "mt1"), choices=["mt1", "mt10", "mt50"],
                        help="Suite: mt1, mt10, or mt50 (default from config)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging (enabled by default)")
    parser.add_argument("--wandb-tag", type=str, action="append", default=None,
                        help="W&B run tag (e.g. name:alice); can be repeated")
    parser.add_argument("--wandb-save-model", action="store_true", help="Upload final checkpoint to W&B as artifact")
    args = parser.parse_args()
    if args.no_lr_decay:
        args.lr_decay_epoch = None
    if args.config is not None and args.approach == "baseline":
        cfg = load_baseline_config(args.config)

    approach_dir = os.path.join(args.approach, "scripts")
    if not os.path.exists(approach_dir):
        print(f"Approach directory not found: {approach_dir}")
        sys.exit(1)
    sys.path.insert(0, approach_dir)
    try:
        from train import train_model, load_train_config
    except ImportError:
        print(f"Could not import train_model from {approach_dir}/train.py")
        sys.exit(1)
    if args.approach == "baseline":
        base_cfg = load_train_config(args.config) if hasattr(load_train_config, "__call__") else cfg
        use_wandb = not args.no_wandb and base_cfg.get("use_wandb", True)
        wandb_project = base_cfg.get("wandb_project") or "cs229-metaworld"
    else:
        use_wandb = not args.no_wandb
        wandb_project = "cs229-metaworld"

    print(f"\n{'='*70}")
    print("Training Policy")
    print(f"{'='*70}")
    print(f"Approach:        {args.approach}")
    print(f"Learning Rate:   {args.lr}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch Size:      {args.batch}")
    print(f"Hidden Layers:   {args.hidden}")
    print(f"Model Name:      {args.name}")
    print(f"Clip Actions:    {'No' if args.no_clip else 'Yes'}")
    print(f"Suite:           {args.suite}")
    if args.end_upsample:
        end_info = f"upsampling (last {args.end_fraction*100:.0f}% x{int(args.end_weight)}"
        if args.end_inner_weight is not None and args.end_inner_fraction and args.end_inner_fraction > 0:
            end_info += f", inner {args.end_inner_fraction*100:.0f}% x{int(args.end_inner_weight)}"
        end_info += ")"
        print(f"End emphasis:    {end_info}")
    else:
        end_info = f"{args.end_weight} (last {args.end_fraction*100:.0f}% of traj)"
        if args.end_inner_weight is not None and args.end_inner_fraction and args.end_inner_fraction > 0:
            end_info += f"; inner {args.end_inner_weight}x last {args.end_inner_fraction*100:.0f}%"
        print(f"End weight:      {end_info}")
    print(f"{'='*70}\n")

    train_model(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch,
        hidden_sizes=args.hidden,
        save_name=args.name,
        clip_actions=not args.no_clip,
        end_weight=args.end_weight,
        end_fraction=args.end_fraction,
        end_inner_weight=args.end_inner_weight,
        end_inner_fraction=args.end_inner_fraction,
        save_run=not args.no_save_run,
        keep_runs=args.keep_runs,
        eval_seed=args.eval_seed,
        lr_decay_epoch=args.lr_decay_epoch,
        lr_decay_gamma=args.lr_decay_gamma,
        end_upsample=args.end_upsample,
        mt10=args.mt10,
        suite=args.suite,
        use_wandb=use_wandb,
        wandb_tags=args.wandb_tag,
        wandb_project=wandb_project,
        wandb_save_model=args.wandb_save_model,
    )

if __name__ == "__main__":
    main()
