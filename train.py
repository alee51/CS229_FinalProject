#!/usr/bin/env python
"""
Unified train script for CS229 project policies

Usage:
    python train.py --approach baseline --epochs 20 --lr 0.0003
    python train.py --approach baseline --epochs 50 --lr 0.001 --name my_model.pth
    python train.py --approach baseline --lr 0.001 --epochs 50 --clip
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Train policies for CS229 project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --approach baseline
  python train.py --approach baseline --epochs 50 --lr 0.001 --name improved.pth
  python train.py --approach baseline --clip --epochs 100
        """
    )
    
    parser.add_argument('--approach', type=str, default='baseline',
                        choices=['baseline', 'vae', 'tce', 'hybrid'],
                        help='Which approach to train (default: baseline)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate (default: 0.0003)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 256, 128],
                        help='Hidden layer sizes (default: 256 256 128)')
    parser.add_argument('--name', type=str, default='cloned_policy.pth',
                        help='Model save name (default: cloned_policy.pth)')
    parser.add_argument('--clip', action='store_true',
                        help='Clip actions to [-1, 1]')
    parser.add_argument('--end-weight', type=float, default=3.0,
                        help='Weight for last fraction of each trajectory (1.0 = no weighting)')
    parser.add_argument('--end-fraction', type=float, default=0.3,
                        help='Fraction of each trajectory to up-weight from end (e.g. 0.3 = last 30%%)')
    parser.add_argument('--no-save-run', action='store_true',
                        help='Do not log run or copy model to baseline/models/runs/')
    parser.add_argument('--keep-runs', type=int, default=50,
                        help='Max run copies to keep in baseline/models/runs/ (default: 50; 0 = keep all)')
    
    args = parser.parse_args()
    
    # Import train function from the appropriate approach
    approach_dir = os.path.join(args.approach, 'scripts')
    if not os.path.exists(approach_dir):
        print(f"❌ Approach directory not found: {approach_dir}")
        sys.exit(1)
    
    sys.path.insert(0, approach_dir)
    
    try:
        from train import train_model
    except ImportError:
        print(f"❌ Could not import train_model from {approach_dir}/train.py")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Training Policy")
    print(f"{'='*70}")
    print(f"Approach:        {args.approach}")
    print(f"Learning Rate:   {args.lr}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch Size:      {args.batch}")
    print(f"Hidden Layers:   {args.hidden}")
    print(f"Model Name:      {args.name}")
    print(f"Clip Actions:    {'Yes' if args.clip else 'No'}")
    print(f"End weight:      {args.end_weight} (last {args.end_fraction*100:.0f}%% of traj)")
    print(f"{'='*70}\n")
    
    train_model(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch,
        hidden_sizes=args.hidden,
        save_name=args.name,
        clip_actions=args.clip,
        end_weight=args.end_weight,
        end_fraction=args.end_fraction,
        save_run=not args.no_save_run,
        keep_runs=args.keep_runs
    )

if __name__ == "__main__":
    main()
