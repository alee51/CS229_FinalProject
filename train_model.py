import numpy as np
import torch
import argparse
import os
import sys

from contrastive_data import TCEDataset
from contrastive_model import TCEAgent
from contrastive_trainer import TCETrainer


def load_data(task_name):
    filepath = f"data/expert_{task_name}.npz"

    if not os.path.exists(filepath):
        print(f"Error: Data file '{filepath}' not found.")
        print(f"  Run: python collect_data.py --task {task_name}")
        sys.exit(1)

    print(f"Loading data from {filepath}...")
    data = np.load(filepath)
    return (
        data['states'],
        data['actions'],
        data['next_states'],
        data['rewards'],
        data['traj_ids'],
    )


def main():
    parser = argparse.ArgumentParser(description="Train TCE + CRTR + RFF policy")
    parser.add_argument('--task',              type=str,   default='reach-v3',
                        help='Metaworld task name')
    parser.add_argument('--epochs',            type=int,   default=50)
    parser.add_argument('--batch_size',        type=int,   default=256)
    parser.add_argument('--lr',                type=float, default=3e-4)
    parser.add_argument('--latent_dim',        type=int,   default=64)
    parser.add_argument('--fourier_dim',       type=int,   default=256)
    parser.add_argument('--repetition_factor', type=int,   default=4,
                        help='CRTR: samples per trajectory per batch')
    parser.add_argument('--temperature',       type=float, default=0.1)
    parser.add_argument('--alpha',             type=float, default=1.0,
                        help='Weight for contrastive loss J_T')
    parser.add_argument('--beta',              type=float, default=1.0,
                        help='Weight for reward prediction loss J_R')
    parser.add_argument('--gamma',             type=float, default=0.99,
                        help='Discount factor. Used to compute the theory-motivated '
                             'alpha for J_T: alpha = 1/(1-gamma) per Nachum & Yang '
                             'Theorem 2. If --alpha is set explicitly, that overrides.')
    args = parser.parse_args()

    # --- CONFIG ---
    config = {
        'task_name':         args.task,
        'lr':                args.lr,
        'batch_size':        args.batch_size,
        'epochs':            args.epochs,
        'temperature':       args.temperature,
        # Per Nachum & Yang Theorem 2, the coefficient on J_T should be
        # alpha_T = (1-gamma)^{-1} to make the bound tight.
        # For gamma=0.99 this is ~100. If the user passed --alpha explicitly
        # at a non-default value we respect that; otherwise we apply the
        # theory-motivated value derived from gamma.
        'alpha': args.alpha if args.alpha != 1.0 else 1.0 / (1.0 - args.gamma),
        'beta':              args.beta,
        'repetition_factor': args.repetition_factor,
    }

    # --- LOAD DATA ---
    s, a, s_next, r, ids = load_data(args.task)

    # --- SPLIT BY TRAJECTORY ID (not flat index) ---
    # Splitting by flat index can cut trajectories in half, leaking consecutive
    # states across the train/val boundary. Splitting by trajectory ID avoids
    # this and makes the val set a true held-out set of full episodes.
    unique_ids  = np.unique(ids)
    n_train     = int(len(unique_ids) * 0.8)
    train_ids   = set(unique_ids[:n_train])

    train_mask  = np.array([i in train_ids for i in ids])
    val_mask    = ~train_mask

    # Pre-load tensors onto the training device inside TCEDataset so that
    # batches don't incur a CPU->GPU transfer on every iteration.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = TCEDataset(
        s[train_mask], a[train_mask], s_next[train_mask],
        r[train_mask], ids[train_mask], device=device,
    )
    val_dataset = TCEDataset(
        s[val_mask], a[val_mask], s_next[val_mask],
        r[val_mask], ids[val_mask], device=device,
    )

    print(f"Train transitions: {len(train_dataset)} | "
          f"Val transitions: {len(val_dataset)}")

    # --- INITIALISE MODEL ---
    agent = TCEAgent(
        input_dim=39,
        action_dim=4,
        latent_dim=args.latent_dim,
        fourier_dim=args.fourier_dim,
    )

    # --- TRAIN ---
    trainer = TCETrainer(agent, train_dataset, val_dataset, config)
    print(f"\nStarting training for task: {args.task}\n")
    trainer.train(epochs=config['epochs'])

    # --- SAVE ---
    os.makedirs('models', exist_ok=True)
    save_path = f"models/tce_policy_{args.task}.pth"
    torch.save(agent.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()