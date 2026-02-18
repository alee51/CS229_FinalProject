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

    # =========================================================
    # ONE-HOT CHANGE 9: Load task_labels from the npz file.
    # =========================================================
    return (
        data['states'],
        data['actions'],
        data['next_states'],
        data['rewards'],
        data['traj_ids'],
        data['task_labels'],   # NEW
    )


def main():
    parser = argparse.ArgumentParser(description="Train TCE + CRTR + RFF policy")
    parser.add_argument('--task',              type=str,   default='mt10')
    parser.add_argument('--epochs',            type=int,   default=50)
    parser.add_argument('--batch_size',        type=int,   default=512)
    parser.add_argument('--lr',                type=float, default=3e-4)
    parser.add_argument('--latent_dim',        type=int,   default=64)
    parser.add_argument('--fourier_dim',       type=int,   default=256)
    parser.add_argument('--num_tasks',         type=int,   default=10,
                        help='Number of tasks — sets one-hot vector length')
    parser.add_argument('--repetition_factor', type=int,   default=4)
    parser.add_argument('--temperature',       type=float, default=0.1)
    parser.add_argument('--alpha',             type=float, default=1.0,
                        help='Weight for J_T; overrides gamma-derived value if != 1.0')
    parser.add_argument('--beta',              type=float, default=1.0,
                        help='Weight for J_R')
    parser.add_argument('--gamma',             type=float, default=0.99,
                        help='Discount factor; sets alpha=1/(1-gamma) per Theorem 2')
    args = parser.parse_args()

    config = {
        'task_name':         args.task,
        'lr':                args.lr,
        'batch_size':        args.batch_size,
        'epochs':            args.epochs,
        'temperature':       args.temperature,
        'alpha': args.alpha if args.alpha != 1.0 else 1.0 / (1.0 - args.gamma),
        'beta':              args.beta,
        'repetition_factor': args.repetition_factor,
    }

    # =========================================================
    # ONE-HOT CHANGE 10: Unpack task_labels from load_data.
    # =========================================================
    s, a, s_next, r, ids, labels = load_data(args.task)

    # --- SPLIT BY TRAJECTORY ID ---
    unique_ids = np.unique(ids)
    n_train    = int(len(unique_ids) * 0.8)
    train_ids  = set(unique_ids[:n_train])

    train_mask = np.array([i in train_ids for i in ids])
    val_mask   = ~train_mask

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================
    # ONE-HOT CHANGE 11: Pass task_labels and num_tasks to TCEDataset.
    # TCEDataset builds the one-hot internally and pre-concatenates
    # to states/next_states — no other file needs to change.
    # =========================================================
    train_dataset = TCEDataset(
        s[train_mask], a[train_mask], s_next[train_mask],
        r[train_mask], ids[train_mask],
        task_labels=labels[train_mask],
        num_tasks=args.num_tasks,
        device=device,
    )
    val_dataset = TCEDataset(
        s[val_mask], a[val_mask], s_next[val_mask],
        r[val_mask], ids[val_mask],
        task_labels=labels[val_mask],
        num_tasks=args.num_tasks,
        device=device,
    )

    print(f"Train: {len(train_dataset)} transitions | Val: {len(val_dataset)}")

    # =========================================================
    # ONE-HOT CHANGE 12: input_dim = state_dim + num_tasks = 39 + 10 = 49.
    # Must match what TCEDataset produced.
    # =========================================================
    state_dim = s.shape[1]                        # 39
    input_dim = state_dim + args.num_tasks        # 49

    agent = TCEAgent(
        input_dim=input_dim,
        action_dim=4,
        latent_dim=args.latent_dim,
        fourier_dim=args.fourier_dim,
    )

    trainer = TCETrainer(agent, train_dataset, val_dataset, config)
    print(f"\nStarting training: {args.task}  (input_dim={input_dim})\n")
    trainer.train(epochs=config['epochs'])

    os.makedirs('models', exist_ok=True)
    save_path = f"models/tce_policy_{args.task}.pth"
    torch.save(agent.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()