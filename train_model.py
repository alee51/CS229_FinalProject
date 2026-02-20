"""
Train TCE + CRTR + FiLM policy using sequential two-phase training.

Usage examples
--------------
# Full sequential run (recommended):
python train_model.py --task mt10 --pretrain_epochs 50 --finetune_epochs 100

# Phase 1 only (pretrain representation):
python train_model.py --task mt10 --pretrain_epochs 80 --finetune_epochs 0

# Phase 2 only (load existing repr checkpoint, train FiLM policy):
python train_model.py --task mt10 --pretrain_epochs 0 --finetune_epochs 100

# Joint training ablation:
python train_model.py --task mt10 --joint --epochs 100
"""

import argparse
import os
import sys

import numpy as np
import torch

from contrastive_data import TCEDataset
from contrastive_model import TCEAgent, KINEMATIC_DIM, CONTEXT_DIM
from contrastive_trainer import TCETrainer


def load_data(task_name):
    filepath = f"data/expert_{task_name}.npz"
    if not os.path.exists(filepath):
        print(f"Error: '{filepath}' not found.  Run: python collect_data.py --task {task_name}")
        sys.exit(1)
    print(f"Loading {filepath}...")
    d = np.load(filepath)
    return d['states'], d['actions'], d['next_states'], d['rewards'], d['traj_ids'], d['task_labels']


def main():
    parser = argparse.ArgumentParser(description="Train TCE + CRTR + FiLM policy")

    # ---- data / model ----
    parser.add_argument('--task',              type=str,   default='mt10')
    parser.add_argument('--num_tasks',         type=int,   default=10)
    parser.add_argument('--latent_dim',        type=int,   default=64)
    parser.add_argument('--fourier_dim',       type=int,   default=256)
    parser.add_argument('--sigma',             type=float, default=1.0,
                        help='RFF bandwidth σ.  Larger = wider/smoother kernel.')

    # ---- optimisation ----
    parser.add_argument('--lr',                type=float, default=3e-4)
    parser.add_argument('--lr_phase2',         type=float, default=3e-4,
                        help='Learning rate for Phase 2 FiLM policy (default = same as --lr)')
    parser.add_argument('--batch_size',        type=int,   default=512)
    parser.add_argument('--weight_decay',      type=float, default=1e-4)
    parser.add_argument('--beta',              type=float, default=1.0,
                        help='Weight for reward prediction loss J_R')
    parser.add_argument('--temperature',       type=float, default=0.5,
                        help='InfoNCE softmax temperature. Use ≥0.5 for unnormalised RFF dots.')
    parser.add_argument('--gamma',             type=float, default=0.99,
                        help='Discount factor; used to compute theory alpha = 1/(1-γ)')
    parser.add_argument('--repetition_factor', type=int,   default=4)

    # ---- training mode ----
    parser.add_argument('--pretrain_epochs',   type=int,   default=50,
                        help='Phase 1 epochs: train encoder+reward with J_T+J_R only')
    parser.add_argument('--finetune_epochs',   type=int,   default=100,
                        help='Phase 2 epochs: train FiLM policy with J_BC only (encoder frozen)')
    parser.add_argument('--joint',             action='store_true',
                        help='Ablation: run joint training instead of sequential')
    parser.add_argument('--epochs',            type=int,   default=100,
                        help='Epochs for joint training (only used with --joint)')

    args = parser.parse_args()

    # In Phase 1 the BC loss is absent, so we can safely use theory alpha = 1/(1-γ)
    theory_alpha = 1.0 / (1.0 - args.gamma)

    config = {
        'task_name':         args.task,
        'lr':                args.lr,
        'lr_phase2':         args.lr_phase2,
        'batch_size':        args.batch_size,
        'weight_decay':      args.weight_decay,
        'temperature':       args.temperature,
        'beta':              args.beta,
        'gamma':             args.gamma,
        'repetition_factor': args.repetition_factor,
        'pretrain_epochs':   args.pretrain_epochs,
        'finetune_epochs':   args.finetune_epochs,
        # Joint-only: keep alpha=1 to avoid destabilising BC (same problem as before)
        'alpha':             1.0 if args.joint else theory_alpha,
    }

    # ---- data split by trajectory (not flat index) ----
    s, a, s_next, r, ids, labels = load_data(args.task)

    unique_ids = np.unique(ids)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(unique_ids)
    n_train   = int(len(unique_ids) * 0.8)
    train_ids = set(unique_ids[:n_train])
    train_mask = np.array([i in train_ids for i in ids])
    val_mask   = ~train_mask

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = TCEDataset(
        s[train_mask], a[train_mask], s_next[train_mask],
        r[train_mask], ids[train_mask],
        task_labels=labels[train_mask],
        num_tasks=args.num_tasks, device=device,
    )
    val_ds = TCEDataset(
        s[val_mask], a[val_mask], s_next[val_mask],
        r[val_mask], ids[val_mask],
        task_labels=labels[val_mask],
        num_tasks=args.num_tasks, device=device,
    )

    print(f"Train: {len(train_ds)} transitions | Val: {len(val_ds)}")

    # ---- model ----
    input_dim = s.shape[1] + args.num_tasks  # 39 + 10 = 49
    agent = TCEAgent(
        input_dim=input_dim,
        action_dim=4,
        latent_dim=args.latent_dim,
        fourier_dim=args.fourier_dim,
        kinematic_dim=KINEMATIC_DIM,
        context_dim=CONTEXT_DIM,
        sigma=args.sigma,
    )

    trainer = TCETrainer(agent, train_ds, val_ds, config)

    if args.joint:
        print(f"\nRunning joint training ablation ({args.epochs} epochs, alpha=1.0)")
        print("Note: val/loss_bc is expected to diverge ~epoch 16.  Use sequential instead.\n")
        trainer.train_joint(args.epochs)
    else:
        if args.pretrain_epochs > 0:
            trainer.train_phase1(args.pretrain_epochs)
        else:
            # Skip Phase 1 — load existing repr checkpoint
            ckpt = trainer.ckpt_repr
            if os.path.exists(ckpt):
                agent.load_state_dict(
                    torch.load(ckpt, map_location=device, weights_only=True)
                )
                print(f"Skipping Phase 1 — loaded existing checkpoint: {ckpt}")
            else:
                print(f"Warning: --pretrain_epochs=0 but no checkpoint at {ckpt}. "
                      f"Starting Phase 2 from random init.")

        if args.finetune_epochs > 0:
            trainer.train_phase2(args.finetune_epochs)

    print("\nDone.")


if __name__ == "__main__":
    main()