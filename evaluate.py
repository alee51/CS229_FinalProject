"""
Evaluate a trained TCE + CRTR + FiLM policy on Meta-World MT-10.

Also includes a linear probe to verify task erasure in the encoder
(the key empirical validation of the Context-Bypass assumption).

Usage
-----
# Full MT-10 sweep (best Phase-2 checkpoint):
python evaluate.py

# Single task:
python evaluate.py --task pick-place-v3

# Verify task erasure after Phase 1:
python evaluate.py --linear_probe --model models/tce_mt10_repr.pth --task mt10
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import metaworld

from contrastive_model import TCEAgent, KINEMATIC_DIM, CONTEXT_DIM

MT10_TASKS = [
    'reach-v3',
    'push-v3',
    'pick-place-v3',
    'door-open-v3',
    'door-close-v3',
    'drawer-open-v3',
    'drawer-close-v3',
    'button-press-topdown-v3',
    'lever-pull-v3',
    'window-open-v3',
]


def make_one_hot(task_name, num_tasks=10):
    idx = MT10_TASKS.index(task_name)
    v   = np.zeros(num_tasks, dtype=np.float32)
    v[idx] = 1.0
    return v


def build_agent(model_path, device, latent_dim=64, fourier_dim=256, num_tasks=10):
    agent = TCEAgent(
        input_dim=39 + num_tasks,
        action_dim=4,
        latent_dim=latent_dim,
        fourier_dim=fourier_dim,
        kinematic_dim=KINEMATIC_DIM,
        context_dim=CONTEXT_DIM,
    ).to(device)
    if not os.path.exists(model_path):
        print(f"Error: model not found at {model_path}")
        sys.exit(1)
    agent.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    agent.eval()
    return agent


# =============================================================================
# POLICY EVALUATION
# =============================================================================

def evaluate_policy(task_name, agent, num_episodes=50, num_tasks=10):
    device    = next(agent.parameters()).device
    mt1       = metaworld.MT1(task_name)
    env       = mt1.train_classes[task_name]()
    tasks     = mt1.test_tasks if len(mt1.test_tasks) > 0 else mt1.train_tasks
    one_hot   = make_one_hot(task_name, num_tasks)
    successes = 0

    for _ in range(num_episodes):
        env.set_task(random.choice(tasks))
        obs, _ = env.reset()
        done, steps, success = False, 0, False

        while not done and steps < 500:
            obs_aug = np.concatenate([obs, one_hot])
            s_t = torch.FloatTensor(obs_aug).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent(s_t).cpu().numpy().flatten()
            obs, _, terminated, truncated, info = env.step(action)
            if info.get('success', 0.0) > 0.0:
                success = True
            steps += 1
            done = terminated or truncated

        if success:
            successes += 1

    rate = (successes / num_episodes) * 100
    print(f"  {task_name:<35} {rate:5.1f}%  ({'#' * int(rate / 5)})")
    return rate


def evaluate_mt10(agent, num_episodes=50, num_tasks=10):
    results = {}
    print("\n" + "=" * 50)
    print("MT-10 Evaluation Results")
    print("=" * 50)
    for task in MT10_TASKS:
        results[task] = evaluate_policy(task, agent, num_episodes, num_tasks)
    mean = sum(results.values()) / len(results)
    print("-" * 50)
    print(f"  {'Mean Success Rate':<35} {mean:5.1f}%")
    print("=" * 50)
    return results, mean


# =============================================================================
# LINEAR PROBE  (task-erasure verification)
# =============================================================================

def run_linear_probe(agent, data_path, device, num_tasks=10,
                     probe_epochs=50, batch_size=512):
    """
    Train a linear classifier z → task_id on frozen encoder outputs.

    Interpretation:
      ~10% accuracy  → task erasure confirmed; Context-Bypass is necessary.
      >50% accuracy  → encoder leaks task identity; increase repetition_factor.

    Args:
        agent:      Trained TCEAgent (encoder is frozen during probing).
        data_path:  Path to the .npz data file used for training.
        device:     torch device.
    """
    if not os.path.exists(data_path):
        print(f"Linear probe skipped: data not found at {data_path}")
        return

    print(f"\n{'='*50}")
    print("Linear Probe: Task Erasure Verification")
    print(f"{'='*50}")
    print("Freezing encoder, training linear z → task_id classifier...")

    d      = np.load(data_path)
    states = torch.FloatTensor(d['states']).to(device)          # (N, 39)
    labels = torch.LongTensor(d['task_labels'].astype(int)).to(device)  # (N,)

    # Augment states with a dummy zero one-hot (context stripped internally)
    one_hot_zeros = torch.zeros(states.shape[0], num_tasks, device=device)
    states_aug    = torch.cat([states, one_hot_zeros], dim=1)   # (N, 49)

    # Collect latent representations in batches
    agent.eval()
    z_list = []
    with torch.no_grad():
        for i in range(0, len(states_aug), batch_size):
            z_list.append(agent.encode(states_aug[i:i+batch_size]))
    Z = torch.cat(z_list, dim=0)  # (N, latent_dim)

    # 80/20 split
    n      = len(Z)
    n_tr   = int(n * 0.8)
    idx    = torch.randperm(n)
    Z_tr,  Z_val  = Z[idx[:n_tr]],      Z[idx[n_tr:]]
    y_tr,  y_val  = labels[idx[:n_tr]],  labels[idx[n_tr:]]

    probe = nn.Linear(Z.shape[1], num_tasks).to(device)
    opt   = optim.Adam(probe.parameters(), lr=1e-3)

    best_val_acc = 0.0
    for ep in range(probe_epochs):
        probe.train()
        for i in range(0, n_tr, batch_size):
            opt.zero_grad()
            logits = probe(Z_tr[i:i+batch_size].detach())
            loss   = nn.CrossEntropyLoss()(logits, y_tr[i:i+batch_size])
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            preds    = probe(Z_val.detach()).argmax(dim=1)
            val_acc  = (preds == y_val).float().mean().item() * 100
        best_val_acc = max(best_val_acc, val_acc)

    chance = 100.0 / num_tasks
    print(f"\nLinear Probe Result:  {best_val_acc:.1f}%  (chance = {chance:.1f}%)")

    if best_val_acc < chance * 1.5:
        print("✓ Near-chance accuracy → task erasure confirmed.")
        print("  The encoder is purely kinematic; Context-Bypass is working correctly.")
    elif best_val_acc > 50.0:
        print("✗ High accuracy → task information is leaking into z.")
        print("  Consider increasing repetition_factor in CRTRBatchSampler.")
    else:
        print("~ Partial task information in z.")
        print("  Context-Bypass is partially mitigating task erasure.")

    return best_val_acc


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',         type=str,  default=None,
                        help='Single task name, or omit for full MT-10 sweep')
    parser.add_argument('--model',        type=str,  default=None,
                        help='Path to checkpoint. Defaults to best Phase-2 or Phase-1 ckpt.')
    parser.add_argument('--episodes',     type=int,  default=50)
    parser.add_argument('--latent_dim',   type=int,  default=64)
    parser.add_argument('--fourier_dim',  type=int,  default=256)
    parser.add_argument('--num_tasks',    type=int,  default=10)
    parser.add_argument('--linear_probe', action='store_true',
                        help='Run task-erasure linear probe (Phase 1 checkpoint recommended)')
    parser.add_argument('--data',         type=str,  default='data/expert_mt10.npz',
                        help='Data file for linear probe')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint priority: CLI arg > Phase-2 best > Phase-1 repr > fallback
    if args.model:
        model_path = args.model
    else:
        candidates = [
            "models/tce_mt10_policy.pth",
            "models/tce_mt10_repr.pth",
            "models/tce_policy_mt10_best.pth",
            "models/tce_policy_mt10_final.pth",
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

    agent = build_agent(model_path, device, args.latent_dim, args.fourier_dim, args.num_tasks)
    print(f"Loaded: {model_path}")

    if args.linear_probe:
        run_linear_probe(agent, args.data, device, args.num_tasks)

    if args.task:
        evaluate_policy(args.task, agent, args.episodes, args.num_tasks)
    else:
        evaluate_mt10(agent, args.episodes, args.num_tasks)