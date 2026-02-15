import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import json
import shutil
from datetime import datetime


def eval_50_goals(policy, task_name="reach-v3", clip_actions=True, eval_seed=42):
    """Run 50 episodes (1 per goal), return success_rate, goal_success (list of 50 bools), failed_goals (list of indices).
    If eval_seed is not None, env.reset(seed=eval_seed+goal_idx) for reproducible eval (match test.py --seed N)."""
    import metaworld
    # Match test.py: fix RNG so eval is reproducible and matches test.py --seed N
    if eval_seed is not None:
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    policy.eval()
    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()
    goal_success = []
    for goal_idx in range(min(50, len(mt1.train_tasks))):
        task = mt1.train_tasks[goal_idx]
        env.set_task(task)
        if eval_seed is not None:
            try:
                out = env.reset(seed=eval_seed + goal_idx)
            except TypeError:
                out = env.reset()
        else:
            out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        if isinstance(obs, tuple):
            obs = obs[0] if len(obs) > 0 else obs
        obs = np.asarray(obs).flatten()
        done = False
        steps = 0
        while not done and steps < 500:
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                action = policy(obs_t).numpy()
            action = np.asarray(action).flatten().astype(np.float64)
            if clip_actions:
                action = np.clip(action, -1.0, 1.0)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, _, term, trunc, info = step_out
            else:
                obs, _, done, info = step_out
                term, trunc = done, False
            done = term or trunc
            obs = np.asarray(obs).flatten() if not done else obs
            steps += 1
        goal_success.append(bool(info.get("success", False)))
    failed_goals = [i for i, s in enumerate(goal_success) if not s]
    success_rate = sum(goal_success) / len(goal_success) * 100
    return success_rate, goal_success, failed_goals

class ClonePolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=None):
        super(ClonePolicy, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_model(learning_rate=0.0003, num_epochs=20, batch_size=64, hidden_sizes=None,
                save_name='cloned_policy.pth', clip_actions=True, data_path=None,
                end_weight=3.0, end_fraction=0.3, end_inner_weight=None, end_inner_fraction=0.0,
                save_run=True, keep_runs=3, eval_seed=42, lr_decay_epoch=None, lr_decay_gamma=0.5,
                end_upsample=False):
    """Train a behavioral cloning policy.
    
    Args:
        learning_rate: Adam learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        hidden_sizes: List of hidden layer sizes
        save_name: Name of the model file to save
        clip_actions: Whether to clip actions to [-1, 1]
        data_path: Path to expert data .npz file
        end_weight: Weight for (s,a) pairs in the last end_fraction of each trajectory (1.0 = no weighting).
        end_fraction: Fraction of each trajectory (from the end) to up-weight (e.g. 0.3 = last 30%%).
        end_inner_weight: If set (e.g. 5.0), the last end_inner_fraction of each traj gets this weight (inner tier).
        end_inner_fraction: Fraction for inner tier (e.g. 0.05 = last 5%%, 0.1 = last 10%%). Used only if end_inner_weight is set.
        save_run: If True, append run stats to baseline/training_runs.json and save model to models/runs/run_TS.pth.
        keep_runs: Max run copies to keep in models/runs/ (oldest deleted). Default 50; use 0 to keep all.
        eval_seed: Seed for post-training 50-goal eval (env.reset(seed=eval_seed+goal_idx)). Use same in test.py --seed for match.
        lr_decay_epoch: If set (e.g. 250), multiply learning rate by lr_decay_gamma every this many epochs.
        lr_decay_gamma: Factor for LR decay (default 0.5). Used only if lr_decay_epoch is set.
        end_upsample: If True, duplicate last segments in the dataset and train with uniform MSE instead of weighted MSE.
    """
    
    if hidden_sizes is None:
        hidden_sizes = [256, 256, 128]
    
    if data_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '..', 'data', 'expert_data_reach-v3.npz')
    
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, save_name)
    
    print(f"Loading data from: {data_path}")
    print(f"Will save model to: {save_path}")
    
    data = np.load(data_path, allow_pickle=True)
    states_list = list(data['states'])
    actions_list = list(data['actions'])
    
    # Build per-sample weights or upsampled data: two-tier optional.
    use_inner = (end_inner_weight is not None and end_inner_fraction > 0 and end_inner_weight != 1.0
                 and end_inner_fraction < end_fraction)
    if end_upsample:
        # Duplicate last segments so they appear end_weight / end_inner_weight times; train with uniform MSE.
        X_parts = []
        Y_parts = []
        for traj_states, traj_actions in zip(states_list, actions_list):
            L = len(traj_states)
            thresh_outer = max(0, int(L * (1 - end_fraction)))
            thresh_inner = max(0, int(L * (1 - end_inner_fraction))) if use_inner else thresh_outer
            # Segment 1: [0:thresh_outer] once
            if thresh_outer > 0:
                X_parts.append(traj_states[:thresh_outer])
                Y_parts.append(traj_actions[:thresh_outer])
            # Segment 2: [thresh_outer:thresh_inner] end_weight times
            if thresh_inner > thresh_outer:
                for _ in range(int(end_weight)):
                    X_parts.append(traj_states[thresh_outer:thresh_inner])
                    Y_parts.append(traj_actions[thresh_outer:thresh_inner])
            # Segment 3: [thresh_inner:L] end_inner_weight times if use_inner else end_weight times
            if L > thresh_inner:
                n_inner = int(end_inner_weight) if use_inner else int(end_weight)
                for _ in range(n_inner):
                    X_parts.append(traj_states[thresh_inner:])
                    Y_parts.append(traj_actions[thresh_inner:])
        X_train = np.concatenate(X_parts)
        Y_train = np.concatenate(Y_parts)
        if clip_actions:
            Y_train = np.clip(Y_train, -1.0, 1.0)
        W_train = np.ones(len(X_train), dtype=np.float32)
        num_samples = min(50000, len(X_train))
        if num_samples < len(X_train):
            indices = np.random.choice(len(X_train), num_samples, replace=False)
            X_train = X_train[indices]
            Y_train = Y_train[indices]
            W_train = W_train[indices]
        print(f"Training on {num_samples} samples")
        upmsg = f"End-of-trajectory upsampling: last {end_fraction*100:.0f}% repeated {int(end_weight)}x"
        if use_inner:
            upmsg += f"; last {end_inner_fraction*100:.0f}% repeated {int(end_inner_weight)}x (uniform MSE)."
        else:
            upmsg += " (uniform MSE)."
        print(upmsg)
    else:
        weights_list = []
        for traj_states in states_list:
            L = len(traj_states)
            thresh_outer = max(0, int(L * (1 - end_fraction)))
            thresh_inner = max(0, int(L * (1 - end_inner_fraction))) if use_inner else thresh_outer
            w = np.ones(L, dtype=np.float32)
            if end_weight != 1.0 and L > 0:
                w[thresh_outer:] = end_weight
            if use_inner and L > 0:
                w[thresh_inner:] = float(end_inner_weight)
            weights_list.append(w)
        W_train = np.concatenate(weights_list)
        X_train = np.concatenate(states_list)
        Y_train = np.concatenate(actions_list)
        if clip_actions:
            Y_train = np.clip(Y_train, -1.0, 1.0)
        num_samples = min(50000, len(X_train))
        if num_samples < len(X_train):
            indices = np.random.choice(len(X_train), num_samples, replace=False)
            X_train = X_train[indices]
            Y_train = Y_train[indices]
            W_train = W_train[indices]
        print(f"Training on {num_samples} samples")
        if end_weight != 1.0:
            print(f"End-of-trajectory weighting: last {end_fraction*100:.0f}% of each traj weighted {end_weight}x", end="")
            if use_inner:
                print(f"; last {end_inner_fraction*100:.0f}% weighted {end_inner_weight}x (inner tier)")
            else:
                print()

    X_tensor = torch.FloatTensor(X_train)
    Y_tensor = torch.FloatTensor(Y_train)
    W_tensor = torch.FloatTensor(W_train)

    dataset = TensorDataset(X_tensor, Y_tensor, W_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = ClonePolicy(X_tensor.shape[1], Y_tensor.shape[1], hidden_sizes=hidden_sizes)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    use_weights = not end_upsample and (end_weight != 1.0 or (use_inner and end_inner_weight != 1.0))

    print(f"\nTraining: LR={learning_rate}, Epochs={num_epochs}, Batch={batch_size}")
    if lr_decay_epoch is not None:
        print(f"LR decay: every {lr_decay_epoch} epochs × {lr_decay_gamma}")
    print(f"Hidden sizes: {hidden_sizes}, Clip actions: {clip_actions}\n")

    final_loss = None
    for epoch in range(num_epochs):
        if lr_decay_epoch is not None and epoch > 0 and epoch % lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_gamma
            print(f"  LR decay at epoch {epoch+1}: new LR = {optimizer.param_groups[0]['lr']:.6f}")
        total_loss = 0
        for batch in dataloader:
            batch_x, batch_y, batch_w = batch
            predictions = policy(batch_x)
            if use_weights:
                mse_per_sample = ((predictions - batch_y) ** 2).mean(dim=1)
                loss = (batch_w * mse_per_sample).sum() / batch_w.sum().clamp(min=1e-8)
            else:
                loss = ((predictions - batch_y) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        final_loss = total_loss / len(dataloader)
        if (epoch + 1) % 50 == 0 or (epoch + 1) == num_epochs:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {final_loss:.6f}")

    if save_run:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_dir = os.path.join(model_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        # Descriptive run filename: end3_inner5x10_upsample_clip.pth
        name_parts = [f"end{int(end_weight)}"]
        if use_inner and end_inner_weight is not None and end_inner_fraction is not None:
            name_parts.append(f"inner{int(end_inner_weight)}x{int(end_inner_fraction * 100)}")
        if end_upsample:
            name_parts.append("upsample")
        name_parts.append("clip" if clip_actions else "noclip")
        run_fname = f"run_{ts}_{'_'.join(name_parts)}.pth"
        run_path = os.path.join(runs_dir, run_fname)
        torch.save(policy.state_dict(), run_path)
        shutil.copy2(run_path, save_path)
        print(f"\nModel saved to {run_path} (latest copied to {save_path})")
    else:
        torch.save(policy.state_dict(), save_path)
        print(f"\nModel saved to {save_path}")

    if save_run:
        print("Running 50-goal eval for run record..." + (f" (seed={eval_seed})" if eval_seed is not None else ""))
        success_rate, goal_success, failed_goals = eval_50_goals(policy, clip_actions=clip_actions, eval_seed=eval_seed)
        print(f"  Eval: {success_rate:.1f}% ({len(goal_success) - len(failed_goals)}/50)")

        run_record = {
            "timestamp": ts,
            "lr": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "hidden_sizes": hidden_sizes,
            "end_weight": end_weight,
            "end_fraction": end_fraction,
            "end_inner_weight": end_inner_weight if use_inner else None,
            "end_inner_fraction": end_inner_fraction if use_inner else None,
            "end_upsample": end_upsample,
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_gamma": lr_decay_gamma if lr_decay_epoch is not None else None,
            "clip_actions": clip_actions,
            "eval_seed": eval_seed,
            "save_name": save_name,
            "final_loss": float(final_loss) if final_loss is not None else None,
            "num_samples": num_samples,
            "success_rate": round(success_rate, 2),
            "goal_success": goal_success,
            "failed_goals": failed_goals,
            "run_path": os.path.abspath(run_path),
        }
        log_path = os.path.join(model_dir, "..", "training_runs.json")
        log_path = os.path.abspath(log_path)
        runs_list = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    runs_list = json.load(f)
            except Exception:
                runs_list = []
        runs_list.append(run_record)
        with open(log_path, "w") as f:
            json.dump(runs_list, f, indent=2)
        print(f"Run logged to {log_path}")

        # RUNS_SUMMARY.md: all runs, with "Latest batch" (last 5) separated by a line
        summary_path = os.path.join(model_dir, "..", "RUNS_SUMMARY.md")
        summary_path = os.path.abspath(summary_path)
        max_runs = 60  # show up to 60 runs
        all_recent = runs_list[-max_runs:]
        all_recent.reverse()  # newest first
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
        print(f"Summary written to {summary_path}")

        if keep_runs > 0:
            run_files = [os.path.join(runs_dir, f) for f in os.listdir(runs_dir) if f.endswith(".pth")]
            run_files.sort(key=os.path.getmtime, reverse=True)
            for old in run_files[keep_runs:]:
                try:
                    os.remove(old)
                    print(f"Removed old run copy {os.path.basename(old)}")
                except Exception:
                    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a behavioral cloning policy')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 256, 128], 
                        help='Hidden layer sizes')
    parser.add_argument('--name', type=str, default='cloned_policy.pth', 
                        help='Model save name')
    parser.add_argument('--no-clip', action='store_true', help='Do not clip actions (default: clip to [-1, 1])')
    parser.add_argument('--data', type=str, help='Path to expert data')
    parser.add_argument('--end-weight', type=float, default=3.0,
                        help='Weight for last fraction of each trajectory (1.0 = no weighting)')
    parser.add_argument('--end-fraction', type=float, default=0.3,
                        help='Fraction of each trajectory to up-weight from the end (e.g. 0.3 = last 30%%)')
    parser.add_argument('--end-inner-weight', type=float, default=None,
                        help='Inner tier weight for last end-inner-fraction (e.g. 5.0); optional')
    parser.add_argument('--end-inner-fraction', type=float, default=0.05,
                        help='Fraction for inner tier (e.g. 0.05 = last 5%%, 0.1 = last 10%%)')
    parser.add_argument('--no-save-run', action='store_true',
                        help='Do not log run to training_runs.json or copy model to runs/')
    parser.add_argument('--keep-runs', type=int, default=50,
                        help='Max run copies to keep in models/runs/ (default: 50; 0 = keep all)')
    parser.add_argument('--eval-seed', type=int, default=42,
                        help='Seed for post-training 50-goal eval; use test.py --seed N with same N to match')
    parser.add_argument('--lr-decay-epoch', type=int, default=None,
                        help='Decay LR by --lr-decay-gamma every N epochs (e.g. 250); optional')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.5,
                        help='LR decay factor (default 0.5); used only if --lr-decay-epoch is set')
    parser.add_argument('--end-upsample', action='store_true',
                        help='Use end upsampling (duplicate last segments) instead of weighted MSE')
    
    args = parser.parse_args()
    
    train_model(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch,
        hidden_sizes=args.hidden,
        save_name=args.name,
        clip_actions=not args.no_clip,
        data_path=args.data,
        end_weight=args.end_weight,
        end_fraction=args.end_fraction,
        end_inner_weight=args.end_inner_weight,
        end_inner_fraction=args.end_inner_fraction,
        save_run=not args.no_save_run,
        keep_runs=args.keep_runs,
        eval_seed=args.eval_seed,
        lr_decay_epoch=args.lr_decay_epoch,
        lr_decay_gamma=args.lr_decay_gamma,
        end_upsample=args.end_upsample
    )