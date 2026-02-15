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


def eval_50_goals(policy, task_name="reach-v3", clip_actions=True):
    """Run 50 episodes (1 per goal), return success_rate, goal_success (list of 50 bools), failed_goals (list of indices)."""
    import metaworld
    policy.eval()
    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()
    goal_success = []
    for goal_idx in range(min(50, len(mt1.train_tasks))):
        task = mt1.train_tasks[goal_idx]
        env.set_task(task)
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
                save_name='cloned_policy.pth', clip_actions=False, data_path=None,
                end_weight=3.0, end_fraction=0.3, save_run=True, keep_runs=3):
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
        save_run: If True, append run stats to baseline/training_runs.json and save model to models/runs/run_TS.pth.
        keep_runs: Max run copies to keep in models/runs/ (oldest deleted). Default 50; use 0 to keep all.
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
    
    # Build per-sample weights: last end_fraction of each trajectory gets end_weight, rest 1.0
    weights_list = []
    for traj_states in states_list:
        L = len(traj_states)
        thresh = max(0, int(L * (1 - end_fraction)))
        w = np.ones(L, dtype=np.float32)
        if end_weight != 1.0 and L > 0:
            w[thresh:] = end_weight
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
        print(f"End-of-trajectory weighting: last {end_fraction*100:.0f}% of each traj weighted {end_weight}x")

    X_tensor = torch.FloatTensor(X_train)
    Y_tensor = torch.FloatTensor(Y_train)
    W_tensor = torch.FloatTensor(W_train)

    dataset = TensorDataset(X_tensor, Y_tensor, W_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = ClonePolicy(X_tensor.shape[1], Y_tensor.shape[1], hidden_sizes=hidden_sizes)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    use_weights = end_weight != 1.0

    print(f"\nTraining: LR={learning_rate}, Epochs={num_epochs}, Batch={batch_size}")
    print(f"Hidden sizes: {hidden_sizes}, Clip actions: {clip_actions}\n")

    final_loss = None
    for epoch in range(num_epochs):
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
        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {final_loss:.6f}")

    if save_run:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_dir = os.path.join(model_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        run_path = os.path.join(runs_dir, f"run_{ts}.pth")
        torch.save(policy.state_dict(), run_path)
        shutil.copy2(run_path, save_path)
        print(f"\nModel saved to {run_path} (latest copied to {save_path})")
    else:
        torch.save(policy.state_dict(), save_path)
        print(f"\nModel saved to {save_path}")

    if save_run:
        print("Running 50-goal eval for run record...")
        success_rate, goal_success, failed_goals = eval_50_goals(policy, clip_actions=clip_actions)
        print(f"  Eval: {success_rate:.1f}% ({len(goal_success) - len(failed_goals)}/50)")

        run_record = {
            "timestamp": ts,
            "lr": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "hidden_sizes": hidden_sizes,
            "end_weight": end_weight,
            "end_fraction": end_fraction,
            "clip_actions": clip_actions,
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

        # RUNS_SUMMARY.md: last 10 runs
        summary_path = os.path.join(model_dir, "..", "RUNS_SUMMARY.md")
        summary_path = os.path.abspath(summary_path)
        last_n = runs_list[-10:]
        last_n.reverse()
        lines = [
            "# Last 10 training runs",
            "",
            "Full history in `training_runs.json`. Per-run models in `models/runs/run_YYYYMMDD_HHMMSS.pth`.",
            "",
            "| timestamp | epochs | end_weight | final_loss | success_rate | failed_goals |",
            "|-----------|--------|------------|------------|--------------|--------------|",
        ]
        for r in last_n:
            ts_ = r.get("timestamp", "")
            ep = r.get("epochs", "")
            ew = r.get("end_weight", "")
            fl = r.get("final_loss")
            fl = f"{fl:.6f}" if fl is not None else "—"
            sr = r.get("success_rate")
            sr = f"{sr}%" if sr is not None else "—"
            fg = r.get("failed_goals", [])
            fg_str = ",".join(str(x) for x in fg[:15]) + ("..." if len(fg) > 15 else "")
            lines.append(f"| {ts_} | {ep} | {ew} | {fl} | {sr} | {fg_str} |")
        with open(summary_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Summary (last 10) written to {summary_path}")

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
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 256, 128], 
                        help='Hidden layer sizes')
    parser.add_argument('--name', type=str, default='cloned_policy.pth', 
                        help='Model save name')
    parser.add_argument('--clip', action='store_true', help='Clip actions to [-1, 1]')
    parser.add_argument('--data', type=str, help='Path to expert data')
    parser.add_argument('--end-weight', type=float, default=3.0,
                        help='Weight for last fraction of each trajectory (1.0 = no weighting)')
    parser.add_argument('--end-fraction', type=float, default=0.3,
                        help='Fraction of each trajectory to up-weight from the end (e.g. 0.3 = last 30%%)')
    parser.add_argument('--no-save-run', action='store_true',
                        help='Do not log run to training_runs.json or copy model to runs/')
    parser.add_argument('--keep-runs', type=int, default=50,
                        help='Max run copies to keep in models/runs/ (default: 50; 0 = keep all)')
    
    args = parser.parse_args()
    
    train_model(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch,
        hidden_sizes=args.hidden,
        save_name=args.name,
        clip_actions=args.clip,
        data_path=args.data,
        end_weight=args.end_weight,
        end_fraction=args.end_fraction,
        save_run=not args.no_save_run,
        keep_runs=args.keep_runs
    )