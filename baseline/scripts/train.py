import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import json
import shutil
import sys
import time
from datetime import datetime

# Shared task registry (baseline/tasks.py)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASELINE_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_BASELINE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from baseline.tasks import get_tasks, num_tasks, policy_input_dim, obs_dim

# Backward compatibility: MT10_TASKS for code that imports it (e.g. test.py)
MT10_TASKS = get_tasks("mt10")


def one_hot_task(task_id, num_tasks=10):
    """Return one-hot vector of shape (num_tasks,) for the given task index."""
    out = np.zeros(num_tasks, dtype=np.float32)
    out[task_id] = 1.0
    return out


def load_train_config(config_path=None):
    """Load training default config from baseline/train_config.yaml. Returns a dict."""
    if config_path is None:
        config_path = os.path.join(_BASELINE_DIR, "train_config.yaml")
    if not os.path.isfile(config_path):
        return {}
    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def get_device(device_prefer="auto"):
    """Resolve device: 'auto' (cuda -> xpu -> cpu), 'cuda', 'xpu', or 'cpu'.
    Works for NVIDIA (cuda), Intel Arc (xpu), and CPU-only machines."""
    if device_prefer == "cpu":
        return torch.device("cpu")
    if device_prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_prefer == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    if device_prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return torch.device("xpu")
    return torch.device("cpu")


def one_hot_task(task_id, num_tasks=10):
    """Return one-hot vector of shape (num_tasks,) for the given task index."""
    out = np.zeros(num_tasks, dtype=np.float32)
    out[task_id] = 1.0
    return out


def eval_50_goals(policy, task_name="reach-v3", clip_actions=True, eval_seed=42, device=None):
    """Run 50 episodes (1 per goal), return success_rate, goal_success (list of 50 bools), failed_goals (list of indices).
    If eval_seed is not None, env.reset(seed=eval_seed+goal_idx) for reproducible eval (match test.py --seed N)."""
    import metaworld
    if device is None:
        device = torch.device("cpu")
    # Match test.py: fix RNG so eval is reproducible and matches test.py --seed N
    if eval_seed is not None:
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            try:
                xpu.manual_seed_all(eval_seed)
            except Exception:
                pass
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
            obs_t = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                action = policy(obs_t).cpu().numpy()
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


def eval_multitask(policy, suite, clip_actions=True, eval_seed=42, device=None):
    """Run 50 episodes (1 per goal) for each task in suite; input = concat(obs, one_hot(task_id)).
    Returns success_rate_per_task (list of floats), success_rate_avg (float)."""
    import metaworld
    if device is None:
        device = torch.device("cpu")
    task_list = get_tasks(suite)
    n_tasks = len(task_list)
    if eval_seed is not None:
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            try:
                xpu.manual_seed_all(eval_seed)
            except Exception:
                pass
    policy.eval()
    success_rate_per_task = []
    for task_id, task_name in enumerate(task_list):
        mt1 = metaworld.MT1(task_name)
        env = mt1.train_classes[task_name]()
        goal_success = []
        n_goals = min(50, len(mt1.train_tasks))
        for goal_idx in range(n_goals):
            task = mt1.train_tasks[goal_idx]
            env.set_task(task)
            if eval_seed is not None:
                try:
                    out = env.reset(seed=eval_seed + task_id * 1000 + goal_idx)
                except TypeError:
                    out = env.reset()
            else:
                out = env.reset()
            obs = out[0] if isinstance(out, tuple) else out
            if isinstance(obs, tuple):
                obs = obs[0] if len(obs) > 0 else obs
            obs = np.asarray(obs).flatten()
            oh = one_hot_task(task_id, num_tasks=n_tasks)
            done = False
            steps = 0
            while not done and steps < 500:
                x = np.concatenate([obs, oh]).astype(np.float32)
                obs_t = torch.FloatTensor(x).to(device)
                with torch.no_grad():
                    action = policy(obs_t).cpu().numpy()
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
        rate = sum(goal_success) / len(goal_success) * 100
        success_rate_per_task.append(rate)
    success_rate_avg = sum(success_rate_per_task) / len(success_rate_per_task)
    return success_rate_per_task, success_rate_avg


def eval_mt10(policy, clip_actions=True, eval_seed=42, device=None):
    """Backward-compat wrapper: eval_multitask(..., suite='mt10')."""
    return eval_multitask(policy, "mt10", clip_actions=clip_actions, eval_seed=eval_seed, device=device)


class ClonePolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=None):
        super(ClonePolicy, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        
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


def infer_baseline_policy_architecture(model_path):
    """Load a baseline .pth once and return (input_dim, output_dim, hidden_sizes, state_dict).
    Callers can build ClonePolicy(input_dim, output_dim, hidden_sizes=hidden_sizes) and
    load_state_dict(state_dict) without reading the file again.
    MT-50 ready: input_dim is taken from the checkpoint, so MT1 (39), MT10 (49), or MT50
    (when added) are supported without code changes.
    """
    import re
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {model_path} is not a state_dict or dict with 'state_dict'")
    # Optional: use saved metadata if present (future extended format)
    if isinstance(ckpt, dict) and "hidden_sizes" in ckpt and "input_dim" in ckpt and "output_dim" in ckpt:
        return (
            ckpt["input_dim"],
            ckpt["output_dim"],
            ckpt["hidden_sizes"],
            ckpt.get("state_dict", state),
        )
    # Infer from ClonePolicy's net.0, net.2, net.4, ... (Linear layers at even indices)
    weight_keys = [k for k in state if re.match(r"net\.\d+\.weight", k)]
    even_indices = sorted([int(k.split(".")[1]) for k in weight_keys if int(k.split(".")[1]) % 2 == 0])
    if not even_indices:
        raise ValueError(f"Checkpoint at {model_path} has no 'net.<even>.weight' keys (not a ClonePolicy?)")
    ordered = [f"net.{i}.weight" for i in sorted(even_indices)]
    for k in ordered:
        if k not in state:
            raise ValueError(f"Checkpoint at {model_path} missing key {k}")
    input_dim = state["net.0.weight"].shape[1]
    output_dim = state[ordered[-1]].shape[0]
    hidden_sizes = [state[k].shape[0] for k in ordered[:-1]]
    return (input_dim, output_dim, hidden_sizes, state)


def train_model(learning_rate=0.0003, num_epochs=20, batch_size=64, hidden_sizes=None,
                save_name='cloned_policy.pth', clip_actions=True, data_path=None,
                end_weight=3.0, end_fraction=0.3, end_inner_weight=None, end_inner_fraction=0.0,
                save_run=True, keep_runs=3, eval_seed=42, lr_decay_epoch=None, lr_decay_gamma=0.5,
                end_upsample=False, suite=None, device="auto",
                use_wandb=True, wandb_tags=None, wandb_project=None, wandb_save_model=False):
    """Train a behavioral cloning policy.
    
    Args:
        ... (same as before)
        suite: 'mt1', 'mt10', or 'mt50'. If None, defaults to 'mt1'.
        use_wandb: If True, log to W&B (enabled by default; use --no-wandb to disable).
        wandb_tags: Optional list of tags for the run.
        wandb_project: W&B project name (default from config or 'CS229_FinalProject').
        wandb_save_model: If True, upload final checkpoint as W&B artifact.
    """
    if suite is None:
        suite = "mt1"
    multitask = suite in ("mt10", "mt50")
    n_tasks = num_tasks(suite)
    in_dim = policy_input_dim(suite)

    device = get_device(device)
    if hidden_sizes is None:
        hidden_sizes = [256, 256, 128]

    script_dir = _SCRIPT_DIR
    if data_path is None:
        if multitask:
            data_path = os.path.join(script_dir, "..", "data", f"expert_data_{suite}.npz")
        else:
            data_path = os.path.join(script_dir, "..", "data", "expert_data_reach-v3.npz")
    
    model_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, save_name)
    
    print(f"Loading data from: {data_path}")
    print(f"Will save model to: {save_path}")
    print(f"Device: {device}")
    if multitask:
        print(f"Multi-task mode ({suite}): {in_dim}-dim input (obs + one-hot task)")

    # W&B: init (enabled by default; disable with use_wandb=False or WANDB_MODE=disabled)
    run = None
    if use_wandb:
        try:
            import wandb
            lr_str = f"{learning_rate:.0e}".replace("-0", "-").replace("e-", "e-")
            name_parts = [f"{suite}-lr{lr_str}-e{num_epochs}", f"end{int(end_weight)}"]
            if end_inner_weight is not None and end_inner_fraction and end_inner_fraction > 0:
                name_parts.append(f"inner{int(end_inner_weight)}x{int(end_inner_fraction*100)}")
            run_name = "-".join(name_parts)
            proj = wandb_project or "CS229_FinalProject"
            approach = "baseline"
            wandb_tags_list = [approach, suite] + list(wandb_tags or [])
            run = wandb.init(project=proj, name=run_name, job_type="train", tags=wandb_tags_list, reinit=True)
            wandb.config.update({
                "approach": approach,
                "lr": learning_rate, "epochs": num_epochs, "batch_size": batch_size,
                "hidden_sizes": hidden_sizes, "end_weight": end_weight, "end_fraction": end_fraction,
                "end_inner_weight": end_inner_weight, "end_inner_fraction": end_inner_fraction,
                "lr_decay_epoch": lr_decay_epoch, "lr_decay_gamma": lr_decay_gamma,
                "clip_actions": clip_actions, "eval_seed": eval_seed, "suite": suite,
                "end_upsample": end_upsample, "save_name": save_name,
            }, allow_val_change=True)
        except Exception as e:
            print(f"W&B init skipped: {e}")
            run = None

    data = np.load(data_path, allow_pickle=True)
    states_list = list(data['states'])
    actions_list = list(data['actions'])
    
    if multitask:
        if 'task_ids' not in data:
            raise ValueError(f"Multi-task mode ({suite}) requires 'task_ids' in the npz file.")
        task_ids = np.asarray(data['task_ids'])
        if len(task_ids) != len(states_list):
            raise ValueError("task_ids length must match number of trajectories.")
        # Append one-hot(task_id) to each state in each trajectory
        new_states_list = []
        for i, (traj_s, tid) in enumerate(zip(states_list, task_ids)):
            oh = one_hot_task(int(tid), num_tasks=n_tasks)
            traj_s = np.asarray(traj_s)
            if traj_s.ndim == 1:
                traj_s = traj_s.reshape(1, -1)
            oh_broadcast = np.broadcast_to(oh, (len(traj_s), len(oh)))
            new_states_list.append(np.hstack([traj_s.astype(np.float32), oh_broadcast]))
        states_list = new_states_list
    
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
    policy = policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    use_weights = not end_upsample and (end_weight != 1.0 or (use_inner and end_inner_weight != 1.0))

    train_start = time.perf_counter()
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
            batch_x, batch_y, batch_w = batch[0].to(device), batch[1].to(device), batch[2].to(device)
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
        if run is not None:
            try:
                step = epoch + 1
                run.log({"train/loss": final_loss, "epoch": step}, step=step)
                if lr_decay_epoch is not None:
                    run.log({"train/lr": optimizer.param_groups[0]["lr"]}, step=step)
            except Exception:
                pass
        if (epoch + 1) % 50 == 0 or (epoch + 1) == num_epochs:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {final_loss:.6f}")

    train_elapsed = time.perf_counter() - train_start
    print(f"\nTraining completed in {train_elapsed:.1f}s ({train_elapsed/60:.1f} min)")

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
        if multitask:
            name_parts.append(suite)
        run_fname = f"run_{ts}_{'_'.join(name_parts)}.pth"
        run_path = os.path.join(runs_dir, run_fname)
        torch.save(policy.state_dict(), run_path)
        shutil.copy2(run_path, save_path)
        print(f"\nModel saved to {run_path} (latest copied to {save_path})")
    else:
        torch.save(policy.state_dict(), save_path)
        print(f"\nModel saved to {save_path}")

    if save_run:
        task_list = get_tasks(suite)
        if multitask:
            print(f"Running {suite} eval for run record..." + (f" (seed={eval_seed})" if eval_seed is not None else ""))
            success_rate_per_task, success_rate_avg = eval_multitask(policy, suite, clip_actions=clip_actions, eval_seed=eval_seed, device=device)
            per_task_str = ", ".join(f"{name}: {r:.0f}%" for name, r in zip(task_list, success_rate_per_task))
            print(f"  Eval: avg={success_rate_avg:.1f}% | per-task: " + per_task_str)
        else:
            print("Running 50-goal eval for run record..." + (f" (seed={eval_seed})" if eval_seed is not None else ""))
            success_rate, goal_success, failed_goals = eval_50_goals(policy, clip_actions=clip_actions, eval_seed=eval_seed, device=device)
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
            "run_path": os.path.abspath(run_path),
            "suite": suite,
            "mt10": multitask and suite == "mt10",  # backward compat
            "train_duration_seconds": round(train_elapsed, 1),
        }
        if multitask:
            run_record["success_rate_per_task"] = [round(r, 2) for r in success_rate_per_task]
            run_record["success_rate_avg"] = round(success_rate_avg, 2)
        else:
            run_record["success_rate"] = round(success_rate, 2)
            run_record["goal_success"] = goal_success
            run_record["failed_goals"] = failed_goals

        # W&B: training run gets only train summary and optional model artifact; auto eval is a separate eval run
        training_run_id = None
        if run is not None:
            try:
                run.log({
                    "train/final_loss": float(final_loss) if final_loss is not None else None,
                    "train/duration_seconds": round(train_elapsed, 1),
                    "train/num_samples": num_samples,
                })
                if wandb_save_model:
                    wandb.save(run_path, base_path=os.path.dirname(run_path), policy="end")
                training_run_id = run.id
            except Exception:
                pass

        # W&B: create a separate eval run for the auto eval (linked to training run via config.training_run_id)
        if run is not None and use_wandb and training_run_id is not None:
            try:
                import wandb
                proj = wandb_project or "CS229_FinalProject"
                approach = "baseline"
                # Use actual saved run file (run_path), not save_name, for model identity
                model_stem = os.path.splitext(os.path.basename(run_path))[0]
                lr_str = f"{learning_rate:.0e}".replace("-0", "-").replace("e-", "e-")
                param_suffix = f"-lr{lr_str}-e{num_epochs}-end{int(end_weight)}"
                if multitask:
                    eval_run_name = f"eval-{suite}-{model_stem}-{50 * n_tasks}ep{param_suffix}"
                    task_or_suite = suite
                    episodes = 50 * n_tasks
                else:
                    task_name = task_list[0]  # mt1 -> reach-v3
                    eval_run_name = f"eval-mt1-{task_name}-{model_stem}-50ep{param_suffix}"
                    task_or_suite = task_name
                    episodes = 50
                eval_tags = [approach, suite]
                wandb.init(project=proj, job_type="eval", name=eval_run_name, tags=eval_tags, reinit=True)
                wandb.config.update({
                    "approach": approach,
                    "suite": suite,
                    "model": os.path.basename(run_path),
                    "task_or_suite": task_or_suite,
                    "episodes": episodes,
                    "seed": eval_seed,
                    "clip_actions": clip_actions,
                    "training_run_id": training_run_id,
                    "source": "auto",
                    # Training hyperparameters (so Runs table and grouping show them)
                    "lr": learning_rate,
                    "epochs": num_epochs,
                    "batch_size": batch_size,
                    "hidden_sizes": hidden_sizes,
                    "end_weight": end_weight,
                    "end_fraction": end_fraction,
                    "end_inner_weight": end_inner_weight,
                    "end_inner_fraction": end_inner_fraction,
                    "lr_decay_epoch": lr_decay_epoch,
                    "lr_decay_gamma": lr_decay_gamma,
                    "end_upsample": end_upsample,
                    "save_name": save_name,
                }, allow_val_change=True)
                if multitask:
                    wandb.log({"eval/success_rate_avg": success_rate_avg})
                    for i, r in enumerate(success_rate_per_task):
                        wandb.log({f"eval/success_rate_{task_list[i]}": r})
                else:
                    wandb.log({"eval/success_rate": success_rate})
                wandb.finish()
            except Exception as e:
                print(f"W&B auto-eval run skipped: {e}")

        if run is not None:
            try:
                run.finish()
            except Exception:
                pass
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
            run_file = r.get("save_name") or (os.path.basename(run_path) if run_path else "—")
            ep = r.get("epochs", "")
            ew = r.get("end_weight", "")
            ei_str = (f"{r.get('end_inner_weight')}@{r.get('end_inner_fraction', 0)*100:.0f}%" 
                      if (r.get("end_inner_weight") and r.get("end_inner_fraction") is not None) else "—")
            ca = r.get("clip_actions")
            clip_str = "yes" if ca is True else ("no" if ca is False else "—")
            fl = r.get("final_loss")
            fl_display = f"{fl * 1e6:.4f}" if fl is not None else "—"
            td = r.get("train_duration_seconds")
            time_str = f"{td}s" if td is not None else "—"
            if r.get("mt10") or r.get("suite", "").startswith("mt"):
                sr_per = r.get("success_rate_per_task") or []
                reach = f"{sr_per[0]}%" if len(sr_per) > 0 else "—"
                avg = f"{r.get('success_rate_avg', '—')}%" if r.get("success_rate_avg") is not None else "—"
                return f"| {run_file} | {ep} | {ew} | {ei_str} | {clip_str} | {fl_display} | {time_str} | MT10 reach: {reach} | avg: {avg} |"
            sr = r.get("success_rate")
            sr = f"{sr}%" if sr is not None else "—"
            fg = r.get("failed_goals", [])
            fg_str = ",".join(str(x) for x in fg[:15]) + ("..." if len(fg) > 15 else "")
            return f"| {run_file} | {ep} | {ew} | {ei_str} | {clip_str} | {fl_display} | {time_str} | {sr} | {fg_str} |"

        lines = [
            "# Training runs (all recent)",
            "",
            "Full history in `training_runs.json`. Per-run models in `models/runs/` with descriptive names (end weight, inner tier, clip/noclip). MT-10 runs show reach and avg success.",
            "",
            "**Latest batch** (most recent runs):",
            "",
            "| run_file | epochs | end_weight | end_inner | clip | final_loss (*10e6) | train_time | success_rate | failed_goals / MT10 |",
            "|----------|--------|------------|-----------|------|-----------------|------------|--------------|---------------------|",
        ]
        for r in latest_batch:
            lines.append(row(r))
        lines.extend([
            "",
            "---",
            "",
            "**Older runs:**",
            "",
            "| run_file | epochs | end_weight | end_inner | clip | final_loss (*10e6) | train_time | success_rate | failed_goals / MT10 |",
            "|----------|--------|------------|-----------|------|-----------------|------------|--------------|---------------------|",
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
    cfg = load_train_config()
    parser = argparse.ArgumentParser(description="Train a behavioral cloning policy")
    parser.add_argument("--config", type=str, default=None, help="Path to train config YAML (default: baseline/train_config.yaml)")
    parser.add_argument("--lr", type=float, default=cfg.get("lr", 0.0003), help="Learning rate")
    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 500), help="Number of epochs")
    parser.add_argument("--batch", type=int, default=cfg.get("batch_size", 64), help="Batch size")
    parser.add_argument("--hidden", type=int, nargs="+", default=cfg.get("hidden_sizes", [256, 256, 128]),
                        help="Hidden layer sizes")
    parser.add_argument("--name", type=str, default=cfg.get("save_name", "cloned_policy.pth"),
                        help="Model save name")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip actions (default: clip to [-1, 1])")
    parser.add_argument("--data", type=str, default=None, help="Path to expert data")
    parser.add_argument("--end-weight", type=float, default=cfg.get("end_weight", 3.0),
                        help="Weight for last fraction of each trajectory (1.0 = no weighting)")
    parser.add_argument("--end-fraction", type=float, default=cfg.get("end_fraction", 0.3),
                        help="Fraction of each trajectory to up-weight from the end (e.g. 0.3 = last 30%%)")
    parser.add_argument("--end-inner-weight", type=float, default=cfg.get("end_inner_weight"),
                        help="Inner tier weight for last end-inner-fraction (e.g. 5.0); optional")
    parser.add_argument("--end-inner-fraction", type=float, default=cfg.get("end_inner_fraction", 0.05),
                        help="Fraction for inner tier (e.g. 0.05 = last 5%%, 0.1 = last 10%%)")
    parser.add_argument("--no-save-run", action="store_true",
                        help="Do not log run to training_runs.json or copy model to runs/")
    parser.add_argument("--keep-runs", type=int, default=cfg.get("keep_runs", 50),
                        help="Max run copies to keep in models/runs/ (default: 50; 0 = keep all)")
    parser.add_argument("--eval-seed", type=int, default=cfg.get("eval_seed", 42),
                        help="Seed for post-training 50-goal eval; use test.py --seed N with same N to match")
    parser.add_argument("--lr-decay-epoch", type=int, default=cfg.get("lr_decay_epoch"),
                        help="Decay LR by --lr-decay-gamma every N epochs (e.g. 250); optional")
    parser.add_argument("--lr-decay-gamma", type=float, default=cfg.get("lr_decay_gamma", 0.5),
                        help="LR decay factor (default 0.5); used only if --lr-decay-epoch is set")
    parser.add_argument("--end-upsample", action="store_true", default=cfg.get("end_upsample", False),
                        help="Use end upsampling (duplicate last segments) instead of weighted MSE")
    parser.add_argument("--suite", type=str, default=cfg.get("suite", "mt1"), choices=["mt1", "mt10", "mt50"],
                        help="Suite: mt1 (single task), mt10, or mt50 (default from config)")
    parser.add_argument("--device", type=str, default=cfg.get("device", "auto"),
                        choices=["auto", "cuda", "xpu", "cpu"],
                        help="Device: auto (prefer GPU), cuda (NVIDIA), xpu (Intel Arc), or cpu (default: auto)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging (enabled by default)")
    parser.add_argument("--wandb-tag", type=str, action="append", default=None,
                        help="W&B run tag (e.g. name:alice); can be repeated")
    parser.add_argument("--wandb-save-model", action="store_true", help="Upload final checkpoint to W&B as artifact")
    args = parser.parse_args()

    if args.config is not None:
        cfg = load_train_config(args.config)
        # Re-apply CLI overrides are already in args
    use_wandb = not args.no_wandb and cfg.get("use_wandb", True)
    wandb_project = cfg.get("wandb_project") or "CS229_FinalProject"

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
        end_upsample=args.end_upsample,
        suite=args.suite,
        device=args.device,
        use_wandb=use_wandb,
        wandb_tags=args.wandb_tag,
        wandb_project=wandb_project,
        wandb_save_model=args.wandb_save_model,
    )