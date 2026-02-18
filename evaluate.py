import numpy as np
import torch
import random
import metaworld
import argparse
import sys
import os
import wandb

from contrastive_model import TCEAgent

# =============================================================
# MUST match the order in collect_data.py and merge_data.py.
# Index position = which column of the one-hot vector is set to 1.
# =============================================================
MT10_TASKS = [
    'reach-v3',                   # 0
    'push-v3',                    # 1
    'pick-place-v3',              # 2
    'door-open-v3',               # 3
    'door-close-v3',              # 4
    'drawer-open-v3',             # 5
    'drawer-close-v3',            # 6
    'button-press-topdown-v3',    # 7
    'lever-pull-v3',              # 8
    'window-open-v3',             # 9
]


def make_one_hot(task_name, num_tasks=10):
    """
    Build a float32 numpy array of shape (num_tasks,) with a 1 at the
    index corresponding to task_name and 0s elsewhere.
    This must be concatenated to every observation at inference time
    to match the 49D input the model was trained on.
    """
    task_idx = MT10_TASKS.index(task_name)
    one_hot  = np.zeros(num_tasks, dtype=np.float32)
    one_hot[task_idx] = 1.0
    return one_hot


def evaluate_policy(task_name, model_path, num_episodes=50,
                    latent_dim=64, fourier_dim=256, num_tasks=10):
    print(f"\nEvaluating: {task_name}")

    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()
    test_tasks = mt1.test_tasks if len(mt1.test_tasks) > 0 else mt1.train_tasks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================
    # ONE-HOT CHANGE 13: input_dim = 39 + num_tasks = 49.
    # Must match what was used during training.
    # =========================================================
    input_dim = 39 + num_tasks
    agent = TCEAgent(
        input_dim=input_dim,
        action_dim=4,
        latent_dim=latent_dim,
        fourier_dim=fourier_dim,
    ).to(device)

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        agent.load_state_dict(state_dict)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    agent.eval()

    # =========================================================
    # ONE-HOT CHANGE 14: Pre-build the one-hot for this task.
    # At every step we concatenate it to the raw 39D observation
    # before passing to the agent — exactly mirroring what
    # TCEDataset did during training.
    # =========================================================
    one_hot = make_one_hot(task_name, num_tasks)  # (10,)

    success_count = 0

    for i in range(num_episodes):
        env.set_task(random.choice(test_tasks))
        obs, _ = env.reset()
        done    = False
        steps   = 0
        success = False

        while not done and steps < 500:
            # Concatenate one-hot to observation: (39,) + (10,) = (49,)
            obs_aug      = np.concatenate([obs, one_hot])
            state_tensor = torch.FloatTensor(obs_aug).unsqueeze(0).to(device)

            with torch.no_grad():
                action = agent(state_tensor).cpu().numpy().flatten()

            next_obs, reward, terminated, truncated, info = env.step(action)

            if info.get('success', 0.0) > 0.0:
                success = True

            obs   = next_obs
            steps += 1
            done  = terminated or truncated

        if success:
            success_count += 1

    success_rate = (success_count / num_episodes) * 100
    print(f"  {task_name:<35} {success_rate:5.1f}%  ({success_count}/{num_episodes})")
    return success_rate


def evaluate_mt10(model_path, num_episodes=50, latent_dim=64,
                  fourier_dim=256, num_tasks=10, log_wandb=False):
    """
    Evaluate on all 10 MT-10 tasks and print a summary table that matches
    the teammate's format for direct comparison.
    """
    if log_wandb:
        wandb.init(
            project="tce-metaworld",
            name=f"eval_mt10_{os.path.basename(model_path)}",
            job_type="eval",
        )

    results = {}
    for task in MT10_TASKS:
        rate = evaluate_policy(
            task, model_path, num_episodes, latent_dim, fourier_dim, num_tasks
        )
        results[task] = rate

    mean_rate = sum(results.values()) / len(results)

    print("\n" + "=" * 45)
    print("MT-10 Evaluation Results")
    print("=" * 45)
    for task, rate in results.items():
        bar = "#" * int(rate / 5)
        print(f"  {task:<35} {rate:5.1f}%  {bar}")
    print("-" * 45)
    print(f"  {'Mean Success Rate':<35} {mean_rate:5.1f}%")
    print("=" * 45)

    if log_wandb:
        log_dict = {f"eval/{task}": rate for task, rate in results.items()}
        log_dict["eval/mean_success_rate"] = mean_rate
        wandb.log(log_dict)
        wandb.finish()

    return results, mean_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained TCE policy")
    parser.add_argument('--task',        type=str,  default=None,
                        help='Single task to evaluate, or omit for full MT-10 sweep')
    parser.add_argument('--episodes',    type=int,  default=50)
    parser.add_argument('--latent_dim',  type=int,  default=64)
    parser.add_argument('--fourier_dim', type=int,  default=256)
    parser.add_argument('--num_tasks',   type=int,  default=10,
                        help='Must match --num_tasks used in train_model.py')
    parser.add_argument('--wandb', action='store_true',
                        help='Log results to Weights & Biases')
    args = parser.parse_args()

    model_path = "models/tce_policy_mt10.pth"

    if args.task:
        evaluate_policy(args.task, model_path, args.episodes,
                        args.latent_dim, args.fourier_dim, args.num_tasks)
    else:
        evaluate_mt10(model_path, args.episodes,
                      args.latent_dim, args.fourier_dim,
                      args.num_tasks, log_wandb=args.wandb)