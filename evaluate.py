import numpy as np
import torch
import random
import metaworld
import argparse
import sys
import os

# ============================================================
# CHANGE 8: Added wandb import for logging eval results
# ============================================================
import wandb

from contrastive_model import TCEAgent

# The canonical MT-10 task list — matches collect_data.py and teammate's setup
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


def evaluate_policy(task_name, model_path, num_episodes=50,
                    latent_dim=64, fourier_dim=256, log_wandb=False):
    print(f"\nEvaluating: {task_name}")

    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()
    test_tasks = mt1.test_tasks if len(mt1.test_tasks) > 0 else mt1.train_tasks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TCEAgent(
        input_dim=39,
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
    success_count = 0

    for i in range(num_episodes):
        env.set_task(random.choice(test_tasks))
        obs, _ = env.reset()
        done    = False
        steps   = 0
        success = False

        while not done and steps < 500:
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
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
    print(f"  {task_name}: {success_rate:.1f}%  ({success_count}/{num_episodes})")
    return success_rate


# ============================================================
# CHANGE 9: New evaluate_mt10() function
# Runs all 10 tasks, prints a summary table matching your
# teammate's format, and logs everything to wandb.
# ============================================================
def evaluate_mt10(model_path, num_episodes=50, latent_dim=64,
                  fourier_dim=256, log_wandb=False):
    """
    Evaluate the model on all 10 MT-10 tasks and report per-task
    success rates plus the mean — directly comparable to teammate's results.
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
            task, model_path, num_episodes, latent_dim, fourier_dim
        )
        results[task] = rate

    mean_rate = sum(results.values()) / len(results)

    print("\n" + "=" * 40)
    print("MT-10 Evaluation Results")
    print("=" * 40)
    for task, rate in results.items():
        bar = "#" * int(rate / 5)   # simple ASCII bar out of 20 chars
        print(f"  {task:<30} {rate:5.1f}%  {bar}")
    print("-" * 40)
    print(f"  {'Mean Success Rate':<30} {mean_rate:5.1f}%")
    print("=" * 40)

    if log_wandb:
        # Log each task individually so you get a bar chart in wandb
        log_dict = {f"eval/{task}": rate for task, rate in results.items()}
        log_dict["eval/mean_success_rate"] = mean_rate
        wandb.log(log_dict)
        wandb.finish()

    return results, mean_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained TCE policy")
    parser.add_argument('--task',       type=str, default=None,
                        help='Single task name, or omit to run all MT-10 tasks')
    parser.add_argument('--episodes',   type=int, default=50)
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Must match the value used in train_model.py')
    parser.add_argument('--fourier_dim', type=int, default=256,
                        help='Must match the value used in train_model.py')
    # ============================================================
    # CHANGE 10: Added --wandb flag to evaluate.py CLI
    # Pass --wandb when you want results logged; omit for quick local checks
    # ============================================================
    parser.add_argument('--wandb', action='store_true',
                        help='Log evaluation results to Weights & Biases')
    args = parser.parse_args()

    model_path = f"models/tce_policy_mt10.pth"

    if args.task:
        # Single-task evaluation (useful for quick checks)
        evaluate_policy(args.task, model_path, args.episodes,
                        args.latent_dim, args.fourier_dim)
    else:
        # Full MT-10 sweep — the main evaluation mode
        evaluate_mt10(model_path, args.episodes,
                      args.latent_dim, args.fourier_dim,
                      log_wandb=args.wandb)