import argparse
import numpy as np
import torch
import metaworld

from models import VAEPolicy


@torch.no_grad()
def eval_success_rate(
    checkpoint: str,
    task_name: str = "reach-v3",
    episodes_per_task: int = 10,
    max_steps: int = 300,
    device: str = "cpu",
):
    device = torch.device(device)

    model = VAEPolicy(input_dim=39, action_dim=4, latent_dim=16).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()

    total_eps = 0
    total_success = 0
    returns = []

    # Evaluate across ALL training task instances (different goals)
    for task in mt1.train_tasks:
        env.set_task(task)

        for _ in range(episodes_per_task):
            obs, _ = env.reset()

            ep_ret = 0.0
            ep_success = False

            for _ in range(max_steps):
                state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                action_pred, _, _, _ = model(state_t)
                action = torch.tanh(action_pred).squeeze(0).cpu().numpy()

                # Safety clip (should already be in [-1,1], but keep it robust)
                action = np.clip(action, -1.0, 1.0)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += float(reward)

                if isinstance(info, dict) and info.get("success", 0) == 1:
                    ep_success = True

                if terminated or truncated:
                    break

            total_eps += 1
            total_success += int(ep_success)
            returns.append(ep_ret)

    success_rate = total_success / max(total_eps, 1)
    avg_return = float(np.mean(returns)) if returns else 0.0

    print("Success-rate evaluation (across mt1.train_tasks):")
    print(f"  task: {task_name}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  episodes_per_task: {episodes_per_task}")
    print(f"  total_episodes: {total_eps}")
    print(f"  successes: {total_success}")
    print(f"  success_rate: {success_rate:.4f}")
    print(f"  avg_return: {avg_return:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="reach-v3")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    eval_success_rate(
        checkpoint=args.checkpoint,
        task_name=args.task,
        episodes_per_task=args.episodes_per_task,
        max_steps=args.max_steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()