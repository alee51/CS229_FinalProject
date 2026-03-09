import argparse
import numpy as np
import torch
import metaworld

from models import VAEPolicy


@torch.no_grad()
def eval_success_rate(
    checkpoint: str,
    task_name: str = "reach-v3",
    n_episodes: int = 50,
    max_steps: int = 200,
    device: str = "cpu",
    render: bool = False,
):
    device = torch.device(device)

    # Load model
    model = VAEPolicy(input_dim=39, action_dim=4, latent_dim=16).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    mt1 = metaworld.MT1(task_name)
    env = mt1.train_classes[task_name]()

    successes = 0
    returns = []

    for ep in range(n_episodes):
        # Use a consistent task instance (you can randomize across mt1.train_tasks if you want)
        env.set_task(mt1.train_tasks[0])

        obs, _ = env.reset()
        ep_return = 0.0
        ep_success = False

        for _ in range(max_steps):
            if render:
                env.render()

            state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1,39)
            action_pred, _, _, _ = model(state_t)

            # Bound to [-1, 1]
            action = torch.tanh(action_pred).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)

            # MetaWorld typically uses info["success"] == 1 when successful
            if isinstance(info, dict) and info.get("success", 0) == 1:
                ep_success = True

            if terminated or truncated:
                break

        successes += int(ep_success)
        returns.append(ep_return)

    success_rate = successes / max(n_episodes, 1)
    avg_return = float(np.mean(returns)) if returns else 0.0

    print("Success-rate evaluation:")
    print(f"  task: {task_name}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  episodes: {n_episodes}")
    print(f"  successes: {successes}")
    print(f"  success_rate: {success_rate:.4f}")
    print(f"  avg_return: {avg_return:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="reach-v3")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    eval_success_rate(
        checkpoint=args.checkpoint,
        task_name=args.task,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        render=args.render,
    )


if __name__ == "__main__":
    main()