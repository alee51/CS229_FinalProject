import argparse
import numpy as np
import torch
import metaworld

from models_mt10_vae import TaskConditionedVAEPolicy


@torch.no_grad()
def eval_mt10_vae_policy(
    checkpoint: str,
    episodes_per_task: int = 20,
    max_steps: int = 300,
    device: str = "cpu",
    use_mu_for_policy: bool = True,
):
    device = torch.device(device)

    ckpt = torch.load(checkpoint, map_location=device)
    task_names = ckpt["task_names"]
    latent_dim = ckpt.get("latent_dim", 16)
    num_tasks = ckpt.get("num_tasks", len(task_names))

    model = TaskConditionedVAEPolicy(
        state_dim=39,
        action_dim=4,
        latent_dim=latent_dim,
        num_tasks=num_tasks,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bench = metaworld.MT10()
    total_eps = 0
    total_succ = 0

    print("Per-task breakdown:")
    for task_id, task_name in enumerate(task_names):
        env_cls = bench.train_classes[task_name]
        env = env_cls()
        matching_tasks = [t for t in bench.train_tasks if t.env_name == task_name]

        succ = 0
        for ep in range(episodes_per_task):
            task = matching_tasks[ep % len(matching_tasks)]
            env.set_task(task)

            obs, _ = env.reset()
            ep_success = False

            for _ in range(max_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                task_t = torch.tensor([task_id], dtype=torch.long, device=device)

                action, _, _, _ = model(obs_t, task_t, use_mu_for_policy=use_mu_for_policy)
                action = action.squeeze(0).cpu().numpy()
                action = np.clip(action, -1.0, 1.0)

                obs, reward, terminated, truncated, info = env.step(action)

                if isinstance(info, dict) and info.get("success", 0) == 1:
                    ep_success = True

                if terminated or truncated:
                    break

            succ += int(ep_success)

        total_eps += episodes_per_task
        total_succ += succ
        print(f"  {task_name:22s} | succ_rate={succ/episodes_per_task:.3f} ({succ}/{episodes_per_task})")

    print("\nMT10 Task-conditioned VAE+policy evaluation (train tasks):")
    print(f"  checkpoint: {checkpoint}")
    print(f"  episodes_per_task: {episodes_per_task}")
    print(f"  max_steps: {max_steps}")
    print(f"  total_episodes: {total_eps}")
    print(f"  total_successes: {total_succ}")
    print(f"  overall_success_rate: {total_succ / max(total_eps,1):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-mu-for-policy", action="store_true")
    args = parser.parse_args()

    eval_mt10_vae_policy(
        checkpoint=args.checkpoint,
        episodes_per_task=args.episodes_per_task,
        max_steps=args.max_steps,
        device=args.device,
        use_mu_for_policy=args.use_mu_for_policy,
    )


if __name__ == "__main__":
    main()