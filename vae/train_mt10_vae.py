import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils import collect_expert_data_mt
from dataset_mt10 import MetaWorldMTDataset
from models_mt10_vae import TaskConditionedVAEPolicy


def train_mt10_vae(
    num_episodes_per_task: int = 200,
    latent_dim: int = 16,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 3e-4,
    lambda_recon: float = 0.1,
    beta_max: float = 0.05,
    warmup_epochs: int = 20,
    use_mu_for_policy: bool = True,
    device: str = "cpu",
    save_path: str = "mt10_task_conditioned_vae_policy.pt",
    seed: int = 0,
):
    device = torch.device(device)
    torch.manual_seed(seed)

    obs, acts, task_ids, task_names = collect_expert_data_mt(
        benchmark="MT10",
        num_episodes_per_task=num_episodes_per_task,
        use_all_train_task_variations=True,
        seed=seed,
    )

    dataset = MetaWorldMTDataset(obs, acts, task_ids, obs_dim=39, act_dim=4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TaskConditionedVAEPolicy(
        state_dim=39,
        action_dim=4,
        latent_dim=latent_dim,
        num_tasks=len(task_names),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()

        # KL warmup
        beta = beta_max * min(1.0, (epoch + 1) / max(1, warmup_epochs))

        tot = imit = rec = kl = 0.0
        n = 0

        for states, expert_actions, task_id in loader:
            states = states.to(device)
            expert_actions = expert_actions.to(device)
            task_id = task_id.to(device)

            optimizer.zero_grad()

            pred_actions, recon_states, mu, logvar = model(
                states, task_id, use_mu_for_policy=use_mu_for_policy
            )

            imitation_loss = torch.nn.functional.mse_loss(pred_actions, expert_actions, reduction="mean")
            recon_loss = torch.nn.functional.mse_loss(recon_states, states, reduction="mean")

            # KL mean per sample
            kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kld_loss = kl_per_sample.mean()

            total_loss = imitation_loss + (lambda_recon * recon_loss) + (beta * kld_loss)
            total_loss.backward()
            optimizer.step()

            tot += total_loss.item()
            imit += imitation_loss.item()
            rec += recon_loss.item()
            kl += kld_loss.item()
            n += 1

        print(
            f"Epoch {epoch+1:03d}/{epochs} | beta={beta:.4f} | "
            f"total={tot/max(n,1):.6f} | imit={imit/max(n,1):.6f} | "
            f"recon={rec/max(n,1):.6f} | kl={kl/max(n,1):.6f}"
        )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "task_names": task_names,
            "latent_dim": latent_dim,
            "num_tasks": len(task_names),
        },
        save_path,
    )
    print(f"Saved MT10 VAE+policy checkpoint to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes-per-task", type=int, default=200)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-recon", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--use-mu-for-policy", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="mt10_task_conditioned_vae_policy.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_mt10_vae(
        num_episodes_per_task=args.num_episodes_per_task,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_recon=args.lambda_recon,
        beta_max=args.beta_max,
        warmup_epochs=args.warmup_epochs,
        use_mu_for_policy=args.use_mu_for_policy,
        device=args.device,
        save_path=args.save_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()