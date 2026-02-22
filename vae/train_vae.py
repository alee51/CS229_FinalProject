import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils import collect_expert_data
from models import VAEPolicy
from dataset import MetaWorldDataset


def train(
    task_name: str = "reach-v3",
    num_episodes: int = 50,
    latent_dim: int = 16,
    beta: float = 0.1,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    device: str = "cpu",
    save_path: str = "vae_policy_reach_v3.pt",
):
    device = torch.device(device)

    # Initialize Model & Optimizer
    model = VAEPolicy(input_dim=39, action_dim=4, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Collect expert data
    obs, acts = collect_expert_data(task_name=task_name, num_episodes=num_episodes)
    dataset = MetaWorldDataset(obs, acts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(epochs):
        epoch_total = 0.0
        epoch_imitation = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for states, expert_actions in loader:
            states = states.to(device)
            expert_actions = expert_actions.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_actions, recon_states, mu, logvar = model(states)

            # Bound actions to [-1, 1] (MetaWorld action space)
            pred_actions = torch.tanh(pred_actions)

            # 1) Action prediction loss (MSE)
            imitation_loss = torch.nn.functional.mse_loss(pred_actions, expert_actions, reduction="mean")

            # 2) State reconstruction (MSE)
            recon_loss = torch.nn.functional.mse_loss(recon_states, states, reduction="mean")

            # 3) KL divergence (mean per sample in batch)
            kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)
            kld_loss = kl_per_sample.mean()

            total_loss = imitation_loss + recon_loss + (beta * kld_loss)
            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_imitation += imitation_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kld_loss.item()
            n_batches += 1

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"total={epoch_total/max(n_batches,1):.6f} | "
            f"imit={epoch_imitation/max(n_batches,1):.6f} | "
            f"recon={epoch_recon/max(n_batches,1):.6f} | "
            f"kl={epoch_kl/max(n_batches,1):.6f}"
        )

    # Save checkpoint
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="reach-v3")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="vae_policy_reach_v3.pt")
    args = parser.parse_args()

    train(
        task_name=args.task,
        num_episodes=args.num_episodes,
        latent_dim=args.latent_dim,
        beta=args.beta,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()