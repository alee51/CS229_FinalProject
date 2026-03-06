"""
VAE policy training. Uses core for task list and env via data_utils.
Root train.py expects: train_model(**kwargs), load_train_config(); test expects ClonePolicy.
"""
import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# When run from root, approach_dir is vae/scripts; when run as __main__ from vae/scripts, same
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VAE_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_VAE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_utils import collect_expert_data
from models import VAEPolicy
from dataset import MetaWorldDataset


# Root test.py expects ClonePolicy(x).forward(obs) to return a single tensor (action)
class ClonePolicy(VAEPolicy):
    """Wrapper so forward returns only action for compatibility with root test.py."""

    def forward(self, state):
        action_pred, recon_state, mu, logvar = super().forward(state)
        return action_pred


def load_train_config(config_path=None):
    """Load config. Returns dict (VAE can add a train_config.yaml later)."""
    return {}


def train(
    task_name: str = "reach-v3",
    num_episodes: int = 50,
    latent_dim: int = 16,
    beta: float = 0.1,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    device: str = "cpu",
    save_path: str = None,
):
    if save_path is None:
        save_path = os.path.join(_VAE_DIR, "models", "latest_policy.pth")
    device = torch.device(device)

    model = VAEPolicy(input_dim=39, action_dim=4, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

            pred_actions, recon_states, mu, logvar = model(states)
            pred_actions = torch.tanh(pred_actions)

            imitation_loss = torch.nn.functional.mse_loss(pred_actions, expert_actions, reduction="mean")
            recon_loss = torch.nn.functional.mse_loss(recon_states, states, reduction="mean")
            kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
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

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to: {save_path}")


def train_model(
    learning_rate=1e-3,
    num_epochs=50,
    batch_size=64,
    save_name="latest_policy.pth",
    **kwargs,
):
    """Entrypoint for root train.py --approach vae. Ignores baseline-specific kwargs."""
    task_name = kwargs.get("task_name", "reach-v3")
    latent_dim = kwargs.get("latent_dim", 16)
    beta = kwargs.get("beta", 0.1)
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(_VAE_DIR, "models", save_name)
    train(
        task_name=task_name,
        num_episodes=50,
        latent_dim=latent_dim,
        beta=beta,
        learning_rate=learning_rate,
        epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        save_path=save_path,
    )


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
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    save_path = args.save_path or os.path.join(_VAE_DIR, "models", "latest_policy.pth")
    train(
        task_name=args.task,
        num_episodes=args.num_episodes,
        latent_dim=args.latent_dim,
        beta=args.beta,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
