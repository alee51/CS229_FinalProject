import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import collect_expert_data
from models import VAEPolicy
from dataset import MetaWorldDataset

def train():
    # Hyperparameters
    latent_dim = 16
    beta = 0.1 # Weight for KL divergence [cite: 20, 22]
    learning_rate = 1e-3
    epochs = 50
    
    # Initialize Model & Optimizer
    model = VAEPolicy(input_dim=39, action_dim=4, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Placeholder for Meta-World data collection [cite: 40]
    obs, acts = collect_expert_data(task_name="reach-v3", num_episodes=50)
    dataset = MetaWorldDataset(obs, acts)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for states, expert_actions in loader:
            optimizer.zero_grad()
            
            # Forward pass
            pred_actions, recon_states, mu, logvar = model(states)
            
            # Joint Loss Objective 
            # 1. Action Prediction (MSE) [cite: 21]
            imitation_loss = torch.nn.functional.mse_loss(pred_actions, expert_actions)
            # 2. State Reconstruction [cite: 20]
            recon_loss = torch.nn.functional.mse_loss(recon_states, states)
            # 3. KL Divergence for smoothness [cite: 10, 20]
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            total_loss = imitation_loss + recon_loss + (beta * kld_loss)
            
            total_loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch}: Loss {total_loss.item()}")

if __name__ == "__main__":
    train()