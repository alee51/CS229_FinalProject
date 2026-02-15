import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class ClonePolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClonePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_model_improved():
    """Improved baseline with action normalization and better hyperparameters"""
    
    # Load data
    data = np.load('expert_data_reach-v3.npz', allow_pickle=True)
    X_train = np.concatenate(data['states'])
    Y_train = np.concatenate(data['actions'])
    
    print(f"📊 Dataset: {len(X_train)} samples")
    print(f"   Action range (raw): [{Y_train.min():.3f}, {Y_train.max():.3f}]")

    # CRITICAL FIX: Normalize actions to [-1, 1] range
    action_min = Y_train.min(axis=0, keepdims=True)
    action_max = Y_train.max(axis=0, keepdims=True)
    Y_train_normalized = 2 * (Y_train - action_min) / (action_max - action_min) - 1
    
    print(f"   Action range (normalized): [{Y_train_normalized.min():.3f}, {Y_train_normalized.max():.3f}]")
    
    # Use all available data (or most of it) instead of just 50k
    num_samples = min(90000, len(X_train))  # Use 90k of ~92k
    indices = np.random.choice(len(X_train), num_samples, replace=False)
    X_train = X_train[indices]
    Y_train_normalized = Y_train_normalized[indices]
    
    print(f"   Using {num_samples} samples for training")

    X_tensor = torch.FloatTensor(X_train)
    Y_tensor = torch.FloatTensor(Y_train_normalized)

    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Smaller batch size for better updates

    policy = ClonePolicy(X_tensor.shape[1], Y_tensor.shape[1])
    optimizer = optim.Adam(policy.parameters(), lr=0.0005)  # Slightly higher LR
    loss_fn = nn.MSELoss()

    # Train for more epochs
    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            predictions = policy(batch_x)
            loss = loss_fn(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss/len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")

    # Save normalization parameters for inference
    torch.save({
        'policy_state': policy.state_dict(),
        'action_min': action_min,
        'action_max': action_max
    }, 'baseline_improved.pth')
    print("✅ Improved baseline saved to baseline_improved.pth")

if __name__ == "__main__":
    train_model_improved()
