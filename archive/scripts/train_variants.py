import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time

class ClonePolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 256, 128]):
        super(ClonePolicy, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_variant(learning_rate, num_epochs, batch_size, hidden_sizes, save_name, clip_actions=False):
    """Train a variant and save it"""
    
    # Load data
    data = np.load('expert_data_reach-v3.npz', allow_pickle=True)
    X_train = np.concatenate(data['states'])
    Y_train = np.concatenate(data['actions'])
    
    # Clip actions if requested (for train/test consistency)
    if clip_actions:
        Y_train = np.clip(Y_train, -1.0, 1.0)

    num_samples = 50000
    indices = np.random.choice(len(X_train), num_samples, replace=False)
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    X_tensor = torch.FloatTensor(X_train)
    Y_tensor = torch.FloatTensor(Y_train)

    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = ClonePolicy(39, 4, hidden_sizes=hidden_sizes)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    print(f"\n{'='*60}")
    print(f"Training: {save_name}")
    print(f"  LR={learning_rate}, Epochs={num_epochs}, Batch={batch_size}, Hidden={hidden_sizes}")
    print(f"{'='*60}")

    start_time = time.time()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            predictions = policy(batch_x)
            loss = loss_fn(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss/len(dataloader)
        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.1f}s")
    
    torch.save(policy.state_dict(), save_name)
    print(f"✅ Saved to {save_name}")

if __name__ == "__main__":
    # Original baseline (no clipping)
    train_variant(learning_rate=0.0003, num_epochs=20, batch_size=64, 
                  hidden_sizes=[256, 256, 128], 
                  save_name='baseline_original.pth', clip_actions=False)
    
    # Variant 1: Higher LR, more epochs (no clipping)
    train_variant(learning_rate=0.001, num_epochs=50, batch_size=64,
                  hidden_sizes=[256, 256, 128],
                  save_name='baseline_lr001_e50.pth', clip_actions=False)
    
    # Variant 2: Even higher LR (no clipping)
    train_variant(learning_rate=0.005, num_epochs=50, batch_size=64,
                  hidden_sizes=[256, 256, 128],
                  save_name='baseline_lr005_e50.pth', clip_actions=False)
    
    # Variant 3: Larger network (no clipping)
    train_variant(learning_rate=0.001, num_epochs=50, batch_size=64,
                  hidden_sizes=[512, 512, 256],
                  save_name='baseline_larger_e50.pth', clip_actions=False)
    
    # Variant 4: Smaller batch size (no clipping)
    train_variant(learning_rate=0.001, num_epochs=50, batch_size=32,
                  hidden_sizes=[256, 256, 128],
                  save_name='baseline_lr001_b32_e50.pth', clip_actions=False)
    
    # Variant 5: LR 0.001, 50 epochs WITH action clipping (train/test consistency)
    train_variant(learning_rate=0.001, num_epochs=50, batch_size=64,
                  hidden_sizes=[256, 256, 128],
                  save_name='baseline_lr001_e50_CLIPPED.pth', clip_actions=True)
    
    print("\n" + "="*60)
    print("All variants trained! Now test each with test_variants.py")
    print("="*60)
