import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 1. Define the Brain (Same as before)
class ClonePolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClonePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),  # Increased neurons to 256 for more brainpower
            nn.ReLU(),
            nn.Linear(256, 256),        # Added deeper layers
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_model():
    print("🚀 Loading massive expert dataset...")
    try:
        data = np.load('expert_data_reach-v3.npz', allow_pickle=True)
    except FileNotFoundError:
        print("❌ Error: expert_data_reach-v3.npz not found.")
        return

    # Prepare Data
    X_train = np.concatenate(data['states'])
    Y_train = np.concatenate(data['actions'])
    
    # Convert to Tensors
    X_tensor = torch.FloatTensor(X_train)
    Y_tensor = torch.FloatTensor(Y_train)

    print(f"📚 Training on {len(X_tensor)} examples...")

    # --- THE BIG FIX: Mini-Batch Loader ---
    # We chop the data into small batches of 64 examples
    dataset = TensorDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # --------------------------------------

    # Setup Model
    input_dim = X_tensor.shape[1]
    output_dim = Y_tensor.shape[1]
    
    policy = ClonePolicy(input_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.0003) # Lower learning rate for stability
    loss_fn = nn.MSELoss()

    # Training Loop
    num_epochs = 50  # We need fewer epochs now because batching updates more often
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_x, batch_y in dataloader:
            # A. Guess
            predictions = policy(batch_x)
            
            # B. Error
            loss = loss_fn(predictions, batch_y)
            
            # C. Learn
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Calculate average loss for this epoch
        avg_loss = total_loss / len(dataloader)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Average Loss: {avg_loss:.6f}")

    print("💾 Saving smarter brain to 'cloned_policy.pth'...")
    torch.save(policy.state_dict(), 'cloned_policy.pth')
    print("✅ Done!")

if __name__ == "__main__":
    train_model()