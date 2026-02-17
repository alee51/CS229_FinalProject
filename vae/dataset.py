import torch
from torch.utils.data import Dataset
import numpy as np

class MetaWorldDataset(Dataset):
    def __init__(self, observations, actions):
        """
        observations: Numpy array of shape (N, 39) [cite: 18]
        actions: Numpy array of shape (N, action_dim)
        """
        self.observations = torch.tensor(observations, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]