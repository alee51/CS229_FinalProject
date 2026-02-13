import torch
from torch.utils.data import Dataset


class TCEDataset(Dataset):
    """
    A custom dataset for Temporal Contrastive Encoding (TCE).
    It holds the expert data and retrieves specific transitions (s, a, r, s')
    along with trajectory IDs for CRTR negative sampling.
    """

    def __init__(self, states, actions, next_states, rewards, traj_ids):
        # 1. Store Data as Tensors
        # We convert numpy arrays to PyTorch FloatTensors immediately.
        # Doing this once in __init__ is much faster than doing it
        # every time we fetch a batch during training.
        self.states = torch.FloatTensor(states)  # Shape: (N, 39)
        self.actions = torch.FloatTensor(actions)  # Shape: (N, 4)
        self.next_states = torch.FloatTensor(next_states)  # Shape: (N, 39)
        self.rewards = torch.FloatTensor(rewards)  # Shape: (N, 1)

        # 2. Store Trajectory IDs
        # 'LongTensor' is used for integers (IDs) - this is important.
        # This is CRITICAL for the CRTR strategy. We need to know which
        # "Episode" (Trajectory) a specific data point belongs to.
        # This allows us to compare State A (time 10) vs State B (time 50)
        # from the SAME episode later in the loss function.
        self.traj_ids = torch.LongTensor(traj_ids)  # Shape: (N, 1)

    def __len__(self):
        # 3. The Size Contract
        # PyTorch needs to know how many total samples exist so it knows
        # when an "epoch" is finished.
        return len(self.states)

    def __getitem__(self, idx):
        # 4. The Retrieval Contract (The Override)
        # This function is called millions of times. It fetches the i-th datapoint.
        # We return a dictionary because it is cleaner to access in the training loop
        # (e.g., batch['state']) than remembering tuple indices (batch).
        return {
            'state': self.states[idx],  # s_t
            'action': self.actions[idx],  # a_t
            'next_state': self.next_states[idx],  # s_{t+1} (The "Positive" match)
            'reward': self.rewards[idx],  # r_t (For Reward Prediction Loss)
            'traj_id': self.traj_ids[idx]  # ID (For masking "Negative" matches)
        }