import numpy as np
import torch
from torch.utils.data import Dataset


class MetaWorldMTDataset(Dataset):
    """
    Multi-task dataset for MetaWorld where each transition has:
      obs: (obs_dim,)
      act: (act_dim,)
      task_id: int in [0, num_tasks-1]
    """
    def __init__(self, observations, actions, task_ids, obs_dim=39, act_dim=4):
        assert len(observations) == len(actions) == len(task_ids)

        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        task_ids = np.asarray(task_ids, dtype=np.int64)

        assert observations.shape[1] == obs_dim, f"Expected obs_dim={obs_dim}, got {observations.shape[1]}"
        assert actions.shape[1] == act_dim, f"Expected act_dim={act_dim}, got {actions.shape[1]}"

        self.observations = torch.tensor(observations, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
        self.task_ids = torch.tensor(task_ids, dtype=torch.long)

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.task_ids[idx]