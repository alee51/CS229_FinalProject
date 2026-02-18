import torch
import numpy as np
from torch.utils.data import Dataset, Sampler


class TCEDataset(Dataset):
    """
    Dataset for TCE + CRTR. Stores expert transitions and handles
    one-hot task encoding so the model knows which task it is executing.

    States and next_states are stored as (N, state_dim + num_tasks) after
    pre-concatenating the one-hot vector in __init__. This means the rest
    of the pipeline — trainer, model — never needs to know about task IDs;
    they just see wider state vectors.

    Args:
        states      : (N, state_dim)   raw proprioceptive states
        actions     : (N, action_dim)
        next_states : (N, state_dim)
        rewards     : (N, 1)
        traj_ids    : (N,)             trajectory IDs for CRTRBatchSampler
        task_labels : (N,)             integer task index 0–(num_tasks-1)
        num_tasks   : int              total number of tasks (10 for MT-10)
        device      : str              'cpu' or 'cuda'
    """

    def __init__(self, states, actions, next_states, rewards,
                 traj_ids, task_labels, num_tasks=10, device='cpu'):

        n = len(states)

        # =============================================================
        # ONE-HOT CHANGE 7: Build one-hot matrix and pre-concatenate.
        #
        # one_hot shape: (N, num_tasks)
        # After concatenation:
        #   states     : (N, state_dim + num_tasks)  e.g. (N, 49) for MT-10
        #   next_states: (N, state_dim + num_tasks)
        #
        # Both state and next_state get the SAME one-hot (the task doesn't
        # change within a transition). Pre-concatenating here means __getitem__
        # and contrastive_trainer.py need zero changes.
        # =============================================================
        task_labels_np = np.array(task_labels, dtype=np.int64).flatten()
        one_hot = np.zeros((n, num_tasks), dtype=np.float32)
        one_hot[np.arange(n), task_labels_np] = 1.0

        states_aug      = np.concatenate([states,      one_hot], axis=1)
        next_states_aug = np.concatenate([next_states, one_hot], axis=1)

        # Pre-load everything onto the target device. Keep num_workers=0
        # in DataLoader when device='cuda' (CUDA tensors can't be shared
        # across worker processes).
        self.states      = torch.FloatTensor(states_aug).to(device)      # (N, 49)
        self.actions     = torch.FloatTensor(actions).to(device)         # (N, 4)
        self.next_states = torch.FloatTensor(next_states_aug).to(device) # (N, 49)
        self.rewards     = torch.FloatTensor(rewards).to(device)         # (N, 1)
        self.traj_ids    = torch.LongTensor(traj_ids).to(device)         # (N,)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state':      self.states[idx],       # (49,) with one-hot appended
            'action':     self.actions[idx],       # (4,)
            'next_state': self.next_states[idx],   # (49,) with one-hot appended
            'reward':     self.rewards[idx],       # (1,)
            'traj_id':    self.traj_ids[idx],      # scalar
        }


class CRTRBatchSampler(Sampler):
    """
    Custom Batch Sampler for CRTR.

    For every batch, guarantees that `repetition_factor` samples are drawn
    from each of `batch_size // repetition_factor` unique trajectories.
    This ensures within-trajectory hard negatives exist in every batch,
    which is the core requirement of CRTR.

    Example: batch_size=512, repetition_factor=4
      → 128 unique trajectories × 4 samples each per batch.
      Each anchor has 3 same-context, different-time hard negatives.
    """

    def __init__(self, traj_ids, batch_size, repetition_factor=4):
        self.batch_size       = batch_size
        self.repetition_factor = repetition_factor

        if batch_size % repetition_factor != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"repetition_factor ({repetition_factor})"
            )

        self.samples_per_traj = batch_size // repetition_factor

        if torch.is_tensor(traj_ids):
            traj_ids = traj_ids.cpu().numpy()
        traj_ids = traj_ids.flatten()

        self.traj_groups = {}
        for idx, t_id in enumerate(traj_ids):
            t_id = int(t_id)
            if t_id not in self.traj_groups:
                self.traj_groups[t_id] = []
            self.traj_groups[t_id].append(idx)

        self.traj_groups  = {k: v for k, v in self.traj_groups.items() if len(v) >= 1}
        self.unique_trajs = list(self.traj_groups.keys())
        self.num_batches  = len(traj_ids) // batch_size

    def __iter__(self):
        can_draw_unique = len(self.unique_trajs) >= self.samples_per_traj

        for _ in range(self.num_batches):
            batch_indices = []

            selected_trajs = np.random.choice(
                self.unique_trajs,
                size=self.samples_per_traj,
                replace=not can_draw_unique,
            )

            for t_id in selected_trajs:
                indices  = self.traj_groups[t_id]
                selected = np.random.choice(indices, self.repetition_factor, replace=True)
                batch_indices.extend(selected.tolist())

            np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches