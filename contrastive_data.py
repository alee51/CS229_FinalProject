import torch
import numpy as np
from torch.utils.data import Dataset, Sampler


class TCEDataset(Dataset):
    """
    A custom dataset for Temporal Contrastive Encoding (TCE).
    It holds the expert data and retrieves specific transitions (s, a, r, s')
    along with trajectory IDs for CRTR negative sampling.
    """


    def __init__(self, states, actions, next_states, rewards, traj_ids, device="cpu"):
        # 1. Store Data as Tensors
        # We convert numpy arrays to PyTorch FloatTensors immediately.
        # Doing this once in __init__ is much faster than doing it
        # every time we fetch a batch during training.
        self.states = torch.FloatTensor(states).to(device)       # (N, 39)
        self.actions = torch.FloatTensor(actions).to(device)      # (N, 4)
        self.next_states = torch.FloatTensor(next_states).to(device)  # (N, 39)
        self.rewards = torch.FloatTensor(rewards).to(device)      # (N, 1)

        # 2. Store Trajectory IDs
        # 'LongTensor' is used for integers (IDs) - this is important.
        # This is CRITICAL for the CRTR strategy. We need to know which
        # "Episode" (Trajectory) a specific data point belongs to.
        # This allows us to compare State A (time 10) vs State B (time 50)
        # from the SAME episode later in the loss function.
        self.traj_ids = torch.LongTensor(traj_ids).to(device)      # (N,)

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
class CRTRBatchSampler(Sampler):
    """
    Custom Batch Sampler for CRTR.

    Requirement: To implement in-trajectory negatives, a batch must contain multiple
    samples from the same trajectory. Standard random sampling fails to guarantee this
    in large datasets.

    Logic:
    1. Group all data indices by their trajectory ID.
    2. For each batch, select 'samples_per_traj' unique trajectories.
    3. For each selected trajectory, sample 'repetition_factor' indices.

    This ensures that for every anchor in the batch, there are (repetition_factor - 1)
    other samples from the same trajectory to serve as hard negatives.

    Example: batch_size=64, repetition_factor=4 -> 16 trajectories x 4 samples each.
    Each sample has 3 within-trajectory hard negatives in every batch.
    """

    def __init__(self, traj_ids, batch_size, repetition_factor=4):
        self.batch_size = batch_size
        self.repetition_factor = repetition_factor

        # Ensure batch size is divisible by repetition factor for balanced batches
        if batch_size % repetition_factor != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be divisible by "
                f"repetition factor ({repetition_factor})"
            )

        # How many distinct trajectories we pull per batch
        self.samples_per_traj = batch_size // repetition_factor

        # --- FIX 1: Convert traj_ids to a flat 1-D numpy array ---
        # traj_ids may arrive as a torch.LongTensor of shape (N,) or (N,1),
        # or as a numpy array of either shape. We normalise so that
        # dictionary keys are consistent plain Python ints.
        if torch.is_tensor(traj_ids):
            traj_ids = traj_ids.cpu().numpy()

        # Flatten in case shape is (N, 1) instead of (N,)
        traj_ids = traj_ids.flatten()

        # --- FIX 2: traj_groups initialisation was missing [] ---
        # Gemini's line:  self.traj_groups[t_id] =   <- SyntaxError
        self.traj_groups = {}
        for idx, t_id in enumerate(traj_ids):
            # int() handles both numpy scalars (ndim==0) and single-element
            # arrays (ndim==1) uniformly after flatten -- no branching needed.
            # Gemini's original branch   `t_id.item() if ndim==0 else t_id`
            # was also wrong: the else case left t_id as a numpy object,
            # causing inconsistent dict keys.
            t_id = int(t_id)
            if t_id not in self.traj_groups:
                self.traj_groups[t_id] = []          # <- was missing []
            self.traj_groups[t_id].append(idx)

        # Filter out any degenerate trajectories with zero samples
        self.traj_groups = {
            k: v for k, v in self.traj_groups.items() if len(v) >= 1
        }

        self.unique_trajs = list(self.traj_groups.keys())
        self.num_batches = len(traj_ids) // batch_size

    def __iter__(self):
        # For each batch, independently sample `samples_per_traj` UNIQUE
        # trajectories using np.random.choice without replacement.
        # This is the simplest correct approach and directly mirrors the
        # CRTR paper: each batch gets a fresh, unbiased draw of trajectories
        # with no risk of the same trajectory appearing twice in one batch.
        #
        # Edge case: if the dataset has fewer unique trajectories than
        # samples_per_traj (e.g. only 3 trajs but samples_per_traj=4),
        # we fall back to replace=True with a warning. In practice your
        # Metaworld data will have many trajectories so this won't trigger.
        can_draw_unique = len(self.unique_trajs) >= self.samples_per_traj

        for _ in range(self.num_batches):
            # --- FIX 3: batch_indices was missing [] (Gemini SyntaxError) ---
            batch_indices = []

            # Draw unique trajectories for this batch
            selected_trajs = np.random.choice(
                self.unique_trajs,
                size=self.samples_per_traj,
                replace=not can_draw_unique,   # False in the normal case
            )

            # For each chosen trajectory, sample 'repetition_factor' timestep
            # indices. replace=True handles short trajectories gracefully.
            for t_id in selected_trajs:
                indices = self.traj_groups[t_id]
                selected = np.random.choice(
                    indices, self.repetition_factor, replace=True
                )
                batch_indices.extend(selected.tolist())

            # Shuffle within the batch so within-traj pairs are not always
            # adjacent -- prevents the model using position as a shortcut
            np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches