import torch
import torch.nn as nn
import numpy as np


class RandomFourierProjection(nn.Module):
    """
    Implements Random Fourier Features (RFF) for kernel approximation.

    Maps a latent vector z (dim d) to a higher-dimensional feature space (dim D)
    such that the inner product approximates a shift-invariant kernel:
        K(x, y) approx phi(x)^T phi(y)

    The feature map is defined as (Bochner's Theorem for Gaussian/RBF kernel):
        phi(z) = sqrt(2/D) * cos(Wz + b)
    where:
        W in R^(d x D)  sampled from N(0, 1/sigma^2 I)
        b in R^D        sampled uniformly from [0, 2*pi]

    CRITICAL -- Fixed Weights:
    W and b are registered as *buffers*, not nn.Parameters. This means:
    1. The optimizer will NOT update them (preserving the kernel approximation).
    2. They ARE included in state_dict (saved and loaded with the model).
    3. They move to the correct device when you call .to(device).
    """

    def __init__(self, input_dim, output_dim, sigma=1.0):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.sigma      = sigma

        w_init = torch.randn(input_dim, output_dim) / sigma
        b_init = torch.rand(output_dim) * 2.0 * np.pi

        self.register_buffer('W', w_init)
        self.register_buffer('b', b_init)

    def forward(self, x):
        proj     = torch.matmul(x, self.W) + self.b
        features = torch.cos(proj)
        features = features * np.sqrt(2.0 / self.output_dim)
        return features


class TCEAgent(nn.Module):
    """
    Temporal Contrastive Encoding Agent with Random Fourier Features.

    Architecture overview:
      1. encoder      : MLP  s_t (49D) -> z_t (latent_dim)         [TRAINABLE]
      2. rff          : RFF  z_t -> phi(z_t) (fourier_dim)         [FIXED BUFFERS]
      3. dynamics_net : MLP  [phi(z_t), a_t] -> phi(z_{t+1})_pred  [TRAINABLE]
      4. reward_net   : MLP  [z_t, a_t] -> r_t_pred                [TRAINABLE]
      5. policy_net   : MLP  z_t -> a_t                            [TRAINABLE]

    Input states are 49D = 39D proprioception + 10D one-hot task ID.
    The one-hot is pre-concatenated in TCEDataset so the encoder simply
    receives a 49D vector — no special handling needed here.

    At inference time only encoder + policy_net are used (forward()).
    """

    # =========================================================
    # ONE-HOT CHANGE 8: input_dim default updated 39 -> 49.
    # 39D proprioceptive state + 10D one-hot task encoding = 49D.
    # Without this the encoder's first Linear layer has wrong dimensions
    # and will either crash or silently ignore the task signal.
    # =========================================================
    def __init__(self, input_dim=49, action_dim=4, latent_dim=64, fourier_dim=256):
        super().__init__()
        self.input_dim   = input_dim
        self.action_dim  = action_dim
        self.latent_dim  = latent_dim
        self.fourier_dim = fourier_dim

        # 1. STATE ENCODER
        # Input is now 49D (39 proprioception + 10 one-hot task ID).
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # 2. RANDOM FOURIER PROJECTION (Fixed)
        self.rff = RandomFourierProjection(latent_dim, fourier_dim, sigma=1.0)

        # 3. DYNAMICS HEAD (Transition Model in Fourier space)
        self.dynamics_net = nn.Sequential(
            nn.Linear(fourier_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, fourier_dim),
        )

        # 4. REWARD HEAD
        self.reward_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 5. POLICY HEAD (Behavioural Cloning — inference only)
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state):
        """Inference path: (49D state with one-hot) -> action."""
        z = self.encoder(state)
        return self.policy_net(z)

    def get_fourier_features(self, state):
        """Encode state all the way to Fourier space."""
        z = self.encoder(state)
        return self.rff(z)

    def compute_dynamics(self, z, action):
        """
        Predict next Fourier features given latent z and action.

        Args:
            z      (Tensor): (N, latent_dim)
            action (Tensor): (N, action_dim)
        Returns:
            pred_phi_next (Tensor): (N, fourier_dim)
        """
        phi   = self.rff(z)
        phi_a = torch.cat([phi, action], dim=1)
        return self.dynamics_net(phi_a)