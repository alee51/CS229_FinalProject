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

    Args:
        input_dim  (int):   Dimension d of the input latent vector z.
        output_dim (int):   Dimension D of the projected Fourier feature space.
        sigma      (float): Bandwidth of the RBF kernel. Controls the
                            length-scale of the approximated kernel. sigma=1.0
                            is a sensible default; tune if dynamics are very
                            fast (smaller sigma) or very slow (larger sigma).
    """

    def __init__(self, input_dim, output_dim, sigma=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma

        # W shape: (input_dim, output_dim) so that z @ W gives (batch, output_dim).
        # Sampling from N(0, 1/sigma^2) means std = 1/sigma.
        w_init = torch.randn(input_dim, output_dim) / sigma

        # b shape: (output_dim,), sampled uniformly from [0, 2*pi].
        b_init = torch.rand(output_dim) * 2.0 * np.pi

        # register_buffer: not a parameter (optimizer ignores it),
        # but moves with .to(device) and is saved in state_dict.
        self.register_buffer('W', w_init)
        self.register_buffer('b', b_init)

    def forward(self, x):
        # Linear projection: (batch, input_dim) @ (input_dim, output_dim) -> (batch, output_dim)
        proj = torch.matmul(x, self.W) + self.b            # Wz + b
        features = torch.cos(proj)                         # cos(Wz + b)
        features = features * np.sqrt(2.0 / self.output_dim)  # sqrt(2/D) normalisation
        return features


class TCEAgent(nn.Module):
    """
    Temporal Contrastive Encoding Agent with Random Fourier Features.

    Architecture overview:
      1. encoder      : MLP  s_t (39D) -> z_t (latent_dim)         [TRAINABLE]
      2. rff          : RFF  z_t -> phi(z_t) (fourier_dim)         [FIXED BUFFERS]
      3. dynamics_net : MLP  [phi(z_t), a_t] -> phi(z_{t+1})_pred  [TRAINABLE]
      4. reward_net   : MLP  [z_t, a_t] -> r_t_pred                [TRAINABLE]
      5. policy_net   : MLP  z_t -> a_t                            [TRAINABLE]

    The contrastive loss operates in Fourier space (dynamics_net output vs
    rff(z_{t+1}_true)), satisfying the linear-dynamics-in-kernel-space requirement
    of Nachum & Yang (Theorem 3).

    At inference time only encoder + policy_net are used (forward()).

    NOTE: At inference time, only encoder + policy_net are used (forward()).
    The RFF projection (rff) and dynamics_net are only active during training.
    """

    def __init__(self, input_dim=39, action_dim=4, latent_dim=64, fourier_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.fourier_dim = fourier_dim

        # -----------------------------------------------------------
        # 1. STATE ENCODER (Phi)
        # -----------------------------------------------------------
        # Maps raw 39D state -> compact latent z (latent_dim).
        # LayerNorm after each hidden layer stabilises training and keeps
        # z on a well-conditioned manifold for the RFF projection.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim)   # normalised output aids kernel approx.
        )

        # -----------------------------------------------------------
        # 2. RANDOM FOURIER PROJECTION (Fixed)
        # -----------------------------------------------------------
        # Maps z -> phi(z) in the kernel feature space.
        # Weights are frozen buffers -- NOT updated by the optimizer.
        self.rff = RandomFourierProjection(latent_dim, fourier_dim, sigma=1.0)

        # -----------------------------------------------------------
        # 3. DYNAMICS HEAD (Transition Model in Fourier space)
        # -----------------------------------------------------------
        # Input:  [phi(z_t), a_t]  (fourier_dim + action_dim)
        # Output: predicted phi(z_{t+1})  (fourier_dim)
        #
        # Theory: operating in Fourier space approximates the linear
        # dynamics requirement T(phi(z_t), a_t) approx phi(z_{t+1}) from
        # Nachum & Yang section 5.2. A shallow MLP provides stability over
        # a strict linear layer at the cost of a small theory relaxation.
        self.dynamics_net = nn.Sequential(
            nn.Linear(fourier_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, fourier_dim)
        )

        # -----------------------------------------------------------
        # 4. REWARD HEAD
        # -----------------------------------------------------------
        # Input:  [z_t, a_t]  (latent_dim + action_dim)
        # Output: predicted scalar reward r_t
        #
        # Reward prediction is necessary to bound J_R in Theorem 2.
        # Operates on raw latent z (not Fourier features) because reward
        # is a direct function of state, not kernel similarity.
        self.reward_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # -----------------------------------------------------------
        # 5. POLICY HEAD (Behavioural Cloning)
        # -----------------------------------------------------------
        # Input:  z_t  (latent_dim)
        # Output: action a_t  (action_dim)
        #
        # Intentionally kept identical in size to the baseline BC MLP
        # so that any performance gain is attributable to the richer
        # representation z, not a larger policy network.
        # This is the ONLY module used at test/deployment time.
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        """
        Standard inference path: state -> action.
        This is the only path executed at evaluation/deployment time.
        """
        z = self.encoder(state)
        return self.policy_net(z)

    def get_fourier_features(self, state):
        """Convenience: encode a state all the way to Fourier space."""
        z = self.encoder(state)
        return self.rff(z)

    def compute_dynamics(self, z, action):
        """
        Predict the next state's Fourier features given current latent z and action.

        Args:
            z      (Tensor): shape (N, latent_dim)  -- output of encoder
            action (Tensor): shape (N, action_dim)

        Returns:
            pred_phi_next (Tensor): shape (N, fourier_dim)
                Predicted phi(z_{t+1}), used as the query in the contrastive loss.
        """
        phi = self.rff(z)                              # z -> phi(z)
        phi_a = torch.cat([phi, action], dim=1)        # [phi(z), a]
        return self.dynamics_net(phi_a)                # -> phi(z_{t+1}) predicted