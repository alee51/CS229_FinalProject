"""
TCE + CRTR + Context-Bypass + FiLM agent for Meta-World MT-10.

References:
  [1] Nachum & Yang 2021 – Provable Rep. Learning for Imitation (TCE / RFF)
  [2] arXiv 2508.13113  – Contrastive Representations for Temporal Reasoning (CRTR)
  [3] Perez et al. 2018 – FiLM: Visual Reasoning with a General Conditioning Layer
  [4] Benchmarking MT-RL for Robotics, RLJ/RLC 2025 – MT-10 obs layout §2.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# MT-10 STATE-SPACE SPLIT
# ============================================================================
# 49D input = [kinematic (36D) | goal xyz (3D) | task one-hot (10D)]
#
# CRTR's in-trajectory negatives erase any feature constant within an episode.
# In MT-10 that is: the 10D task one-hot and the 3D goal xyz — both frozen for
# the entire 500-step trajectory.  Erasing them from the encoder is correct
# behavior for CRTR, but catastrophic for the policy.
#
# Context-Bypass: encoder sees only the 36D kinematics; the 13D context
# (goal + task id) bypasses the encoder entirely and goes to the policy via
# FiLM layers (not simple concatenation — see FiLMPolicyNet below).
#
# Adjust KINEMATIC_DIM if your Metaworld version has a different obs layout.
# Constraint: KINEMATIC_DIM + CONTEXT_DIM == 49.
# ============================================================================
KINEMATIC_DIM = 36     # dynamic arm kinematics  (obs[:36])
GOAL_DIM      = 3      # goal xyz                (obs[36:39])
ONEHOT_DIM    = 10     # task one-hot            (obs[39:49])
CONTEXT_DIM   = GOAL_DIM + ONEHOT_DIM   # 13D


# ============================================================================
# FiLM POLICY NET
# ============================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (Perez et al., AAAI 2018).

        h_out = gamma(context) * h + beta(context)

    FiLM lets the context signal (goal xyz + task one-hot) modulate how
    kinematic features are processed at *every hidden layer*, not just at
    the input.  This gives the policy the representational capacity to solve
    goal-conditioned multi-stage tasks (pick-place, drawer-open, lever-pull)
    where the relationship between arm kinematics and the goal is non-linear
    and task-specific at every step of the trajectory.

    Simple concatenation [z, context] → MLP only injects context at the input;
    FiLM propagates it through the entire computation graph.

    Weights are initialised to the identity (γ=1, β=0) so training starts from
    a pure kinematic baseline and learns modulation incrementally.
    """

    def __init__(self, context_dim: int, feature_dim: int):
        super().__init__()
        self.gamma_net = nn.Linear(context_dim, feature_dim)
        self.beta_net  = nn.Linear(context_dim, feature_dim)
        # Identity init: start at gamma=1, beta=0 for stable early training
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.ones_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_net(context)   # (N, feature_dim)
        beta  = self.beta_net(context)    # (N, feature_dim)
        return gamma * h + beta


class FiLMPolicyNet(nn.Module):
    """
    Goal-conditioned policy using FiLM conditioning (Perez et al. 2018).

    Architecture:
        z (latent_dim) → Linear(128) → ReLU → FiLM(context)
                       → Linear(128) → ReLU → FiLM(context)
                       → Linear(action_dim) → Tanh

    The 128×128 hidden dimensions match the baseline policy for fair comparison.
    """

    def __init__(self, latent_dim: int, context_dim: int,
                 action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1   = nn.Linear(latent_dim, hidden_dim)
        self.film1 = FiLMLayer(context_dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.film2 = FiLMLayer(context_dim, hidden_dim)
        self.fc3   = nn.Linear(hidden_dim, action_dim)

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.film1(self.fc1(z), context))
        h = F.relu(self.film2(self.fc2(h), context))
        return torch.tanh(self.fc3(h))


# ============================================================================
# RANDOM FOURIER FEATURES
# ============================================================================

class RandomFourierProjection(nn.Module):
    """
    Random Fourier Features for Gaussian RBF kernel approximation.

        phi(z) = sqrt(2/D) * cos(W z + b)
        phi(x)^T phi(y)  ≈  exp(-||x-y||² / 2σ²)    [Bochner's theorem]

    W ~ N(0, 1/σ² I),  b ~ Uniform[0, 2π].  Registered as buffers (not
    parameters) so the optimizer never updates them [1, §2.2].

    DO NOT L2-normalise the output before computing the InfoNCE dot product.
    Normalisation maps vectors to the unit hypersphere, turning the dot product
    into cosine similarity and severing the link to the Gaussian kernel.
    The RBF approximation requires the raw Euclidean inner product [1, §3].
    """

    def __init__(self, input_dim: int, output_dim: int, sigma: float = 1.0):
        super().__init__()
        self.register_buffer('W', torch.randn(input_dim, output_dim) / sigma)
        self.register_buffer('b', torch.rand(output_dim) * 2.0 * np.pi)
        self._scale = np.sqrt(2.0 / output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(torch.matmul(x, self.W) + self.b) * self._scale


# ============================================================================
# TCE AGENT
# ============================================================================

class TCEAgent(nn.Module):
    """
    TCE + CRTR + Context-Bypass + FiLM agent for Meta-World MT-10.

    Data flow:
                     ┌──────────────────────────────────────────┐
                     │           49D augmented state             │
                     └──────┬──────────────────┬────────────────┘
                            │ kinematic (36D)  │ context (13D)
                            ▼                  │ goal xyz + one-hot
                       encoder f               │
                            │ z (64D)          │
                  ┌─────────┴──────────┐       │
                  │                    │       │
                  ▼                    ▼       ▼
              rff(z)=phi       reward_net    FiLMPolicyNet
              (contrastive)  [z,a,ctx]→r̂  [z,ctx]→action
                  │
                  ▼
            InfoNCE vs psi = rff(g_net(kinematic_next, action))

    Gradient routing:
      encoder  ← J_T (contrastive) + J_R (reward)  [NOT J_BC]
      g_net    ← J_T only
      policy   ← J_BC only  (receives z.detach() in trainer)
      reward   ← J_R only
    """

    def __init__(
        self,
        input_dim: int = 49,
        action_dim: int = 4,
        latent_dim: int = 64,
        fourier_dim: int = 256,
        kinematic_dim: int = KINEMATIC_DIM,
        context_dim: int = CONTEXT_DIM,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.input_dim     = input_dim
        self.action_dim    = action_dim
        self.latent_dim    = latent_dim
        self.fourier_dim   = fourier_dim
        self.kinematic_dim = kinematic_dim
        self.context_dim   = context_dim

        assert kinematic_dim + context_dim == input_dim, (
            f"kinematic_dim ({kinematic_dim}) + context_dim ({context_dim}) "
            f"must equal input_dim ({input_dim}).  "
            f"Adjust KINEMATIC_DIM / CONTEXT_DIM if your obs layout differs."
        )

        # 1. Kinematic encoder  f: R^36 → R^latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(kinematic_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),           nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, latent_dim),    nn.LayerNorm(latent_dim),
        )

        # 2. Shared Random Fourier Projection (fixed buffers)
        self.rff = RandomFourierProjection(latent_dim, fourier_dim, sigma=sigma)

        # 3. g-network  g: [R^36, R^4] → R^latent_dim  (separate from encoder)
        self.g_net = nn.Sequential(
            nn.Linear(kinematic_dim + action_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),                        nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, latent_dim),                 nn.LayerNorm(latent_dim),
        )

        # 4. Reward head  [z, action, context] → scalar
        self.reward_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # 5. FiLM policy  z → (FiLM-conditioned on context) → action
        self.policy_net = FiLMPolicyNet(latent_dim, context_dim, action_dim)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def split_state(self, state: torch.Tensor):
        """Split 49D state into (kinematic 36D, context 13D)."""
        return state[:, :self.kinematic_dim], state[:, self.kinematic_dim:]

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """f(kinematic(state)) → z."""
        kinematic, _ = self.split_state(state)
        return self.encoder(kinematic)

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """(49D state) → action in [-1, 1].  Used at evaluation time."""
        kinematic, context = self.split_state(state)
        z = self.encoder(kinematic)
        return self.policy_net(z, context)

    # -------------------------------------------------------------------------
    # Contrastive helpers
    # -------------------------------------------------------------------------

    def get_fourier_features(self, state: torch.Tensor) -> torch.Tensor:
        """phi(s) = RFF(f(kinematic(s))) — anchor for InfoNCE."""
        return self.rff(self.encode(state))

    def compute_keys(self, next_state: torch.Tensor,
                     action: torch.Tensor) -> torch.Tensor:
        """psi(s', a) = RFF(g(kinematic(s'), a)) — key for InfoNCE."""
        kinematic_next, _ = self.split_state(next_state)
        return self.rff(self.g_net(torch.cat([kinematic_next, action], dim=1)))