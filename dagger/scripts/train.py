"""
DAgger approach — stub.
Use core for task list and env/eval when implemented.
"""
import os
import sys

# Project root for core
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DAGGER_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_DAGGER_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Stub: minimal policy compatible with baseline interface for future DAgger implementation
import torch
import torch.nn as nn


class ClonePolicy(nn.Module):
    """Stub policy (same interface as baseline). Replace with DAgger policy when implemented."""
    def __init__(self, input_dim, output_dim, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_train_config(config_path=None):
    """Load config. Returns dict (empty for stub)."""
    return {}


def train_model(**kwargs):
    """Stub: DAgger not implemented yet. Saves a placeholder to dagger/models/latest_policy.pth."""
    print("DAgger is not implemented yet.")
    model_dir = os.path.join(_DAGGER_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    save_name = kwargs.get("save_name", "latest_policy.pth")
    save_path = os.path.join(model_dir, save_name)
    policy = ClonePolicy(39, 4, hidden_sizes=[64, 64])
    torch.save(policy.state_dict(), save_path)
    print(f"Placeholder stub saved to {save_path} (implement DAgger to train).")
