import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskConditionedStateVAE(nn.Module):
    def __init__(self, state_dim=39, latent_dim=16, num_tasks=10, hidden=(128, 64)):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.num_tasks = num_tasks

        enc_in = state_dim + num_tasks
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden[1], latent_dim)

        dec_in = latent_dim + num_tasks
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], state_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state, task_id):
        """
        state: (B, state_dim)
        task_id: (B,) long
        returns: recon_state, mu, logvar, z
        """
        task_oh = F.one_hot(task_id, num_classes=self.num_tasks).to(dtype=state.dtype)
        enc_x = torch.cat([state, task_oh], dim=-1)

        h = self.encoder(enc_x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        dec_x = torch.cat([z, task_oh], dim=-1)
        recon_state = self.decoder(dec_x)
        return recon_state, mu, logvar, z


class TaskConditionedVAEPolicy(nn.Module):
    def __init__(self, state_dim=39, action_dim=4, latent_dim=16, num_tasks=10, hidden_policy=(128, 128)):
        super().__init__()
        self.vae = TaskConditionedStateVAE(
            state_dim=state_dim,
            latent_dim=latent_dim,
            num_tasks=num_tasks,
            hidden=(128, 64),
        )

        pol_in = latent_dim + num_tasks
        self.policy_head = nn.Sequential(
            nn.Linear(pol_in, hidden_policy[0]),
            nn.ReLU(),
            nn.Linear(hidden_policy[0], hidden_policy[1]),
            nn.ReLU(),
            nn.Linear(hidden_policy[1], action_dim),
        )

        self.num_tasks = num_tasks

    def forward(self, state, task_id, use_mu_for_policy: bool = True):
        """
        If use_mu_for_policy=True, policy uses mu (deterministic, often more stable).
        Otherwise uses sampled z (stochastic).
        Returns:
          pred_actions (bounded to [-1,1]),
          recon_state, mu, logvar
        """
        recon_state, mu, logvar, z = self.vae(state, task_id)

        task_oh = F.one_hot(task_id, num_classes=self.num_tasks).to(dtype=state.dtype)
        lat = mu if use_mu_for_policy else z
        pol_x = torch.cat([lat, task_oh], dim=-1)

        action_logits = self.policy_head(pol_x)
        pred_actions = torch.tanh(action_logits)  # bound to [-1, 1]
        return pred_actions, recon_state, mu, logvar