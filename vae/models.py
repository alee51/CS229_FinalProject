import torch
import torch.nn as nn
import torch.nn.functional as F

class StateVAE(nn.Module):
    def __init__(self, input_dim=39, latent_dim=16): # 39D baseline [cite: 18]
        super(StateVAE, self).__init__()
        
        # Encoder: Maps raw state to latent parameters [cite: 9, 20]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder: Enforces the bottleneck structure [cite: 20]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class VAEPolicy(nn.Module):
    def __init__(self, input_dim=39, action_dim=4, latent_dim=16):
        super(VAEPolicy, self).__init__()
        self.vae = StateVAE(input_dim, latent_dim)
        # Fixed policy architecture to vary only the encoder [cite: 17]
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        recon_state, mu, logvar = self.vae(state)
        action_pred = self.policy_head(mu) 
        return action_pred, recon_state, mu, logvar