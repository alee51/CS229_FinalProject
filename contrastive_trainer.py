import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# ============================================================
# CHANGE 3: Added wandb import
# ============================================================
import wandb

from contrastive_data import CRTRBatchSampler


class TCETrainer:
    """
    Trainer for the TCE + CRTR + RFF pipeline.

    Responsibilities:
      - Builds DataLoaders (using CRTRBatchSampler for training).
      - Defines the composite loss: contrastive (InfoNCE in Fourier space)
        + reward prediction (J_R) + behavioural cloning (J_BC).
      - Runs the train / validation loops.
      - Logs all metrics to Weights & Biases.

    The training logic lives here; train_model.py is responsible only for
    loading data, constructing datasets, and calling trainer.train().
    """

    def __init__(self, agent, train_dataset, val_dataset, config):
        self.agent  = agent
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.to(self.device)

        # ------------------------------------------------------------------
        # Optimizer — only trainable parameters (RFF buffers W, b are excluded
        # automatically because they are registered as buffers, not parameters)
        # ------------------------------------------------------------------
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config['lr'])

        # ------------------------------------------------------------------
        # Training DataLoader — CRTRBatchSampler enforces within-trajectory
        # negatives. Every batch is guaranteed to contain `repetition_factor`
        # samples from each included trajectory, giving the InfoNCE loss the
        # hard negatives CRTR requires.
        # ------------------------------------------------------------------
        train_sampler = CRTRBatchSampler(
            train_dataset.traj_ids,
            batch_size=config['batch_size'],
            repetition_factor=config.get('repetition_factor', 4),
        )
        # num_workers=0 required when TCEDataset pre-loads tensors onto CUDA
        self.train_loader = DataLoader(
            train_dataset, batch_sampler=train_sampler, num_workers=0
        )

        # Validation uses standard batching — CRTR sampling not needed here
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
        )

        # ============================================================
        # CHANGE 4: Initialise wandb run
        # project="tce-metaworld" groups all your runs together on wandb.
        # config= logs all hyperparameters so you can filter/compare runs.
        # name= gives the run a readable label in the wandb dashboard.
        # ============================================================
        wandb.init(
            project="tce-metaworld",
            config=config,
            name=f"{config.get('task_name', 'task')}_bs{config['batch_size']}_rep{config.get('repetition_factor', 4)}",
        )
        # Track gradients and parameter histograms every 50 steps — useful
        # for spotting exploding/vanishing gradients in the encoder.
        wandb.watch(self.agent, log="gradients", log_freq=50)

    # ----------------------------------------------------------------------
    # Loss helpers
    # ----------------------------------------------------------------------

    def _compute_contrastive_loss(self, pred_phi, target_phi):
        """
        InfoNCE loss operating in Fourier feature space.

        Positive pairs are the diagonal (sample i predicted vs sample i true).
        Off-diagonal entries are negatives. Because CRTRBatchSampler has
        placed multiple samples from the same trajectory in the batch, some
        off-diagonal entries are within-trajectory hard negatives (same
        context, different time), satisfying the CRTR requirement without
        any extra masking logic.
        """
        temperature = self.config.get('temperature', 0.1)

        pred_norm   = F.normalize(pred_phi,   dim=1)
        target_norm = F.normalize(target_phi, dim=1)

        logits = torch.matmul(pred_norm, target_norm.T) / temperature
        labels = torch.arange(logits.size(0), device=self.device)
        return F.cross_entropy(logits, labels)

    def _compute_batch_losses(self, batch):
        """
        Run one forward pass and return all three component losses.

        Returns:
            loss_t  : contrastive loss in Fourier space   (J_T)
            loss_r  : reward prediction MSE               (J_R)
            loss_bc : behavioural cloning MSE             (J_BC)
        """
        state      = batch['state'].to(self.device)
        action     = batch['action'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        reward     = batch['reward'].to(self.device)

        # A. Encode current state
        z = self.agent.encoder(state)

        # B. Predict phi(z_{t+1}) via dynamics head in Fourier space
        pred_phi_next = self.agent.compute_dynamics(z, action)

        # C. Ground-truth phi(z_{t+1}) — no gradients through target
        with torch.no_grad():
            z_next_true     = self.agent.encoder(next_state)
            target_phi_next = self.agent.rff(z_next_true)

        # D. Losses
        loss_t  = self._compute_contrastive_loss(pred_phi_next, target_phi_next)
        loss_r  = F.mse_loss(
            self.agent.reward_net(torch.cat([z, action], dim=1)), reward
        )
        loss_bc = F.mse_loss(self.agent.policy_net(z), action)

        return loss_t, loss_r, loss_bc

    # ----------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------

    def train(self, epochs):
        alpha = self.config.get('alpha', 1.0)
        beta  = self.config.get('beta',  1.0)

        for epoch in range(epochs):
            # ------ Training ------
            self.agent.train()
            total_loss = total_loss_t = total_loss_r = total_loss_bc = 0.0

            for batch in self.train_loader:
                self.optimizer.zero_grad()

                loss_t, loss_r, loss_bc = self._compute_batch_losses(batch)
                loss = loss_bc + (alpha * loss_t) + (beta * loss_r)

                loss.backward()
                # ============================================================
                # CHANGE 5: Gradient clipping
                # Prevents large InfoNCE gradient spikes early in training
                # from pushing the encoder weights into a bad region.
                # ============================================================
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss    += loss.item()
                total_loss_t  += loss_t.item()
                total_loss_r  += loss_r.item()
                total_loss_bc += loss_bc.item()

            n = len(self.train_loader)
            avg_train    = total_loss    / n
            avg_train_t  = total_loss_t  / n
            avg_train_r  = total_loss_r  / n
            avg_train_bc = total_loss_bc / n

            # ------ Validation ------
            self.agent.eval()
            val_loss = val_loss_t = val_loss_r = val_loss_bc = 0.0

            with torch.no_grad():
                for batch in self.val_loader:
                    loss_t, loss_r, loss_bc = self._compute_batch_losses(batch)
                    val_loss    += (loss_bc + (alpha * loss_t) + (beta * loss_r)).item()
                    val_loss_t  += loss_t.item()
                    val_loss_r  += loss_r.item()
                    val_loss_bc += loss_bc.item()

            m = len(self.val_loader)
            avg_val    = val_loss    / m
            avg_val_t  = val_loss_t  / m
            avg_val_r  = val_loss_r  / m
            avg_val_bc = val_loss_bc / m

            print(
                f"Epoch {epoch + 1:03d}/{epochs} | "
                f"Train: {avg_train:.4f} | "
                f"Val:   {avg_val:.4f}"
            )

            # ============================================================
            # CHANGE 6: Log all metrics to wandb each epoch
            # Breaking out each loss component lets you see in wandb whether
            # e.g. the contrastive loss is converging while BC stalls, etc.
            # ============================================================
            wandb.log({
                "epoch":                epoch + 1,
                "train/loss_total":     avg_train,
                "train/loss_contrastive": avg_train_t,
                "train/loss_reward":    avg_train_r,
                "train/loss_bc":        avg_train_bc,
                "val/loss_total":       avg_val,
                "val/loss_contrastive": avg_val_t,
                "val/loss_reward":      avg_val_r,
                "val/loss_bc":          avg_val_bc,
            })

        # ============================================================
        # CHANGE 7: Close the wandb run cleanly when training finishes
        # ============================================================
        wandb.finish()