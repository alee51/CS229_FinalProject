"""
TCE Trainer with sequential two-phase training.

Phase 1 (Representation):  Train encoder + g_net + reward_net with J_T + J_R.
                           Policy is locked. alpha=100 (theory) is safe here.
Phase 2 (Policy):          Freeze encoder. Train FiLM policy with J_BC only.
                           Stationary z eliminates the Epoch-16 val divergence.

Why sequential?  In joint training, the encoder continuously updates from J_T
and J_R.  The policy (which receives z.detach()) overfits to the current
training-set representations.  At validation time, z has drifted into regions
the policy has never seen, causing val/loss_bc to explode.  Freezing the
encoder in Phase 2 gives the policy a stationary target — the U-shaped curve
cannot appear when z does not move.

Joint training is still available via trainer.train_joint(epochs) for ablation.
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from contrastive_data import CRTRBatchSampler


class TCETrainer:

    def __init__(self, agent, train_dataset, val_dataset, config):
        self.agent  = agent
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.to(self.device)

        # CRTRBatchSampler for Phase 1 (contrastive needs in-trajectory negatives)
        train_sampler = CRTRBatchSampler(
            train_dataset.traj_ids,
            batch_size=config['batch_size'],
            repetition_factor=config.get('repetition_factor', 4),
        )
        self.train_loader_crtr = DataLoader(
            train_dataset, batch_sampler=train_sampler, num_workers=0
        )
        # Standard shuffle loader for Phase 2 (BC just needs diverse transitions)
        self.train_loader_bc = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
        )

        os.makedirs('models', exist_ok=True)
        task = config.get('task_name', 'task')
        self.ckpt_repr   = f"models/tce_{task}_repr.pth"   # best Phase-1 checkpoint
        self.ckpt_policy = f"models/tce_{task}_policy.pth" # best Phase-2 checkpoint
        self.ckpt_joint  = f"models/tce_{task}_joint.pth"  # joint training checkpoint

        wandb.init(
            project="tce-metaworld",
            config=config,
            name=(f"{task}_p1-{config.get('pretrain_epochs',0)}"
                  f"_p2-{config.get('finetune_epochs',0)}"),
            reinit=True,
        )

    # =========================================================================
    # LOSS HELPERS
    # =========================================================================

    def _contrastive_loss(self, phi, psi):
        """
        InfoNCE between anchor phi(s_t) and key psi(s_{t+1}, a_t).

        Raw dot product (NO L2 normalisation) preserves the RFF kernel
        approximation: phi(x)^T phi(y) ≈ exp(-||x-y||² / 2σ²) [TCE §3].
        Normalising would reduce this to cosine similarity and void TCE Theorem 3.

        The RFF scale factor sqrt(2/D) keeps raw dot products near [-2, 2].
        Temperature τ=0.5 (vs the previous 0.1) softens the unnormalised
        logit distribution and reduces softmax saturation risk.
        """
        tau    = self.config.get('temperature', 0.5)
        logits = torch.matmul(phi, psi.T) / tau
        labels = torch.arange(logits.size(0), device=self.device)
        return F.cross_entropy(logits, labels)

    def _repr_losses(self, batch):
        """Compute J_T and J_R (no BC).  Used in Phase 1."""
        state      = batch['state'].to(self.device)
        action     = batch['action'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        reward     = batch['reward'].to(self.device)

        _, context = self.agent.split_state(state)

        z   = self.agent.encode(state)
        phi = self.agent.rff(z)
        psi = self.agent.compute_keys(next_state, action)

        loss_t = self._contrastive_loss(phi, psi)
        loss_r = F.mse_loss(
            self.agent.reward_net(torch.cat([z, action, context], dim=1)), reward
        )
        return loss_t, loss_r

    def _bc_loss(self, batch, freeze_encoder=True):
        """Compute J_BC.  Used in Phase 2 (and joint training)."""
        state  = batch['state'].to(self.device)
        action = batch['action'].to(self.device)

        _, context = self.agent.split_state(state)

        if freeze_encoder:
            # Encoder is frozen externally; still use no_grad for safety
            with torch.no_grad():
                z = self.agent.encode(state)
        else:
            # Joint training: z.detach() blocks BC gradient from entering encoder
            z = self.agent.encode(state).detach()

        pred = self.agent.policy_net(z, context)
        return F.mse_loss(pred, action)

    # =========================================================================
    # PHASE 1 — REPRESENTATION PRETRAINING
    # =========================================================================

    def train_phase1(self, epochs: int):
        """
        Train encoder + g_net + reward_net using J_T + J_R.

        Policy is not updated here.  With no BC loss to destabilise, alpha
        can safely be set to the theoretical value 1/(1-γ) ≈ 100 [TCE Thm 2],
        giving the encoder the full contrastive signal it needs.

        Early stopping on val/loss_contrastive.
        """
        gamma = self.config.get('gamma', 0.99)
        alpha = 1.0 / (1.0 - gamma)          # ≈ 100 for γ=0.99
        beta  = self.config.get('beta', 1.0)

        print(f"\n{'='*60}")
        print(f"  Phase 1: Representation Pretraining  ({epochs} epochs)")
        print(f"  alpha={alpha:.0f}  beta={beta}  (theory alpha, safe without BC)")
        print(f"{'='*60}")

        repr_params = (
            list(self.agent.encoder.parameters())
            + list(self.agent.g_net.parameters())
            + list(self.agent.reward_net.parameters())
        )
        optimizer = optim.Adam(
            repr_params,
            lr=self.config['lr'],
            weight_decay=self.config.get('weight_decay', 1e-4),
        )

        best_val_t = float('inf')

        for epoch in range(epochs):
            self.agent.train()
            tr_t = tr_r = 0.0

            for batch in self.train_loader_crtr:
                optimizer.zero_grad()
                loss_t, loss_r = self._repr_losses(batch)
                loss = alpha * loss_t + beta * loss_r
                loss.backward()
                torch.nn.utils.clip_grad_norm_(repr_params, max_norm=1.0)
                optimizer.step()
                tr_t += loss_t.item()
                tr_r += loss_r.item()

            n = len(self.train_loader_crtr)

            self.agent.eval()
            vl_t = vl_r = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    lt, lr = self._repr_losses(batch)
                    vl_t += lt.item()
                    vl_r += lr.item()
            m = len(self.val_loader)

            improved = (vl_t / m) < best_val_t
            if improved:
                best_val_t = vl_t / m
                torch.save(self.agent.state_dict(), self.ckpt_repr)

            print(
                f"P1 {epoch+1:03d}/{epochs} | "
                f"Ctr train={tr_t/n:.4f} val={vl_t/m:.4f} | "
                f"Rew train={tr_r/n:.4f}"
                + (" <- best" if improved else "")
            )
            wandb.log({
                "phase": 1, "epoch": epoch + 1,
                "p1/train_contrastive": tr_t / n,
                "p1/train_reward":      tr_r / n,
                "p1/val_contrastive":   vl_t / m,
                "p1/val_reward":        vl_r / m,
            })

        # Load best repr checkpoint for Phase 2
        self.agent.load_state_dict(
            torch.load(self.ckpt_repr, map_location=self.device, weights_only=True)
        )
        print(f"\nPhase 1 done. Best val/contrastive = {best_val_t:.4f}")
        print(f"Best repr checkpoint: {self.ckpt_repr}")

    # =========================================================================
    # PHASE 2 — POLICY FINETUNING  (encoder frozen)
    # =========================================================================

    def train_phase2(self, epochs: int):
        """
        Freeze encoder + g_net + reward_net.  Train only the FiLM policy.

        Because the encoder is frozen, z does not drift.  The policy trains
        against a stationary latent manifold — the Epoch-16 val divergence
        cannot occur.  A fresh standard DataLoader is used (CRTR sampling is
        irrelevant for BC).

        Early stopping on val/loss_bc.
        """
        print(f"\n{'='*60}")
        print(f"  Phase 2: Policy Finetuning  ({epochs} epochs, encoder frozen)")
        print(f"{'='*60}")

        # Freeze representation networks
        for m in (self.agent.encoder, self.agent.g_net, self.agent.reward_net,
                  self.agent.rff):
            for p in m.parameters():
                p.requires_grad_(False)

        optimizer = optim.Adam(
            self.agent.policy_net.parameters(),
            lr=self.config.get('lr_phase2', self.config['lr']),
            weight_decay=self.config.get('weight_decay', 1e-4),
        )

        best_val_bc = float('inf')

        for epoch in range(epochs):
            self.agent.train()
            tr_bc = 0.0

            for batch in self.train_loader_bc:
                optimizer.zero_grad()
                loss_bc = self._bc_loss(batch, freeze_encoder=True)
                loss_bc.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.policy_net.parameters(), max_norm=1.0
                )
                optimizer.step()
                tr_bc += loss_bc.item()

            n = len(self.train_loader_bc)

            self.agent.eval()
            vl_bc = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    vl_bc += self._bc_loss(batch, freeze_encoder=True).item()
            m = len(self.val_loader)

            improved = (vl_bc / m) < best_val_bc
            if improved:
                best_val_bc = vl_bc / m
                torch.save(self.agent.state_dict(), self.ckpt_policy)

            print(
                f"P2 {epoch+1:03d}/{epochs} | "
                f"BC  train={tr_bc/n:.4f}  val={vl_bc/m:.4f}"
                + (" <- best" if improved else "")
            )
            wandb.log({
                "phase": 2, "epoch": epoch + 1,
                "p2/train_bc": tr_bc / n,
                "p2/val_bc":   vl_bc / m,
            })

        # Unfreeze (so model can be inspected / retrained later if needed)
        for m in (self.agent.encoder, self.agent.g_net, self.agent.reward_net,
                  self.agent.rff):
            for p in m.parameters():
                p.requires_grad_(True)

        print(f"\nPhase 2 done. Best val/loss_bc = {best_val_bc:.4f}")
        print(f"Best policy checkpoint: {self.ckpt_policy}")
        wandb.finish()

    # =========================================================================
    # JOINT TRAINING  (kept for ablation comparison)
    # =========================================================================

    def train_joint(self, epochs: int):
        """
        Joint training baseline.  All losses computed simultaneously.

        z.detach() blocks BC gradients from the encoder, but representation
        drift (encoder updating while policy trains) still causes val/loss_bc
        to diverge around epoch 16.  Kept for ablation — prefer sequential.
        """
        alpha = self.config.get('alpha', 1.0)
        beta  = self.config.get('beta',  1.0)

        print(f"\n{'='*60}")
        print(f"  Joint Training  ({epochs} epochs, alpha={alpha}, beta={beta})")
        print(f"  NOTE: sequential training is recommended over joint.")
        print(f"{'='*60}")

        optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config.get('weight_decay', 1e-4),
        )
        best_val_bc = float('inf')

        for epoch in range(epochs):
            self.agent.train()
            tr_t = tr_r = tr_bc = 0.0

            for batch in self.train_loader_crtr:
                optimizer.zero_grad()
                loss_t, loss_r = self._repr_losses(batch)
                loss_bc        = self._bc_loss(batch, freeze_encoder=False)
                loss = loss_bc + alpha * loss_t + beta * loss_r
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), max_norm=1.0
                )
                optimizer.step()
                tr_t  += loss_t.item()
                tr_r  += loss_r.item()
                tr_bc += loss_bc.item()

            n = len(self.train_loader_crtr)

            self.agent.eval()
            vl_t = vl_r = vl_bc = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    lt, lr = self._repr_losses(batch)
                    lb     = self._bc_loss(batch, freeze_encoder=False)
                    vl_t  += lt.item()
                    vl_r  += lr.item()
                    vl_bc += lb.item()
            m = len(self.val_loader)

            improved = (vl_bc / m) < best_val_bc
            if improved:
                best_val_bc = vl_bc / m
                torch.save(self.agent.state_dict(), self.ckpt_joint)

            print(
                f"Joint {epoch+1:03d}/{epochs} | "
                f"BC train={tr_bc/n:.4f} val={vl_bc/m:.4f} | "
                f"Ctr={tr_t/n:.4f}"
                + (" <- best" if improved else "")
            )
            wandb.log({
                "phase": 0, "epoch": epoch + 1,
                "joint/train_bc":          tr_bc / n,
                "joint/train_contrastive": tr_t  / n,
                "joint/train_reward":      tr_r  / n,
                "joint/val_bc":            vl_bc / m,
            })

        print(f"\nJoint done. Best val/loss_bc = {best_val_bc:.4f}")
        wandb.finish()