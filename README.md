# TCE + CRTR + Random Fourier Features — CS 229 Project

**Branch:** `jonathan/contrastive-encoder`  
**Author:** Jonathan Lu  
**Baseline:** Nancy's multi-task BC branch (`main`)

---

## What this branch does

This branch upgrades the behavioural cloning (BC) baseline with a structured
representation learning objective. The goal is to learn a latent state
representation that is more robust to distribution shift and task interference
than plain BC — evaluated on the same MT-10 benchmark as the baseline.

Three methods from the literature are combined:

**TCE — Temporal Contrastive Encoding** (Nachum & Yang)  
Adds a contrastive loss that trains the encoder to predict future states in
latent space, rather than just mapping states to actions. This forces the
encoder to capture task-relevant dynamics rather than memorising the training
distribution.

**CRTR — Contrastive Representations for Temporal Reasoning** (Ziarko et al.)  
Standard contrastive learning can be gamed by the encoder learning to
distinguish trajectory *contexts* (table colour, lighting) rather than
*dynamics*. CRTR fixes this by guaranteeing that every training batch contains
multiple samples from the same trajectory — forcing the encoder to tell apart
states that share the same visual context but differ in temporal position.
Implemented via a custom `CRTRBatchSampler`.

**Random Fourier Features** (Bochner's Theorem)  
Nachum & Yang's performance bound requires that latent dynamics be
approximable by a linear model. A standard MLP dynamics head does not satisfy
this. RFF projects the latent state into a higher-dimensional space that
approximates an RBF kernel, where the linear dynamics assumption holds. The
projection weights are **fixed** (not trained) — this is critical for
preserving the kernel property throughout training.

---

## Architecture

```
s_t (39D)
   │
   ▼
encoder φ          3-layer MLP, LayerNorm     [TRAINABLE]  → z_t (64D)
   │
   ├──────────────────────────────────────────► policy_net  → action (4D)  [inference only]
   │
   ▼
rff                cos(Wz + b), fixed weights [FROZEN]     → φ(z_t) (256D)
   │
   ├── [φ(z_t), a_t] → dynamics_net → φ(z_{t+1})_pred     [TRAINABLE]
   │
   └── [z_t,   a_t] → reward_net   → r_t_pred             [TRAINABLE]
```

**Loss function:**

```
J_total = J_BC  +  α · J_T  +  β · J_R

J_T  = InfoNCE( φ(z_{t+1})_pred , φ(z_{t+1})_true )   ← contrastive in Fourier space
J_R  = MSE( r_pred , r_true )                          ← reward prediction
J_BC = MSE( a_pred , a_expert )                        ← behavioural cloning

α = 1 / (1 - γ)   per Nachum & Yang Theorem 2  (≈ 100 for γ = 0.99)
β = 1.0
```

---

## Files

| File | Purpose |
|---|---|
| `collect_data.py` | Collect expert demonstrations using Metaworld scripted policies |
| `merge_data.py` | Combine per-task `.npz` files into a single MT-10 dataset |
| `contrastive_data.py` | `TCEDataset` + `CRTRBatchSampler` (enforces within-trajectory negatives) |
| `contrastive_model.py` | `RandomFourierProjection` + `TCEAgent` architecture |
| `contrastive_trainer.py` | Training loop with composite loss, gradient clipping, W&B logging |
| `train_model.py` | Entry point: data loading, trajectory-based train/val split, model init |
| `evaluate.py` | Per-task and full MT-10 evaluation with W&B logging |

---

## Setup

```bash
pip install torch metaworld wandb
wandb login    # paste API key from wandb.ai/authorize
```

---

## Reproducing results

### Step 1 — Collect expert data

```bash
python collect_data.py --task reach-v3                --episodes 100
python collect_data.py --task push-v3                 --episodes 100
python collect_data.py --task pick-place-v3           --episodes 100
python collect_data.py --task door-open-v3            --episodes 100
python collect_data.py --task door-close-v3           --episodes 100
python collect_data.py --task drawer-open-v3          --episodes 100
python collect_data.py --task drawer-close-v3         --episodes 100
python collect_data.py --task button-press-topdown-v3 --episodes 100
python collect_data.py --task lever-pull-v3           --episodes 100
python collect_data.py --task window-open-v3          --episodes 100
```

### Step 2 — Merge into one MT-10 dataset

```bash
python merge_data.py
```

### Step 3 — Train

```bash
python train_model.py --task mt10 --epochs 100 --batch_size 512
```

Training logs to W&B automatically. Key hyperparameters:

| Argument | Default | Notes |
|---|---|---|
| `--epochs` | 50 | 100 recommended for MT-10 |
| `--batch_size` | 256 | 512 for MT-10 |
| `--latent_dim` | 64 | Encoder output dimension |
| `--fourier_dim` | 256 | RFF projection dimension |
| `--repetition_factor` | 4 | CRTR: samples per trajectory per batch |
| `--temperature` | 0.1 | InfoNCE temperature |
| `--gamma` | 0.99 | Discount factor; sets α = 1/(1-γ) automatically |

### Step 4 — Evaluate

```bash
# Full MT-10 sweep with W&B logging (matches teammate's eval format)
python evaluate.py --episodes 50 --wandb

# Single task quick check (no W&B)
python evaluate.py --task reach-v3 --episodes 20
```

---

## Baseline comparison

The baseline branch trains a standard 2-layer BC MLP directly on (state, action)
pairs with no representation learning. MT-10 results from the baseline:

| Task | Baseline BC |
|---|---|
| reach-v3 | 22.0% |
| push-v3 | 28.0% |
| pick-place-v3 | 18.0% |
| door-open-v3 | 72.0% |
| door-close-v3 | 66.0% |
| drawer-open-v3 | 100.0% |
| drawer-close-v3 | 100.0% |
| button-press-topdown-v3 | 92.0% |
| lever-pull-v3 | 0.0% |
| window-open-v3 | 100.0% |
| **Mean** | **59.8%** |

The baseline shows strong task interference — drawer/window tasks hit 100% while
reach, push, pick-place, and lever-pull are near-zero. The hypothesis is that
CRTR's context-removal signal will reduce this variance by learning dynamics-
based rather than context-based representations.

---

## References

1. Nachum, O. & Yang, M. — *Provable Representation Learning for Imitation with Contrastive Fourier Features* (TCE + RFF)
2. Ziarko et al. — *Contrastive Representations for Temporal Reasoning* (CRTR)
3. PyTorch `torch.load` security: `weights_only=True` flag (PyTorch ≥ 1.13)
