TRAINING_NOTES.md

### Current Status
Trying to improve baseline policy; standard 3 layer, no tail upsampling.
- 2000 epochs 256 256 128 got 52%.
- 64 64 with 2000 epochs got 98%!!! 

###### How to further improve
- tune this model. 

#### Longer-term future steps

### Past Attempts
<details>
    <summary> tail upsampling fail </summary>
I tried upsampling the tail, but even though training loss got super low, it didn't reflect in the success rate. I think what happened is that even though the model gets super close, the tail chunk I'm taking is too big, so I upsample both times the arm is close to the goal; when the arm is moving fast-ish and super slow towards the goal. Even with very good training, the success rate only got up to 54%. 
</details>
<details>
  <summary>Successful 98% baseline for MT-1</summary>

Here’s a full spec you can drop into your notes for **no_end_baseline_64x64.pth**:

---

## no_end_baseline_64x64.pth – full training config

**Command (what you ran):**
```bash
python train.py --name no_end_baseline_64x64.pth --end-weight 1.0 --epochs 2000 --hidden 64 64
```

**All parameters (explicit + defaults):**

| Parameter | Value | CLI flag |
|-----------|--------|----------|
| save_name | no_end_baseline_64x64.pth | --name |
| end_weight | 1.0 | --end-weight |
| num_epochs | 2000 | --epochs |
| hidden_sizes | [64, 64] | --hidden 64 64 |
| learning_rate | 0.0003 | --lr (default) |
| batch_size | 64 | --batch (default) |
| clip_actions | True | (default; not --no-clip) |
| data_path | baseline/data/expert_data_reach-v3.npz | --data (default) |
| end_fraction | 0.3 | --end-fraction (default) |
| end_inner_weight | None | --end-inner-weight (default) |
| end_inner_fraction | 0.05 | --end-inner-fraction (default) |
| end_upsample | False | (default; no --end-upsample) |
| save_run | True | (default; not --no-save-run) |
| keep_runs | 50 | --keep-runs (default) |
| eval_seed | 42 | --eval-seed (default) |
| lr_decay_epoch | None | --lr-decay-epoch (default) |
| lr_decay_gamma | 0.5 | --lr-decay-gamma (default) |

**Model (ClonePolicy):**

- **Architecture:** MLP: `Linear(input_dim → 64) → ReLU → Linear(64 → 64) → ReLU → Linear(64 → output_dim)`.
- **input_dim:** from expert data (states in `expert_data_reach-v3.npz`); for reach-v3 this is the flattened observation dimension.
- **output_dim:** from expert data (actions in same .npz); for reach-v3 this is the action dimension.
- **Activation:** ReLU after each hidden layer; no activation on the output (regression).
- **Optimizer:** Adam, lr=0.0003.
- **Loss:** MSE; with end_weight=1.0 there is no trajectory-end weighting (uniform over all (s,a) pairs).

**Data:** `baseline/data/expert_data_reach-v3.npz` (default path).

**Equivalent full command with all defaults written out:**
```bash
python train.py --name no_end_baseline_64x64.pth --end-weight 1.0 --epochs 2000 --hidden 64 64 \
  --lr 0.0003 --batch 64 --end-fraction 0.3 --end-inner-fraction 0.05 \
  --keep-runs 50 --eval-seed 42 --lr-decay-gamma 0.5
```
(Omitting `--no-clip`, `--data`, `--end-inner-weight`, `--no-save-run`, `--lr-decay-epoch`, `--end-upsample` leaves them at the defaults above.)
</details>

