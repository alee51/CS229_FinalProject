# Weights & Biases (W&B) integration

Training runs are logged to **Weights & Biases** by default so the team can compare runs and tune hyperparameters in one place.

## Setup

1. **Install:** `pip install wandb` (or use `requirements.txt`).
2. **API key (recommended):** Set your key in the environment so W&B can log in without interactive `wandb login`.
   - Get your key from [wandb.ai/authorize](https://wandb.ai/authorize).
   - **Unix / macOS (bash/zsh):** `export WANDB_API_KEY=your_key_here` (or add to `~/.bashrc` / `~/.zshrc`).
   - **Windows (cmd):** `set WANDB_API_KEY=your_key_here`
   - **Windows (PowerShell):** `$env:WANDB_API_KEY="your_key_here"`
   - **Using a `.env` file:** Copy `.env.example` to `.env`, put your real key in `.env`, and load it before running (e.g. `source .env` on Unix, or use `python-dotenv`). **Do not commit `.env`** — it is in `.gitignore`. Only `.env.example` (no real key) is committed.
   - **Option B:** Run `wandb login` once and paste your API key (stores it in wandb’s local config; no env var needed).
3. **Project:** All runs go to the project **`CS229_FinalProject`** (configurable in `baseline/train_config.yaml` or via your W&B account).

**Security:** Do not commit API keys to the repo. Use the env var or `wandb login` (which stores the key locally only). The file with your real key (e.g. `.env`) is listed in `.gitignore` so it is never committed.

**Job types and tags:** Training runs use **job_type `train`** and evaluation runs use **job_type `eval`**, so you can filter by type in the W&B UI. Runs are automatically tagged by **approach** (baseline, vae, tce) and **suite** (mt1, mt10, mt50) so you can quickly filter (e.g. all baseline runs or all mt10 runs). All sweep hyperparameters (epochs, lr, hidden_sizes, lr_decay_*, etc.) are in **config** for table columns and sweep comparison, not in tags.

## Single runs (default: W&B on)

From project root:

```bash
python train.py --approach baseline
python train.py --approach baseline --suite mt10 --epochs 500
```

- Each run logs: **config** (lr, epochs, batch_size, end_weight, suite, etc.), **train/loss** every epoch, **eval/success_rate** (or **eval/success_rate_avg** for multi-task) at the end.
- Run names are derived from hyperparameters (e.g. `mt10-lr3e-04-e500-end3`).

**Disable W&B for a run:**

```bash
python train.py --approach baseline --no-wandb
# or
set WANDB_MODE=disabled
python train.py --approach baseline
```

**Tag runs** (e.g. your name or experiment id) so teammates can filter in the UI:

```bash
python train.py --approach baseline --wandb-tag name:alice --wandb-tag exp:mt10-baseline
```

**Upload the trained checkpoint to W&B** (optional):

```bash
python train.py --approach baseline --wandb-save-model
```

## Logging test runs (test.py)

Evaluation runs from **test.py** are logged to the same project **`CS229_FinalProject`** with **job type `eval`**, so you can track results for models you have already trained. In the W&B UI, filter by job type “eval” to see only test runs, or compare them with training runs.

**Auto eval at end of training:** The post-training 50-goal eval is logged as a **separate eval run** (not on the training run), with **config.training_run_id** set to the training run's id so you can link this eval to that training run. It also sets **config.source = "auto"** so you can filter auto evals from test.py evals. All eval runs (auto and test.py) log **config.model** (the .pth filename) so you can group evals by policy. Policies trained before W&B have no training run; only their test.py eval runs appear.

From project root:

```bash
python test.py --approach baseline --model cloned_policy.pth
python test.py --approach baseline --model my_model.pth --suite mt10
```

- Each test run logs: **config** (approach, model, task/suite, episodes, seed, clip_actions), **eval/success_rate** (single-task) or **eval/success_rate_avg** and **eval/success_rate_{task_name}** (mt10/mt50). Metric names match training so eval runs are comparable in the same charts.

**Disable W&B for a test run:**

```bash
python test.py --approach baseline --model my.pth --no-wandb
```

**Tag test runs** (e.g. to identify which checkpoint or experiment):

```bash
python test.py --approach baseline --model my.pth --wandb-tag model:mt10-500ep --wandb-tag name:alice
```

## Parallel plot 0–100% scaling

For the MT10 parallel coordinates plot, W&B auto-scales each axis to the data range. To force all success-rate axes to 0–100%, run once from project root:

```bash
python log_wandb_scale_anchors.py
```

This creates two dummy runs (`scale-anchor-0` and `scale-anchor-100`) with all MT10 success-rate metrics at 0 and 100. Include these runs in the parallel coordinates panel's run set so all axes scale 0–100%. Filter by tag `scale-anchor` or `mt10` to hide these runs in other panels or to find them for the parallel plot. You only need to run it once per project unless you delete the anchor runs.

## Config file (single source of truth)

Default hyperparameters live in **`baseline/train_config.yaml`**. The CLI overrides only what you pass. To use a different config file:

```bash
python train.py --approach baseline --config path/to/my_config.yaml
```

## Sweeps (hyperparameter search)

1. **Create a sweep** (from project root; uses `baseline/sweep.yaml` or the MT-10 YAMLs below):

   ```bash
   wandb sweep baseline/sweep.yaml
   ```

   Copy the printed sweep ID (e.g. `entity/CS229_FinalProject/abc123`).

2. **Run one or more agents** (each run executes one trial):

   ```bash
   wandb agent <entity>/CS229_FinalProject/<sweep_id>
   ```

   Multiple group members can run `wandb agent` with the same sweep ID to share the work.

Sweep defaults for non-swept parameters come from `baseline/train_config.yaml`; the sweep YAML only defines the search space (e.g. `lr`, `epochs`, `end_weight`). Edit `baseline/sweep.yaml` to change the search method (random/grid/bayesian) or parameters.

### Overnight run (MT-10 hybrid: 2 small + 1 big sweep, 20 runs)

Run three one-at-a-time / full-grid sweeps for MT-10 (reference baseline: 500 ep, 128×128, end_inner 0.01). From project root:

1. Create each sweep (copy the printed sweep ID each time):
   ```bash
   wandb sweep baseline/sweep_mt10_arch.yaml
   wandb sweep baseline/sweep_mt10_epochs.yaml
   wandb sweep baseline/sweep_mt10_end_inner_full.yaml
   ```
2. Run one agent per sweep (e.g. three PowerShell windows, or run sequentially):
   ```bash
   wandb agent <entity>/CS229_FinalProject/<sweep_id_for_arch>
   wandb agent <entity>/CS229_FinalProject/<sweep_id_for_epochs>
   wandb agent <entity>/CS229_FinalProject/<sweep_id_for_end_inner_full>
   ```

Total: 2 + 2 + 16 = 20 runs. **Parallel agents:** W&B does not auto-start multiple agents. To run in parallel, start multiple `wandb agent` processes manually (e.g. one per sweep in separate terminals).

### Running sweeps from the W&B web UI

Create a new sweep in the project; set the **program** to `run_sweep.py` and ensure the run command executes from the **project root** (so `run_sweep.py` and `baseline/` resolve correctly). Sweep parameter names the entrypoint expects: `hidden_dim`, `epochs`, `end_inner_fraction`, `suite`, `lr`, `batch_size`, `end_weight`, `end_fraction`, `end_inner_weight`, `lr_decay_epoch`, `lr_decay_gamma`. You can paste or adapt `baseline/sweep.yaml` or the MT-10 sweep YAMLs in the UI.

## Summary

| What              | Command / location |
|-------------------|--------------------|
| Project name      | `CS229_FinalProject` (set in `baseline/train_config.yaml`) |
| Job type          | Train: `job_type=train`; Eval: `job_type=eval` |
| Auto tags         | approach (baseline/vae/tce), suite (mt1/mt10/mt50); add more with `--wandb-tag` |
| Sweep hyperparams | In **config** (epochs, lr, hidden_sizes, etc.) for table columns and sweep comparison |
| Disable W&B       | `--no-wandb` or `WANDB_MODE=disabled` (train and test) |
| Test runs (eval)  | `python test.py --approach baseline --model <name>` (job_type=eval) |
| Config defaults   | `baseline/train_config.yaml` |
| Sweep definition  | `baseline/sweep.yaml`; MT-10 overnight: `sweep_mt10_arch.yaml`, `sweep_mt10_epochs.yaml`, `sweep_mt10_end_inner_full.yaml` |
| Sweep entrypoint  | `run_sweep.py` (loads config + `wandb.config`) |
| MT-10 baseline model | `baseline/models/mt10_baseline.pth` (300 ep, 128×128, end_inner 0.01); use with `test.py --model mt10_baseline.pth --suite mt10` |
