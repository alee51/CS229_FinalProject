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
python test.py --approach baseline --model latest_policy.pth
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
python scripts_all/log_wandb_scale_anchors.py
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

### Create a sweep from the W&B web UI

You can initialize sweeps in the W&B web UI instead of writing local YAML files. The agent still runs locally; it pulls the sweep config from W&B.

**Steps to create a sweep in the UI**

1. Open your project at [wandb.ai](https://wandb.ai) (e.g. **CS229_FinalProject**).
2. Go to **Sweeps** in the project, then **Create sweep** / **New sweep**.
3. In the sweep creator you will see either:
   - A **YAML config** text area: paste a full sweep config (see templates below), or
   - A **method / metric / parameters** form: set method (e.g. grid), metric name `eval/success_rate_avg`, goal **maximize**, then add parameters by name and type.
4. Set the **program** to `scripts_all/run_sweep.py` (or the command that runs it, e.g. `python scripts_all/run_sweep.py`). When you run the agent from your machine with `wandb agent ...` from the **project root**, the working directory is already correct.
5. Save/create the sweep. Copy the sweep ID, then from project root run: `wandb agent <entity>/CS229_FinalProject/<sweep_id>`. No YAML file is needed on disk for the sweep definition.

**Parameter reference** (for the UI form or when editing pasted YAML)

Parameters that `scripts_all/run_sweep.py` reads from the sweep config. Any parameter not set in the sweep falls back to `baseline/train_config.yaml`.

| Parameter | Type | Example (YAML) |
|-----------|------|----------------|
| `suite` | string | `value: mt10` |
| `end_weight` | float | `value: 1.0` |
| `end_fraction` | float | `value: 0.3` |
| `epochs` | integer | `values: [500, 1000]` |
| `hidden_dim` | integer | `values: [128, 256]` (builds [dim, dim] in code) |
| `end_inner_fraction` | float | `values: [0, 0.01]` |
| `end_inner_weight` | float or null | omit or `value: null` |
| `lr` | float | optional |
| `batch_size` | integer | optional |
| `lr_decay_epoch`, `lr_decay_gamma` | integer, float | optional |

**Copy-paste YAML templates**

Paste one of these into the UI’s YAML config editor, then edit the `parameters` section as needed.

**Template 1 — MT-10 architecture sweep (grid, 2 runs):**

```yaml
program: scripts_all/run_sweep.py
method: grid
metric:
  name: eval/success_rate_avg
  goal: maximize
parameters:
  suite:
    value: mt10
  end_weight:
    value: 1.0
  epochs:
    value: 500
  hidden_dim:
    values: [128, 256]
  end_inner_fraction:
    value: 0.01
```

**Template 2 — Small custom grid (edit values in UI):**

```yaml
program: scripts_all/run_sweep.py
method: grid
metric:
  name: eval/success_rate_avg
  goal: maximize
parameters:
  suite:
    value: mt10
  end_weight:
    value: 1.0
  epochs:
    values: [500, 1000]
  hidden_dim:
    values: [128, 256]
  end_inner_fraction:
    values: [0, 0.01]
```

**Running the agent:** After creating the sweep in the UI, run locally from project root: `wandb agent <entity>/CS229_FinalProject/<sweep_id>`.

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
| Sweep entrypoint  | `scripts_all/run_sweep.py` (loads config + `wandb.config`) |
| MT-10 baseline model | `baseline/models/mt10_baseline.pth` (300 ep, 128×128, end_inner 0.01); use with `test.py --model mt10_baseline.pth --suite mt10` |
