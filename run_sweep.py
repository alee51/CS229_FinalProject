#!/usr/bin/env python
"""
W&B sweep entrypoint: load baseline/train_config.yaml, overlay wandb.config, call train_model.
Run from project root:
  wandb sweep baseline/sweep.yaml
  wandb agent <entity>/<project>/<sweep_id>
"""
import os
import sys

# When wandb agent runs "python run_sweep.py", it may invoke conda's python while VIRTUAL_ENV
# points at the project venv (where torch is installed). Prepend venv site-packages so imports work.
_venv = os.environ.get("VIRTUAL_ENV")
if _venv and os.path.isdir(_venv):
    if sys.platform == "win32":
        _venv_site = os.path.join(_venv, "Lib", "site-packages")
    else:
        _venv_site = os.path.join(_venv, "lib", "python{}.{}".format(*sys.version_info[:2]), "site-packages")
    if os.path.isdir(_venv_site) and _venv_site not in sys.path:
        sys.path.insert(0, _venv_site)

# Run from project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add baseline/scripts so we can import train_model and load_train_config
sys.path.insert(0, os.path.join(PROJECT_ROOT, "baseline", "scripts"))
from train import train_model, load_train_config

def main():
    import wandb
    run = wandb.init()
    cfg = wandb.config
    # Load defaults from baseline/train_config.yaml
    default_cfg = load_train_config(os.path.join(PROJECT_ROOT, "baseline", "train_config.yaml"))
    # Merge: sweep params override defaults
    def get(key, default=None):
        if hasattr(cfg, key):
            return getattr(cfg, key)
        return default_cfg.get(key, default)
    lr = get("lr", 0.0003)
    epochs = get("epochs", 500)
    batch_size = get("batch_size", 64)
    # Prefer scalar hidden_dim (W&B / UI friendly); else hidden_sizes list; else train_config default
    if get("hidden_dim") is not None:
        hd = get("hidden_dim")
        hidden_sizes = [hd, hd]
    else:
        hidden_sizes = get("hidden_sizes", default_cfg.get("hidden_sizes", [64, 64]))
        if isinstance(hidden_sizes, list):
            pass
        else:
            hidden_sizes = list(hidden_sizes) if hasattr(hidden_sizes, "__iter__") else default_cfg.get("hidden_sizes", [64, 64])
    save_name = get("save_name", "cloned_policy.pth")
    end_weight = get("end_weight", 3.0)
    end_fraction = get("end_fraction", 0.3)
    end_inner_weight = get("end_inner_weight")
    end_inner_fraction = get("end_inner_fraction", 0.05)
    lr_decay_epoch = get("lr_decay_epoch", 250)
    lr_decay_gamma = get("lr_decay_gamma", 0.5)
    suite = get("suite", "mt10")
    train_model(
        learning_rate=lr,
        num_epochs=epochs,
        batch_size=batch_size,
        hidden_sizes=hidden_sizes,
        save_name=save_name,
        clip_actions=True,
        data_path=None,
        end_weight=end_weight,
        end_fraction=end_fraction,
        end_inner_weight=end_inner_weight,
        end_inner_fraction=end_inner_fraction,
        save_run=True,
        keep_runs=50,
        eval_seed=42,
        lr_decay_epoch=lr_decay_epoch,
        lr_decay_gamma=lr_decay_gamma,
        end_upsample=False,
        suite=suite,
        device="auto",
        use_wandb=True,
        wandb_tags=None,
        wandb_project=default_cfg.get("wandb_project") or "CS229_FinalProject",
        wandb_save_model=False,
    )


if __name__ == "__main__":
    main()
