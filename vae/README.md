# VAE approach

The **root** `train.py` and `test.py` use **`vae/scripts/`** as the entrypoint: they call `train_model`, `load_train_config`, and the `ClonePolicy` wrapper from `vae/scripts/train.py`.

The original VAE implementation (e.g. `train_vae.py`, `data_utils.py`, `models.py`, `dataset.py` in this directory) is unchanged and remains the source for VAE training logic; `vae/scripts/` wraps it for the unified root entrypoints.
