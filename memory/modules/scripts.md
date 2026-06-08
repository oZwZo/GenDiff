# Module: scripts (training entrypoint + data prep CLIs)
> Path: script/  |  Part of: [codebase.md](../codebase.md)

## Purpose
Command-line glue: the training entrypoint that wires a YAML config to data → model →
sampler → `pl.Trainer`, plus one-off data-conversion / label-transfer / sampling utilities.
This module owns no model logic — it composes `src/`.

## Files
| File | Role |
|------|------|
| script/main_train_with_PL_trainer.py | **Main training entrypoint.** YAML → loaders/model/sampler → Trainer.fit/validate/test |
| script/path_n_util.py | Reads `machine_config.json` (main/data/pth dirs); `main_train_parser()` argparse |
| script/Old_train.py | Legacy training loop (pre-Lightning); reference only |
| script/sample_delta_x.py | CLI to sample ΔX from a checkpoint (pairs with `_diffplot.sample_Delta_X`) |
| script/perturb.py | CLI for in-silico perturbation runs |
| script/visualize_velocity_perturb.py | Render velocity/perturbation figures |
| script/TFA2h5ad.py, TF_atlas_raw.py | Build the TF-Atlas `.h5ad` from raw downloads |
| script/h5ad_to_10xraw.py, convert_large_csv_mtx.py, loom_ndarray.py, read_loom.py | Format conversion (h5ad↔10x↔loom↔csv) |
| script/KNN_transfer_label.py, RF_transfer_label.py | Transfer cell-type labels (kNN / random-forest) from a reference |
| script/chatgpt_cvae.py | Scratch/experimental CVAE |

## Key Functions — `script/main_train_with_PL_trainer.py`
- **`dl_from_config(configs, shuffle, n_workers)`**: `sc.read` the AnnData, build train/val/test
  `Dataset`s via `eval(_reader.<dataset_class>)(**dataset_kwargs)`, wrap in `DataLoader`s with a
  custom `my_collate` (drops 8-tuple ODE samples whose time-span ≠ `collect_time_span`).
- **`get_model_from_config(configs, cuda)`**: `eval(_epsilon_module.<epsilon_class>)(**epsilon_kwargs)`;
  optionally loads/freezes a `pretrain_embedder_pth`.
- **`get_sampler_from_configs(configs, eps_net)`**: `eval(_learner.<sampler_class>)(**sampler_kwargs, model=eps_net)`.
- `__main__`: builds everything, names the run from the config path, logs under
  `pth_dir/<config_subdir>/<epsilon_class>_<run_name>`, trains 200 epochs with
  `ModelCheckpoint(monitor=val_loss)` + `EarlyStopping(patience=50)`.

## How to run
```bash
python script/main_train_with_PL_trainer.py \
    --model_config configs/TF_atlas/TF_diff/H256_d3.yaml \
    --CUDA 0 --n_workers 10
```

## Dependencies on Other Modules
- Imports the whole `src/` package; instantiates classes by **string name via `eval`**, so a
  config string must exactly match a class in [models.md](models.md)/[diffusion.md](diffusion.md)/[data.md](data.md).
- `path_n_util` / `PATH` require `machine_config.json` at repo root or `src/` (gitignored).

## Module-Level Gotchas
- Class selection is `eval`-based — typos in a config surface as `AttributeError`, not validation.
- `main_train_with_PL_trainer` imports `path_n_util` as a top-level module, so it must be run
  from the `script/` directory (or with it on `PYTHONPATH`).
- Trainer hardcodes `accelerator='gpu', devices=1, max_epochs=200` (the `epochs` config key is unused there).
