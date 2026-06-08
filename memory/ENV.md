# Environment: GenDiff
> Generated: 2026-06-08 | Verified by import + forward-pass smoke test (below). Part of: [codebase.md](codebase.md)

## Chosen environment: `PINN_env` (micromamba)
- Manager: **micromamba** 1.5.6, `MAMBA_ROOT_PREFIX=/rds/user/wz369/hpc-work/LIBS/mamba`
- Env path: `/rds/user/wz369/hpc-work/LIBS/mamba/envs/PINN_env`
- Python: **3.9.22** | interpreter: `…/envs/PINN_env/bin/python`
- Activate: `micromamba activate PINN_env`  ·  one-off: `micromamba run -n PINN_env <cmd>`

### Why this env (of 14 available)
It is the only env carrying GenDiff's full train **and** analysis stack. Decisively, it has **pytorch_lightning 1.9.5**, which matches GenDiff's PL-1.x API (`pl.Trainer(auto_lr_find=…, auto_select_gpus=…)` in `script/main_train_with_PL_trainer.py` — those args were removed in PL 2.x, so the 2.x envs `mioflow`/`scarches`/`multiome` would break training). It also uniquely bundles **scvelo 0.3.3** and **cellrank 2.0.7** (required by `src/_diffplot.py` and the trajectory analysis). Other envs were missing torch (`clonalflow`,`moscot`) or scvelo/cellrank (all others).

### Key package versions
| Package | Version | Needed by |
|---------|---------|-----------|
| torch | 2.7.0+cu126 | models, samplers, learners |
| pytorch_lightning | **1.9.5** | `_learner.py`, training script (1.x API) |
| torchmetrics | 1.7.1 | `AE_learner` metrics |
| einops | **0.8.2** (installed by this setup) | `_helper_net`, `_epsilon_module`, `_sampler` |
| scanpy | 1.10.3 | `_reader`, preprocessing |
| anndata | 0.10.9 | dataset IO |
| scvelo | 0.3.3 | `_diffplot` velocity/trajectory |
| cellrank | 2.0.7 | macrostate / absorption analysis |
| torchdyn | present | `ODE_learner` (NeuralODE) |
| scikit-learn, seaborn | present | `_pp_fun`, `_diffplot` |
| numpy | 1.26.4 | — |

## What this setup changed (2026-06-08)
1. **Installed `einops` 0.8.2** into `PINN_env` (`pip install einops`) — the only missing core dependency. Pure-Python, no version conflicts.
2. **Editable-installed GenDiff** (`pip install -e .` from repo root). setup.py exposes `packages=['src']`, so `import src` works from any cwd. Records: `__editable__.gendiff-0.1.pth`, `gendiff-0.1.dist-info` in the env site-packages. (pip warns the legacy `setup.py develop` mode is deprecated — harmless; pin `pip<25.3` or add a `pyproject.toml` later if needed.)
3. **Created `machine_config.json`** in two places (required, `.gitignore`d, machine-specific):
   - `src/machine_config.json` — read by `src/__init__.py` and `src/PATH.py`
   - `machine_config.json` (repo root) — read by `script/path_n_util.py`
   Both contain:
   ```json
   {"main_dir":"/rds/user/wz369/hpc-work/GenDiff/",
    "data_dir":"/rds/user/wz369/hpc-work/GenDiff/data/",
    "pth_dir":"/rds/user/wz369/hpc-work/GenDiff/pth/"}
   ```
4. **Created `pth/`** (checkpoint/log dir referenced by `pth_dir`).

> No source code was modified. The stray `from turtle import forward` lines import cleanly here (tkinter is present in `PINN_env`), so they were left as-is.

## Verified import test (PASSED)
```bash
cd /tmp   # proves cwd-independence
micromamba run -n PINN_env python -c "
import src
from src import _helper_net,_epsilon_module,_sampler,_learner,_reader,_configure,_linear
import torch
from src._epsilon_module import Epsilon_Linear
from src._learner import Equivalent_Diffuse_Sampler
net = Epsilon_Linear(gene_dim=50,time_emb_dim=0,n_base_perturbs=10,condition_emb_dim=16,
                     use_batch_index=False,hidden_size=[32,16,32])
samp = Equivalent_Diffuse_Sampler(model=net,scheduler='linear_beta_schedule',loss_type='huber',timesteps=200)
out = net(torch.randn(8,50), torch.randint(0,10,(8,1)), None, torch.randint(0,200,(8,)))
print('OK', tuple(out.shape))"
```
Result: `import src OK`, all core submodules import, `Epsilon_Linear` forward returns `(8, 50)`. `from src import _diffplot` also succeeds (it self-appends `script/` to `sys.path`).

## Gotchas
- **GPU:** `torch.cuda.is_available()` is `False` on the login/CPU node; the cu126 build will use GPU on a GPU compute node. Training (`script/main_train_with_PL_trainer.py`) hardcodes `accelerator='gpu', devices=1` → run it on a GPU node (the script falls back to CPU only where it checks `torch.cuda.is_available()`).
- **Optional deps NOT installed:** `scvi` and `cpa` are absent → `src/my_VAE.py` will not import (standalone NB-VAE experiment, not part of the core pipeline). Install only if that module is needed.
- **Config `anndata_path` is stale** (`/home/wergillius/...`). Repoint to `data/` files before training (see [Dataset.md](Dataset.md)).
- **Two config files must agree:** if `main_dir`/`pth_dir` change, update both `src/machine_config.json` and the repo-root `machine_config.json`.
- **Run training from `script/`** (or with `script/` on `PYTHONPATH`) because `main_train_with_PL_trainer.py` does a bare `import path_n_util`.

## Quick start
```bash
micromamba activate PINN_env
cd /rds/user/wz369/hpc-work/GenDiff/script
python main_train_with_PL_trainer.py --model_config ../configs/TF_atlas/TF_diff/H256_d3.yaml --CUDA 0 --n_workers 10
# (first edit the config's anndata_path to /rds/user/wz369/hpc-work/GenDiff/data/differentiated_model_velo.h5ad)
```
