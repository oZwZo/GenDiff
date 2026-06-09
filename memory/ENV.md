# Environment: GenDiff
> Generated: 2026-06-08 | Updated: 2026-06-08 (added GenDiff_env clone; added the manuscript
> "quick_train" reproduction path — the paper's TF-Atlas figures come from there, not from the training
> script). Verified by import + forward-pass test. Part of: [codebase.md](codebase.md)

## Primary environment: `GenDiff_env` (micromamba)
A full clone of `PINN_env`, created so GenDiff has a dedicated env. GenDiff is installed in **both** envs.
- Manager: **micromamba** 1.5.6, `MAMBA_ROOT_PREFIX=/rds/user/wz369/hpc-work/LIBS/mamba`
- Env path: `/rds/user/wz369/hpc-work/LIBS/mamba/envs/GenDiff_env`  (18 GB, 100,616 files)
- Python: **3.9.22** | interpreter: `…/envs/GenDiff_env/bin/python`
- Activate: `micromamba activate GenDiff_env`  ·  one-off: `micromamba run -n GenDiff_env <cmd>`
- `PINN_env` (the source) still exists and is identical; either works.

### Key package versions (identical in both envs)
| Package | Version | Needed by |
|---------|---------|-----------|
| torch | 2.7.0+cu126 | models, samplers, learners |
| pytorch_lightning | **1.9.5** | `_learner.py`, training script (1.x API) |
| torchmetrics | 1.7.1 | `AE_learner` metrics |
| einops | 0.8.2 | `_helper_net`, `_epsilon_module`, `_sampler` |
| scanpy / anndata | 1.10.3 / 0.10.9 | `_reader`, dataset IO |
| scvelo | 0.3.3 | `_diffplot` velocity/trajectory |
| cellrank | 2.0.7 | macrostate / absorption analysis |
| torchdyn | 1.0.6 | `ODE_learner` (NeuralODE) |
| scikit-learn / seaborn | present | `_pp_fun`, `_diffplot` |
| numpy | 1.26.4 | — |

## How GenDiff_env was created (provenance)
micromamba has **no `--clone`**, there is no `conda`/`mamba` binary in this root, and **305 of 342
packages are pip-installed** (torch, scvelo, cellrank, the cuda/rapids/jax stack, GenDiff) — so a
conda explicit-spec recreate would drop ~89% of the env. The faithful, reliable method was a
directory copy + prefix-fix (the conda-pack approach):
1. `cp -a envs/PINN_env envs/GenDiff_env` — byte-for-byte copy (independent, not hardlinked).
2. Rewrote the `…/PINN_env` → `…/GenDiff_env` prefix in **109 files** under `bin/` and `etc/`
   (console-script shebangs, conda activation scripts) so the clone's `pip` and entry points
   target GenDiff_env. Verified: `bin/pip` shebang → `…/GenDiff_env/bin/python3.9`; 0 old-prefix
   refs remain in `bin/etc`; `pip --version` reports the GenDiff_env site-packages.
3. `python -m pip install -e .` — reinstalled GenDiff editable (idempotent; the editable finder
   points at the repo root, so it is prefix-independent).
4. `machine_config.json` (repo root + `src/`) already exists from the earlier setup; env-independent.

> Cosmetic note: conda-meta JSON / dist-info RECORD / some compiled libs inside GenDiff_env still
> contain the literal `PINN_env` path. This does not affect functionality (conda libs use
> `$ORIGIN`-relative RPATHs, and only `bin/`+`etc/` shebangs/activation matter at runtime). Deleting
> PINN_env later is safe for GenDiff_env, but re-run the import test afterward to be sure.

## Disk-quota episode (why this took space management)
The first copy failed with **"Disk quota exceeded"** — the binding limit is the **inode (file-count)
quota = 1,048,576**, not bytes (block space had ~270 GB free). Before the copy the env area was at
~1.045 M files (only ~3.6k inodes free), so cloning 100k files failed instantly. To make room
(user-approved):
- Deleted micromamba envs **`moscot` (5.7 G), `scarches` (6.8 G), `inDecay` (9.5 G)** → freed ~22 GB and ~154k inodes.
- Moved **`adata_ori_metaclone_20250425_final.h5ad` (36 GB)** from `/rds/user/wz369/hpc-work/` to
  `~/playground` (→ `/rds/project/rds-SDzz0CATGms/users/wz369`, a separate filesystem) → freed 36 GB on rds-d7.

After cloning, rds-d7 inode usage is **991,539 / 1,048,576 (57,037 free, 95% used)** — functional but
tight. Watch the inode quota: another full env clone (~100k files) would not fit without freeing more.
`df -i /rds/user/wz369` and `lfs quota -uh $USER /rds/user/wz369` are the commands to monitor it.

## Verified import test (PASSED in GenDiff_env)
```bash
cd /tmp   # cwd-independent
micromamba run -n GenDiff_env python -c "
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
Result: python resolves to `…/GenDiff_env/bin/python`; core submodules import; forward pass → `(8, 50)`.
`from src import _diffplot` also works (it self-appends `script/` to `sys.path`).

## machine_config.json (required, gitignored, env-independent)
`src/machine_config.json` (read by `src/__init__.py`, `src/PATH.py`) and repo-root `machine_config.json`
(read by `script/path_n_util.py`), both:
```json
{"main_dir":"/rds/user/wz369/hpc-work/GenDiff/",
 "data_dir":"/rds/user/wz369/hpc-work/GenDiff/data/",
 "pth_dir":"/rds/user/wz369/hpc-work/GenDiff/pth/"}
```

## Gotchas
- **GPU:** `torch.cuda.is_available()` is `False` on the login node; the cu126 build uses GPU on a GPU
  compute node. Training hardcodes `accelerator='gpu', devices=1` → run on a GPU node.
- **Inode quota is at 95%** (57k free) — checkpoints/logs during training write many files; free more
  inodes (delete an unused env, clear `pth/` logs) before large runs.
- **Optional deps NOT installed:** `scvi`, `cpa` → `src/my_VAE.py` will not import (standalone NB-VAE
  experiment, not part of the core pipeline).
- **Config `anndata_path` is stale** (`/home/wergillius/...`) → repoint to `data/` files (see [Dataset.md](Dataset.md)).
- **Run training from `script/`** (or with `script/` on `PYTHONPATH`): `main_train_with_PL_trainer.py`
  does a bare `import path_n_util`.

## Quick start (the diffusion/ε framework — NOT what made the paper figures)
```bash
micromamba activate GenDiff_env
cd /rds/user/wz369/hpc-work/GenDiff/script
python main_train_with_PL_trainer.py --model_config ../configs/TF_atlas/TF_diff/H256_d3.yaml --CUDA 0 --n_workers 10
# (first set the config anndata_path to /rds/user/wz369/hpc-work/GenDiff/data/differentiated_model_velo.h5ad)
```

## Reproducing the manuscript TF-Atlas results (quick_train — Figs 4–6)
The paper's TF-Atlas figures are produced by `some_tutorials/TF-atlas/quick_train/`, using the
plain-Lightning models in `src/_linear.py` with a bare `pl.Trainer` (no YAML/configurer). The published
GenDiff is `_linear.Pretrained_GenDiff` = a pretrained expression autoencoder + a per-gene Ridge prior;
see codebase.md "MANUSCRIPT MODEL" for the exact definition.
- **Data file:** notebooks read `…/data/TFAtlas/GSE217460_210322_TFAtlas_differentiated.h5ad`
  (local equivalent `data/differentiated_model_velo.h5ad`, 28825×4806 — see [Dataset.md](Dataset.md)).
  **Prediction target** = `adata.layers['a3_PathSampled_X']`; **split** = `adata.obs['split']`.
- **Training entry points** (run on a GPU node, from repo root with `src` importable):
  - `quick_train_reconX.py` → pretrained expression AE (`Embedding_model`), ckpt under
    `{pth_dir}/TF_atlas_quick/Embedding_CAE_deep_model_reconX/`.
  - `quick_train_CAE.py` → CAE baseline, ckpt under `{pth_dir}/quick_CVAE/`.
  - `fit_linear_each_gene.py` → per-gene sklearn Ridge → writes `ridgemodel_params.npy` (GenDiff's prior)
    and `ridge_model_r.csv`.
  - then run the notebooks in order `2 → 4 → 5 → 5.1 → 6 → 5.2`.
- **These scripts hardcode an old machine:** the data path
  (`/home/wergillius/Project/diffuse_differentiate/data/TFAtlas/…`) and the GPU index
  (`gpus=[0]` in reconX, `gpus=[1]` in CAE). The notebooks load checkpoints from old absolute paths
  (`/home/wergillius/data/diffuse_differentiate/…`). None of those `.ckpt`s are in the repo — retrain
  or repoint before the eval/trajectory notebooks will run. Repoint the data path to
  `data/TFAtlas/…` (or `data/differentiated_model_velo.h5ad`) and fix the GPU index first.

### BarRNA-seq (Figs 2–3) — hybrid, two models
- **Benchmark (Fig 2a–c): quick-train**, in `some_tutorials/Barcodelet/` — `2.quick_train.ipynb`
  (Linear / CAE / LatentAdd baselines), `3.quick_train_AE.ipynb` (pretrained `CAE_model` reconX),
  `4.get_output_prior.ipynb` (per-gene `RidgeCV` prior → `prior_coef_Jul2.npy`/`prior_intcpt_Jul2.npy`),
  assembled and plotted in `5.Figure2a-c.ipynb`. Data `data/.../integrated_mesc_group0_Nov7.h5ad`;
  condition = the 7 pathway columns; target `adata.obsm['Tr_SampledX_r100']`.
- **Trajectory + latent (Fig 2f–g, 3): the diffusion pipeline.** Train with
  `main_train_with_PL_trainer.py --model_config configs/Barcodelet/mesc_5layer_k=15_traverse_notime.yaml`
  (→ `Epsilon_Linear` + `Equivalent_Diffuse_Sampler`), then analyse via `1.sample_DeltaX.ipynb`
  (`dfp.extrapolate` / `plot_representation`, `use_rep='z_c'`). This is the one manuscript result that
  uses the training-script pipeline rather than quick_train.
