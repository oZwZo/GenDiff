# Environment: GenDiff
> Generated: 2026-06-08 | Updated: 2026-06-08 (added GenDiff_env clone; added the manuscript
> "quick_train" reproduction path — the paper's TF-Atlas figures come from there, not from the training
> script); 2026-06-10 (added the `cellot` and `cellflow` baseline/comparison envs — see end of file);
> 2026-06-11 (added the `cpa` baseline/comparison env; added `cinemaot` into the existing `cospar`
> env — see end of file).
> Verified by import + forward-pass test. Part of: [codebase.md](codebase.md)

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

---

# Baseline / comparison environments
> Two extra micromamba envs in the same root (`MAMBA_ROOT_PREFIX=/rds/user/wz369/hpc-work/LIBS/mamba`),
> built 2026-06-10 to run external perturbation-response models on the GenDiff datasets (TF-Atlas,
> BarRNA-seq) as comparisons. Each has its own deliberately-pinned stack; do **not** install either
> package into `GenDiff_env`/`PINN_env` (version conflicts both ways). Both verified on the cluster's
> **A100-SXM4-80GB** (driver 595.71.05).

## `cellot` (micromamba) — CellOT, neural optimal-transport baseline
Runs [CellOT](https://github.com/bunnech/cellot) (Bunne et al.) — learns a control→one-perturbation
transport map (one model per perturbation). PyTorch, deliberately old pins (the repo's `requirements.txt`).
- Env path: `/rds/user/wz369/hpc-work/LIBS/mamba/envs/cellot` (~49,400 files) | Python **3.9.5**
- Activate: `micromamba activate cellot` · one-off: `micromamba run -n cellot <cmd>`
- Repo (installed `-e`): `/rds/user/wz369/hpc-work/cellot` (cloned, then `python setup.py develop`)
- **Memory + harness:** [`cellot/memory/codebase.md`](../../cellot/memory/codebase.md),
  [`cellot/memory/HARNESS.md`](../../cellot/memory/HARNESS.md) — train/evaluate tool definitions.

### Key package versions
| Package | Version | Note |
|---------|---------|------|
| torch | **1.11.0+cu113** | swapped from the PyPI default cu102 (see GPU gotcha) |
| numpy | 1.19.5 | repo pin |
| pandas | 1.2.5 | repo pin |
| anndata / scanpy | 0.7.6 / 1.8.1 | old AnnData API |
| scipy / scikit-learn | 1.8.1 / 1.1.1 | repo pins |
| ml-collections | 0.1.0 | config_flags system |
| requests | 2.32.5 | pulled in by deps |

### Provenance (how it was built)
1. `micromamba create -y -n cellot python=3.9.5 pip setuptools wheel`
2. `pip install -r requirements.txt` (exact pins above) → `python setup.py develop`
3. **GPU fix:** PyPI `torch==1.11.0` is a **cu102** build (arch list sm_37…sm_70) and dies on the A100
   (sm_80) with `no kernel image is available`. Uninstalled it and installed the same version's cu113
   build: `pip install --no-cache-dir "torch==1.11.0+cu113" -f https://download.pytorch.org/whl/torch_stable.html`
   (arch list now includes sm_80/sm_86). Verified A100 matmul + forward/backward.

### Gotchas (full list in HARNESS.md)
- Run scripts from the **repo root** — task configs use relative data paths.
- `data.target` is **not** in the task YAML → always pass `--config.data.target <condition>`.
- Plain-CellOT eval needs `--embedding ""` or `evaluate.py` tries to load a sibling scGen autoencoder.
- `cache/model.pt` only appears after the first `eval_freq` step (250); the train/test split is
  regenerated at runtime — `obs['split']` is **ignored** (fix `datasplit.random_state` for reproducibility).

## `cellflow` (micromamba) — CellFlow, JAX flow-matching model
Runs [CellFlow](https://github.com/theislab/cellflow) (pip pkg **`cellflow-tools`**) — conditional
OT flow-matching over perturbation conditions (JAX/flax/ott-jax/diffrax). Models many conditions in
one model via a learned condition embedding.
- Env path: `/rds/user/wz369/hpc-work/LIBS/mamba/envs/cellflow` (~39,000 files) | Python **3.11.15**
- Activate: `micromamba activate cellflow` · one-off: `micromamba run -n cellflow <cmd>`
- Repo clone (docs/tutorials): `/rds/user/wz369/hpc-work/cellflow` (the package itself is pip-installed)
- **Memory + harness:** [`cellflow/memory/codebase.md`](../../cellflow/memory/codebase.md),
  [`cellflow/memory/HARNESS.md`](../../cellflow/memory/HARNESS.md) — train/predict tool definitions (Python API).

### Key package versions
| Package | Version | Note |
|---------|---------|------|
| cellflow-tools | **0.0.9** | latest; needs Python ≥3.11 |
| jax / jaxlib | 0.10.1 / 0.10.1 | GPU backend |
| jax-cuda12-plugin / -pjrt | 0.10.1 | bundled CUDA 12.9 + cuDNN 9.23 |
| ott-jax / flax / diffrax | 0.6.0 / 0.12.7 / 0.7.2 | solver stack (OTFlowMatching default) |
| numpy / pandas | 2.4.6 / 2.3.3 | modern stack |
| anndata / scanpy | 0.12.16 / 1.11.5 | modern AnnData API |
| scikit-learn | 1.5.1 | pinned by cellflow |
| requests | 2.34.2 | **manually added** (undeclared import, see gotcha) |

### Provenance (how it was built)
1. `micromamba create -y -n cellflow python=3.11 pip`
2. `pip install --no-cache-dir cellflow-tools "jax[cuda12]"` (resolves jax+jaxlib+cuda12 plugin together)
3. `pip install --no-cache-dir requests` — CellFlow's `preprocessing/_gene_emb.py` imports `requests`
   unconditionally but doesn't declare it; import fails without this.
- Verified: `jax.default_backend()=='gpu'`, `jax.devices()==[CudaDevice(id=0)]`, A100 matmul finite;
  `cellflow.model.CellFlow` + preprocessing/training/data submodules import. (The `hwloc/numa
  cpubind failed` lines JAX prints on this cluster are cosmetic.)

### Gotchas (full list in HARNESS.md)
- **`requests` is an undeclared runtime dep** (added above) — re-add it if the env is rebuilt.
- `control_key` is a **boolean** obs column (`True`=control), not a string label — build it explicitly,
  e.g. `adata.obs['is_control'] = adata.obs['condition']=='control'`.
- Prediction AnnData must contain **only control cells** (`_verify_prediction_data` raises otherwise).
- `ad.concat` drops `adata.uns` (which holds condition embeddings) — reassign `uns` after any concat/subset.
- **No automatic checkpointing** — call `cf.save(<dir>)` manually after `train()`.

### Eligibility note
CellFlow requires **Python ≥3.11**, so it cannot go in `GenDiff_env`/`PINN_env` (3.9.22), `cellot`
(3.9.5), or `mfm`/`mioflow` (3.10 — would also churn their jax/ott/sklearn). It needs its own env,
which is this one.

## `cpa` (micromamba) — CPA, Compositional Perturbation Autoencoder
Runs [CPA](https://github.com/theislab/CPA) (Lotfollahi et al. 2023, *Mol Syst Biol*; pip pkg
**`cpa-tools`**) — a scvi-tools-based autoencoder that learns a disentangled additive latent
(`z = z_basal + z_pert + z_covs`) with adversarial classifiers, for counterfactual perturbation-response
prediction (drugs, doses, combinations, CRISPR, context transfer, batch correction).
- Env path: `/rds/user/wz369/hpc-work/LIBS/mamba/envs/cpa` | Python **3.10.20**
- Activate: `micromamba activate cpa` · one-off: `micromamba run -n cpa <cmd>`
- Repo clone (source + tutorials): `/rds/user/wz369/hpc-work/external/CPA` (commit `fbd7c02`, depth-1)
- **Memory + harness:** [`external/CPA/memory/CODEBASE_MEMORY.md`](../../external/CPA/memory/CODEBASE_MEMORY.md),
  [`external/CPA/memory/HARNESS.md`](../../external/CPA/memory/HARNESS.md) — codebase map + train/predict tool
  definitions. Also `external/CPA/memory/smoke_test.py` (working end-to-end test) and
  `external/CPA/memory/requirements-lock.txt` (`pip freeze`).

### Key package versions
| Package | Version | Note |
|---------|---------|------|
| cpa-tools | **0.8.8** | needs Python <3.11 |
| scvi-tools | 0.20.3 | CPA subclasses scvi `BaseModelClass`/`TrainingPlan`; do not cross the 1.0 boundary |
| torch / torchaudio | **2.0.0+cu117** / 2.0.1 | CUDA 11.7 wheel (`nvidia-*-cu11`) |
| lightning / pytorch-lightning | 2.2.5 / **1.9.5** | scvi+CPA import `pytorch_lightning` (1.x API) |
| anndata / scanpy | 0.9.2 / 1.10.4 | old AnnData API |
| numpy / scipy | 1.23.5 / 1.12.0 | repo pins (`numpy<1.24`) |
| jax / jaxlib | 0.4.23 / 0.4.23 | **CPU-only** jaxlib (only the optional Ray tuner uses jax) |
| ray | 2.9.3 | hyperparameter tuner (`run_autotune`) |
| pyarrow | **14.0.2** | **manually pinned** — see provenance step 3 |

### Provenance (how it was built)
1. `micromamba create -y -n cpa -c conda-forge python=3.10 pip`
2. `pip install cpa-tools` (pip resolves the tight pins above; chose torch 2.0.0+cu117, scvi 0.20.3, etc.)
3. **Required fix — pyarrow:** `cpa-tools` pins `ray 2.9.x` but leaves `pyarrow` uncapped, so the fresh
   install pulled pyarrow 24, and ray 2.9.3 then crashes on `import` with
   `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'` (removed in pyarrow 16).
   Fixed with `pip install "pyarrow<15"` → pyarrow 14.0.2. **`import cpa` fails without this — re-apply if rebuilt.**
- Verified by full-stack import + `memory/smoke_test.py`: trains 2 epochs and predicts on synthetic data
  (`CPA_pred` shape `(600, 40)`, exit 0). Run on the **login node (CPU)**; torch is the cu117 GPU build for
  GPU nodes, but a GPU train has not yet been exercised (unlike cellot/cellflow which were A100-verified).

### Gotchas (full list in HARNESS.md)
- **Class-level encoder state:** `CPA.pert_encoder`/`covars_encoder`/`pert_smiles_map` are *class*
  attributes set only when `None` → a 2nd `setup_anndata` in the same process silently reuses the 1st
  dataset's encoders. Reset all three to `None` before setting up a new dataset.
- **`recon_loss` must match data scale:** `'nb'`/`'zinb'` = raw counts (row sums > 0); `'gauss'` = log-norm.
- **Split naming trap:** `split_key` values `train`/`test`/`ood` where **`test` is the *validation* split**
  and `ood` is the held-out test set (override via `CPA(train_split=, valid_split=, test_split=)`).
- **`cpa_metric` only logs in validation** → set `check_val_every_n_epoch <= max_epochs` (1 for short runs)
  or early-stopping/`SaveBestState` break.
- **`tests/test_cpa.py` is a dead API** (`drug_key`/`dose_key` + `scvi.data.setup_anndata`) — use
  `memory/smoke_test.py` as the canonical example.
- **`micromamba` is a shell function**, absent in non-interactive scripts — use the binary
  `$MAMBA_EXE` = `/rds/user/wz369/hpc-work/LIBS/mamba/micromamba`.

### Eligibility note
CPA needs **Python <3.11** *and* a tightly-pinned old stack (`scvi-tools<1.0`, `torch<=2.0.1`,
`numpy<1.24`, `jax<0.4.24`, `ray 2.9.x`). No existing env fit: `GenDiff_env`/`PINN_env` (torch 2.7),
`cellot` (torch 1.11 / numpy 1.19 / anndata 0.7), `cospar` (numpy 1.26), `cellflow` (Python 3.11).
Installing CPA into any of them would have broken it — hence its own env.

## `cinemaot` (installed into the existing `cospar` env) — CINEMA-OT
Runs [CINEMA-OT](https://github.com/vandijklab/CINEMA-OT) (Dong et al., *Nat Methods* 2023; pip pkg
**`cinemaot`**) — a **causal** perturbation-effect method: separates confounder variation (FastICA +
Chatterjee-ξ independence test) from treatment variation, then matches treated↔control cells by
entropy-regularized **optimal transport** (a bundled Sinkhorn–Knopp, *not* POT) to get single-cell
individual treatment effects, synergy, and per-gene confounder-vs-effect attribution. Pure-Python, light.
- **Reused env (not dedicated):** `cospar` — `/rds/user/wz369/hpc-work/LIBS/mamba/envs/cospar` | Python **3.9.23**
- Activate: `micromamba activate cospar` · one-off: `micromamba run -n cospar <cmd>`
- Repo clone (editable source + tutorial): `/rds/user/wz369/hpc-work/external/CINEMA-OT` (commit `949bc3f`)
- **Memory + harness:** [`external/CINEMA-OT/memory/CODEBASE_MEMORY.md`](../../external/CINEMA-OT/memory/CODEBASE_MEMORY.md),
  [`external/CINEMA-OT/memory/HARNESS.md`](../../external/CINEMA-OT/memory/HARNESS.md), plus
  `external/CINEMA-OT/memory/smoke_test.py` (working end-to-end test).

### Why cospar (preferred existing, per request) instead of a new env
CINEMA-OT's deps are **unpinned** and pure-Python (`numpy/pandas/scanpy/scikit-learn/scipy/statsmodels/
anndata` — no torch/jax/POT), so it drops into any modern scanpy env with **zero dependency churn**.
All of `cospar`, `cpa`, `GenDiff_env`, `mfm` already satisfied the 7 core deps; **`cospar` was picked
because it additionally already had `gseapy` (for `cinemaot.utils`) and `leidenalg`+`igraph` (for
`cinemaot_weighted`)** — nearly the whole optional stack was present. cinemaot version 0.0.4 (its
`__version__` string still says 0.0.3 — cosmetic).

### Provenance (how it was added)
1. `pip install -e /rds/user/wz369/hpc-work/external/CINEMA-OT --no-deps` — `--no-deps` guarantees **no**
   existing cospar package was up/downgraded; only the `cinemaot` editable package was added.
2. `pip install plotly` → plotly 6.8.0 (+ pure-python `narwhals`) — the only missing `utils` dep
   (gseapy/leidenalg/igraph were already present).
- **Not installed (optional):** `scib`+`harmonypy` (only `cinemaot.benchmark` baselines need them),
  `scsim` (only `simulation.py`). Add on demand.
- Verified by import + `memory/smoke_test.py` on the **login node (CPU)**: ICA→ξ→Sinkhorn-OT→ITE runs,
  outputs `cf (500×9)`, `ot_matrix (250×250)`, `TE (250×200)`, and recovers the planted perturbed genes
  (top-20 effect recovery 1.00), exit 0. (CINEMA-OT is CPU-only — no GPU path.)

### Gotchas (full list in HARNESS.md)
- **`adata.obsm['X_pca']` is required** (ICA input) — run `sc.pp.pca` first.
- **`thres` (ξ cutoff) is the main knob** and its default *differs by function*: `0.15`
  (`cinemaot_unweighted`) vs `0.75` (`cinemaot_weighted`). `ValueError: No confounder components
  identified` → raise `thres`.
- **Pairwise only** (one control vs one treatment per call) — loop for multiple perturbations.
- `cinemaot_weighted` needs Leiden (leidenalg/igraph — present); `attribution_scatter`/`NBregression`
  need `adata.obsm['cf']` + a **sparse** `adata.raw`.
- `Xi` tie-breaking RNG is unseeded → set `np.random.seed(...)` for reproducibility.

### Footprint note
This is the first comparison tool **co-located** in an existing env rather than getting its own
(cellot/cellflow/cpa each needed dedicated envs due to hard version conflicts; CINEMA-OT has none).
If cospar is ever rebuilt, re-run the two install steps above to restore cinemaot.
