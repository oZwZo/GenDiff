# GenDiff — Diffusion Modeling on the Differentiation Landscape

<img src="difusoin_waddington.jpg"  width='700'>

GenDiff learns a **displacement field** over single-cell state space: for each cell it predicts ΔX,
the change in expression that moves the cell forward along its differentiation trajectory under a given
perturbation (transcription factor, drug, knockout). Training supervision is built by a neighbour
sampler that, for every cell, looks at slightly-more-differentiated neighbours and records the
expression difference. At inference the learned net maps `(expression, condition, pseudotime) → ΔX` in a
single forward pass, so you can read off the predicted direction of motion for any cell under any
condition, integrate it into a trajectory, or compare it against RNA velocity.

The recommended way to use GenDiff is the scvi-tools-style high-level API in the `gendiff` package.

## Installation

Requirements: Python ≥ 3.9, and a CUDA GPU for training (preprocessing and inference run on CPU).
Dependencies are declared in `setup.py`: PyTorch, PyTorch-Lightning, scanpy, anndata,
numpy/scipy/scikit-learn, pandas, matplotlib. `pip install -e .` registers the `gendiff`, `gendiff_dev`,
and `src` packages so `import gendiff` resolves from any directory.

**Option A — existing cluster environment (recommended on the HPC).** The `GenDiff_env` mamba env already
has the scientific stack and a CUDA-matched PyTorch; just register the package without touching the
curated dependencies:

```bash
cd /rds/user/wz369/hpc-work/GenDiff
/rds/user/wz369/hpc-work/LIBS/mamba/envs/GenDiff_env/bin/pip install -e . --no-deps
```

**Option B — fresh install with pip + venv.** Install a PyTorch build matching your CUDA *first*, so the
editable install does not pull a mismatched wheel:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or your CUDA; omit URL for CPU
cd /rds/user/wz369/hpc-work/GenDiff
pip install -e .                                                       # remaining deps from setup.py
```

**Option C — fresh install with uv** (same idea, faster resolver):

```bash
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu126   # or your CUDA; omit URL for CPU
cd /rds/user/wz369/hpc-work/GenDiff
uv pip install -e .
```

Verify the install:

```bash
python -c "import gendiff; print(gendiff.GenDiff)"
```

## Quickstart

```python
from gendiff import GenDiff

# 1. register the data and build the ΔX target (runs the knn sampler once)
GenDiff.setup_anndata(
    adata,
    condition_key="condition",        # obs column: perturbation label
    pseudotime_key="dpt_pseudotime",  # obs column: differentiation axis (control sits low)
    use_rep="X_pca",                  # obsm key: representation for the kNN graph
    control="ctrl",                   # the baseline label -> token 0
    sampler_config="config1",         # geodesic, M=15, repeat=3 (recommended)
)

# 2. build and train
model = GenDiff(adata, hidden_size=(512, 512, 256, 512, 512))
model.train(max_epochs=200, batch_size=128, lr=1e-3)

# 3. predict and persist
dX  = model.predict_delta(adata, condition="Sox2")        # (n_obs, n_genes) displacement field
Xp  = model.predict_population(adata, condition="Sox2")   # X + ΔX
model.save("runs/sox2")
model = GenDiff.load("runs/sox2", adata)
```

## What GenDiff needs from your AnnData

`setup_anndata` checks for four things. The tutorial notebook shows how to compute each if missing.

| Requirement | Where | Notes |
|---|---|---|
| Normalized log-expression | `adata.X` | This is both the model input and the space ΔX lives in. `normalize_total` + `log1p`. |
| Low-dim representation | `adata.obsm[use_rep]` | Used to build the kNN graph for the sampler. `X_pca`, or a batch-corrected `X_pca_harmony`. |
| Pseudotime | `adata.obs[pseudotime_key]` | The differentiation axis; controls should sit low. `sc.tl.dpt` with a control cell as root. |
| Condition column | `adata.obs[condition_key]` | Perturbation labels; one of them is the control (`control=`, inferred if omitted). |

## Sampler configs

The ΔX supervision is built once inside `setup_anndata` by the revised `knn_sampler`. Pick the
parameter combination with `sampler_config`:

| `sampler_config` | metric | M | repeat | meaning |
|---|---|---|---|---|
| `"config1"` (default) | geodesic | 15 | 3 | the recommended r3 setting |
| `"config2"` | euclid | 30 | 3 | metric-tie reference (r3_euclid_m30) |
| `"others"` | your choice | your choice | your choice | pass `sampler_kwargs={"metric":..., "M":..., "repeat":...}` |

`same_condition` (whether neighbours must share the perturbation) defaults to `"auto"`: it is set True
when conditions are densely sampled (median ≥ 50 cells/condition) and False otherwise. The sampler
prints the exact settings it runs, and how many cells had no eligible higher-pseudotime neighbour
(those get ΔX = 0).

## API reference (the parts you call)

- `GenDiff.setup_anndata(adata, *, condition_key, pseudotime_key, use_rep, control=None, layer="X",
  split_key=None, batch_key=None, sampler_config="config1", same_condition="auto",
  sampler_kwargs=None, target_obsm="gendiff_dX", rebuild=False, ...)` — validates fields, builds the
  ΔX target into `adata.obsm["gendiff_dX"]`, writes `adata.obs["discrete_time"]`, a `gendiff_split`
  column (if `split_key` is None), and registers everything under `adata.uns["gendiff_setup"]` /
  `adata.uns["gendiff_token_dict"]`. Re-run with `rebuild=True` to recompute the target.
- `GenDiff(adata, *, hidden_size=(512,512,512,256,512,512,512), condition_emb_dim=256, time_emb_dim=64,
  timesteps=200, loss_type="huber", ...)` — builds the epsilon net + diffusion learner. `condition_emb_dim`
  should appear in `hidden_size` (it becomes the latent layer) or it is inserted with a warning.
- `model.train(max_epochs=200, batch_size=128, lr=1e-3, accelerator="auto", devices=1, patience=50,
  early_stopping=True, default_root_dir=None, **trainer_kwargs)` — wraps a PyTorch-Lightning Trainer,
  monitors `val_loss`, checkpoints the best epoch.
- `model.predict_delta(adata=None, condition=None, *, batch_size=1024, device=None)` — one forward pass
  per cell; returns the (n_obs, n_genes) ΔX field aligned to obs order. `condition=None` uses the
  control (token 0).
- `model.predict_population(adata=None, condition=None, ...)` — `X + predict_delta(...)`.
- `model.save(dir)` / `GenDiff.load(dir, adata)` — `model.pt` (state dict) + `attr.json`
  (setup, init params, token dict).

## Tutorials

- [`tutorials/1_preprocessing_and_training.ipynb`](tutorials/1_preprocessing_and_training.ipynb) — from a
  raw-ish AnnData to a trained, saved model: the four preprocessing requirements, `setup_anndata`,
  inspecting the built ΔX target, and training.
- [`tutorials/2_downstream_analysis.ipynb`](tutorials/2_downstream_analysis.ipynb) — load a trained model
  and analyze it: per-condition ΔX and predicted populations, top moving genes, displacement vs
  pseudotime, plotting the field on the embedding, integrating trajectories, and comparing to RNA
  velocity.

The notebooks are runnable templates; they have not been executed here (training needs a GPU and the
full datasets). Expect a 2k-cell dataset to train in a few minutes on one GPU.

## How the code is organized

- `gendiff/` — the high-level API (`GenDiff` facade). The only package most users touch.
- `src/` — the model internals: epsilon net (`_epsilon_module.py`), diffusion learner (`_learner.py`),
  datasets (`_reader.py`, including the new `Precomputed_Delta_Dataset` that serves the precomputed ΔX).
- `gendiff_dev/` — the sampler and the development/benchmark registries
  (`targets/builders.py::knn_sampler`, the dataset and target registries).

The old on-the-fly `Path_Diffuse` sampler is kept in `src/_reader.py` but **deprecated**; it is only for
reproducing the manuscript. New training should go through `gendiff.GenDiff`, which precomputes ΔX with
the knn sampler and is more robust to the choice of k and traversal depth.
