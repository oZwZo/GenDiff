# Module: analysis (sampling, perturbation, trajectory & velocity)
> Files: src/_diffplot.py, plus scripts in script/  |  Part of: [codebase.md](../codebase.md)

## Purpose
Everything done with a **trained** model: reload a checkpoint, extract latent
representations, generate ΔX displacement fields, simulate in-silico TF perturbations, turn
displacements into RNA-velocity-style flows, and infer/visualise differentiation trajectories
(often handing off to scVelo and CellRank). This is the "what was analysed" layer.

## Files
| File | Role |
|------|------|
| src/_diffplot.py | Main analysis/plotting API (reload, sample ΔX, perturb, velocity, trajectories) |
| script/sample_delta_x.py | Batch ΔX sampling for a config/checkpoint (CLI) |
| script/perturb.py | Run in-silico perturbation sweeps |
| script/visualize_velocity_perturb.py | Velocity/perturbation visualisation |

## Key Functions — `src/_diffplot.py`
- **`get_ckpt_path(relative_path)`**: resolve a `.ckpt` under `pth_dir/<relative>/checkpoints`.
- **`reload_sampler(yaml_file, ckpt_path, device)`**: rebuild ε-net from YAML and load a
  Lightning checkpoint into the inference sampler.
- **`plot_representation(yaml_path, ckpt_path, use_rep='z_c'|'z'|'c', ...)`**: run the model
  over all cells, store latent in `obsm`, recompute neighbours/UMAP → returns `(adata_zc, model)`.
- **`sample_Delta_X(yaml_path, sampling_repeat=100, ...)`**: Monte-Carlo sample the predicted
  displacement ΔX for every cell (stochastic VAE/diffusion) and average.
- **`perturbation(yaml_path, ckpt_path, perturbation, ...)`**: swap the condition token to a
  chosen TF and predict the resulting ΔX field (counterfactual perturbation).
- **`compute_velocity(adata, velocity_matrix, ...)`**: load ΔX as a velocity layer for scVelo.
- **`extrapolate(T_extrapolate, ...)`**: iterate the displacement field forward/backward to
  roll out a trajectory over multiple steps.
- **`diffuse_cell(annData, Sampler, sampled_time=range(20), ...)`**: run the sampler's reverse
  process on cells and collect the stepwise generated matrices.
- **`get_cell_transitions` / `plot_transit_cells`**: build/plot transition paths between cells.
- **`rank_TF_by_absortion_p(...)`**: rank TFs by CellRank absorption probability into a macrostate.
- Plot/util: `condition_on_umap`, `Xmat_by_time`, `triple_plot`, `merge_with_control`,
  `get_dataloader`, `get_annDataLoader`. Marker-gene dicts `mesc_marker_genes(_thress)` (mouse mesendoderm).

## Internal Data Flow
```
YAML + .ckpt ──reload_sampler──► trained ε-net
   │                                   │
   ├─ plot_representation ──► latent (z/z_c/c) in obsm ──► UMAP / clustering
   ├─ sample_Delta_X / perturbation ──► ΔX matrix ──► compute_velocity ──► scVelo
   └─ extrapolate / diffuse_cell ──► simulated trajectory ──► CellRank (absorption, macrostates)
```

## Dependencies on Other Modules
- Imports `_epsilon_module`, `_reader`, `_sampler`, `_learner`, `_configure`, `PATH` and the
  three config-builder functions from `script/main_train_with_PL_trainer`.
- External: `scvelo`, `cellrank` (via notebooks), `scanpy`, `anndata`, `seaborn`, `matplotlib`.

## Module-Level Gotchas
- `_diffplot` imports `from script.main_train_with_PL_trainer import ...`, so `script/` must be
  on `sys.path` (it appends `PATH.main_dir/script`) and `machine_config.json` must exist.
- Some helpers have stragglers (`save_pred_sampled_deltaX` references an undefined `path`).
- The bulk of the *actual* analysis lives in notebooks — see [tutorials.md](tutorials.md).
