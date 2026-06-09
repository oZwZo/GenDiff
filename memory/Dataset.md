# Datasets: GenDiff `data/`
> Generated: 2026-06-08 | Location: `/rds/user/wz369/hpc-work/GenDiff/data/` (also reachable via `/home/wz369/rds/hpc-work/GenDiff/data/`)
> Inspected with anndata backed mode (`PINN_env`); X matrices not loaded. Part of: [codebase.md](codebase.md)

Four `.h5ad` (AnnData) files, ~12.5 GB total. Two correspond to the manuscript's **TF-Atlas** (`DS:tfatlas`, human hESC TF overexpression) and two to the **BarRNA-seq** mouse germ-layer study (`DS:barrnaseq`, 5 signalling pathways). All carry the kNN graph + pseudotime + perturbation labels that `src/_reader.py` datasets consume (see [modules/data.md](modules/data.md)).

## Summary
| File | Size | Cells × Genes | Species / domain | condition_key | Tokens | Training-ready? | Manuscript DS |
|------|------|---------------|------------------|---------------|--------|-----------------|---------------|
| `differentiated_model_velo.h5ad` | 6.2 GB | 28825 × 4806 | human TF-Atlas (differentiated) | `TF` | uns dict: 3369 | **Yes** (split, discrete_time, token dict) | DS:tfatlas |
| `210717_TFAtlas1M2_diffed.h5ad` | 4.2 GB | 28825 × 5000 | human TF-Atlas + RNA-velocity | `TF` | — (no dict) | No (velocity source) | DS:tfatlas (raw/velo) |
| `integrated_mesc_group0_Nov7.h5ad` | 1.2 GB | 5745 × 2000 | mouse ESC (BarRNA-seq, root = mESC) | `condition` | uns dict: 33 | **Yes** | DS:barrnaseq |
| `integrated_mesendo_group_Nov4.h5ad` | 1.25 GB | 5802 × 2000 | mouse mesendoderm (BarRNA-seq) | `condition` | uns dict: 33 | **Yes** (+ `time_slot` for ODE) | DS:barrnaseq |

## Per-file detail

### `differentiated_model_velo.h5ad` — TF-Atlas, training-ready  ★ primary TF-Atlas file
- 28,825 cells × **4,806 genes** — matches `gene_dim: 4806` in the TF-Atlas configs and the manuscript's "4806 genes".
- `obs`: `TF` (2,535 distinct TF-ORF tokens incl. `ctrl`/GFP), `batch` (0/1), `louvain` (25, incl. `7-1`,`8-0`…), `dpt_pseudotime`, **`discrete_time`** (0–200), **`split`** (train/test/val), `course_louvain` (18), **`course_cell_type`** (17: stromal_1-5, endoderm_1-2, fibroblast_1-3, ectoderm_2-3, …), `Cluster Enriched TFs`, `is_large`, `velocity_self_transition`.
- `layers`: `X`, **`velocity`** (= GenDiff-predicted ΔX used as the velocity matrix for scVelo/CellRank).
- ⚠ **Manuscript training target `a3_PathSampled_X` is NOT in this local file** (its only layers are
  `X`, `velocity`). The quick_train scripts/notebooks read `adata.layers['a3_PathSampled_X']` from the
  old-machine file `…/data/TFAtlas/GSE217460_210322_TFAtlas_differentiated.h5ad` — which is **absent
  here** (no `data/TFAtlas/` dir). Provenance of that layer: it is the **supervised ΔX target**, a
  per-cell `delta_X = X_next − X` emitted by the GenDiff **path sampler** (`Path_Diffuse` in
  `src/_reader.py` — the "Path"; `alpha=3` → the "a3" prefix; an `a2_`/α=2 variant also existed),
  collected and averaged over `repeat` samplings by `src/_diffplot.sample_Delta_X`. It is **not a model
  output** — it is the ground truth that GenDiff and the Ridge/CAE baselines are all trained/scored
  against. No code in this repo writes the key (cf. `_sc_explore_fn.save_pred_sampled_deltaX`, which
  writes a hand-named `obsm[key]`); it was baked into the saved h5ad, so retraining requires
  regenerating it with the path sampler. (BarRNA-seq's analogue is `obsm['Tr_SampledX_r100']` =
  traverse sampler, repeat 100 — see below.)
- `obsm`: `X_pca`, `X_pca_harmony`, `X_umap`, `X_diffmap`, `velocity_umap`. `obsp`: `connectivities`, `distances`.
- `uns`: **`iroot` = 28820**, **`unique_token_dict`** (3,369 entries, token→index, index 0 = control), `neighbors`, `velocity_graph`(+neg), `rank_genes_groups`.
- → Used by `Diffuse_Dataset` / `Path_Diffuse` / `Root_Diffuse` (condition_key `TF`, layers `X`, pseudotime `discrete_time`). This is the file behind Figs 4–6.

### `210717_TFAtlas1M2_diffed.h5ad` — TF-Atlas RNA-velocity source
- 28,825 cells × **5,000** HVGs. `obs`: `TF` (2,535), `batch`, `louvain`, `dpt_pseudotime`, `m3_pseudotime`, `velocity_pseudotime`, `v_pseudotime_2`, spliced/unspliced size factors.
- `layers`: **`spliced`, `unspliced`, `Ms`, `Mu`, `velocity`, `variance_velocity`** — full scVelo dynamical output.
- `var`: velocity_gamma/r2/genes. `uns`: `iroot`=12980, `velocity_params`, `neighbors`. **No `split`, no `discrete_time`, no `unique_token_dict`** → not directly trainable by GenDiff; it is the scVelo-processed source (different gene set: 5000 vs 4806).
- → **Relevant to revision point R2.1** (scVelo benchmark): this is the only file with spliced/unspliced layers, so a real RNA-velocity comparison on the TF-Atlas must start here.

### `integrated_mesc_group0_Nov7.h5ad` — BarRNA-seq, mESC-rooted
- 5,745 cells × 2,000 genes (mouse Ensembl IDs; `var` has `gene_name`/`Gene_Symbol`). `starting.state` = mESCs.
- `obs`: 5 signalling pathways as binary cols **`RA`,`Wnt`,`TgfB`,`Bmp`,`Fgf`** (+ `Notch`,`Shh` ≈ 0 in this group), `assignment` (33), **`condition`** (32 combos like `Bmp+Fgf`,`ctrl`), `dose_val`, `control`, `dpt_pseudotime`, **`discrete_time`** (0–200), **`split`**, `dpt_order`.
- `obsm`: **`Tr_SampledX_r100`** (GenDiff traverse-sampled ΔX, repeat=100), `X_pca`, `X_umap`, `X_diffmap`. `obsp`: `connectivities`,`distances`, **`diff_connectivities`,`diff_distances`** (diffuse-sampler graph).
- `uns`: **`iroot`=4086**, **`unique_token_dict`** (33: `Control`=0 + 32 combos), `diff`, `neighbors`.
- → BarRNA-seq germ-layer dataset (Figs 2–3). condition_key `condition`, max_multiplexing reflects `+`-delimited combos.

### `integrated_mesendo_group_Nov4.h5ad` — BarRNA-seq, mesendoderm-rooted
- 5,802 cells × 2,000 genes. `starting.state` = mesendoderm. Same 5 pathways (+`Notch`,`Shh` varying here), `assignment` (33), `dpt_groups` (4).
- `obs`: `dpt_pseudotime`, **`discrete_time`** (0–200), **`split`**, **`time_slot`** (0/50/100/150/200 — the bins consumed by `ODE_dataset`/`ODE_learner`).
- `obsm`: `X_pca`,`X_umap`,`X_diffmap`. `obsp`: `connectivities`,`distances`,`diff_connectivities`,`diff_distances`. `uns`: **`iroot`=2902**, **`unique_token_dict`** (33), `neighbors`.
- → BarRNA-seq mesendoderm variant; supports the `Barcodelet/mesendo` and `ODE_learner` configs.

## AnnData → dataset contract (cross-check vs [modules/data.md](modules/data.md))
Trainable files satisfy the loader contract: `obs[condition_key]`, `obs['discrete_time']`, `obs['split']`, `obsp['connectivities'|'distances']`, `uns['iroot']`, `uns['unique_token_dict']` (passed via config `unique_token_dict: unique_token_dict`). `Depth_from_root` (needed by `Root_Diffuse`/`Path_Diffuse`) is **not precomputed** in any file — `src/_pp_fun.infer_knn_depth` must be run first for root/path samplers.

## Notes / discrepancies
- **GEO accession:** manuscript `§METH.data` cites TF-Atlas = GSE216481; repo configs reference GSE217460. These local files are named `210717_TFAtlas1M2` / `differentiated_model_velo` (no accession in filename) — confirm provenance before answering reviewer data-availability points.
- **Token/embedding sizes:** `unique_token_dict` here has 3,369 entries; configs set `n_base_perturbs: 3551` (embedding-table size, ≥ tokens); only 2,535 TFs are actually observed among differentiated cells. All three numbers are internally consistent (table ≥ dict ≥ observed) but worth stating precisely.
- **Config paths are stale:** configs point `anndata_path` at `/home/wergillius/Project/diffuse_differentiate/data/...`. To train here, repoint `anndata_path` to these `data/` files (e.g. `differentiated_model_velo.h5ad` for TF-Atlas, `integrated_mesc_group0_Nov7.h5ad` for BarRNA-seq).
- **scVelo benchmark (R2.1):** spliced/unspliced layers exist **only** in `210717_TFAtlas1M2_diffed.h5ad`; the BarRNA-seq files have no spliced/unspliced, so a fair scVelo comparison there is not possible without re-quantification.
- **Manuscript TF-Atlas target layer missing:** `a3_PathSampled_X` (the supervised ΔX target, see the
  `differentiated_model_velo.h5ad` detail above) is **not present** in any local h5ad and the
  notebooks' source file `data/TFAtlas/GSE217460_210322_TFAtlas_differentiated.h5ad` does not exist
  here — regenerate it with the GenDiff path sampler (`Path_Diffuse` → `sample_Delta_X`) before
  retraining/reproducing Figs 4–6. (BarRNA-seq's target `obsm['Tr_SampledX_r100']` **is** present in
  `integrated_mesc_group0_Nov7.h5ad`.)
