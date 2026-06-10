# GenDiff ΔX-sampler improvement — agent handoff

Self-contained handoff for a new agent. Covers findings, exact code paths (new sampler + metrics), data,
how to run, and open items. Worktree: `/rds/user/wz369/hpc-work/GenDiff-sampler-auto` (git branch
`sampler-auto`, uncommitted). Env python: `/rds/user/wz369/hpc-work/LIBS/mamba/envs/GenDiff_env/bin/python`.

## 0. TL;DR findings
- **r100 → r3**: averaging 3 neighbour draws matches/beats the shipped 100-draw target at 1/30 the cost;
  r100 over-amplifies mean reversion. **This is the main, dataset-general recommendation.**
- **Bug fixed (BarRNA-seq)**: all `obsp` graphs in `integrated_mesc_group0_Nov7.h5ad` are stored FULLY DENSE
  (avg deg ~5744). The earlier "geodesic" screen was Dijkstra on a fully-connected graph (= direct distance).
  Scripts now rebuild genuine sparse PCA/diffmap kNN graphs.
- **geodesic/Fermat M15 neighbours** beat euclidean on BarRNA-seq target condition-direction (D_pear 0.39 vs
  0.27) — but this does **NOT** replicate on TF-Atlas (euclid≈geodesic≈Fermat). Dataset-specific.
- **Reversion confound**: in BarRNA-seq the differentiation flow is ~85% collinear with −X (`rev_collin`=0.85),
  so CBDir/ICCoh and all-gene r² are reversion-dominated (trivial −X tops CBDir). On TF-Atlas rev_collin=0.54.
  **Use `pt_drift` (forward motion under the velocity transition matrix), not CBDir / all-gene r², as the
  trajectory metric.** On TF-Atlas the gold real RNA velocity scores CBDir≈0.02 — CBDir is invalid there.
- **Model is the bottleneck for condition-specificity** (BarRNA-seq): a competent model (VNet) fits the
  geodesic target at val cosine 0.908 but still predicts condition direction only at chance.
- **TF-Atlas verdict**: where a real RNA-velocity field already exists, the kNN-displacement sampler does NOT
  beat it (real velocity pt_drift +0.014 / ICCoh 0.82 vs sampler −0.008 / 0.02). The sampler is for datasets
  WITHOUT velocity (BarRNA-seq).

## 1. Read these first (the md / report files to pass on)
| File | What it is |
|------|-----------|
| `/rds/user/wz369/hpc-work/GenDiff-sampler-auto/HANDOFF.md` | THIS file — start here. |
| `/rds/user/wz369/hpc-work/GenDiff-sampler-auto/AUTORUN_PLAN.md` | Run plan + decisions + recovery state. |
| `/rds/user/wz369/hpc-work/GenDiff/GenDiff-manuscript/response/R1.3/sampler_report.html` | Full technical report (background, challenge, metric definitions, BarRNA-seq + TF-Atlas results). |
| `/rds/user/wz369/hpc-work/GenDiff/GenDiff-manuscript/response/R1.3/findings.md` | Condensed findings (R1.3 rebuttal + sampler + TF-Atlas sections). |
| `/rds/user/wz369/hpc-work/GenDiff/GenDiff-manuscript/response/R1.3/progress.md` | Dated progress log (full chronology incl. bugs/fixes). |
| `/rds/user/wz369/hpc-work/GenDiff/GenDiff-manuscript/response/R1.3/task_plan.md` | Task plan + phases + errors table + OPEN items (S5). |
| `/rds/user/wz369/hpc-work/GenDiff/GenDiff-manuscript/response/R1.3/sampling_metric_design.md` | Why the composite delta-specificity metric was chosen. |

## 2. Data
- BarRNA-seq: `/rds/user/wz369/hpc-work/GenDiff/data/integrated_mesc_group0_Nov7.h5ad`
  (5745×2000, z-scored X, `obs['condition']` 32 cond, `dpt_pseudotime`, 7 pathway one-hot cols, `obs['split']`,
  target `obsm['Tr_SampledX_r100']`; cell_type uniform = mESCs; **obsp graphs are dense — rebuild kNN**).
- TF-Atlas: `/rds/user/wz369/hpc-work/GenDiff/data/210717_TFAtlas1M2_diffed.h5ad`
  (28825×5000, log-norm X, `obs['TF']` 2535 perturbations ~4 cells each, controls GFP/mCherry,
  `velocity_pseudotime`, `obsm['X_pca_harmony']`, real `layers['Ms','velocity']`, `obs['louvain']` 25 clusters;
  no split — created in code; obsp graphs genuinely sparse).

## 3. CODE PATH — the new sampler
All under `/rds/user/wz369/hpc-work/GenDiff-sampler-auto/auto/`.

### BarRNA-seq
- **`sweep.py`** — the core. Importable module + CLI (`python sweep.py all`). Contains:
  - sparse kNN graph build (`_knn_graph`, `geo(power)` Dijkstra) — lines ~47-66.
  - **`draw(metric, M, alpha, repeat)`** — same-condition, higher-pseudotime neighbour sampler (the NEW
    sampler). metric ∈ {euclid, diffmap, geodesic, fermat}. lines ~68-101.
  - `ot_draw` (OT coupling), `target_from_draws` (ΔX = mean_r(x_neighbour) − x_cell).
  - `train_lite` (linear-skip + MLP velocity model), config grid, `run_one`.
- `gen_velocity.py` — builds finalist fields (`geo_m15_r3`, `fermat_m15_r3`, `old_r100`, `minusX`) → `fields/`.
- `otcfm_field.py` — OT-CFM + 1-step velreg baselines (torchcfm) → `fields/otcfm_pred.npy`, `velreg_pred.npy`.
- `train_vnet_on_target.py` — competent VNet on an arbitrary target npy (model-transfer test).

### TF-Atlas (adapted)
- **`tf_sweep.py`** — TF-Atlas core. **`draw(...)` is the ROOT/GLOBAL sampler** (higher-pseudotime neighbours
  from the global pool, condition-agnostic — matches GenDiff `Root_Diffuse`). TF-embedding lite model
  (`Net`). Creates the 80/10/10 split. CLI `python tf_sweep.py all`.
- `tf_gen_velocity.py` — finalist fields + real velocity reference + −X → `tf_fields/`.

## 4. CODE PATH — the metrics
- **Composite delta-specificity** (condition direction): `sweep.py::composite(D, idx)` and
  `eval_composite_fields.py` (runs it on every field in `fields/`). Returns D_pear (Pearson of
  reversion-removed condition signature vs perturbed−control effect on top-50 DEGs), D_dir (sign agreement),
  PDS (argmax discrimination); `revcorr` = reversion level. TF version: `tf_sweep.py::composite` (per-TF, ≥20
  cells, vs GFP/mCherry control).
- **Velocity-direction proxy**: `sweep.py::velo_dir(V, cells)` → CBDir (cosine vs forward expr flow) + ICCoh.
- **Trajectory eval (the arbiter)**:
  - BarRNA-seq: **`traj_eval.py`** — CBDir, ICCoh, **`rev_collin`** (confound diagnostic), **`pt_drift`**
    (E[Δpseudotime] under the CellRank velocity transition matrix — THE valid metric), macrostate↔condition
    ARI. `run_traj.py` runs it over all `fields/`.
  - TF-Atlas: **`tf_traj.py`** — same metrics but branch recovery vs **louvain**, CellRank on real `Ms`
    moments, run on a 6000-cell louvain-stratified subsample (full GPCCA is O(N³)). `tf_run_traj.py` driver.

Metric definitions in prose: section 4.2 of `sampler_report.html`.

## 5. Results files
- BarRNA-seq: `auto/sweep_results.csv`, `auto/traj_results.csv`, `auto/composite_fields.csv`, `auto/fields/*.npy`.
- TF-Atlas: `auto/tf_sweep_results.csv`, `auto/tf_traj_results.csv`, `auto/tf_fields/*.npy`.
- Geodesic caches: `auto/.cache/` (BarRNA-seq), `auto/.tfcache/` (TF-Atlas).

## 6. How to reproduce
```
PY=/rds/user/wz369/hpc-work/LIBS/mamba/envs/GenDiff_env/bin/python
cd /rds/user/wz369/hpc-work/GenDiff-sampler-auto/auto
$PY sweep.py all            # BarRNA-seq sweep      -> sweep_results.csv
$PY gen_velocity.py; $PY otcfm_field.py             # fields/
$PY run_traj.py; $PY eval_composite_fields.py       # traj_results.csv, composite_fields.csv
$PY tf_sweep.py all         # TF-Atlas sweep        -> tf_sweep_results.csv  (geodesic build is slow, ~10min x2)
$PY tf_gen_velocity.py; $PY tf_run_traj.py          # tf_fields/, tf_traj_results.csv
```
GPU recommended (A100 in-session worked). For batch: `sbatch auto/job.slurm auto/<script>.py all`.

## 7. OPEN items (next agent)
1. Confirm **r3** target inside the FULL GenDiff diffusion pipeline (`Epsilon_Linear` +
   `Equivalent_Diffuse_Sampler`, `script/main_train_with_PL_trainer.py`) — current results use lite/VNet stand-ins.
2. Address the **condition-specificity bottleneck** (BarRNA-seq): condition-aware head / residual-on-reversion
   training (the sampler's target signal does not survive learning).
3. Benchmark **CellFlow** (no installable env in the window; OT-CFM was the proxy).
4. **Merge** the `sampler-auto` worktree into the main tree (another Claude session was editing code separately).
