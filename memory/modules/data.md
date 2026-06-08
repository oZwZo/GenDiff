# Module: data (datasets, neighbour sampling, preprocessing)
> Files: src/_reader.py, src/_pp_fun.py, src/_sc_explore_fn.py, src/_configure.py  |  Part of: [codebase.md](../codebase.md)

## Purpose
Turns a single-cell `AnnData` (genes × cells, with a kNN graph, pseudotime, and perturbation
labels) into PyTorch training pairs. The central idea: for a cell `i` with condition `c` and
pseudotime `t`, **find a "next" cell along the differentiation trajectory** (same condition,
later pseudotime, graph-near) and emit the displacement **ΔX = x_next − x_i** as the training
target. Different dataset classes implement different neighbour-search strategies. Also owns
the YAML→kwargs config bridge and light preprocessing helpers.

## Files
| File | Role |
|------|------|
| src/_reader.py | All `torch.utils.data.Dataset` classes + token/condition helpers |
| src/_configure.py | `Yaml_configurer`: YAML → `dataset_kwargs`/`epsilon_kwargs`/`sampler_kwargs` |
| src/_pp_fun.py | Preprocessing: token dict, pseudotime discretisation, train/val/test split, root-depth |
| src/_sc_explore_fn.py | Exploration/IO helpers (TF-atlas cell types, reachability, adata→10x export) |

## Classes & Functions
### `src/_reader.py` — datasets (all return a 5-tuple `(X, batch_idx, condition_idx, ΔX/target, t)`)
- **`AnnDataSet`**: base. Selects a `layers`/`obsm`/`X` matrix and a split (`train`/`val`/`test`/`All`),
  densifies sparse matrices, `get_root_cell()` resolves the trajectory root.
- **`Condition_AnnDataSet`** (AnnDataSet): adds perturbation handling. `unique_token_dict`
  maps tokens→indices (**0 = control**); splits multiplexed conditions on `delimiter`, pads to
  `max_multiplexing`. `__getitem__` returns `(X, batch, condition_idx, [], [])`.
- **`Dummy_condition_AnnDataSet`**: returns the condition as a **one-hot** vector (for
  `Epsilon_Categorical`).
- **`Supervised_AnnDataSet`**: target ΔX comes from a precomputed `label_key` (obs/obsm/layer)
  rather than neighbour search; optional `input_rep` (e.g. PCA). Used by the `_Supervised` configs.
- **`Traverse_Dataset`**: neighbour search by **graph random-walk** (`traverse_neighbor`) or
  `expand_neighbor`; target is `raw_X[neighbour] − x`. `_diffuse_neighbor` enforces two passes:
  (1) same condition, (2) larger pseudotime; samples final neighbour by connectivity.
- **`Diffuse_Dataset`** (Condition_AnnDataSet): the main diffusion dataset. Expands kNN to
  `max_degree`, intersects "same condition" ∩ "later time", samples one neighbour weighted by
  distance (`_sample_by_distance`, with `alpha` sharpening). Falls back to root/self when empty.
- **`Path_Diffuse`** (Diffuse_Dataset): trajectory-aware sampler used by **Perturb-kinetics /
  PathSampler** configs. `_neighbor_path` classifies each cell into one of six scenarios
  (around_root / orphan / sample-backward-by-condition / sample-forward-by-time /
  single-future-state / condition+time) using Dijkstra distance & depth; sets ΔX and `t`
  accordingly. Has `alpha`, `repeat` knobs.
- **`Root_Diffuse`** (Diffuse_Dataset): ΔX = x_i − root_x (displacement straight from the
  trajectory root); `t` = `Depth_from_root`. Used by **RootSampler** configs.
- **`Fix_Degree_Diffuse`** (Diffuse_Dataset): WIP (calls `self.super()`, references undefined
  vars) — divides ΔX by graph degree.
- **`ODE_dataset`** (Diffuse_Dataset): returns `(X_t0, t0, batch, c, X_t1, t1, pass1, pass2)`
  (8-tuple) with coarse `time_slot` bins, for `ODE_learner`.
- Helpers: `demultiplex_token`, `get_embedding_index`.

### `src/_configure.py`
- **`Yaml_configurer`**: loads a YAML and sets each key as an attribute. Properties build the
  exact kwargs each component needs, dispatching on `dataset_class`/`epsilon_class`/`sampler_class`:
  `dataset_kwargs`, `epsilon_kwargs`, `sampler_kwargs`. This is the contract between YAML and code.

### `src/_pp_fun.py`
- `token_labels()` (TF-atlas token dict from `tokens.txt`), `discrete_time()` (pseudotime×250 → int
  step, with jitter), `random_split()` (80/10/10 into `obs['split']`),
  `infer_knn_depth()` (Dijkstra depth from `iroot` → `obs['Depth_from_root']`).

### `src/_sc_explore_fn.py`
- `TF_atlas_useful_celltype` list; `hightlightcell`, RGB helpers, `TFAtlas_anno_control`,
  `find_unreachable` (cells disconnected from root), `convert_adata_to_10x` (export for Seurat).

## Required AnnData contents (per config)
- `obs[condition_key]` (e.g. `TF`, `target_genes`) — perturbation label string.
- `obs[pseudotime_key]` / `obs['discrete_time']` — pseudotime; `obs['Depth_from_root']` for root samplers.
- `obsp['...connectivities']`, `obsp['...distances']` — kNN graph.
- `uns['iroot']`, `uns[unique_token_dict]` — root cell index and token→index map.

## Dependencies on Other Modules
- Consumed by `script/main_train_with_PL_trainer.dl_from_config` (builds train/val/test loaders).
- 5-tuple output order must match the learner's `training_step` unpacking ([diffusion.md](diffusion.md)).

## Module-Level Gotchas
- Variable-length returns: most datasets emit 5-tuples, `ODE_dataset` emits 8, `check_samples=True`
  emits extra debug fields — `dl_from_config.my_collate` filters/aligns these.
- ΔX **sign matters** and is computed differently per class (neighbour−cell vs cell−root); a flagged
  "# flip !!!" comment in `Path_Diffuse` shows this was iterated on.
- `Fix_Degree_Diffuse`, `Diffuse_Dataset._diffuse_keepi`, and `Latent_Interaction` contain
  unfinished/buggy code paths — verify before reuse.
- Neighbour search reads `raw` (unsplit) graph/labels so trajectories can cross split boundaries.
