# Module: configs (experiment registry = datasets × methods)
> Path: configs/  |  Part of: [codebase.md](../codebase.md)

## Purpose
Every experiment is one YAML file. The directory structure encodes **which dataset** and
**which method variant** was run; the file is consumed by `Yaml_configurer` ([data.md](data.md))
and dispatched by `main_train_with_PL_trainer` ([scripts.md](scripts.md)). Reading the config
tree is the fastest way to see what has actually been tried.

## YAML schema (see configs/template.yaml)
`epsilon_class` · `sampler_class` · `dataset_class` (the three `eval`'d class names) ·
`anndata_path` · `gene_dim` · `n_base_perturbs` (# perturbation tokens, incl. control) ·
`condition_emb_dim` · `hidden_size` (list → depth) · `condition_key` · `pseudotime_key` ·
`max_degree` · `search_strategy` · `n_neighbor` · `alpha`/`repeat` (Path samplers) ·
`scheduler`/`timesteps`/`beta_start`/`beta_end`/`loss_type` · `kl_weight` (CVAE) · `lr`.

## Experiment groups (datasets)
| Dir | Dataset / domain | Typical classes | Notes |
|-----|------------------|-----------------|-------|
| `TF_atlas/` | **TF-Atlas** (Joung GSE217460, hESC TF-overexpression atlas; gene_dim 4806, 3551 TFs) | `Epsilon_Linear` + `Equivalent_Diffuse_Sampler` + `Diffuse_Dataset`/Path/Root/Traverse | Primary dataset; many depth/width/sampler variants |
| `TF_atlas_AE/` | TF-Atlas | `Epsilon_CAE`/`Epsilon_CVAE` + `AE_learner` | Autoencoder/VAE conditioning, KL-weight sweeps |
| `TF_atlas_Supervised/` | TF-Atlas | `Epsilon_*` + supervised + `Supervised_AnnDataSet` | ΔX from precomputed labels; regularisation sweeps |
| `TF_atlas_Dummy_condition/` | TF-Atlas | `Epsilon_Categorical` (one-hot) | One-hot conditioning baseline |
| `Perturb-kinetics/` | **Perturb-sci** nascent-RNA perturbation (condition `target_genes`, gene_dim 4806) | Path/Diffuse samplers, `Epsilon_Linear` | `a3r10` = alpha 3, repeat 10 |
| `Barcodelet/` | **mESC → mesendoderm** lineage-barcoded differentiation (mouse) | `Epsilon_Linear`, traverse | 3/5-layer, k=15 variants |
| `Collide-seq/` | Collide-seq (afmos) perturbation | Traversal | preprocessing in notebooks |
| `ODE_learner/` | (method study) | `ODE_eps` + `ODE_learner` + `ODE_dataset` | Neural-ODE formulation |
| `GSM_eps_linear/` | (method study) | linear ε | debug/linear scheduler |

## Dependencies on Other Modules
- Pure data files. Read by `Yaml_configurer`; class strings must match `src/` classes exactly.

## Module-Level Gotchas
- `anndata_path` values are **absolute paths to an old machine** (`/home/wergillius/Project/...`)
  — must be repointed (and `machine_config.json` recreated) on a new host.
- The number of variants ≫ the number of distinct ideas; filenames encode the knob being swept
  (e.g. `H256_d3` = hidden 256 / depth 3, `kl1e-4`, `a3r10`, `depth33`/`depth44`, `Neg`/`Pos`).
- `configs/template.yaml` has a duplicated YAML header (two `---` blocks) — copy with care.
