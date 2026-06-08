# Module: tutorials / notebooks (the analyses performed)
> Path: some_tutorials/, validation/  |  Part of: [codebase.md](../codebase.md)

## Purpose
The notebooks are the actual research record: data preprocessing, training drivers, and all
downstream analysis/figures per dataset. They consume `src/` + `script/`. Grouped by dataset.

## By dataset
### `some_tutorials/TF-atlas/` (20 nbs — the main study)
- `3.preprocess_subTF_fortraining.ipynb`, `4.TF_diff_newiroot.ipynb` — build training AnnData, set root.
- `5.visualize_model_on_diff.ipynb`, `6.TF_sub_embedding.ipynb`, `6.GenDiff_representation.ipynb` —
  latent representation & embedding analysis.
- `7.Course_annotaion.ipynb`, `Differentiated.ipynb` — cell-type annotation.
- `8.Try_CellRank.ipynb`, `5.1.GenDiff+Cellrank.ipynb`, `5.infer_trajectory.ipynb` —
  trajectory inference / fate probabilities via CellRank.
- `9.Perturbation_velo.ipynb`, `5.2Perturbation.ipynb` — in-silico TF perturbation → velocity.
- `quick_train/` — `1.linear_model`, `2.quick_train`, `3.quick_train_PCA`, `4.evaluate_in_testset`,
  plus `quick_train_CAE.py`, `quick_train_reconX.py`, `fit_linear_each_gene.py` (per-gene linear prior).
- Integration helpers: `Harmony_integration.py`, `common_gene_int.py`, `Read_Fetal.ipynb`, `TF_fetal_harmony.ipynb`.

### `some_tutorials/Barcodelet/` (mESC mesendoderm, 5 nbs)
`1.sample_DeltaX` · `2.quick_train` · `3.quick_train_AE` · `4.get_output_prior` ·
`5.Figure2a-c.ipynb` (paper figure).

### `some_tutorials/Perturb_sci/` (nascent-RNA perturbation, 4 nbs)
`1.read_nascent_data` · `2.preprocess_multimodal` · `2-1.preprocess_fewer_HVG` · `3.sampling`.

### `some_tutorials/Cao_Fetal_ref/` (human fetal reference)
`1.Integration_with_TFatlas` · `2.convert_int_h5ad_2_robject` — annotate TF-Atlas against a fetal atlas.

### `some_tutorials/collide-seq/`
`1.preprocess_afmos_fortraining`.

### `some_tutorials/DDPM_tutorial/`
`annotated_diffusion.ipynb` + `my_implemented.ipynb` — educational DDPM reference (not project data).

### `some_tutorials/test_notebook/` (9 nbs — evaluation/dev)
`evaluate_diffrecon`, `evaluate_Equivalent_model`, `test_epsilon_sampler`, `dataset_sigma`,
`Equivalent_mesc`, `Qsample_BarletSeq`, `norman_normalize`, `zebrafish` (extra datasets explored).

### `validation/validation.ipynb`
Standalone validation/sanity checks.

## Module-Level Gotchas
- Notebooks reference `machine_config.json` paths and saved checkpoints under `pth_dir` — none
  are in the repo; outputs won't reproduce without the data + trained models.
- Numbering (1→9) is the intended reading order within each dataset folder.
- `test_notebook/PATH.py` is a local copy of `src/PATH.py` so notebooks can import without install.
