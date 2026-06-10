# gendiff_dev — four-layer model-development package

Implements the reshape in `GenDiff-manuscript/MODEL_DEV_ARCHITECTURE.md`. Each sub-direction owns one
layer behind a frozen contract; a new direction enters as a **registry entry + a config**, not an edit to
anyone else's code. Additive and non-breaking: `trajectory_model/` and `auto/` keep working unchanged.

```
gendiff_dev/
  data/registry.py        dataset name -> spec            (tfatlas_velo, tfatlas_1m2, barrnaseq)
  targets/registry.py     target name -> versioned .npy + manifest   (r100, r3, r3_euclid_m30,
            builders.py     the consolidated neighbour sampler         real_velocity, zero_aug)
  models/registry.py      predictor name -> predictor     (identity_null, condition_mean, ridge,
                            + bridge to trajectory_model/baselines.py zoo)
  eval/panel.py           ONE metric panel: L1/L2/L3 + composite + pt_drift + rev_collin
  experiments/runner.py   (dataset, target, model) -> results
  results/store.py        long format: (dataset, target, model, metric, value, stratum)
```

## Run an experiment
```bash
cd /rds/user/wz369/hpc-work/GenDiff          # so `import gendiff_dev` resolves
python -m gendiff_dev.experiments.runner --dataset barrnaseq --target r3 --model ridge --trajectory
```
The open §5 question — does `real_velocity` or `r3` beat `r100`? — is a 3-row sweep:
```bash
for t in r100 r3 real_velocity; do
  python -m gendiff_dev.experiments.runner --dataset tfatlas_1m2 --target $t --model ridge --trajectory
done
# compare in results/runs.csv
```

## Contracts (the frozen interfaces)
- **dataset spec**: `path, n_genes, condition_key, pseudotime_key, velocity_key, moments_key,
  celltype_key, use_rep, split_key, control_values`.
- **target**: `ndarray (n_obs, n_genes)` + manifest `{builder, params, source_hash, revcorr, ...}`;
  **never mutated in place** — a new sampler ships a new name (dissolves collision C2).
- **predictor**: `fit(X_gene, dx, cond) -> self`, `predict_delta(X_gene, cond) -> (n, G)`,
  optional `predict_population(adata, condition) -> (m, d)`.
- **eval**: population-vs-observed only (never vs the sampled target); every trajectory claim reports
  `pt_drift` and `rev_collin` beside any CBDir/overshoot (reversion confound handled identically).

## Add a …
- **dataset**: one entry in `data/registry.py::SPECS`.
- **target**: one builder in `targets/builders.py` + one branch in `targets/registry.py::_build`.
- **model**: one class with the predictor contract in `models/registry.py` (or register in
  `baselines.REGISTRY` and reach it via the bridge).
- **metric**: one function in `eval/panel.py` (two-reviewer per CODEOWNERS).

## Migration status (vs §7)
- **Done**: eval panel unified (L1/L2/L3 + pt_drift/rev_collin/revcorr/velo_dir/composite); data registry
  (3 datasets); target registry + consolidated sampler (r100/r3/r3_euclid_m30/real_velocity/zero_aug,
  versioned artifacts); native predictor contract + 3 reference models; runner over (dataset,target,model);
  long-format results store.
- **Incremental TODO**: (§7.3) migrate each `track_*.py` / `baselines.py` predictor to the native contract
  (the `BaselinesAdapter` bridges them now; validate parity vs `run_scorecard.py` per §7.5);
  (§7.5) re-run the terminal-location scorecard through the runner to confirm parity, then run the
  `target ∈ {r100, r3, real_velocity}` sweep; (§7.6) once `auto/`'s unique logic lives here, retire the
  `sampler-auto` worktree. `pt_drift` needs scvelo+cellrank (degrades to NaN otherwise).

## Provenance
- `eval/panel.py` L1/L2/L3 ported from `trajectory_model/metrics.py`; confound/composite/pt_drift from
  `auto/traj_eval.py` + `auto/sweep.py`.
- `targets/builders.py::knn_sampler` consolidates `auto/sweep.py::draw` + `auto/tf_sweep.py::draw`.
- `models/registry.py` bridges `trajectory_model/baselines.py::REGISTRY` (+ `track_*.py`).
