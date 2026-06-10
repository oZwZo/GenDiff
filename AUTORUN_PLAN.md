# Autonomous run plan — improved ΔX sampler (worktree: sampler-auto)

> Window ~7.5 h from 2026-06-09. User offline. Worktree isolates code edits from a parallel Claude
> session on the main tree; merge next day. Data referenced by absolute path
> (/rds/user/wz369/hpc-work/GenDiff/data/integrated_mesc_group0_Nov7.h5ad). Env: GenDiff_env.

## Aligned decisions
- Trajectory stop-criterion metrics: **velocity-direction correctness** + **terminal-state / branch recovery**.
- Models: **lite** velocity model for the sweep; **full GenDiff diffusion pipeline** (Epsilon_Linear +
  Equivalent_Diffuse_Sampler) for the final head-to-head.
- Compute: **batch A100 jobs, ≤4 concurrent** (gottgens-sl2-gpu/ampere); session GPU OK for fast lite work.
- Stop when: trajectory metrics BEAT OT-CFM and CellFlow, AND other metrics ≥ near-parity.
- CellFlow installing in cellflow_env; fallback = OT-CFM proxy if blocked.

## Key prior result + CAVEAT
- Round-1 screen: geodesic/Fermat neighbours >> Euclidean on direction (D_pear 0.05→0.23).
- Round-2: **de-reversion** (emit ΔX − (a_g+b_g·X)) sends revcorr→0 and composite to ~0.82/0.93/0.97.
- ⚠ LEAKAGE RISK: with b_g≈−1, de-rev ΔX ≈ x_neighbour; the composite "truth" (meanX_c−ctrl) also encodes
  condition identity, so they correlate near-trivially. MUST verify with (a) a velocity-based truth and
  (b) the downstream trajectory metrics. Do not declare de-rev the winner on the composite alone.

## Phases
- [ ] P0 Infra (session): verify cellflow; build sampler module, metric module (composite + velocity-based
      + trajectory: CBDir/ICCoh velocity-direction, CellRank terminal/branch), SLURM template. Cache geodesic.
- [ ] P1 Lite sweep (session GPU + small batch): parameter grid {neighbour metric, M, repeat, de-rev,
      coupling, α}. Score composite + velocity-based + fast trajectory proxy. Rank; pick 2-3 finalists.
      Adversarially verify the de-rev leakage with a velocity-truth metric.
- [ ] P2 Finalists (batch A100): regenerate target with finalist sampler; train full GenDiff pipeline;
      run scVelo + CellRank trajectory metrics. Baselines: current sampler, OT-CFM, CellFlow.
- [ ] P3 Stop-check + synthesis: trajectory > OT-CFM/CellFlow & others ≥ near-parity? Write up; commit.

## Time budget guard
- Deadline ≈ start+7.5h. If P2 (full pipeline + CellRank) won't finish, fall back to lite-model trajectory
  metrics for the head-to-head and clearly flag. Always leave ~30 min for synthesis + commit.

## State (for recovery if context is summarized)
- Authored & ready: auto/sweep.py (lite sweep, ~25 configs), auto/traj_eval.py (scVelo+CellRank terminal
  recovery + CBDir/ICCoh), auto/job.slurm (A100 batch template, gottgens-sl2-gpu/ampere). Worktree =
  /rds/user/wz369/hpc-work/GenDiff-sampler-auto. PYBIN=/rds/.../LIBS/mamba/envs/GenDiff_env/bin/python.
- NEXT once Bash works: (1) smoke-test `micromamba run -n GenDiff_env python auto/sweep.py 0`; fix bugs;
  (2) run full sweep on session GPU (or `sbatch auto/job.slurm auto/sweep.py all`); (3) pick finalists by
  CBDir/ICCoh (+ composite, leakage-flagged); (4) sbatch traj_eval + full-pipeline + OT-CFM/CellFlow;
  (5) stop-check vs OT-CFM/CellFlow; synthesize; commit on branch sampler-auto.
- Geodesic cache reused from response/R1.3/.Dgeo_cache.npy if present.

## Progress (2026-06-10, Bash recovered)
- BUG FOUND: every obsp graph in the adata is stored FULLY DENSE (avg deg ~5744). This (a) made the
  prototype's "geodesic" = Dijkstra on a fully-connected graph ≈ direct distance, and (b) made `euclid`
  use all cells via two_hop (150B ops -> 10-min timeout). FIX: build genuine sparse PCA/diffmap kNN
  graphs in sweep.py + traj_eval.py; geodesic/Fermat = Dijkstra on the sparse PCA-kNN graph.
- LITE SWEEP DONE (22 configs, auto/sweep_results.csv). On the corrected graph:
  * repeat=3 is the dominant lever: CBDir ~0.79 (r3) vs ~0.69 (r1). r100 over-amplifies reversion
    (revcorr -> -0.8). So r3 ~= r100 on direction at 1/30 the cost. KEY RESULT.
  * geodesic/fermat M15 modestly beat euclid on TARGET delta-direction (tgt_D_pear 0.37-0.39 vs 0.27);
    on the LEARNED field CBDir they're ~tied (~0.79). Large M dilutes specificity.
  * OT coupling: worst direction (CBDir 0.52), best discrimination (PDS 0.28, ICCoh 0.098).
  * de-rev helps only r1, never beats r3. Leakage flag stands.
  * pred_D_pear ~= 0 for ALL configs: lite model does not learn condition-specific delta direction
    regardless of sampler. Sampler gains live in the target; full GenDiff model needed to exploit them.
  * Finalists: geodesic M15 r3, fermat M15 r3.
- FIELDS generated (auto/fields/*_pred.npy, learned lite fields + raw targets):
  geo_m15_r3, fermat_m15_r3, old_r100 (current sampler -> lite), minusX (trivial).
- OT-CFM + 1-step velreg trained on Tr_SampledX_r100 (auto/otcfm_field.py): otcfm val_cos 0.286,
  velreg val_cos 0.671 (direct regression >> OT-CFM, consistent with R1.3). Fields saved.
- IN PROGRESS: auto/run_traj.py -> CBDir/ICCoh + CellRank terminal recovery on all 6 fields
  (auto/traj_results.csv). Then stop-check vs otcfm; update md files; HTML report if promising.

## Errors / notes
| time | note |
|------|------|
| start | Bash classifier (Fable 5) unavailable after model switch; Read/Write work. Authored all scripts; |
|       | retrying Bash via ScheduleWakeup until classifier recovers, then execute. RESOLVED: Bash recovered. |
| 10:00 | All obsp graphs dense -> rebuilt sparse kNN graphs. No cellflow_env exists -> OT-CFM is the FM proxy. |
