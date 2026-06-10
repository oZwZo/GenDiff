"""
gendiff_dev.experiments.runner — one runner for the whole package (architecture §5 / §7.4).

An experiment is a tuple (dataset, target, model[, split, eval]) resolved from the registries. This
generalises trajectory_model/run_scorecard.py (`--predictor` only) to `--dataset --target --model`, so the
open question "does training on real_velocity or r3 beat r100?" is a 3-row sweep, not a code merge.

Results land in ONE long-format store: (dataset, target, model, metric, value, stratum), so every
direction's numbers are directly comparable (architecture §6.6).

CLI:  python -m gendiff_dev.experiments.runner --dataset barrnaseq --target r3 --model ridge --trajectory
"""
from __future__ import annotations
import argparse, numpy as np, scipy.sparse as sp
from ..data import registry as data_reg
from ..targets import registry as tgt_reg
from ..models import registry as model_reg
from ..eval import panel
from ..results import store


def _dense(X):
    return (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)


def run(dataset, target, model, *, max_conditions=30, k_rollout=10, trajectory=False,
        out=None, seed=0, adata=None):
    spec = data_reg.get(dataset)
    if adata is None: adata = data_reg.load(dataset)
    Xg = _dense(adata.X)
    cond = adata.obs[spec["condition_key"]].astype(str).to_numpy()
    pt = np.asarray(adata.obs[spec["pseudotime_key"]], float)
    split = adata.obs["split"].astype(str).to_numpy()
    tr = split != "test"; te = split == "test"
    ctrl = data_reg.control_mask(adata, dataset)

    tgt, manifest = tgt_reg.get(dataset, target, adata=adata)
    pred = model_reg.get(model)
    pred.fit(Xg[tr], tgt[tr], cond[tr])

    rows = []
    def add(metric, value, stratum="all"):
        if value is None or (isinstance(value, float) and np.isnan(value)): return
        rows.append(dict(dataset=dataset, target=target, model=model,
                         metric=metric, value=round(float(value), 6), stratum=stratum))

    # ---- per-cell predicted field (delta models) -> condition-specificity + trajectory ----
    V = None
    try:
        V = pred.predict_delta(Xg, cond)
    except NotImplementedError:
        pass
    if V is not None:
        t_c, deg, _ = panel.condition_effects(Xg, cond, ctrl)
        comp = panel.composite_delta_specificity(V, np.where(te)[0], cond, t_c, deg)
        for k in ("D_pear", "D_dir", "PDS"): add(f"composite.{k}", comp[k])
        add("composite.revcorr", panel.revcorr(V, Xg, stride=max(1, adata.n_obs // 2000)))
        if trajectory:
            adj = panel.knn_adjacency(adata.obsm[spec["use_rep"]], k=15)
            grp = adata.obs[spec["celltype_key"]].astype(str).to_numpy()
            add("traj.rev_collin", panel.rev_collin(Xg, pt, adj))
            vd = panel.velo_dir(V, Xg, pt, adj, grp)
            add("traj.CBDir", vd["CBDir"]); add("traj.ICCoh", vd["ICCoh"])
            d = panel.pt_drift(adata, V, spec["pseudotime_key"], use_rep=spec["use_rep"],
                               xkey=spec.get("moments_key") or "_expr", group_key=spec["celltype_key"])
            add("traj.pt_drift", d.get("pt_drift")); add("traj.pt_fwd_frac", d.get("pt_fwd_frac"))

    # ---- per-condition population scorecard (L1 gene-space + L2b DEG + L3 overshoot) ----
    rng = np.random.default_rng(seed)
    conds = [c for c in np.unique(cond) if not data_reg.control_mask(adata, dataset)[cond == c].all()
             and (cond == c).sum() >= 10 and c not in (spec.get("control_values") or [])]
    if len(conds) > max_conditions:
        conds = list(rng.choice(conds, max_conditions, replace=False))
    ctrl_pb = Xg[ctrl].mean(0)
    from sklearn.neighbors import NearestNeighbors
    repm = np.asarray(adata.obsm[spec["use_rep"]])
    nn_pt = NearestNeighbors(n_neighbors=15).fit(repm)
    nn_gene = NearestNeighbors(n_neighbors=1).fit(Xg)   # map rolled gene-space endpoints -> use_rep, fit once
    qhi = np.quantile(pt, 0.8); qlo = np.quantile(pt, 0.2)
    L = {"delta_pearson": [], "delta_sign": [], "deg_f1": [], "edist": [], "overshoot_signed": []}
    for c in conds:
        m = cond == c
        early = np.where(m & (pt <= qlo))[0];  obs_t = np.where(m & (pt >= qhi))[0]
        if early.size < 3 or obs_t.size < 3: continue
        if V is not None:                                   # Euler rollout of early cells
            xp = Xg[early].copy()
            for _ in range(k_rollout):
                xp = xp + pred.predict_delta(xp, np.array([c] * len(xp)))
        else:
            xp = Xg[obs_t]                                  # population model fallback (no delta)
        pred_pb = xp.mean(0); obs_pb = Xg[obs_t].mean(0)
        L["delta_pearson"].append(panel.delta_pearson(pred_pb, obs_pb, ctrl_pb))
        L["delta_sign"].append(panel.delta_sign_accuracy(pred_pb, obs_pb, ctrl_pb))
        L["deg_f1"].append(panel.topk_deg_overlap(pred_pb, obs_pb, ctrl_pb)["f1"])
        a = xp[:200]; b = Xg[obs_t][:200]
        L["edist"].append(panel.energy_distance(a, b))
        # endpoint pseudotime: map rolled endpoints to use_rep via gene-space NN -> overshoot vs terminal
        rep_xp = repm[early] if V is None else repm[nn_gene.kneighbors(xp, return_distance=False)[:, 0]]
        _, ii = nn_pt.kneighbors(rep_xp)
        L["overshoot_signed"].append(panel.pseudotime_overshoot(pt[ii].mean(1), pt[obs_t])["signed_mean"])
    add("L2b.delta_pearson", np.nanmean(L["delta_pearson"]) if L["delta_pearson"] else None)
    add("L2b.delta_sign_accuracy", np.nanmean(L["delta_sign"]) if L["delta_sign"] else None)
    add("L2b.topk_deg_f1", np.nanmean(L["deg_f1"]) if L["deg_f1"] else None)
    add("L1.energy_distance", np.nanmean(L["edist"]) if L["edist"] else None)
    add("L3.pseudotime_overshoot_signed", np.nanmean(L["overshoot_signed"]) if L["overshoot_signed"] else None)

    store.append(rows, out=out)
    return rows


def main():
    ap = argparse.ArgumentParser(description="gendiff_dev experiment runner (dataset, target, model)")
    ap.add_argument("--dataset", required=True, choices=data_reg.list_datasets())
    ap.add_argument("--target", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_conditions", type=int, default=30)
    ap.add_argument("--k_rollout", type=int, default=10)
    ap.add_argument("--trajectory", action="store_true", help="compute pt_drift + confound panel (needs scvelo/cellrank)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = run(a.dataset, a.target, a.model, max_conditions=a.max_conditions,
               k_rollout=a.k_rollout, trajectory=a.trajectory, out=a.out)
    print(f"[runner] {a.dataset}/{a.target}/{a.model}: {len(rows)} metrics ->")
    for r in rows: print(f"  {r['metric']:34s} {r['value']:>10}  ({r['stratum']})")


if __name__ == "__main__":
    main()
