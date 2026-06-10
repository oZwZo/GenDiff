"""
gendiff_dev.eval.panel — the single, unified evaluation panel (architecture §4.4 / §7.1).

This is the shared contract both model-development directions score through. It is the UNION of:
  - terminal-location metrics (ported verbatim from trajectory_model/metrics.py): L1 distribution,
    L2 important-gene, L3 terminal location;
  - sampler-direction additions (ported from auto/traj_eval.py + auto/sweep.py): the reversion-confound
    diagnostics (`rev_collin`, `revcorr`, `variance_ratio`), the valid trajectory axis (`pt_drift`,
    forward motion under the CellRank transition matrix), and condition-specificity
    (`composite_delta_specificity` -> D_pear / D_dir / PDS on reversion-removed residuals).

Two rules both directions already agree on (do not violate):
  1. Score predicted POPULATIONS vs OBSERVED cells, never against the sampled ΔX target (circular).
  2. Every trajectory claim reports `pt_drift` AND `rev_collin` beside any overshoot/CBDir number, so the
     reversion confound is handled identically everywhere. CBDir / per-cell-ΔX accuracy are LOCAL
     diagnostics only (a trivial −X reversion field can top them) — never the headline.

Only numpy / scipy / sklearn / POT(ot) are required. `pt_drift` additionally needs scvelo + cellrank and
degrades to NaN (with a reason) if they are unavailable.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, spearmanr, pearsonr, rankdata


# ===========================================================================================
# L1 — distribution distances (from trajectory_model/metrics.py)
# ===========================================================================================
def energy_distance(A, B, sqeuclidean=True):
    """scPerturb E-distance: 2*mean d(A,B) - mean d(A,A') - mean d(B,B'). Larger = more different."""
    metric = "sqeuclidean" if sqeuclidean else "euclidean"
    dab = cdist(A, B, metric=metric).mean()
    daa = cdist(A, A, metric=metric).mean()
    dbb = cdist(B, B, metric=metric).mean()
    return float(2 * dab - daa - dbb)


def e_test(A, B, n_perm=1000, seed=0, sqeuclidean=True):
    """Permutation significance for the E-distance (scPerturb E-test). Returns (E_obs, p_value)."""
    rng = np.random.default_rng(seed)
    e_obs = energy_distance(A, B, sqeuclidean)
    pooled = np.vstack([A, B]); nA = len(A); count = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        count += (energy_distance(pooled[idx[:nA]], pooled[idx[nA:]], sqeuclidean) >= e_obs)
    return e_obs, float((count + 1) / (n_perm + 1))


def mmd_rbf(A, B, gamma=None):
    """Squared MMD, RBF kernel (median-distance gamma). Sensitive to mean AND variance shift."""
    if gamma is None:
        d = cdist(A[: min(len(A), 500)], B[: min(len(B), 500)], "euclidean")
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        gamma = 1.0 / (2 * med ** 2 + 1e-12)
    Kaa = np.exp(-gamma * cdist(A, A, "sqeuclidean"))
    Kbb = np.exp(-gamma * cdist(B, B, "sqeuclidean"))
    Kab = np.exp(-gamma * cdist(A, B, "sqeuclidean"))
    return float(Kaa.mean() + Kbb.mean() - 2 * Kab.mean())


def sliced_wasserstein(A, B, n_projections=50, seed=0):
    """Sliced-Wasserstein distance (needs POT)."""
    import ot
    return float(ot.sliced_wasserstein_distance(A, B, n_projections=n_projections, seed=seed))


def variance_ratio(pred, obs):
    """trace(cov(pred))/trace(cov(obs)). <1 => variance compression. Also a confound diagnostic."""
    vp = np.trace(np.cov(pred, rowvar=False)); vo = np.trace(np.cov(obs, rowvar=False))
    return float(vp / (vo + 1e-12))


def perturbation_discrimination(pred_pseudobulk, obs_pseudobulk):
    """PDS. dict[condition]->mean expr. PDS_c = 1 - normalized_rank of true c by L1 distance.
    1=perfect, 0.5=random; effect-size independent. Adds key '_mean'."""
    conds = [c for c in pred_pseudobulk if c in obs_pseudobulk]
    obs_mat = np.stack([obs_pseudobulk[c] for c in conds]); out = {}
    for c in conds:
        d = np.abs(obs_mat - pred_pseudobulk[c][None, :]).sum(1)
        r = rankdata(d, method="min")[conds.index(c)] - 1
        out[c] = float(1 - r / (len(conds) - 1)) if len(conds) > 1 else np.nan
    out["_mean"] = float(np.nanmean([v for k, v in out.items() if k != "_mean"]))
    return out


# ===========================================================================================
# L2 — important-gene  (population DEG + local per-cell)  (from metrics.py)
# ===========================================================================================
def _logfc(pert_pb, ctrl_pb):
    return np.asarray(pert_pb, float) - np.asarray(ctrl_pb, float)


def delta_pearson(pred_pb, obs_pb, ctrl_pb, gene_mask=None):
    dp, do = _logfc(pred_pb, ctrl_pb), _logfc(obs_pb, ctrl_pb)
    if gene_mask is not None: dp, do = dp[gene_mask], do[gene_mask]
    return float(pearsonr(dp, do)[0]) if dp.std() > 0 and do.std() > 0 else np.nan


def delta_sign_accuracy(pred_pb, obs_pb, ctrl_pb, gene_mask=None):
    dp, do = _logfc(pred_pb, ctrl_pb), _logfc(obs_pb, ctrl_pb)
    if gene_mask is not None: dp, do = dp[gene_mask], do[gene_mask]
    return float(np.mean(np.sign(dp) == np.sign(do)))


def topk_deg_overlap(pred_pb, obs_pb, ctrl_pb, k=50):
    dp, do = np.abs(_logfc(pred_pb, ctrl_pb)), np.abs(_logfc(obs_pb, ctrl_pb))
    sp, so = set(np.argsort(dp)[-k:]), set(np.argsort(do)[-k:]); inter = len(sp & so)
    prec = inter / k; rec = inter / k
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1, "jaccard": inter / len(sp | so)}


def effect_size_spearman(pred_pb, obs_pb, ctrl_pb, gene_mask=None):
    dp, do = _logfc(pred_pb, ctrl_pb), _logfc(obs_pb, ctrl_pb)
    if gene_mask is not None: dp, do = dp[gene_mask], do[gene_mask]
    return float(spearmanr(dp, do)[0])


def marker_recovery_auroc(pred_endpoint, marker_idx, background):
    from sklearn.metrics import roc_auc_score
    score = pred_endpoint.mean(0) - background.mean(0)
    y = np.zeros(score.shape[0], int); y[list(marker_idx)] = 1
    if y.sum() == 0 or y.sum() == len(y): return np.nan
    return float(roc_auc_score(y, score))


def per_cell_cosine(dx_true, dx_pred, gene_mask=None, eps=1e-8):
    """LOCAL step-direction accuracy (sampler-dependent diagnostic, NOT terminal location)."""
    a = np.asarray(dx_true, float); b = np.asarray(dx_pred, float)
    if gene_mask is not None: a = a[:, gene_mask]; b = b[:, gene_mask]
    num = (a * b).sum(1); den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
    return num / den


def per_cell_corr(dx_true, dx_pred, gene_mask=None):
    a = np.asarray(dx_true, float); b = np.asarray(dx_pred, float)
    if gene_mask is not None: a = a[:, gene_mask]; b = b[:, gene_mask]
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    num = (a * b).sum(1); den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-12
    return num / den


def per_gene_r2(dx_true, dx_pred, gene_mask=None):
    a = np.asarray(dx_true, float); b = np.asarray(dx_pred, float)
    cols = np.arange(a.shape[1]) if gene_mask is None else np.where(np.asarray(gene_mask))[0]
    out = np.full(len(cols), np.nan)
    for k, i in enumerate(cols):
        if a[:, i].std() > 0 and b[:, i].std() > 0:
            out[k] = pearsonr(a[:, i], b[:, i])[0] ** 2
    return out


# ===========================================================================================
# L3 — terminal location  (from metrics.py)  + pt_drift (the valid trajectory axis, from auto/)
# ===========================================================================================
def pseudotime_overshoot(pt_pred, pt_obs_terminal):
    """SIGNED terminal-location error. Returns w1 (magnitude) and signed_mean (+overshoot / -stall)."""
    pt_pred = np.asarray(pt_pred, float); pt_obs = np.asarray(pt_obs_terminal, float)
    return {"w1": float(wasserstein_distance(pt_pred, pt_obs)),
            "signed_mean": float(pt_pred.mean() - pt_obs.mean())}


def celltype_proportion_match(pred_emb, ref_emb, ref_labels, obs_terminal_labels, k=15):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=k).fit(ref_emb, ref_labels)
    pred_lab = clf.predict(pred_emb); cats = sorted(set(ref_labels))
    def prop(labs): return np.array([np.mean(np.asarray(labs) == c) for c in cats])
    p, o = prop(pred_lab), prop(obs_terminal_labels)
    corr = float(pearsonr(p, o)[0]) if p.std() > 0 and o.std() > 0 else np.nan
    return {"tv": float(0.5 * np.abs(p - o).sum()), "corr": corr,
            "pred_prop": dict(zip(cats, p.round(4))), "obs_prop": dict(zip(cats, o.round(4)))}


def pt_drift(adata, V, pseudotime_key, use_rep="X_pca", xkey="Ms",
             n_sub=6000, seed=0, group_key=None, n_neighbors=15):
    """THE valid trajectory metric: expected one-step change in pseudotime under the velocity-derived
    CellRank transition matrix.  pt_drift = mean_i [(T @ pt)_i - pt_i];  >0 = field flows forward
    (differentiation), <0 = backward (de-differentiation / reversion).
    V: (n_obs, n_genes) predicted velocity. For large n_obs the CellRank macrostate step is O(N^3), so
    the transition matrix is built on a deterministic (optionally group-stratified) subsample of n_sub.
    Returns {pt_drift, pt_fwd_frac, n_sub} or {pt_drift: nan, reason: ...} if scvelo/cellrank missing."""
    try:
        import scanpy as sc, scvelo as scv, cellrank as cr, scipy.sparse as sp
    except Exception as e:
        return {"pt_drift": float("nan"), "pt_fwd_frac": float("nan"), "reason": f"no scvelo/cellrank: {e!r}"[:120]}
    pt = np.asarray(adata.obs[pseudotime_key], float)
    rng = np.random.default_rng(seed)
    if adata.n_obs > n_sub:
        if group_key is not None and group_key in adata.obs:
            g = adata.obs[group_key].astype(str).to_numpy(); frac = n_sub / adata.n_obs; keep = []
            for lvl in np.unique(g):
                idx = np.where(g == lvl)[0]; kk = max(2, int(round(len(idx) * frac)))
                keep.append(rng.choice(idx, size=min(kk, len(idx)), replace=False))
            sub = np.sort(np.concatenate(keep))
        else:
            sub = np.sort(rng.choice(adata.n_obs, size=n_sub, replace=False))
    else:
        sub = np.arange(adata.n_obs)
    a = adata[sub].copy(); pts = pt[sub]
    a.layers["velocity"] = np.asarray(V)[sub].astype(np.float32)
    if xkey not in a.layers:                      # datasets without RNA-velocity moments: use expression
        a.layers["_expr"] = (a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)).astype(np.float32)
        xkey = "_expr"
    sc.pp.neighbors(a, n_neighbors=n_neighbors, use_rep=use_rep)
    scv.tl.velocity_graph(a, vkey="velocity", xkey=xkey, n_jobs=1)
    vk = cr.kernels.VelocityKernel(a, vkey="velocity", xkey=xkey); vk.compute_transition_matrix()
    T = vk.transition_matrix; T = T.tocsr() if sp.issparse(T) else sp.csr_matrix(T)
    drift = np.asarray(T.dot(pts)).ravel() - pts
    return {"pt_drift": round(float(drift.mean()), 5),
            "pt_fwd_frac": round(float((drift > 0).mean()), 4), "n_sub": int(len(sub))}


# ===========================================================================================
# Reversion-confound diagnostics + condition-specificity  (from auto/traj_eval.py, auto/sweep.py)
# ===========================================================================================
def knn_adjacency(rep, k=15):
    """List of neighbour index arrays (self excluded) from a representation, e.g. adata.obsm['X_pca']."""
    from sklearn.neighbors import NearestNeighbors
    rep = np.asarray(rep); nn = NearestNeighbors(n_neighbors=k + 1).fit(rep); _, idx = nn.kneighbors(rep)
    return [idx[i][1:] for i in range(rep.shape[0])]


def _pear(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum()); return float((x * y).sum() / d) if d > 0 else np.nan


def revcorr(dx, X, stride=1):
    """Mean per-cell Pearson(ΔX_i, X_i). -1 = pure mean reversion (ΔX ≈ −X). Reversion level."""
    dx = np.asarray(dx, float); X = np.asarray(X, float)
    v = [_pear(dx[i], X[i]) for i in range(0, len(X), stride)
         if np.linalg.norm(dx[i]) > 0 and X[i].std() > 0]
    return round(float(np.nanmean(v)), 4) if v else float("nan")


def rev_collin(X, pt, adj):
    """CONFOUND DIAGNOSTIC: mean cosine(−X_i, local forward expr flow toward higher-pseudotime kNN).
    High (~0.85 BarRNA-seq) => differentiation flow is reversion-like => CBDir / all-gene r2 are
    reversion-dominated and must not be the headline. Lower (~0.54 TF-Atlas) => less confounded."""
    X = np.asarray(X, float); pt = np.asarray(pt, float); out = []
    for i in range(len(X)):
        nb = adj[i]; fwd = nb[pt[nb] > pt[i]]; xi = X[i]; nx = np.linalg.norm(xi)
        if fwd.size and nx > 0:
            u = (X[fwd] - xi).mean(0); nu = np.linalg.norm(u)
            if nu > 0: out.append(float((-xi * u).sum() / (nx * nu)))
    return round(float(np.mean(out)), 4) if out else float("nan")


def velo_dir(V, X, pt, adj, group_labels=None):
    """LOCAL diagnostic (confounded — report only beside pt_drift/rev_collin):
    CBDir = cos(V_i, forward expr flow toward higher-pt kNN); ICCoh = within-group coherence of V."""
    V = np.asarray(V, float); X = np.asarray(X, float); pt = np.asarray(pt, float)
    cb = []; ic = []
    for i in range(len(X)):
        nb = adj[i]; fwd = nb[pt[nb] > pt[i]]; vi = V[i]; nv = np.linalg.norm(vi)
        if fwd.size and nv > 0:
            u = (X[fwd] - X[i]).mean(0); nu = np.linalg.norm(u)
            if nu > 0: cb.append(float((vi * u).sum() / (nv * nu)))
        if group_labels is not None and nv > 0:
            same = nb[group_labels[nb] == group_labels[i]]
            c = [float((vi * V[j]).sum() / (nv * np.linalg.norm(V[j]) + 1e-9))
                 for j in same if np.linalg.norm(V[j]) > 0]
            if c: ic.append(np.mean(c))
    return {"CBDir": round(float(np.mean(cb)), 4) if cb else float("nan"),
            "ICCoh": round(float(np.mean(ic)), 4) if ic else float("nan")}


def condition_effects(X, condition, control_mask, k_deg=50, min_cells=20):
    """Per-condition observed effect t_c = mean_c(X) − mean_control(X) and its top-k DEG indices.
    Returns (t_c: dict[cond]->vec, deg: dict[cond]->idx, conds: list). Control-relative, effect truth."""
    X = np.asarray(X, float); condition = np.asarray(condition).astype(str)
    ctrl = X[np.asarray(control_mask, bool)].mean(0)
    conds = [c for c in np.unique(condition) if (condition == c).sum() >= min_cells]
    t_c = {c: X[condition == c].mean(0) - ctrl for c in conds}
    deg = {c: np.argsort(np.abs(t_c[c]))[::-1][:k_deg] for c in conds}
    return t_c, deg, conds


def composite_delta_specificity(field, idx, condition, t_c, deg, min_cells=15):
    """Condition-specificity of a per-cell field (target or predicted velocity), reversion-removed.
    For each condition c with >=min_cells in `idx`: resid_c = mean_c(field) − global_mean(field);
    compare to t_c on the condition's top DEGs.
      D_pear = mean_c Pearson(resid_c[deg_c], t_c[deg_c])   (+1 right, 0 none, −1 anti)
      D_dir  = mean_c fraction of DEGs with matching sign   (0.5 = chance)
      PDS    = fraction of c whose resid_c best-correlates with its own t_c (argmax discrimination)
    Score the PREDICTED field for a real claim; the target's score is a construction diagnostic."""
    field = np.asarray(field, float); idx = np.asarray(idx); cond = np.asarray(condition).astype(str)
    g = field[idx].mean(0)
    cs = [c for c in t_c if (cond[idx] == c).sum() >= min_cells]
    if len(cs) < 2:
        return {"D_pear": float("nan"), "D_dir": float("nan"), "PDS": float("nan"), "n_cond": len(cs)}
    resid = {c: field[idx[cond[idx] == c]].mean(0) - g for c in cs}
    dps = [_pear(resid[c][deg[c]], t_c[c][deg[c]]) for c in cs]
    dds = [float((np.sign(resid[c][deg[c]]) == np.sign(t_c[c][deg[c]])).mean()) for c in cs]
    hit = 0; valid = 0                                  # PDS over conditions with a defined argmax
    for c in cs:
        corrs = [_pear(resid[c], t_c[u]) for u in cs]
        if np.all(np.isnan(corrs)): continue            # constant/zero field -> undefined, skip
        valid += 1
        if cs[int(np.nanargmax(corrs))] == c: hit += 1
    dp = np.nan if np.all(np.isnan(dps)) else float(np.nanmean(dps))
    return {"D_pear": round(float(dp), 4) if dp == dp else float("nan"),
            "D_dir": round(float(np.mean(dds)), 4), "PDS": round(hit / valid, 4) if valid else float("nan"),
            "n_cond": len(cs)}


# ===========================================================================================
# helpers (from metrics.py)
# ===========================================================================================
def pseudobulk(X, labels):
    labels = np.asarray(labels)
    return {c: X[labels == c].mean(0) for c in np.unique(labels)}


def hvg_mask(adata, n=200):
    hv = adata.var["dispersions_norm"].nlargest(n).index
    return adata.var.index.isin(hv)


def marker_mask(adata, genes):
    return adata.var.index.isin(list(genes))


def stratify_index(values, bins=(0, 5, 10, 50, np.inf), right=False):
    return np.digitize(np.asarray(values, float), bins=np.asarray(bins, float), right=right)


# ===========================================================================================
# aggregators
# ===========================================================================================
def score_populations(pred_pb, obs_pb, ctrl_pb, pred_pca=None, obs_pca=None, gene_mask=None, k_deg=50):
    """Per-condition population scorecard (L1 latent + L2b DEG). Pass *_pca for distribution distances."""
    out = {"delta_pearson": delta_pearson(pred_pb, obs_pb, ctrl_pb, gene_mask),
           "delta_sign_accuracy": delta_sign_accuracy(pred_pb, obs_pb, ctrl_pb, gene_mask),
           "effect_size_spearman": effect_size_spearman(pred_pb, obs_pb, ctrl_pb, gene_mask),
           **{f"deg_{k}": v for k, v in topk_deg_overlap(pred_pb, obs_pb, ctrl_pb, k_deg).items()}}
    if pred_pca is not None and obs_pca is not None:
        out["energy_distance"] = energy_distance(pred_pca, obs_pca)
        out["mmd_rbf"] = mmd_rbf(pred_pca, obs_pca)
        out["variance_ratio"] = variance_ratio(pred_pca, obs_pca)
    return out


def confound_panel(X, V, pt, adj, group_labels=None):
    """The mandatory confound block printed beside every trajectory claim."""
    out = {"revcorr": revcorr(V, X), "rev_collin": rev_collin(X, pt, adj)}
    out.update(velo_dir(V, X, pt, adj, group_labels))
    return out
