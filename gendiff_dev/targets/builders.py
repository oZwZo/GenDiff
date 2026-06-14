"""
gendiff_dev.targets.builders — ΔX target constructors (architecture §4.2, sampler-owned).

This is the consolidated home for the neighbour-sampler logic that used to be re-implemented in
auto/sweep.py and auto/tf_sweep.py (collisions C1/C5/C7). Every builder returns an ndarray
(n_obs, n_genes) aligned to adata.obs order. `repeat` (C5) is a plain parameter, never a hardcoded 100.
"""
from __future__ import annotations
import warnings
import numpy as np

# Named presets for knn_sampler. A preset fixes (metric, M, repeat); same_condition / use_rep /
# graph_k / alpha are still chosen by the caller (the dataset decides same_condition density).
SAMPLER_CONFIGS = {
    "config1": dict(metric="geodesic", M=15, repeat=3),  # r3  — recommended default
    "config2": dict(metric="euclid",   M=30, repeat=3),  # r3_euclid_m30 — metric-tie reference
}


def _resolve_sampler_config(config, metric, M, repeat, same_condition, use_rep, graph_k, alpha, verbose):
    """Turn (config, explicit params) into the (metric, M, repeat) actually used, and announce it.
    Presets ('config1'/'config2') ignore any explicitly-passed metric/M/repeat (with a warning);
    'others' reads them. Unknown config is a hard error so a typo never silently runs config1."""
    explicit = {k: v for k, v in (("metric", metric), ("M", M), ("repeat", repeat)) if v is not None}
    if config in SAMPLER_CONFIGS:
        vals = dict(SAMPLER_CONFIGS[config])
        if explicit:
            warnings.warn(
                f"knn_sampler: config={config!r} is a fixed preset "
                f"(metric={vals['metric']}, M={vals['M']}, repeat={vals['repeat']}); the explicit "
                f"{explicit} you passed are IGNORED. Pass config='others' to use your own values.",
                stacklevel=3)
        src = f"preset config={config!r}"
    elif config == "others":
        vals = dict(metric=metric or "euclid",
                    M=15 if M is None else M,
                    repeat=3 if repeat is None else repeat)
        src = "config='others' (user-supplied params)"
    else:
        raise ValueError(
            f"knn_sampler: unknown config {config!r}. Choose 'config1' (geodesic, M=15, repeat=3 — "
            f"recommended), 'config2' (euclid, M=30, repeat=3), or 'others' (then pass metric/M/repeat).")
    if verbose:
        print(f"[knn_sampler] {src} -> metric={vals['metric']} M={vals['M']} repeat={vals['repeat']} "
              f"| same_condition={same_condition} use_rep={use_rep!r} graph_k={graph_k} alpha={alpha}",
              flush=True)
    return vals["metric"], vals["M"], vals["repeat"]


def from_obsm(adata, key="Tr_SampledX_r100"):
    """The current shipped target (e.g. r100), read from a precomputed obsm. No recomputation."""
    return np.asarray(adata.obsm[key]).astype(np.float32)


def from_velocity_layer(adata, key="velocity"):
    """The real RNA-velocity field as a 'use-the-velocity' reference target."""
    import scipy.sparse as sp
    v = adata.layers[key]; v = v.toarray() if sp.issparse(v) else np.asarray(v)
    return np.nan_to_num(v).astype(np.float32)


def _knn(rep, k):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(rep); d, i = nn.kneighbors(rep)
    return d, i


def _geodesic(rep, k, power, sources):
    """Dijkstra on a symmetric kNN graph (real manifold geodesic; power=2 => Fermat). O(N^2) memory in
    sources x N — use for one-off artifact builds, cache the result yourself for big N."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    n = rep.shape[0]; d, i = _knn(rep, k)
    rows = np.repeat(np.arange(n), k); cols = i[:, 1:].ravel(); w = d[:, 1:].ravel().astype(np.float64)
    if power != 1: w = w ** power
    g = sp.csr_matrix((w, (rows, cols)), shape=(n, n)); g = g.maximum(g.T)
    return dijkstra(g, directed=False, indices=sources).astype(np.float32)


def knn_sampler(adata, *, use_rep, pseudotime_key, condition_key=None, same_condition=False,
                config="config1", metric=None, M=None, repeat=None, alpha=1.0, seed=0,
                cells=None, graph_k=15, verbose=True):
    """ROOT/GLOBAL or SAME-CONDITION higher-pseudotime neighbour sampler.
      ΔX_i = mean over `repeat` neighbours of (x_neighbour − x_i).
    Neighbours are higher-pseudotime cells, ranked by `metric` ∈ {euclid, geodesic, fermat}, optionally
    restricted to the same condition (same_condition=True -> BarRNA-seq traverse; False -> TF-Atlas root).

    `config` selects the parameter combination and is the recommended entry point:
      'config1' (default) -> metric=geodesic, M=15, repeat=3   (the r3 recommendation)
      'config2'           -> metric=euclid,   M=30, repeat=3   (r3_euclid_m30 metric-tie reference)
      'others'            -> read the explicit metric / M / repeat you pass.
    Under a preset, any metric/M/repeat you also pass are ignored (with a warning); set config='others'
    to use them. The actual settings are printed when verbose=True. same_condition / use_rep / graph_k /
    alpha are honoured under every config (the dataset, not the preset, decides same_condition).

    `graph_k` is the kNN graph degree used for euclid two-hop candidates and the geodesic graph (default
    15 reproduces the shipped artifacts). Cells with no eligible neighbour get ΔX=0. `cells` limits which
    rows are filled (default: all)."""
    import scipy.sparse as sp
    metric, M, repeat = _resolve_sampler_config(
        config, metric, M, repeat, same_condition, use_rep, graph_k, alpha, verbose)
    if same_condition and not condition_key:
        warnings.warn("knn_sampler: same_condition=True but condition_key is None; running global "
                      "(condition-agnostic) sampling.", stacklevel=2)
    rng = np.random.default_rng(seed)
    X = (adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)).astype(np.float32)
    rep = np.asarray(adata.obsm[use_rep]); n, G = X.shape
    pt = np.asarray(adata.obs[pseudotime_key], float)
    cond = adata.obs[condition_key].astype(str).to_numpy() if (same_condition and condition_key) else None
    cells = np.arange(n) if cells is None else np.asarray(cells)
    D = np.zeros((n, G), dtype=np.float32)

    if metric in ("geodesic", "fermat"):
        geo = _geodesic(rep, graph_k, 2 if metric == "fermat" else 1, cells)
        pos = {int(c): p for p, c in enumerate(cells)}
        for c in cells:
            gd = geo[pos[int(c)]]; reach = np.where(np.isfinite(gd) & (gd > 0))[0]
            reach = reach[pt[reach] > pt[c]]
            if cond is not None: reach = reach[cond[reach] == cond[c]]
            if reach.size == 0: continue
            near = reach[np.argsort(gd[reach])[:M]]; dd = gd[near]
            D[c] = _draw(X, near, dd, alpha, repeat, rng) - X[c]
    else:
        _, idx = _knn(rep, graph_k); adj = [idx[i][1:] for i in range(n)]
        for c in cells:
            cand = _two_hop(adj, c); cand = cand[pt[cand] > pt[c]]
            if cond is not None: cand = cand[cond[cand] == cond[c]]
            if cand.size == 0: continue
            dd = np.linalg.norm(rep[cand] - rep[c], axis=1)
            if cand.size > M:
                o = np.argsort(dd)[:M]; cand = cand[o]; dd = dd[o]
            D[c] = _draw(X, cand, dd, alpha, repeat, rng) - X[c]
    if verbose:
        nz = int((np.abs(D[cells]).sum(1) == 0).sum())
        print(f"[knn_sampler] built ΔX {D.shape[0]}x{D.shape[1]}; {nz}/{len(cells)} requested cells "
              f"had no eligible higher-pseudotime neighbour -> ΔX=0", flush=True)
    return D


def _two_hop(adj, i):
    s = set(adj[i].tolist())
    for j in adj[i]: s.update(adj[j].tolist())
    s.discard(i); return np.fromiter(s, int)


def _draw(X, near, d, alpha, repeat, rng):
    w = (d.max() - d + 0.1 * d.min()); w = np.maximum(w, 0) ** alpha
    p = w / w.sum() if w.sum() > 0 else None
    pick = rng.choice(near, size=repeat, p=p)
    return X[pick].mean(0)


# ---------------------------------------------------------------------------------------------------
# smooth_deriv — the SAMPLER_OPT winner: the derivative of a SMOOTHED mean-expression trajectory.
#
# The one-hop neighbour difference (knn_sampler) has E[ΔX | x] dominated by regression-to-the-mean
# (ΔX ≈ −x), so model-free integration of it reverts and collapses cells to the data centroid
# (revcorr ≈ −0.9). Replacing it with d/dt of a smoothed per-gene mean trajectory μ(t) removes that
# radial component by construction (revcorr ≈ 0), keeps the genuine forward kinetics, and self-slows at
# the terminal plateau (dμ/dt → 0). Three modes, increasing in how few priors they need:
#   'nb'    per-cluster NB/Poisson μ(t) in count space, ΔX = dlog1p(μ)/dt — best balance (needs counts).
#   'gauss' per-cluster cubic-spline μ(t) fit directly in adata.X space     — count-free fallback.
#   'local' cluster-free per-cell local linear slope dX/dt over the use_rep kNN — the unbiased variant
#           (no supervised partition); pair with auto_root=True for a fully prior-free target.
# ---------------------------------------------------------------------------------------------------
def _dense(M):
    import scipy.sparse as sp
    return (M.toarray() if sp.issparse(M) else np.asarray(M))


def _spline_basis(t, n_knots=5):
    """Truncated-power cubic basis B(t) and its time-derivative Bp(t); interior knots at t-quantiles."""
    t = np.asarray(t, float)
    ks = np.quantile(t, np.linspace(0, 1, n_knots + 2)[1:-1])
    cols = [np.ones_like(t), t, t ** 2, t ** 3]
    dcols = [np.zeros_like(t), np.ones_like(t), 2 * t, 3 * t ** 2]
    for k in ks:
        rp = np.clip(t - k, 0, None)
        cols.append(rp ** 3); dcols.append(3 * rp ** 2)
    return np.stack(cols, 1), np.stack(dcols, 1)


def _smooth_gauss(X, t, n_knots=5):
    """Spline fit X ~ B(t) per gene (one lstsq for all genes); return dμ/dt = Bp @ coef (X's own space)."""
    B, Bp = _spline_basis(t, n_knots)
    coef, *_ = np.linalg.lstsq(B, X, rcond=None)
    return Bp @ coef


def _smooth_nb(counts, t, n_knots=5, n_iter=3, chunk=400):
    """Poisson/NB-mean spline in count space (vectorised IRLS, gene-chunked): log μ = B β; returns the
    log1p-space derivative dlog1p(μ)/dt = μ'/(1+μ) with μ' = μ·(Bp β)."""
    B, Bp = _spline_basis(t, n_knots)
    n, g = counts.shape; nb = B.shape[1]; V = np.zeros((n, g), np.float64)
    for c0 in range(0, g, chunk):
        c1 = min(c0 + chunk, g); yk = counts[:, c0:c1]
        beta, *_ = np.linalg.lstsq(B, np.log1p(yk), rcond=None)
        for _ in range(n_iter):
            eta = B @ beta; mu = np.exp(np.clip(eta, -20, 20))
            Wz = mu * eta + (yk - mu)
            A = np.einsum("ni,ng,nj->gij", B, mu, B, optimize=True)
            rhs = np.einsum("ni,ng->gi", B, Wz, optimize=True)
            A += 1e-6 * np.eye(nb)[None]
            beta = np.linalg.solve(A, rhs[..., None])[..., 0].T
        eta = B @ beta; mu = np.exp(np.clip(eta, -20, 20)); dmu = mu * (Bp @ beta)
        V[:, c0:c1] = dmu / (1.0 + mu)
    return V


def _local_slope(X, t, rep, k=15, chunk=2000):
    """Cluster-free per-cell local linear trajectory slope dX/dt: distance-kernel-weighted OLS-through-x_i
    of each gene on (t_nb − t_i) over the cell's k neighbours in `rep`. Continuous analogue of per-cluster
    μ(t) with no supervised partition."""
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(rep); d, idx = nn.kneighbors(rep)
    idx = idx[:, 1:]; d = d[:, 1:]; V = np.zeros_like(X)
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n); ii = idx[c0:c1]; dd = d[c0:c1]
        w = np.exp(-dd ** 2 / (np.median(dd, 1, keepdims=True) ** 2 + 1e-9))
        dt = t[ii] - t[c0:c1, None]; dX = X[ii] - X[c0:c1, None, :]
        num = np.einsum("bk,bkg->bg", w * dt, dX); den = np.einsum("bk,bk->b", w, dt * dt) + 1e-9
        V[c0:c1] = num / den[:, None]
    return V


def _resolve_counts(adata, counts_layer):
    """Return (counts_linear, std_log1p) for the NB mode, or (None, None) if no count source is available.
    Source priority: an explicit linear-count layer; else adata.raw (assumed log1p, aligned to var_names).
    std_log1p is the per-gene std of log1p(counts) — used to map the log1p-space derivative into adata.X's
    space when adata.X is a z-score of that same log1p (the manuscript convention)."""
    if counts_layer is not None and counts_layer in (adata.layers or {}):
        counts = _dense(adata.layers[counts_layer]).astype(np.float64)
        std_log1p = np.log1p(np.clip(counts, 0, None)).std(0)
        return counts, std_log1p
    if adata.raw is not None:
        try:
            log1p = _dense(adata.raw[:, adata.var_names].X).astype(np.float64)
        except Exception:
            log1p = _dense(adata.raw.X).astype(np.float64)
        if log1p.shape[1] == adata.n_vars:
            if np.nanmax(log1p) > 30:
                warnings.warn("_resolve_counts: adata.raw.X values exceed 30 — it is assumed to be log1p "
                              "and expm1'd back to counts, but these look like RAW linear counts. expm1 "
                              "will overflow and the NB sampler will degrade to ~0; pass a linear-count "
                              "layer via sampler_kwargs={'counts_layer': ...} or store log1p in raw.",
                              stacklevel=3)
            return np.expm1(log1p), log1p.std(0)
    return None, None


def _potency(adata):
    """Transcriptional-potency proxy = number of EXPRESSED genes per cell (CytoTRACE's gene-counts
    signature; higher = more potent = closer to the root). Robust on z-scored adata.X, where an
    entropy-of-expression proxy is meaningless. Source priority: raw counts (#nonzero), then an obs
    detected-gene column, then (last resort) shifted-expression entropy."""
    import scipy.sparse as sp
    if adata.raw is not None:
        R = adata.raw.X
        nz = np.asarray((R > 0).sum(1)).ravel() if sp.issparse(R) else (np.asarray(R) > 0).sum(1)
        return nz.astype(float)
    for k in ("n_genes", "nFeature_RNA", "n_genes_by_counts", "n_counts"):
        if k in adata.obs:
            return np.asarray(adata.obs[k], float)
    from scipy.stats import entropy
    X = _dense(adata.X).astype(np.float64)
    warnings.warn("smooth_deriv auto_root: no count/detected-gene source; falling back to shifted-"
                  "expression entropy (weak on z-scored X).", stacklevel=3)
    return entropy(X - X.min(0, keepdims=True) + 1e-5, axis=1)


def _auto_root_dpt(adata, use_rep, pseudotime_key, verbose=True):
    """Prior-free root: the most transcriptionally potent cell (kNN-smoothed #expressed-genes — CytoTRACE's
    gene-counts signature), then recompute diffusion pseudotime from it. Falls back to the existing
    pseudotime if the diffusion basis for sc.tl.dpt is unavailable."""
    from sklearn.neighbors import NearestNeighbors
    pot = _potency(adata)
    # smooth over a TRUE embedding kNN (robust to a dense / fully-connected stored obsp graph, which would
    # otherwise reduce the smoothing to a meaningless global mean).
    import scipy.sparse as sp
    rep = np.asarray(adata.obsm[use_rep], float); rep = rep[:, :50] if rep.shape[1] > 50 else rep
    nn = NearestNeighbors(n_neighbors=min(16, len(rep))).fit(rep); _, idx = nn.kneighbors(rep)
    pot = pot[idx].mean(1)
    root = int(np.argmax(pot))
    # Compute dpt from the auto-root. If the stored graph is dense/degenerate (e.g. a fully-connected obsp),
    # the stored diffmap is unreliable, so rebuild neighbors+diffmap from use_rep on a COPY (clean dpt, no
    # mutation). If the graph is already a sane sparse kNN, mutate-and-restore in place (cheap, no copy).
    C = adata.obsp.get("connectivities")
    degenerate = (C is None) or (not sp.issparse(C)) or (C.getnnz(1).mean() > 45)
    try:
        import scanpy as sc
        if degenerate:
            b = adata.copy()
            sc.pp.neighbors(b, n_neighbors=16, use_rep=use_rep)
            sc.tl.diffmap(b)
            b.uns["iroot"] = root; sc.tl.dpt(b)
            pt = np.asarray(b.obs["dpt_pseudotime"], float); del b
        else:
            saved_iroot = adata.uns.get("iroot", None)
            saved_dpt = adata.obs["dpt_pseudotime"].copy() if "dpt_pseudotime" in adata.obs else None
            adata.uns["iroot"] = root; sc.tl.dpt(adata)
            pt = np.asarray(adata.obs["dpt_pseudotime"], float)
            if saved_dpt is not None:
                adata.obs["dpt_pseudotime"] = saved_dpt
            if saved_iroot is not None:
                adata.uns["iroot"] = saved_iroot
            elif "iroot" in adata.uns:
                del adata.uns["iroot"]
        if verbose:
            print(f"[smooth_deriv] auto-root = cell {root} (max smoothed #expressed-genes potency); "
                  f"dpt recomputed{' on rebuilt graph' if degenerate else ''}", flush=True)
    except Exception as e:
        warnings.warn(f"smooth_deriv: auto_root could not recompute dpt ({e!r}); using existing "
                      f"{pseudotime_key!r}.", stacklevel=3)
        pt = np.asarray(adata.obs[pseudotime_key], float)
    return pt


def smooth_deriv(adata, *, pseudotime_key, use_rep, mode="auto", cluster_key="louvain",
                 counts_layer=None, n_knots=5, graph_k=15, auto_root=False, zscored=None,
                 min_cluster=30, seed=0, verbose=True):
    """Smoothed mean-trajectory-derivative ΔX target (the SAMPLER_OPT winner). Returns (n_obs, n_genes) in
    adata.X's units, aligned to obs order, with the mean-reversion confound removed by construction.

    mode : 'nb' | 'gauss' | 'local' | 'auto'. 'auto' picks 'nb' if counts are available, else 'gauss' if
           `cluster_key` is present, else 'local'.
    cluster_key : obs partition for the per-cluster fit ('nb'/'gauss'); clusters smaller than `min_cluster`
           get ΔX=0. Ignored by 'local'. If absent, the per-cluster modes fall back to one global cluster.
    counts_layer : a LINEAR-count layer for 'nb'; if None, adata.raw (log1p) is used. Without any count
           source, 'nb' downgrades to 'gauss'.
    auto_root : recompute pseudotime from an auto-selected potency root instead of trusting the supplied
           root (removes the manual-root prior). zscored : whether adata.X is a z-score of log1p (auto-
           detected from the fraction of negative entries when None) — controls the NB→X-space rescaling.
    """
    X = _dense(adata.X).astype(np.float64)
    n, G = X.shape
    rep = np.asarray(adata.obsm[use_rep], float)
    rep = rep[:, :50] if rep.shape[1] > 50 else rep
    t = _auto_root_dpt(adata, use_rep, pseudotime_key, verbose) if auto_root \
        else np.asarray(adata.obs[pseudotime_key], float)
    counts, std_log1p = _resolve_counts(adata, counts_layer)
    if zscored is None:
        zscored = float((X < 0).mean()) > 0.1

    if mode == "auto":
        mode = "nb" if counts is not None else ("gauss" if cluster_key in adata.obs else "local")
    if mode == "nb" and counts is None:
        warnings.warn("smooth_deriv: mode='nb' but no count source (counts_layer / adata.raw); "
                      "falling back to mode='gauss'.", stacklevel=2)
        mode = "gauss"
    if verbose:
        print(f"[smooth_deriv] mode={mode} cluster_key={cluster_key!r} auto_root={auto_root} "
              f"zscored={zscored} n_knots={n_knots}", flush=True)

    if mode == "local":
        D = _local_slope(X, t, rep, k=graph_k)
    else:
        if cluster_key in adata.obs:
            clusters = adata.obs[cluster_key].astype(str).to_numpy()
        else:
            warnings.warn(f"smooth_deriv: cluster_key {cluster_key!r} not in obs; fitting one global "
                          f"trajectory (mode={mode}).", stacklevel=2)
            clusters = np.zeros(n, dtype=object)
        D = np.zeros_like(X)
        for cl in np.unique(clusters):
            sub = np.where(clusters == cl)[0]
            if sub.size < min_cluster:
                continue
            if mode == "nb":
                vlog = _smooth_nb(counts[sub], t[sub], n_knots)
                D[sub] = vlog / (std_log1p + 1e-9) if zscored else vlog
            else:
                D[sub] = _smooth_gauss(X[sub], t[sub], n_knots)
    D = np.nan_to_num(D).astype(np.float32)
    if verbose:
        nz = int((np.abs(D).sum(1) == 0).sum())
        print(f"[smooth_deriv] built ΔX {D.shape[0]}x{D.shape[1]}; {nz} cells with ΔX=0 "
              f"(unfit clusters / no neighbours)", flush=True)
    return D


def zero_aug(adata, base, pseudotime_key, q=0.8):
    """Terminal ΔX≈0 supervision: copy `base` but zero the displacement for terminal (top-q pseudotime)
    cells, teaching the model that committed cells stop moving."""
    pt = np.asarray(adata.obs[pseudotime_key], float)
    D = np.asarray(base).copy(); D[pt >= np.quantile(pt, q)] = 0.0
    return D.astype(np.float32)


# ---------------------------------------------------------------------------------------------------
# pt_gradient — the SAMPLER_OPT 'ptgrad' field: the gene-space gradient of pseudotime.
#
# Local linear regression of pseudotime on the use_rep kNN gives ∇_h t, the embedding-space direction of
# steepest pseudotime increase per cell. A least-squares gene->embedding linear map W (ref ≈ [X|1]·W) pulls
# it back to gene space: ΔX = ∇_h t · Wᵀ. This is the most FORWARD-reaching model-free field but it
# over-disperses off-manifold and is condition-blind (it points up ∇t regardless of perturbation), so it is
# an aggressive / ablation target, not a default — see SAMPLER_OPT/FINDINGS.md.
# ---------------------------------------------------------------------------------------------------
def pt_gradient(adata, *, use_rep, pseudotime_key, graph_k=15, max_rep=50, chunk=4000, verbose=True):
    """Pseudotime-gradient ΔX target (n_obs, n_genes), in adata.X units, aligned to obs order.

    The gene->embedding map is fit by least squares on ALL cells (an unsupervised linear projector, like
    PCA loadings — no label leakage). `graph_k` is the neighbourhood for the local ∇t regression; `max_rep`
    caps the embedding dimension used (the shipped reps are 30–50-d)."""
    from sklearn.neighbors import NearestNeighbors
    X = _dense(adata.X).astype(np.float64)
    n, G = X.shape
    ref = np.asarray(adata.obsm[use_rep], float)
    ref = ref[:, :max_rep] if ref.shape[1] > max_rep else ref
    t = np.asarray(adata.obs[pseudotime_key], float)

    # gene -> embedding linear map (with bias): ref ≈ [X | 1] @ Wfull ; Wgene = Wfull without the bias row.
    Xb = np.hstack([X, np.ones((n, 1))])
    Wfull, *_ = np.linalg.lstsq(Xb, ref, rcond=None)
    Wgene = Wfull[:-1]                                            # (G, d)
    r2 = 1.0 - ((ref - Xb @ Wfull) ** 2).sum() / (((ref - ref.mean(0)) ** 2).sum() + 1e-9)

    # local linear regression t ~ ref over each cell's kNN -> embedding gradient (batched min-norm lstsq).
    nn = NearestNeighbors(n_neighbors=min(graph_k + 1, n)).fit(ref); _, idx = nn.kneighbors(ref)
    idx = idx[:, 1:]
    Gh = np.zeros((n, ref.shape[1]))
    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        A = ref[idx[c0:c1]] - ref[c0:c1, None, :]                # (b, k, d)
        y = t[idx[c0:c1]] - t[c0:c1, None]                       # (b, k)
        Gh[c0:c1] = np.einsum("bdk,bk->bd", np.linalg.pinv(A), y)
    V = np.nan_to_num(Gh @ Wgene.T).astype(np.float32)
    if verbose:
        print(f"[pt_gradient] built ΔX {V.shape[0]}x{V.shape[1]}; gene->{use_rep} fit R2={r2:.3f}, "
              f"graph_k={graph_k}", flush=True)
    return V


# ---------------------------------------------------------------------------------------------------
# ot_sampler — the SAMPLER_OPT 'ot' field: per-cluster optimal-transport early->late displacement.
#
# Within each cluster, solve the EMD between EARLY (bottom-q pseudotime) and LATE (top-q) cells in use_rep
# space; the barycentric map sends each early cell to its transported late position, ΔX = mapped − x. The
# displacement is then kNN-interpolated from these early anchors to every cell. Because ΔX transports to
# REAL late cells it is naturally manifold-scaled and does not over-push — the study's robust runner-up.
# ---------------------------------------------------------------------------------------------------
def ot_sampler(adata, *, use_rep, pseudotime_key, cluster_key="louvain", graph_k=15,
               q_early=0.33, q_late=0.67, cap=400, min_cluster=20, min_ends=5, seed=0, verbose=True):
    """Optimal-transport early->late ΔX target (n_obs, n_genes), in adata.X units, aligned to obs order.

    `cluster_key` partitions the transport (each cluster fit independently); if absent, one global transport
    is used. Clusters smaller than `min_cluster`, or with fewer than `min_ends` early/late cells, are skipped.
    `cap` subsamples each end for the EMD solve. Requires POT (`pip install pot`)."""
    import ot as POT
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    X = _dense(adata.X).astype(np.float64)
    rep = np.asarray(adata.obsm[use_rep], float)
    t = np.asarray(adata.obs[pseudotime_key], float)
    n, G = X.shape
    if cluster_key in adata.obs:
        louv = adata.obs[cluster_key].astype(str).to_numpy()
    else:
        warnings.warn(f"ot_sampler: cluster_key {cluster_key!r} not in obs; using one global transport.",
                      stacklevel=2)
        louv = np.zeros(n, dtype=object)

    Vsrc = np.zeros((n, G)); have = np.zeros(n, bool)
    for l in np.unique(louv):
        ci = np.where(louv == l)[0]
        if ci.size < min_cluster:
            continue
        q1, q2 = np.quantile(t[ci], [q_early, q_late])
        e = ci[t[ci] < q1]; lt = ci[t[ci] > q2]
        if e.size < min_ends or lt.size < min_ends:
            continue
        if e.size > cap: e = rng.choice(e, cap, replace=False)
        if lt.size > cap: lt = rng.choice(lt, cap, replace=False)
        M = POT.dist(rep[e], rep[lt], metric="sqeuclidean"); M /= M.max() + 1e-9
        Gp = POT.emd(np.ones(len(e)) / len(e), np.ones(len(lt)) / len(lt), M)
        mapped = (Gp / (Gp.sum(1, keepdims=True) + 1e-12)) @ X[lt]
        Vsrc[e] = mapped - X[e]; have[e] = True

    if not have.any():
        warnings.warn("ot_sampler: no cluster had enough early/late cells; returning ΔX=0.", stacklevel=2)
        return np.zeros((n, G), np.float32)
    # interpolate the OT displacement from anchor (early) cells to all cells via use_rep kNN (chunked).
    anc = np.where(have)[0]; Va = Vsrc[anc]
    nn = NearestNeighbors(n_neighbors=min(graph_k, anc.size)).fit(rep[anc])
    V = np.zeros((n, G))
    for c0 in range(0, n, 2000):
        c1 = min(c0 + 2000, n)
        d, ii = nn.kneighbors(rep[c0:c1])
        w = np.exp(-d ** 2 / (np.median(d, 1, keepdims=True) ** 2 + 1e-9))
        V[c0:c1] = (w[:, :, None] * Va[ii]).sum(1) / w.sum(1, keepdims=True)
    V = np.nan_to_num(V).astype(np.float32)
    if verbose:
        print(f"[ot_sampler] built ΔX {V.shape[0]}x{V.shape[1]}; {anc.size} early anchors over "
              f"{len(np.unique(louv))} cluster(s), interpolated to all cells", flush=True)
    return V
