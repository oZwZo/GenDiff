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


def zero_aug(adata, base, pseudotime_key, q=0.8):
    """Terminal ΔX≈0 supervision: copy `base` but zero the displacement for terminal (top-q pseudotime)
    cells, teaching the model that committed cells stop moving."""
    pt = np.asarray(adata.obs[pseudotime_key], float)
    D = np.asarray(base).copy(); D[pt >= np.quantile(pt, q)] = 0.0
    return D.astype(np.float32)
