"""
gendiff_dev.targets.builders — ΔX target constructors (architecture §4.2, sampler-owned).

This is the consolidated home for the neighbour-sampler logic that used to be re-implemented in
auto/sweep.py and auto/tf_sweep.py (collisions C1/C5/C7). Every builder returns an ndarray
(n_obs, n_genes) aligned to adata.obs order. `repeat` (C5) is a plain parameter, never a hardcoded 100.
"""
from __future__ import annotations
import numpy as np


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
                metric="euclid", M=15, repeat=3, alpha=1.0, seed=0, cells=None, graph_k=15):
    """ROOT/GLOBAL or SAME-CONDITION higher-pseudotime neighbour sampler.
      ΔX_i = mean over `repeat` neighbours of (x_neighbour − x_i).
    Neighbours are higher-pseudotime cells, ranked by `metric` ∈ {euclid, geodesic, fermat}, optionally
    restricted to the same condition (same_condition=True -> BarRNA-seq traverse; False -> TF-Atlas root).
    `graph_k` is the kNN graph degree used for euclid two-hop candidates and the geodesic graph (default
    15 reproduces the shipped artifacts). Cells with no eligible neighbour get ΔX=0. `cells` limits which
    rows are filled (default: all)."""
    import scipy.sparse as sp
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
