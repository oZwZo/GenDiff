"""
gendiff_dev.targets.registry — named, versioned ΔX target artifacts (architecture §4.2).

A target is an ndarray on disk + a JSON manifest. **Targets are never mutated in place** — the sampler
ships a NEW named artifact and no consumer's baseline silently shifts (this dissolves collision C2).

Layout under TARGET_ROOT:  <dataset>__<name>.npy  +  <dataset>__<name>.manifest.json
Manifest: {dataset, name, builder, params, n_obs, n_genes, source_path, created, revcorr}.

`get(dataset, name)` returns (array, manifest), building+caching on first request.
Canonical names (resolved per-dataset by DEFS):
  r100          current shipped target (BarRNA-seq obsm)        -> from_obsm
  r3            r=3 neighbour sampler (the recommendation)       -> knn_sampler (geodesic, same-cond auto)
  r3_euclid_m30 r=3 euclid M30 (metric-tie reference)            -> knn_sampler
  real_velocity the use-the-velocity reference                   -> from_velocity_layer
  zero_aug      terminal-ΔX≈0 supervision on top of r3           -> zero_aug(base=r3)
  smooth_nb     per-cluster NB μ(t) derivative (SAMPLER_OPT win)  -> smooth_deriv(mode='nb')
  smooth_unbiased  auto-root (potency) + per-cluster μ(t) deriv     -> smooth_deriv(mode='auto', auto_root)
"""
from __future__ import annotations
import os, json, hashlib, numpy as np
from . import builders
from ..data import registry as data_reg

TARGET_ROOT = os.environ.get(
    "GENDIFF_TARGET_ROOT", "/rds/user/wz369/hpc-work/GenDiff/gendiff_dev/results/targets")


def _paths(dataset, name):
    base = os.path.join(TARGET_ROOT, f"{dataset}__{name}")
    return base + ".npy", base + ".manifest.json"


def list_targets():
    return ["r100", "r3", "r3_euclid_m30", "real_velocity", "zero_aug", "smooth_nb", "smooth_unbiased"]


def _build(dataset, name, adata, spec):
    """Construct a canonical target for a dataset. `same_condition` is True only where conditions are
    densely sampled (BarRNA-seq); on the sparse TF atlases the sampler runs global/root."""
    same_cond = dataset == "barrnaseq"
    common = dict(use_rep=spec["use_rep"], pseudotime_key=spec["pseudotime_key"],
                  condition_key=spec["condition_key"], same_condition=same_cond)
    if name == "r100":
        return builders.from_obsm(adata, "Tr_SampledX_r100"), dict(builder="from_obsm", key="Tr_SampledX_r100")
    if name == "real_velocity":
        return builders.from_velocity_layer(adata, spec.get("velocity_key") or "velocity"), dict(builder="from_velocity_layer")
    if name == "r3":
        p = dict(config="config1", **common)
        return builders.knn_sampler(adata, **p), dict(builder="knn_sampler", config="config1",
                                                      **{k: p[k] for k in ("same_condition",)})
    if name == "r3_euclid_m30":
        p = dict(config="config2", **common)
        return builders.knn_sampler(adata, **p), dict(builder="knn_sampler", config="config2",
                                                      **{k: p[k] for k in ("same_condition",)})
    if name == "zero_aug":
        base, _ = get(dataset, "r3", adata=adata)
        return builders.zero_aug(adata, base, spec["pseudotime_key"], q=0.8), dict(builder="zero_aug", base="r3", q=0.8)
    if name == "smooth_nb":
        p = dict(pseudotime_key=spec["pseudotime_key"], use_rep=spec["use_rep"], mode="nb",
                 cluster_key="louvain")
        return builders.smooth_deriv(adata, **p), dict(builder="smooth_deriv", mode="nb", cluster_key="louvain")
    if name == "smooth_unbiased":
        # only the ROOT is automated; algorithmic louvain + counts are routine preprocessing, not priors.
        p = dict(pseudotime_key=spec["pseudotime_key"], use_rep=spec["use_rep"], mode="auto",
                 cluster_key="louvain", auto_root=True)
        return builders.smooth_deriv(adata, **p), dict(builder="smooth_deriv", mode="auto",
                                                       cluster_key="louvain", auto_root=True)
    raise KeyError(f"unknown target {name!r}; known: {list_targets()}")


def get(dataset, name, adata=None, rebuild=False):
    """Return (array, manifest). Builds + caches on first request; reuses the artifact afterward."""
    npy, mpath = _paths(dataset, name)
    if os.path.exists(npy) and os.path.exists(mpath) and not rebuild:
        with open(mpath) as f: manifest = json.load(f)
        return np.load(npy), manifest
    spec = data_reg.get(dataset)
    if adata is None: adata = data_reg.load(dataset)
    arr, params = _build(dataset, name, adata, spec)
    os.makedirs(TARGET_ROOT, exist_ok=True)
    src_hash = hashlib.md5(f"{spec['path']}:{adata.n_obs}:{adata.n_vars}".encode()).hexdigest()[:12]
    from .. import _now
    manifest = dict(dataset=dataset, name=name, params=params, n_obs=int(arr.shape[0]),
                    n_genes=int(arr.shape[1]), source_path=spec["path"], source_hash=src_hash,
                    created=_now(), revcorr=_revcorr(arr, adata))
    np.save(npy, arr.astype(np.float32))
    with open(mpath, "w") as f: json.dump(manifest, f, indent=2)
    return arr, manifest


def _revcorr(D, adata):
    import scipy.sparse as sp
    from ..eval import panel
    X = (adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)).astype(np.float32)
    return panel.revcorr(D, X, stride=max(1, adata.n_obs // 2000))
