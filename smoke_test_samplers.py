#!/usr/bin/env python
"""Smoke test for the four newly-exposed ΔX samplers (smooth_nb, nb_global, ot, ptgrad) and their
integration into the high-level GenDiff API.

Checks, all on the small BarRNA-seq dataset:
  A. builder level — each field builds, shape (n, n_genes), finite, non-trivial (mean|abs|>0).
     The true per-cluster NB and per-cluster OT code paths are exercised with a SYNTHETIC louvain +
     counts layer (BarRNA-seq ships neither), so the IRLS / EMD branches actually run.
  B. API level — GenDiff.setup_anndata(adata, sampler=name) accepts every name, writes obsm['gendiff_dX']
     with the right shape, records `sampler` in the registry; an unknown sampler raises ValueError.
  C. end-to-end — build (sampler='ptgrad') + construct the model + 1-epoch CPU train, on a subset.

Writes /tmp/smoke_test_samplers.json and exits non-zero on any failure.
"""
from __future__ import annotations
import os, sys, json, traceback, warnings
warnings.filterwarnings("ignore")
import numpy as np

REPO = "/rds/user/wz369/hpc-work/GenDiff"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from gendiff import GenDiff
from gendiff_dev.data import registry as data_reg
from gendiff_dev.targets import builders, registry as tgt_reg

errors, notes = [], []


def check(cond, msg):
    if not cond:
        errors.append(msg)
        print(f"  FAIL: {msg}", flush=True)
    else:
        print(f"  ok:   {msg}", flush=True)


def valid_dx(dX, n, g, name):
    check(isinstance(dX, np.ndarray), f"{name}: returns ndarray")
    check(dX.shape == (n, g), f"{name}: shape {dX.shape} == ({n}, {g})")
    check(np.isfinite(dX).all(), f"{name}: all finite")
    check(float(np.abs(dX).mean()) > 0, f"{name}: non-trivial (mean|abs| {float(np.abs(dX).mean()):.4g} > 0)")


print("[load] barrnaseq ...", flush=True)
adata = data_reg.load("barrnaseq")
spec = data_reg.get("barrnaseq")
USE_REP, PT, COND = spec["use_rep"], spec["pseudotime_key"], spec["condition_key"]
n, g = adata.n_obs, adata.n_vars
print(f"[load] {n} x {g}; use_rep={USE_REP} pt={PT} cond={COND}", flush=True)

# ---- synthetic louvain + counts so the true per-cluster NB / OT branches run ----------------------
import pandas as pd
from sklearn.cluster import KMeans
lab = KMeans(n_clusters=4, n_init=4, random_state=0).fit_predict(np.asarray(adata.obsm[USE_REP]))
adata.obs["louvain"] = pd.Categorical(lab.astype(str))
Xd = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype=np.float64)
adata.layers["counts"] = np.rint(np.clip(np.expm1(Xd - Xd.min(0, keepdims=True)), 0, 1e4)).astype(np.float32)
print(f"[synth] added louvain ({adata.obs['louvain'].nunique()} clusters) + counts layer", flush=True)

# ============================================================ A. builder level
print("\n[A] builder-level field construction", flush=True)
try:
    valid_dx(builders.pt_gradient(adata, use_rep=USE_REP, pseudotime_key=PT, verbose=True), n, g, "ptgrad")
    valid_dx(builders.ot_sampler(adata, use_rep=USE_REP, pseudotime_key=PT, cluster_key="louvain",
                                 verbose=True), n, g, "ot (per-louvain)")
    valid_dx(builders.ot_sampler(adata, use_rep=USE_REP, pseudotime_key=PT, cluster_key="__none__",
                                 verbose=True), n, g, "ot (global fallback)")
    valid_dx(builders.smooth_deriv(adata, use_rep=USE_REP, pseudotime_key=PT, mode="nb",
                                   cluster_key="louvain", counts_layer="counts", verbose=True), n, g,
             "smooth_nb (per-louvain NB, real IRLS)")
    valid_dx(builders.smooth_deriv(adata, use_rep=USE_REP, pseudotime_key=PT, mode="nb",
                                   cluster_key=None, counts_layer="counts", verbose=True), n, g,
             "nb_global (global NB, real IRLS)")
except Exception:
    errors.append("builder-level raised:\n" + traceback.format_exc())
    print(traceback.format_exc(), flush=True)

# ---- determinism: ptgrad is seedless and should be reproducible -----------------------------------
try:
    d1 = builders.pt_gradient(adata, use_rep=USE_REP, pseudotime_key=PT, verbose=False)
    d2 = builders.pt_gradient(adata, use_rep=USE_REP, pseudotime_key=PT, verbose=False)
    check(np.allclose(d1, d2), "ptgrad: deterministic across calls")
except Exception:
    errors.append("ptgrad determinism raised:\n" + traceback.format_exc())

# ============================================================ B. API level
print("\n[B] GenDiff.setup_anndata sampler dispatch", flush=True)
built = {}
for name in ("knn", "smooth_nb", "nb_global", "ot", "ptgrad", "smooth"):
    try:
        a = adata.copy()
        # give NB/smooth a count source through adata.raw. _resolve_counts treats raw.X as log1p
        # (manuscript convention) and expm1's it back, so store log1p(counts), not linear counts.
        import anndata as ad
        a.raw = ad.AnnData(X=np.log1p(a.layers["counts"]).astype(np.float32), var=a.var.copy())
        GenDiff.setup_anndata(a, condition_key=COND, pseudotime_key=PT, use_rep=USE_REP,
                              control="ctrl", sampler=name, rebuild=True, verbose=False)
        dX = np.asarray(a.obsm["gendiff_dX"])
        built[name] = dX
        valid_dx(dX, n, g, f"setup(sampler={name!r})")
        check(a.uns["gendiff_setup"]["sampler"] == name, f"setup(sampler={name!r}): registry records sampler")
    except Exception:
        errors.append(f"setup(sampler={name!r}) raised:\n" + traceback.format_exc())
        print(traceback.format_exc(), flush=True)

# clustering must actually change the NB field (guards against cluster_key being ignored)
if "smooth_nb" in built and "nb_global" in built:
    check(not np.allclose(built["smooth_nb"], built["nb_global"]),
          "smooth_nb != nb_global (per-cluster vs global NB differ)")

# unknown sampler must raise
try:
    GenDiff.setup_anndata(adata.copy(), condition_key=COND, pseudotime_key=PT, use_rep=USE_REP,
                          control="ctrl", sampler="nope", rebuild=True, verbose=False)
    errors.append("unknown sampler did NOT raise")
    print("  FAIL: unknown sampler did not raise", flush=True)
except ValueError:
    print("  ok:   unknown sampler raises ValueError", flush=True)
except Exception:
    errors.append("unknown sampler raised wrong error:\n" + traceback.format_exc())

# dev targets registry parity
try:
    tl = tgt_reg.list_targets()
    for nm in ("smooth_nb", "nb_global", "ot", "ptgrad"):
        check(nm in tl, f"targets.registry lists {nm!r}")
except Exception:
    errors.append("targets.registry raised:\n" + traceback.format_exc())

# ============================================================ C. end-to-end build + 1-epoch train
print("\n[C] end-to-end build + 1-epoch CPU train (sampler='ptgrad', 1200-cell subset)", flush=True)
try:
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(n, 1200, replace=False))
    a = adata[sub].copy()
    GenDiff.setup_anndata(a, condition_key=COND, pseudotime_key=PT, use_rep=USE_REP,
                          control="ctrl", sampler="ptgrad", rebuild=True, verbose=False)
    model = GenDiff(a, hidden_size=(128, 256, 128), condition_emb_dim=256, timesteps=50, verbose=False)
    model.train(max_epochs=1, batch_size=128, accelerator="cpu", devices=1, num_workers=0,
                enable_progress_bar=False, default_root_dir="/tmp/smoke_samplers_run")
    dx = model.predict_delta(a, condition=str(a.obs[COND].iloc[0]))
    check(np.asarray(dx).shape[0] == a.n_obs, "end-to-end: predict_delta returns per-cell ΔX")
    print("  ok:   end-to-end build + train + predict completed", flush=True)
except Exception:
    errors.append("end-to-end raised:\n" + traceback.format_exc())
    print(traceback.format_exc(), flush=True)

# ============================================================ summary
out = {"errors": errors, "n_errors": len(errors), "notes": notes}
json.dump(out, open("/tmp/smoke_test_samplers.json", "w"), indent=2)
print(f"\n{'='*60}\n{'PASS' if not errors else 'FAIL'} — {len(errors)} error(s); wrote /tmp/smoke_test_samplers.json", flush=True)
sys.exit(1 if errors else 0)
