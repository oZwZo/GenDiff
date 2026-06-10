"""
gendiff_dev.models.registry — predictor registry (architecture §4.3, terminal-location-owned).

Native contract (a runner-friendly simplification of trajectory_model/baselines._BasePredictor that takes
the condition explicitly instead of decoding a one-hot tail):

    predictor.fit(X_gene, dx, cond) -> self
    predictor.predict_delta(X_gene, cond) -> (n, n_genes)     # per-cell ΔX
    predictor.predict_population(adata, condition) -> (m, d)  # OPTIONAL, transport models only
    predictor.name : str

`get(name)` returns an instance. Three reference predictors are defined here so the package runs
standalone; the full existing zoo (ridge/cae/otcfm/gate/zerov/ot_anchored/ode_horizon/combo/stoch/...)
in trajectory_model/baselines.py is reachable via `bridge_registry()` and wrapped by `BaselinesAdapter`
(it builds the [gene||one-hot] input those predictors expect). Migrating each track to the native
contract is the incremental §7.3 step; nothing here forces a rewrite.
"""
from __future__ import annotations
import os, sys, numpy as np

LOCAL = ("identity_null", "condition_mean", "ridge")


def list_models():
    names = list(LOCAL)
    try:
        names += [k for k in bridge_registry() if k not in names]
    except Exception:
        pass
    return names


def get(name, **kwargs):
    if name in LOCAL:
        return {"identity_null": IdentityNull, "condition_mean": ConditionMean, "ridge": Ridge}[name](**kwargs)
    return BaselinesAdapter(name, **kwargs)   # existing-zoo bridge


# ---------------------------------------------------------------- reference predictors (native contract)
class _Base:
    name = "base"
    def fit(self, X_gene, dx, cond): return self
    def predict_delta(self, X_gene, cond): raise NotImplementedError


class IdentityNull(_Base):
    """ΔX = 0 (the do-nothing null; the floor every model must clear on local accuracy)."""
    name = "identity_null"
    def fit(self, X_gene, dx, cond): self.G = dx.shape[1]; return self
    def predict_delta(self, X_gene, cond): return np.zeros((len(X_gene), self.G), np.float32)


class ConditionMean(_Base):
    """ΔX = mean training ΔX for the cell's condition (falls back to global mean)."""
    name = "condition_mean"
    def fit(self, X_gene, dx, cond):
        cond = np.asarray(cond).astype(str); self.g = dx.mean(0)
        self.m = {c: dx[cond == c].mean(0) for c in np.unique(cond)}; return self
    def predict_delta(self, X_gene, cond):
        cond = np.asarray(cond).astype(str)
        return np.stack([self.m.get(c, self.g) for c in cond]).astype(np.float32)


class Ridge(_Base):
    """Linear ΔX = W [x || onehot(cond)] (the manuscript's competitive linear baseline)."""
    name = "ridge"
    def __init__(self, alpha=100.0): self.alpha = alpha
    def _oh(self, cond):
        cond = np.asarray(cond).astype(str)
        return np.stack([(cond == c).astype(np.float32) for c in self.vocab], 1) if len(self.vocab) else np.zeros((len(cond), 0), np.float32)
    def fit(self, X_gene, dx, cond):
        from sklearn.linear_model import Ridge as _R
        self.vocab = sorted(set(np.asarray(cond).astype(str)))
        self.model = _R(alpha=self.alpha).fit(np.hstack([X_gene, self._oh(cond)]), dx); return self
    def predict_delta(self, X_gene, cond):
        return self.model.predict(np.hstack([X_gene, self._oh(cond)])).astype(np.float32)


# ---------------------------------------------------------------- bridge to trajectory_model/baselines.py
def _traj_dir():
    return os.environ.get("GENDIFF_TRAJ_DIR",
                          "/rds/user/wz369/hpc-work/GenDiff/GenDiff-manuscript/trajectory_model")


def bridge_registry():
    """Return the existing baselines.REGISTRY (importing the track modules so they self-register)."""
    d = _traj_dir()
    if d not in sys.path: sys.path.insert(0, d)
    import baselines
    for t in ("track_zerov", "track_gate", "track_ot", "track_ode", "track_combo", "track_stoch"):
        try: __import__(t)
        except Exception: pass
    return baselines.REGISTRY


class BaselinesAdapter(_Base):
    """Wrap an existing baselines.REGISTRY predictor (one-hot-tail contract) behind the native contract.
    Best-effort: builds [gene || onehot(cond)] for fit/predict_delta. Validate parity against
    run_scorecard.py before trusting headline numbers (§7.5)."""
    def __init__(self, name, **kwargs):
        self.name = name; self._impl = bridge_registry()[name](**kwargs)
        self.applicable_levels = getattr(self._impl, "applicable_levels", {"L1", "L2a", "L2b", "L3"})
    def _oh(self, cond):
        cond = np.asarray(cond).astype(str)
        return np.stack([(cond == c).astype(np.float32) for c in self.vocab], 1)
    def fit(self, X_gene, dx, cond, adata=None, split_mask=None):
        self.vocab = sorted(set(np.asarray(cond).astype(str)))
        self._impl.fit(np.hstack([X_gene, self._oh(cond)]), dx, adata, split_mask); return self
    def predict_delta(self, X_gene, cond):
        return np.asarray(self._impl.predict_delta(np.hstack([X_gene, self._oh(cond)]))).astype(np.float32)
    def predict_population(self, adata, condition):
        return self._impl.predict_population(adata, condition)
    def has_population(self):
        return hasattr(self._impl, "predict_population")
