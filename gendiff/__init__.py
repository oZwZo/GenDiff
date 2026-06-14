"""
gendiff — high-level, scvi-tools-style API for GenDiff.

    from gendiff import GenDiff
    GenDiff.setup_anndata(adata, condition_key=..., pseudotime_key=..., use_rep=...)
    model = GenDiff(adata); model.train(); model.predict_delta(adata, condition=...)

The ΔX supervision defaults to the revised knn_sampler (config1 = geodesic, M=15, repeat=3). This is
the integration layer above the internal `src` (model/learner) and `gendiff_dev` (sampler) packages.
"""
from .model import GenDiff

__all__ = ["GenDiff"]
