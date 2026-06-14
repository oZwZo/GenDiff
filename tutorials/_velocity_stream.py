import warnings; warnings.filterwarnings("ignore")
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/rds/user/wz369/hpc-work/GenDiff")
import numpy as np
import matplotlib.pyplot as plt
import scvelo as scv
scv.settings.verbosity = 1

from gendiff import GenDiff
from gendiff_dev.data import registry as data_reg

adata = data_reg.load("barrnaseq")
model = GenDiff.load("runs/barrnaseq_gendiff/model", adata)

# ensure discrete_time (same recipe as setup)
pt = adata.obs["dpt_pseudotime"].astype(float).values
ptn = (pt - np.nanmin(pt)) / (np.nanmax(pt) - np.nanmin(pt) + 1e-12)
adata.obs["discrete_time"] = np.clip((ptn * (model.timesteps - 1)).round(), 0, model.timesteps - 1).astype(int)

import scipy.sparse as sp
X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
adata.layers["Ms"] = X.astype(np.float32)            # expression space for the velocity graph
scv.pp.neighbors(adata, use_rep="X_pca", n_neighbors=30)

# linear pseudotime gradient in UMAP, to test if flow points "forward"
emb = np.asarray(adata.obsm["X_umap"])
A = np.column_stack([emb, np.ones(len(emb))])
coef, *_ = np.linalg.lstsq(A, pt, rcond=None)
g = coef[:2]; g = g / (np.linalg.norm(g) + 1e-12)


def stream_for(cond_label, tag):
    dX = model.predict_delta(adata, condition=cond_label).astype(np.float32)
    adata.layers["velocity"] = dX
    scv.tl.velocity_graph(adata, vkey="velocity", xkey="Ms", n_jobs=4)
    scv.tl.velocity_embedding(adata, basis="umap", vkey="velocity")
    V2 = np.asarray(adata.obsm["velocity_umap"])
    nz = np.linalg.norm(V2, axis=1) > 0
    cos = (V2[nz] @ g) / (np.linalg.norm(V2[nz], axis=1) + 1e-12)
    print(f"[{tag}] mean cosine(2D arrow, +pseudotime dir) = {np.mean(cos):+.3f} "
          f"| frac pointing forward = {np.mean(cos > 0):.2f} (n={nz.sum()})", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    scv.pl.velocity_embedding_stream(adata, basis="umap", vkey="velocity", color="dpt_pseudotime",
                                     ax=ax[0], show=False, legend_loc="none", title=f"{tag}: pseudotime")
    scv.pl.velocity_embedding_stream(adata, basis="umap", vkey="velocity", color="cell_type",
                                     ax=ax[1], show=False, legend_loc="right margin", title=f"{tag}: cell_type")
    fig.tight_layout()
    out = f"tutorials/velocity_stream_{tag}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("  saved", out, flush=True)


stream_for("ctrl", "ctrl")
pert = [c for c in adata.uns["gendiff_token_dict"] if c != "ctrl"][0]
stream_for(pert, pert.replace("+", "_"))
print("DONE")
