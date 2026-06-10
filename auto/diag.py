"""Stage-by-stage timing to find the bottleneck in sweep.py's module load + config 0."""
import sys, os, time, numpy as np, anndata as ad, scipy.sparse as sp
t0=time.time()
def mark(m): print(f'[{time.time()-t0:7.1f}s] {m}',flush=True)
DATA='/rds/user/wz369/hpc-work/GenDiff/data/integrated_mesc_group0_Nov7.h5ad'
mark('start import torch'); import torch; mark(f'torch ok cuda={torch.cuda.is_available()}')
import torch.nn as nn
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors
mark('import ot'); import ot; mark('ot ok')
mark('read h5ad'); a=ad.read_h5ad(DATA); mark(f'read ok N={a.n_obs} G={a.n_vars}')
X=(a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)).astype(np.float64)
mark('X dense')
conn=a.obsp['connectivities']
mark(f'conn type={type(conn).__name__} sparse={sp.issparse(conn)}')
if sp.issparse(conn):
    mark(f'conn nnz={conn.nnz} avg_deg={conn.nnz/a.n_obs:.1f}')
    conn_d=np.asarray(conn.todense())
else:
    conn_d=np.asarray(conn); mark(f'conn dense, nonzero_total={np.count_nonzero(conn_d)} avg_deg={np.count_nonzero(conn_d)/a.n_obs:.1f}')
mark('conn dense done')
N=a.n_obs
t=time.time(); adj=[np.nonzero(conn_d[i])[0] for i in range(N)]; mark(f'adj_euclid built in {time.time()-t:.1f}s, mean_deg={np.mean([len(x) for x in adj]):.1f}')
Xdm=np.asarray(a.obsm['X_diffmap'])
t=time.time(); _nn=NearestNeighbors(n_neighbors=16).fit(Xdm); _,_dmnn=_nn.kneighbors(Xdm); mark(f'diffmap NN in {time.time()-t:.1f}s')
# CUDA init
t=time.time(); z=torch.zeros(10,device='cuda' if torch.cuda.is_available() else 'cpu'); torch.cuda.synchronize() if torch.cuda.is_available() else None; mark(f'cuda init {time.time()-t:.1f}s')
mark('ALL STAGES DONE')
