"""Generate TF-Atlas velocity fields for the trajectory head-to-head. Imports tf_sweep (data + Root/global
samplers + TF-embedding lite model). Saves learned fields for the finalist sampler and an r1 low-averaging
reference, plus the dataset's REAL scVelo velocity layer (gold reference) and trivial -X."""
import os, numpy as np
import tf_sweep as T
OUT=os.path.join(T.HERE,'tf_fields'); os.makedirs(OUT,exist_ok=True)

def field(cfg):
    draws=T.draw(cfg['metric'],cfg['M'],cfg.get('alpha',1),cfg['repeat'])
    D,have=T.target_from_draws(draws,cfg['repeat']); return T.train_lite(D,have)

for name,cfg in [('geo_m15_r3',dict(metric='geodesic',M=15,repeat=3,alpha=1)),
                 ('euclid_m15_r1',dict(metric='euclid',M=15,repeat=1,alpha=1))]:
    V=field(cfg); np.save(os.path.join(OUT,f'{name}_pred.npy'),V.astype(np.float32)); print('saved',name,V.shape,flush=True)

# real scVelo velocity layer = gold reference
import scipy.sparse as sp
rv=T.a.layers['velocity']; rv=rv.toarray() if sp.issparse(rv) else np.asarray(rv)
rv=np.nan_to_num(rv).astype(np.float32); np.save(os.path.join(OUT,'realvelo_pred.npy'),rv); print('saved realvelo',rv.shape,flush=True)
# trivial -X
np.save(os.path.join(OUT,'minusX_pred.npy'),(-T.X).astype(np.float32)); print('saved minusX',flush=True)
print('DONE')
