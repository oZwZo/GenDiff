"""
Produce per-cell velocity fields (.npy, N x G, original gene units) for the trajectory head-to-head.
Reuses sweep.py's neighbour samplers + lite velocity model so every method is on equal footing.
Saves both the sampler TARGET field and the LEARNED (lite-model predicted) field. The learned field is
the fair object for CBDir/CellRank (the target's direction trivially matches forward flow).
Usage: python gen_velocity.py
"""
import os, numpy as np
import sweep as S   # importing runs data load + sparse-kNN graph build (geodesic cache reused)
OUT=os.path.join(S.HERE,'fields'); os.makedirs(OUT,exist_ok=True)

def field_for_config(cfg):
    if cfg['coupling']=='ot': draws=S.ot_draw(cfg['M'],cfg['repeat'])
    else: draws=S.draw(cfg['metric'],cfg['M'],cfg['alpha'],cfg['repeat'])
    D,have=S.target_from_draws(draws,cfg['repeat'])
    if cfg.get('derev'):
        Dtr,(aa,bb)=S.de_rev(D,have); V=S.train_lite(Dtr,have); V=V+(aa+bb*S.X)
    else:
        V=S.train_lite(D,have)
    return V, D, have

def save(name, V, D=None):
    np.save(os.path.join(OUT,f'{name}_pred.npy'), V.astype(np.float32))
    if D is not None: np.save(os.path.join(OUT,f'{name}_target.npy'), D.astype(np.float32))
    print('saved', name, 'pred', V.shape, flush=True)

# --- finalists (new sampler) ---
for name,cfg in [('geo_m15_r3',dict(metric='geodesic',M=15,repeat=3,coupling='traverse',alpha=1,derev=False)),
                 ('fermat_m15_r3',dict(metric='fermat',M=15,repeat=3,coupling='traverse',alpha=1,derev=False))]:
    V,D,have=field_for_config(cfg); save(name,V,D)

# --- baseline: current sampler (Tr_SampledX_r100) as target -> lite learned field ---
oldD=np.zeros((S.N,S.G)); oldD[S.tr]=np.asarray(S.a.obsm['Tr_SampledX_r100']).astype(np.float64)[S.tr]
Vold=S.train_lite(oldD,S.tr); save('old_r100',Vold,oldD)

# --- baseline: trivial -X (mean reversion) ---
np.save(os.path.join(OUT,'minusX_pred.npy'), (-S.X).astype(np.float32)); print('saved minusX', flush=True)
print('DONE')
