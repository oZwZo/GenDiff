"""
Trajectory evaluation for a predicted per-cell velocity V (.npy, N x G, all cells, gene space).
NOTE on this dataset: cell_type is uniform ('mESCs') and there is NO germ-layer lineage annotation, so
terminal/branch recovery must be defined against the perturbation `condition` (32 signalling combos that
drive distinct fates). Also, in this z-scored space the bulk differentiation flow is largely collinear
with reversion (-X), so raw direction metrics are confounded; we therefore report several complementary
quantities and a confound diagnostic.

Metrics written to traj_results.csv:
  CBDir      cosine(V_i, local forward expr flow toward higher-pseudotime kNN)  [reversion-confounded]
  ICCoh      within-condition coherence of V (cosine to same-condition neighbours' V)
  rev_collin cosine(-X_i, forward flow u_i) averaged  -- the confound: how reversion-like the diff axis is
  pt_drift   E[pt_j-pt_i] under the velocity-derived CellRank transition matrix (>0 = flows forward in pt)
  pt_fwd_frac fraction of cells with positive pt drift
  n_macro    # CellRank macrostates
  ARI_cond / NMI_cond   agreement of macrostate labels with `condition` on terminal (top-30% pt) cells
  n_branches # macrostates whose terminal-cell majority condition is distinct (non-ctrl)
Usage: python traj_eval.py <velocity.npy> <tag>
"""
import sys, os, json, csv, numpy as np, anndata as ad, scipy.sparse as sp, warnings
warnings.filterwarnings('ignore')
DATA='/rds/user/wz369/hpc-work/GenDiff/data/integrated_mesc_group0_Nov7.h5ad'
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,'traj_results.csv')

def main(vpath, tag):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import pandas as pd
    a=ad.read_h5ad(DATA)
    X=(a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)).astype(np.float64)
    V=np.load(vpath).astype(np.float64); assert V.shape==X.shape, (V.shape,X.shape)
    pt=a.obs['dpt_pseudotime'].to_numpy().astype(float)
    condition=a.obs['condition'].astype(str).to_numpy()
    Xpca=np.asarray(a.obsm['X_pca']); _nn=NearestNeighbors(n_neighbors=16).fit(Xpca); _,_pi=_nn.kneighbors(Xpca)
    adj=[_pi[i][1:] for i in range(a.n_obs)]
    # (1) direction metrics + confound diagnostic
    cb=[]; ic=[]; rc=[]
    for i in range(a.n_obs):
        nb=adj[i]; fwd=nb[pt[nb]>pt[i]]; vi=V[i]; nv=np.linalg.norm(vi); xi=X[i]; nx=np.linalg.norm(xi)
        if fwd.size>0:
            u=(X[fwd]-xi).mean(0); nu=np.linalg.norm(u)
            if nu>0:
                if nv>0: cb.append(float((vi*u).sum()/(nv*nu)))
                if nx>0: rc.append(float((-xi*u).sum()/(nx*nu)))     # reversion vs differentiation flow
        same=nb[condition[nb]==condition[i]]
        if same.size>0 and nv>0:
            cc=[float((vi*V[j]).sum()/(nv*np.linalg.norm(V[j])+1e-9)) for j in same if np.linalg.norm(V[j])>0]
            if cc: ic.append(np.mean(cc))
    res=dict(tag=tag,
             CBDir=round(float(np.mean(cb)),4) if cb else float('nan'),
             ICCoh=round(float(np.mean(ic)),4) if ic else float('nan'),
             rev_collin=round(float(np.mean(rc)),4) if rc else float('nan'))
    # (2) CellRank: pseudotime drift under transition matrix + macrostate<->condition recovery
    try:
        import scanpy as sc, scvelo as scv, cellrank as cr
        sc.pp.neighbors(a, n_neighbors=15, use_rep='X_pca')
        a.layers['velocity']=V.astype(np.float32); a.layers['expr']=X.astype(np.float32)
        scv.tl.velocity_graph(a, vkey='velocity', xkey='expr', n_jobs=1)
        vk=cr.kernels.VelocityKernel(a, vkey='velocity', xkey='expr'); vk.compute_transition_matrix()
        T=vk.transition_matrix; T=T.tocsr() if sp.issparse(T) else sp.csr_matrix(T)
        # expected next-step pseudotime change: (T @ pt) - pt
        drift=np.asarray(T.dot(pt)).ravel()-pt
        res['pt_drift']=round(float(np.mean(drift)),5); res['pt_fwd_frac']=round(float((drift>0).mean()),4)
        nst=8
        try: g=cr.estimators.GPCCA(vk); g.compute_schur(n_components=nst+2)
        except Exception: g=cr.estimators.GPCCA(vk)
        g.compute_macrostates(n_states=nst)
        ms=a.obs['macrostates'].astype(str).to_numpy() if 'macrostates' in a.obs.columns else g.macrostates.cat.codes.to_numpy().astype(str)
        ok=ms!='nan'
        term=ok & (pt>=np.quantile(pt,0.70))                 # terminal = top-30% pseudotime, assigned cells
        if term.sum()>10 and len(set(ms[term]))>1:
            ari=adjusted_rand_score(condition[term],ms[term]); nmi=normalized_mutual_info_score(condition[term],ms[term])
        else:
            ari=nmi=float('nan')
        # branch map: terminal-cell majority condition per macrostate
        df=pd.DataFrame({'ms':ms[term],'c':condition[term]})
        maj={m:df[df.ms==m].c.value_counts().index[0] for m in df.ms.unique()} if len(df) else {}
        nbranch=len(set(v for v in maj.values() if v!='ctrl'))
        res.update(n_macro=len(set(ms[ok])), ARI_cond=round(float(ari),4), NMI_cond=round(float(nmi),4),
                   n_branches=nbranch, branch_map=json.dumps(maj)[:300])
    except Exception as e:
        res.update(n_macro=-1, pt_drift=float('nan'), ARI_cond=float('nan'), traj_err=repr(e)[:200])
    new=not os.path.exists(RES)
    with open(RES,'a',newline='') as f:
        # union of keys so header covers error rows too
        fields=['tag','CBDir','ICCoh','rev_collin','pt_drift','pt_fwd_frac','n_macro','ARI_cond','NMI_cond','n_branches','branch_map','traj_err']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(res)
    print('TRAJ',tag,{k:res.get(k) for k in ['CBDir','rev_collin','pt_drift','pt_fwd_frac','ARI_cond','n_branches']})

if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])
