"""
TF-Atlas trajectory evaluation for a predicted per-cell velocity V (.npy, N x 5000).
Unlike BarRNA-seq, this dataset HAS real RNA-velocity moments (Ms/Mu) and 25 louvain clusters, so CellRank
runs natively (xkey='Ms') and branch recovery is genuine (macrostate <-> louvain agreement on terminal cells).
Differentiation axis = velocity_pseudotime; neighbour graph from X_pca_harmony.
Metrics (-> tf_traj_results.csv):
  CBDir       cosine(V, local forward expr flow toward higher-velocity_pseudotime kNN)
  ICCoh       within-louvain coherence of V
  rev_collin  cosine(-X, forward flow)  [reversion confound diagnostic]
  pt_drift    E[vpt_j - vpt_i] under CellRank transition matrix (>0 = flows forward)
  pt_fwd_frac fraction of cells with positive drift
  ARI_lou/NMI_lou  macrostate vs louvain agreement on terminal (top-30% vpt) cells
  n_macro     # CellRank macrostates ; n_branches  distinct terminal louvain majorities
Usage: python tf_traj.py <velocity.npy> <tag>
"""
import sys, os, json, csv, numpy as np, anndata as ad, scipy.sparse as sp, warnings
warnings.filterwarnings('ignore')
DATA='/rds/user/wz369/hpc-work/GenDiff/data/210717_TFAtlas1M2_diffed.h5ad'
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,'tf_traj_results.csv')

def main(vpath, tag):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import pandas as pd
    a=ad.read_h5ad(DATA)
    X=(a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)).astype(np.float64)
    V=np.load(vpath).astype(np.float64); assert V.shape==X.shape,(V.shape,X.shape)
    pt=a.obs['velocity_pseudotime'].to_numpy().astype(float)
    lou=a.obs['louvain'].astype(str).to_numpy()
    Xh=np.asarray(a.obsm['X_pca_harmony']); _nn=NearestNeighbors(n_neighbors=16).fit(Xh); _,_pi=_nn.kneighbors(Xh)
    adj=[_pi[i][1:] for i in range(a.n_obs)]
    cb=[]; ic=[]; rc=[]
    for i in range(a.n_obs):
        nb=adj[i]; fwd=nb[pt[nb]>pt[i]]; vi=V[i]; nv=np.linalg.norm(vi); xi=X[i]; nx=np.linalg.norm(xi)
        if fwd.size>0:
            u=(X[fwd]-xi).mean(0); nu=np.linalg.norm(u)
            if nu>0:
                if nv>0: cb.append(float((vi*u).sum()/(nv*nu)))
                if nx>0: rc.append(float((-xi*u).sum()/(nx*nu)))
        same=nb[lou[nb]==lou[i]]
        if same.size>0 and nv>0:
            c=[float((vi*V[j]).sum()/(nv*np.linalg.norm(V[j])+1e-9)) for j in same if np.linalg.norm(V[j])>0]
            if c: ic.append(np.mean(c))
    res=dict(tag=tag, CBDir=round(float(np.mean(cb)),4) if cb else float('nan'),
             ICCoh=round(float(np.mean(ic)),4) if ic else float('nan'),
             rev_collin=round(float(np.mean(rc)),4) if rc else float('nan'))
    try:
        import scanpy as sc, scvelo as scv, cellrank as cr
        # CellRank GPCCA densifies the NxN matrix (no petsc/slepc) -> O(N^3); intractable at 28825 cells.
        # Run the kernel + macrostates on a deterministic louvain-stratified subsample (~6000 cells).
        rng=np.random.default_rng(0); NSUB=6000
        if a.n_obs>NSUB:
            frac=NSUB/a.n_obs; keep=[]
            for l in np.unique(lou):
                idx=np.where(lou==l)[0]; k=max(2,int(round(len(idx)*frac)))
                keep.append(rng.choice(idx,size=min(k,len(idx)),replace=False))
            sub=np.sort(np.concatenate(keep))
        else: sub=np.arange(a.n_obs)
        asub=a[sub].copy(); pts=pt[sub]; lous=lou[sub]
        sc.pp.neighbors(asub, n_neighbors=15, use_rep='X_pca_harmony')
        asub.layers['velocity']=V[sub].astype(np.float32)
        scv.tl.velocity_graph(asub, vkey='velocity', xkey='Ms', n_jobs=1)   # Ms moments exist natively
        vk=cr.kernels.VelocityKernel(asub, vkey='velocity', xkey='Ms'); vk.compute_transition_matrix()
        T=vk.transition_matrix; T=T.tocsr() if sp.issparse(T) else sp.csr_matrix(T)
        drift=np.asarray(T.dot(pts)).ravel()-pts
        res['pt_drift']=round(float(np.mean(drift)),5); res['pt_fwd_frac']=round(float((drift>0).mean()),4); res['n_sub']=len(sub)
        try:
            nst=12
            try: g=cr.estimators.GPCCA(vk); g.compute_schur(n_components=nst+3)
            except Exception: g=cr.estimators.GPCCA(vk)
            g.compute_macrostates(n_states=nst)
            ms=asub.obs['macrostates'].astype(str).to_numpy() if 'macrostates' in asub.obs.columns else g.macrostates.cat.codes.to_numpy().astype(str)
            ok=ms!='nan'; term=ok&(pts>=np.quantile(pts,0.70))
            if term.sum()>10 and len(set(ms[term]))>1:
                ari=adjusted_rand_score(lous[term],ms[term]); nmi=normalized_mutual_info_score(lous[term],ms[term])
            else: ari=nmi=float('nan')
            df=pd.DataFrame({'ms':ms[term],'l':lous[term]})
            maj={m:df[df.ms==m].l.value_counts().index[0] for m in df.ms.unique()} if len(df) else {}
            res.update(n_macro=len(set(ms[ok])), ARI_lou=round(float(ari),4), NMI_lou=round(float(nmi),4),
                       n_branches=len(set(maj.values())), branch_map=json.dumps(maj)[:300])
        except Exception as e2:
            res.update(n_macro=-1, ARI_lou=float('nan'), traj_err='gpcca:'+repr(e2)[:160])
    except Exception as e:
        res.update(n_macro=-1, pt_drift=float('nan'), ARI_lou=float('nan'), traj_err=repr(e)[:200])
    new=not os.path.exists(RES)
    with open(RES,'a',newline='') as f:
        fields=['tag','CBDir','ICCoh','rev_collin','pt_drift','pt_fwd_frac','n_sub','n_macro','ARI_lou','NMI_lou','n_branches','branch_map','traj_err']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        if new: w.writeheader()
        w.writerow(res)
    print('TFTRAJ',tag,{k:res.get(k) for k in ['CBDir','rev_collin','pt_drift','ARI_lou','n_branches']},flush=True)

if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])
