"""
Lite-model parameter-sensitivity sweep for the improved ΔX sampler.
Per config: build neighbour graph -> sample ΔX target -> train lite velocity model -> predict velocity on
ALL cells -> score {composite delta-specificity (target quality), velocity-direction correctness (CBDir/
ICCoh, the trajectory proxy), reversion level}. Designed for `python sweep.py <idx>` (SLURM array) or
`python sweep.py all`.

Key design notes:
- The velocity-changing knobs are neighbour metric / horizon M / repeat r / coupling / alpha. De-reversion
  is a TRAINING-TARGET transform: train on residual, add reversion back at inference -> same velocity. We
  test it but rank on velocity-direction (which it cannot game) + composite (flagged for leakage).
- Trajectory proxy here = CBDir (cosine of predicted velocity vs local forward expression flow toward
  higher-pseudotime kNN) + ICCoh (within-cell-type coherence). Terminal/branch recovery (CellRank) is run
  separately for finalists (traj_eval.py).
"""
import sys, os, json, time, csv, numpy as np, anndata as ad, scipy.sparse as sp
import torch, torch.nn as nn
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors
import ot

DATA='/rds/user/wz369/hpc-work/GenDiff/data/integrated_mesc_group0_Nov7.h5ad'
HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'.cache'); os.makedirs(CACHE,exist_ok=True)
# reuse the geodesic cache produced by the round-2 screen if present
GEOCACHE_ALT='/rds/user/wz369/hpc-work/GenDiff/GenDiff-manuscript/response/R1.3/.Dgeo_cache.npy'
RES=os.path.join(HERE,'sweep_results.csv')
SEED=0; np.random.seed(SEED); torch.manual_seed(SEED); rng=np.random.default_rng(SEED)
dev='cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- data ----------------
a=ad.read_h5ad(DATA)
X=(a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)).astype(np.float64)
Xpca=np.asarray(a.obsm['X_pca']); Xdm=np.asarray(a.obsm['X_diffmap'])
cond=a.obs['condition'].astype(str).to_numpy(); pt=a.obs['dpt_pseudotime'].to_numpy().astype(float)
ctype=a.obs['cell_type'].astype(str).to_numpy() if 'cell_type' in a.obs.columns else np.array(['NA']*a.n_obs)
split=a.obs['split'].astype(str).to_numpy()
tr=np.where(split=='train')[0]; te=np.where(split=='test')[0]; N,G=X.shape
C=a.obs[['RA','Wnt','TgfB','Bmp','Fgf','Notch','Shh']].to_numpy().astype(np.float32)
# NOTE: all obsp graphs in this adata are stored FULLY DENSE (avg deg ~5744), so they are NOT usable
# as kNN adjacencies and Dijkstra on them would just return direct distances. We build genuine sparse
# kNN graphs from X_pca / X_diffmap below; the geodesic/Fermat distances are Dijkstra on the sparse
# PCA-kNN graph (a real manifold geodesic), NOT the old dense-distance cache.

trc=cond[tr]; counts={c:int((trc==c).sum()) for c in np.unique(trc)}
DOM=[c for c in sorted(counts,key=counts.get,reverse=True) if counts[c]>=30]
mean_ctrl=X[tr][trc=='ctrl'].mean(0)
t_c={c:X[tr][trc==c].mean(0)-mean_ctrl for c in DOM}
deg={c:np.argsort(np.abs(t_c[c]))[::-1][:50] for c in DOM}

# ---------------- neighbour graphs (built from scratch; obsp graphs are stored fully dense) ----------------
K=15
_nnp=NearestNeighbors(n_neighbors=K+1).fit(Xpca); _pd,_pi=_nnp.kneighbors(Xpca)   # PCA kNN dist + idx
adj_euclid=[_pi[i][1:] for i in range(N)]                                          # drop self
_nnd=NearestNeighbors(n_neighbors=K+1).fit(Xdm); _,_di=_nnd.kneighbors(Xdm); adj_diffmap=[_di[i][1:] for i in range(N)]
def _knn_graph(power):
    rows=np.repeat(np.arange(N),K); cols=_pi[:,1:].ravel(); w=_pd[:,1:].ravel().astype(np.float64)
    if power!=1: w=w**power
    g=sp.csr_matrix((w,(rows,cols)),shape=(N,N)); return g.maximum(g.T)            # symmetrize
def _geo(power):
    f=os.path.join(CACHE,f'geoknn_p{power}.npy')
    if os.path.exists(f): return np.load(f)
    D=dijkstra(_knn_graph(power),directed=False,indices=tr); np.save(f,D); return D
_GEO={}
def geo(power):
    if power not in _GEO: _GEO[power]=_geo(power)
    return _GEO[power]
tr_pos={int(i):p for p,i in enumerate(tr)}

def two_hop(adj,i):
    s=set(adj[i].tolist())
    for j in adj[i]: s.update(adj[j].tolist())
    s.discard(i); return np.fromiter(s,int)

def draw(metric,M,alpha,repeat):
    """return dict i->array of `repeat` neighbour indices (same-cond, higher-pt), per neighbour metric."""
    out={}
    geomat = geo(1) if metric=='geodesic' else (geo(2) if metric=='fermat' else None)
    for i in tr:
        if metric in ('geodesic','fermat'):
            g=geomat[tr_pos[i]]; reach=np.where(np.isfinite(g)&(g>0))[0]
            reach=reach[(cond[reach]==cond[i])&(pt[reach]>pt[i])]
            if reach.size==0: continue
            near=reach[np.argsort(g[reach])[:M]]; d=g[near]
        else:
            adj=adj_euclid if metric=='euclid' else adj_diffmap
            cand=two_hop(adj,i); cand=cand[(cond[cand]==cond[i])&(pt[cand]>pt[i])]
            if cand.size==0: continue
            sp_=Xpca if metric=='euclid' else Xdm
            d=np.linalg.norm(sp_[cand]-sp_[i],axis=1)
            if cand.size>M: near=cand[np.argsort(d)[:M]]; d=d[np.argsort(d)[:M]]
            else: near=cand
        w=(d.max()-d+0.1*d.min()); w=np.maximum(w,0)**alpha; p=w/w.sum() if w.sum()>0 else None
        out[i]=rng.choice(near,size=repeat,p=p)
    return out

def ot_draw(M,repeat):
    """OT coupling earlier->later within condition (PCA cost). single match (repeat ignored)."""
    out={}
    for c in DOM:
        ci=tr[trc==c]; med=np.median(pt[ci]); e=ci[pt[ci]<=med]; l=ci[pt[ci]>med]
        if e.size<3 or l.size<3: continue
        if e.size>300: e=rng.choice(e,300,replace=False)
        if l.size>300: l=rng.choice(l,300,replace=False)
        Mx=ot.dist(Xpca[e],Xpca[l],metric='sqeuclidean'); Mx/=Mx.max()+1e-9
        Gp=ot.emd(np.ones(len(e))/len(e),np.ones(len(l))/len(l),Mx); match=l[Gp.argmax(1)]
        for k,i in enumerate(e): out[i]=np.array([match[k]])
    return out

def target_from_draws(draws,repeat):
    D=np.zeros((N,G)); have=[]
    for i in tr:
        if i in draws: D[i]=X[draws[i][:repeat]].mean(0)-X[i]; have.append(i)
    return D,np.array(have)

# ---------------- lite velocity model ----------------
xmu=X[tr].mean(0).astype(np.float32); xsd=(X[tr].std(0)+1e-6).astype(np.float32)
Xs=((X-xmu)/xsd).astype(np.float32)
Xs_t=torch.tensor(Xs,device=dev); C_t=torch.tensor(C,device=dev)
class Net(nn.Module):
    def __init__(s,h=256):
        super().__init__(); d=G+7
        s.sk=nn.Linear(d,G); s.m=nn.Sequential(nn.Linear(d,h),nn.SiLU(),nn.Dropout(0.1),nn.Linear(h,G))
    def forward(s,x,c): h=torch.cat([x,c],1); return s.sk(h)+s.m(h)

def train_lite(D,have,steps=4000,bs=256):
    Dsd=(D[have].std(0)+1e-6).astype(np.float32); Dsd_t=torch.tensor(Dsd,device=dev)
    Dt=torch.tensor((D/Dsd).astype(np.float32),device=dev); ht=torch.tensor(have,device=dev)
    torch.manual_seed(SEED); net=Net().to(dev); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    for s in range(steps):
        idx=ht[torch.randint(0,len(have),(bs,),device=dev)]
        loss=((net(Xs_t[idx],C_t[idx])-Dt[idx])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        V=(net(Xs_t,C_t)*Dsd_t).cpu().numpy()    # predicted velocity for ALL cells (orig units)
    return V

# ---------------- metrics ----------------
def pear(x,y):
    x=x-x.mean(); y=y-y.mean(); d=np.sqrt((x*x).sum()*(y*y).sum()); return float((x*y).sum()/d) if d>0 else np.nan
def composite(D,idx):
    g=D[idx].mean(0); resid={c:D[idx[cond[idx]==c]].mean(0)-g for c in DOM if (cond[idx]==c).sum()>=20}; cs=list(resid)
    if len(cs)<2: return dict(D_pear=np.nan,D_dir=np.nan,PDS=np.nan)
    dp=np.nanmean([pear(resid[c][deg[c]],t_c[c][deg[c]]) for c in cs])
    dd=np.nanmean([(np.sign(resid[c][deg[c]])==np.sign(t_c[c][deg[c]])).mean() for c in cs])
    hit=sum(1 for c in cs if (not np.all(np.isnan([pear(resid[c],t_c[cp]) for cp in cs]))) and cs[int(np.nanargmax([pear(resid[c],t_c[cp]) for cp in cs]))]==c)
    return dict(D_pear=round(float(dp),3),D_dir=round(float(dd),3),PDS=round(hit/len(cs),3))
def velo_dir(V,cells):
    """CBDir: cos(V_i, local forward expr flow toward higher-pt kNN); ICCoh: within-cell-type coherence."""
    cb=[]; ic=[]
    for i in cells:
        nb=adj_euclid[i]; fwd=nb[pt[nb]>pt[i]]
        if fwd.size>0 and np.linalg.norm(V[i])>0:
            tdir=(X[fwd]-X[i]).mean(0); n=np.linalg.norm(tdir)
            if n>0: cb.append(float((V[i]*tdir).sum()/(np.linalg.norm(V[i])*n)))
        same=nb[ctype[nb]==ctype[i]]
        if same.size>0 and np.linalg.norm(V[i])>0:
            cs=[float((V[i]*V[j]).sum()/(np.linalg.norm(V[i])*np.linalg.norm(V[j])+1e-9)) for j in same if np.linalg.norm(V[j])>0]
            if cs: ic.append(np.mean(cs))
    return round(float(np.mean(cb)),3) if cb else np.nan, round(float(np.mean(ic)),3) if ic else np.nan
def revcorr(D,idx):
    v=[pear(D[i],X[i]) for i in idx[::3] if np.linalg.norm(D[i])>0 and np.std(X[i])>0]; return round(float(np.nanmean(v)),3)

# ---------------- config grid ----------------
def make_configs():
    cfgs=[]
    for metric in ['euclid','geodesic','fermat']:
        for M in [15,30,60]:
            for r in [1,3]:
                cfgs.append(dict(metric=metric,M=M,repeat=r,coupling='traverse',alpha=1,derev=False))
    for r in [1,3]:
        cfgs.append(dict(metric='ot',M=0,repeat=r,coupling='ot',alpha=1,derev=False))
    # de-rev variants on the round-1 front-runners (to confirm velocity-invariance after add-back)
    for metric in ['geodesic']:
        for M in [15,30]:
            cfgs.append(dict(metric=metric,M=M,repeat=1,coupling='traverse',alpha=1,derev=True))
    return cfgs
CONFIGS=make_configs()

def de_rev(D,have):
    Xi=X[have]; Di=D[have]; mx=Xi.mean(0); md=Di.mean(0); vx=Xi.var(0)+1e-12
    b=((Xi-mx)*(Di-md)).mean(0)/vx; a=md-b*mx
    R=np.zeros_like(D); R[have]=Di-(a+b*Xi); return R,(a,b)

def run_one(cfg):
    t0=time.time()
    if cfg['coupling']=='ot': draws=ot_draw(cfg['M'],cfg['repeat'])
    else: draws=draw(cfg['metric'],cfg['M'],cfg['alpha'],cfg['repeat'])
    D,have=target_from_draws(draws,cfg['repeat'])
    comp_target=composite(D,have); rc=revcorr(D,have)
    if cfg['derev']:
        Dtr,(aa,bb)=de_rev(D,have); V=train_lite(Dtr,have); V=V+(aa+bb*X)   # add reversion back -> velocity
    else:
        V=train_lite(D,have)
    comp_pred=composite(V,tr)
    cb,ic=velo_dir(V,te)            # trajectory proxy on held-out test cells
    row=dict(**cfg, n=len(have), revcorr=rc, **{f'tgt_{k}':v for k,v in comp_target.items()},
             **{f'pred_{k}':v for k,v in comp_pred.items()}, CBDir=cb, ICCoh=ic, sec=round(time.time()-t0,1))
    return row

def append(row):
    new=not os.path.exists(RES)
    with open(RES,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if new: w.writeheader()
        w.writerow(row)

if __name__=='__main__':
    arg=sys.argv[1] if len(sys.argv)>1 else 'all'
    print(f'device={dev} nconfigs={len(CONFIGS)}')
    items=range(len(CONFIGS)) if arg=='all' else [int(arg)]
    for i in items:
        cfg=CONFIGS[i]; print(f'[{i}] {cfg} ...',flush=True)
        try:
            row=run_one(cfg); append(row)
            print(f'   -> CBDir={row["CBDir"]} ICCoh={row["ICCoh"]} pred_Dpear={row.get("pred_D_pear")} revcorr={row["revcorr"]} ({row["sec"]}s)',flush=True)
        except Exception as e:
            print(f'   !! {cfg} failed: {e!r}',flush=True)
    print('DONE')
