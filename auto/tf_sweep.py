"""
TF-Atlas adaptation of the ΔX-sampler sweep. Differences from the BarRNA-seq sweep (sweep.py):
  - Dataset: 210717_TFAtlas1M2_diffed.h5ad (28825 x 5000, log-normalised X). Genuinely SPARSE obsp graphs.
  - Perturbation = `TF` (2535 TFs, median ~4 cells/TF). Controls = GFP/mCherry. Condition modelled by a TF
    EMBEDDING (not a 7-pathway one-hot).
  - Sampler = ROOT/GLOBAL: ΔX = mean over r higher-pseudotime neighbours drawn from the GLOBAL pool
    (condition-agnostic), matching GenDiff's Root_Diffuse for TF-Atlas. (Same-TF traverse is infeasible at
    ~4 cells/TF.)
  - Pseudotime axis = velocity_pseudotime (RNA-velocity derived; controls sit below average -> correct
    differentiation direction). Graphs built from X_pca_harmony (batch-corrected).
  - Metrics: trajectory-first (CBDir / ICCoh-by-louvain / rev_collin / CBDir uses velocity_pseudotime);
    condition-specificity (composite) computed ONLY on TFs with >=20 cells vs the GFP/mCherry control.
  - No `split` column in the file -> a deterministic 80/10/10 split is created here.
CLI: python tf_sweep.py <idx> | all   ->  appends to tf_sweep_results.csv
"""
import sys, os, json, time, csv, numpy as np, anndata as ad, scipy.sparse as sp
import torch, torch.nn as nn
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

DATA='/rds/user/wz369/hpc-work/GenDiff/data/210717_TFAtlas1M2_diffed.h5ad'
HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'.tfcache'); os.makedirs(CACHE,exist_ok=True)
RES=os.path.join(HERE,'tf_sweep_results.csv')
SEED=0; np.random.seed(SEED); torch.manual_seed(SEED); rng=np.random.default_rng(SEED)
dev='cuda' if torch.cuda.is_available() else 'cpu'
CTRL_TFS={'TFORF3549-GFP','TFORF3550-mCherry'}

# ---------------- data ----------------
a=ad.read_h5ad(DATA)
X=(a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)).astype(np.float32)
N,G=X.shape
Xrep=np.asarray(a.obsm['X_pca_harmony']); Xdm=np.asarray(a.obsm['X_diffmap_None'])
pt=a.obs['velocity_pseudotime'].to_numpy().astype(float)
tf_str=a.obs['TF'].astype(str).to_numpy()
louvain=a.obs['louvain'].astype(str).to_numpy()
tfs,tf_id=np.unique(tf_str,return_inverse=True)            # tf_id: int label per cell
n_tf=len(tfs)
is_ctrl=np.isin(tf_str,list(CTRL_TFS))

# deterministic 80/10/10 split
perm=rng.permutation(N); ntr=int(0.8*N); nval=int(0.1*N)
split=np.array(['train']*N,dtype=object)
split[perm[ntr:ntr+nval]]='val'; split[perm[ntr+nval:]]='test'
tr=np.where(split=='train')[0]; te=np.where(split=='test')[0]; va=np.where(split=='val')[0]

# condition-specificity setup: TFs with >=20 TRAIN cells (excluding controls), vs control mean
trtf=tf_str[tr]; cnt={t:int((trtf==t).sum()) for t in np.unique(trtf)}
DOMTF=[t for t in cnt if cnt[t]>=20 and t not in CTRL_TFS]
mean_ctrl=X[tr][np.isin(trtf,list(CTRL_TFS))].mean(0) if is_ctrl[tr].any() else X[tr].mean(0)
t_c={t:X[tr][trtf==t].mean(0)-mean_ctrl for t in DOMTF}
deg={t:np.argsort(np.abs(t_c[t]))[::-1][:50] for t in DOMTF}
print(f'N={N} G={G} n_tf={n_tf} train/test/val={len(tr)}/{len(te)}/{len(va)} DOMTF(>=20)={len(DOMTF)} ctrl_cells={int(is_ctrl.sum())}',flush=True)

# ---------------- neighbour graphs (sparse; built from harmony PCA + diffmap) ----------------
K=15
_nnp=NearestNeighbors(n_neighbors=K+1).fit(Xrep); _pd,_pi=_nnp.kneighbors(Xrep)
adj_euclid=[_pi[i][1:] for i in range(N)]
_nnd=NearestNeighbors(n_neighbors=K+1).fit(Xdm); _,_di=_nnd.kneighbors(Xdm); adj_diffmap=[_di[i][1:] for i in range(N)]
def _knn_graph(power):
    rows=np.repeat(np.arange(N),K); cols=_pi[:,1:].ravel(); w=_pd[:,1:].ravel().astype(np.float64)
    if power!=1: w=w**power
    g=sp.csr_matrix((w,(rows,cols)),shape=(N,N)); return g.maximum(g.T)
def _geo(power):
    f=os.path.join(CACHE,f'geoknn_p{power}.npy')
    if os.path.exists(f): return np.load(f)
    print(f'computing geodesic Dijkstra (power={power}) from {len(tr)} sources ...',flush=True)
    D=dijkstra(_knn_graph(power),directed=False,indices=tr).astype(np.float32); np.save(f,D); return D
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
    """ROOT/GLOBAL: per train cell -> r higher-pseudotime neighbours from the global pool (no TF filter)."""
    out={}
    geomat=geo(1) if metric=='geodesic' else (geo(2) if metric=='fermat' else None)
    for i in tr:
        if metric in ('geodesic','fermat'):
            gd=geomat[tr_pos[i]]; reach=np.where(np.isfinite(gd)&(gd>0))[0]
            reach=reach[pt[reach]>pt[i]]
            if reach.size==0: continue
            near=reach[np.argsort(gd[reach])[:M]]; d=gd[near]
        else:
            adj=adj_euclid if metric=='euclid' else adj_diffmap
            cand=two_hop(adj,i); cand=cand[pt[cand]>pt[i]]
            if cand.size==0: continue
            sp_=Xrep if metric=='euclid' else Xdm
            d=np.linalg.norm(sp_[cand]-sp_[i],axis=1)
            if cand.size>M: order=np.argsort(d)[:M]; near=cand[order]; d=d[order]
            else: near=cand
        w=(d.max()-d+0.1*d.min()); w=np.maximum(w,0)**alpha; p=w/w.sum() if w.sum()>0 else None
        out[i]=rng.choice(near,size=repeat,p=p)
    return out

def target_from_draws(draws,repeat):
    D=np.zeros((N,G),dtype=np.float32); have=[]
    for i in tr:
        if i in draws: D[i]=X[draws[i][:repeat]].mean(0)-X[i]; have.append(i)
    return D,np.array(have)

# ---------------- lite velocity model (TF embedding) ----------------
xmu=X[tr].mean(0); xsd=X[tr].std(0)+1e-6
Xs=((X-xmu)/xsd).astype(np.float32)
Xs_t=torch.tensor(Xs,device=dev); tfid_t=torch.tensor(tf_id,device=dev,dtype=torch.long)
class Net(nn.Module):
    def __init__(s,emb=64,h=256):
        super().__init__(); s.emb=nn.Embedding(n_tf,emb); d=G+emb
        s.sk=nn.Linear(d,G); s.m=nn.Sequential(nn.Linear(d,h),nn.SiLU(),nn.Dropout(0.1),nn.Linear(h,G))
    def forward(s,x,tfi): h=torch.cat([x,s.emb(tfi)],1); return s.sk(h)+s.m(h)
def train_lite(D,have,steps=4000,bs=256):
    Dsd=(D[have].std(0)+1e-6).astype(np.float32); Dsd_t=torch.tensor(Dsd,device=dev)
    Dt=torch.tensor((D/Dsd).astype(np.float32),device=dev); ht=torch.tensor(have,device=dev)
    torch.manual_seed(SEED); net=Net().to(dev); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4)
    for s in range(steps):
        idx=ht[torch.randint(0,len(have),(bs,),device=dev)]
        loss=((net(Xs_t[idx],tfid_t[idx])-Dt[idx])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        V=np.empty((N,G),dtype=np.float32); bs2=4096
        for s in range(0,N,bs2):
            sl=slice(s,min(N,s+bs2)); V[sl]=(net(Xs_t[sl],tfid_t[sl])*Dsd_t).cpu().numpy()
    return V

# ---------------- metrics ----------------
def pear(x,y):
    x=x-x.mean(); y=y-y.mean(); d=np.sqrt((x*x).sum()*(y*y).sum()); return float((x*y).sum()/d) if d>0 else np.nan
def composite(D,idx):
    g=D[idx].mean(0); cs=[t for t in DOMTF if (tf_str[idx]==t).sum()>=15]
    resid={t:D[idx[tf_str[idx]==t]].mean(0)-g for t in cs}
    if len(cs)<2: return dict(D_pear=np.nan,D_dir=np.nan,PDS=np.nan,nTF=len(cs))
    dp=np.nanmean([pear(resid[t][deg[t]],t_c[t][deg[t]]) for t in cs])
    dd=np.nanmean([(np.sign(resid[t][deg[t]])==np.sign(t_c[t][deg[t]])).mean() for t in cs])
    hit=sum(1 for t in cs if cs[int(np.nanargmax([pear(resid[t],t_c[u]) for u in cs]))]==t)
    return dict(D_pear=round(float(dp),3),D_dir=round(float(dd),3),PDS=round(hit/len(cs),3),nTF=len(cs))
def velo_dir(V,cells):
    cb=[]; ic=[]
    for i in cells:
        nb=adj_euclid[i]; fwd=nb[pt[nb]>pt[i]]; nv=np.linalg.norm(V[i])
        if fwd.size>0 and nv>0:
            tdir=(X[fwd]-X[i]).mean(0); n=np.linalg.norm(tdir)
            if n>0: cb.append(float((V[i]*tdir).sum()/(nv*n)))
        same=nb[louvain[nb]==louvain[i]]
        if same.size>0 and nv>0:
            c=[float((V[i]*V[j]).sum()/(nv*np.linalg.norm(V[j])+1e-9)) for j in same if np.linalg.norm(V[j])>0]
            if c: ic.append(np.mean(c))
    return round(float(np.mean(cb)),3) if cb else np.nan, round(float(np.mean(ic)),3) if ic else np.nan
def revcorr(D,idx):
    v=[pear(D[i],X[i]) for i in idx[::5] if np.linalg.norm(D[i])>0 and np.std(X[i])>0]; return round(float(np.nanmean(v)),3)

# ---------------- configs ----------------
def make_configs():
    cfgs=[]
    for metric in ['euclid','diffmap','geodesic','fermat']:
        for M in [15,30]:
            for r in [1,3]:
                cfgs.append(dict(metric=metric,M=M,repeat=r,alpha=1))
    return cfgs
CONFIGS=make_configs()

def run_one(cfg):
    t0=time.time(); draws=draw(cfg['metric'],cfg['M'],cfg['alpha'],cfg['repeat'])
    D,have=target_from_draws(draws,cfg['repeat'])
    ct=composite(D,have); rc=revcorr(D,have); V=train_lite(D,have); cp=composite(V,tr); cb,ic=velo_dir(V,te)
    return dict(**cfg,n=len(have),revcorr=rc,**{f'tgt_{k}':v for k,v in ct.items()},
                **{f'pred_{k}':v for k,v in cp.items()},CBDir=cb,ICCoh=ic,sec=round(time.time()-t0,1))
def append(row):
    new=not os.path.exists(RES)
    with open(RES,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if new: w.writeheader()
        w.writerow(row)
if __name__=='__main__':
    arg=sys.argv[1] if len(sys.argv)>1 else 'all'
    print(f'device={dev} nconfigs={len(CONFIGS)}',flush=True)
    items=range(len(CONFIGS)) if arg=='all' else [int(arg)]
    for i in items:
        cfg=CONFIGS[i]; print(f'[{i}] {cfg} ...',flush=True)
        try:
            row=run_one(cfg); append(row)
            print(f'   -> CBDir={row["CBDir"]} ICCoh={row["ICCoh"]} tgt_Dpear={row.get("tgt_D_pear")} pred_Dpear={row.get("pred_D_pear")} rev={row["revcorr"]} ({row["sec"]}s)',flush=True)
        except Exception as e:
            import traceback; print(f'   !! failed: {e!r}',flush=True); traceback.print_exc()
    print('DONE')
