"""
Train OT-CFM (and the trivial velocity-regression reference) on the SAME Tr_SampledX_r100 supervision as
the manuscript benchmark, then save the full-cell predicted velocity field (N x G) for trajectory eval.
Recipe copied verbatim from GenDiff-manuscript/response/R1.3/benchmark_fm.py (z-score input by expression
std, predict velocity / velocity-std, linear skip + MLP residual, early-stop on val cosine, t=0 readout).
Usage: python otcfm_field.py
"""
import os, numpy as np, torch, torch.nn as nn, anndata as ad, scipy.sparse as sp
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
SEED=0; np.random.seed(SEED); torch.manual_seed(SEED)
dev='cuda' if torch.cuda.is_available() else 'cpu'
H5='/rds/user/wz369/hpc-work/GenDiff/data/integrated_mesc_group0_Nov7.h5ad'
PCOLS=['RA','Wnt','TgfB','Bmp','Fgf','Notch','Shh']
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'fields'); os.makedirs(OUT,exist_ok=True)

a=ad.read_h5ad(H5)
X=(a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)).astype(np.float32)
DX=np.asarray(a.obsm['Tr_SampledX_r100']).astype(np.float32)
C=a.obs[PCOLS].to_numpy().astype(np.float32)
cond=a.obs['condition'].astype(str).to_numpy(); split=a.obs['split'].astype(str).to_numpy()
tr=np.where(split=='train')[0]; te=np.where(split=='test')[0]; va=np.where(split=='val')[0]; G=X.shape[1]

class VNet(nn.Module):
    def __init__(s,g,cdim,hidden=256,use_t=True,p=0.1):
        super().__init__(); s.use_t=use_t; din=g+cdim+(1 if use_t else 0)
        s.skip=nn.Linear(din,g); s.mlp=nn.Sequential(nn.Linear(din,hidden),nn.SiLU(),nn.Dropout(p),nn.Linear(hidden,g))
    def forward(s,x,c,t=None):
        h=torch.cat([x,c]+([t.reshape(-1,1)] if s.use_t else []),1); return s.skip(h)+s.mlp(h)

X_t=torch.tensor(X,device=dev); DX_t=torch.tensor(DX,device=dev); C_t=torch.tensor(C,device=dev)
tr_t=torch.tensor(tr,device=dev); va_t=torch.tensor(va,device=dev)
xmu_t=torch.tensor(X[tr].mean(0),device=dev); xsd_t=torch.tensor(X[tr].std(0)+1e-6,device=dev)
dsd_t=torch.tensor((DX[tr].std(0)+1e-6).astype(np.float32),device=dev)
def stdzx(x): return (x-xmu_t)/xsd_t
groups={}
for i in tr: groups.setdefault(cond[i],[]).append(i)
gkeys=[k for k,v in groups.items() if len(v)>=2]; gidx={k:torch.tensor(v,device=dev) for k,v in groups.items()}
gprob=np.array([len(groups[k]) for k in gkeys],float); gprob/=gprob.sum()
def predict(net,use_t,idx_t,n):
    net.eval()
    with torch.no_grad():
        v=net(stdzx(X_t[idx_t]),C_t[idx_t],torch.zeros(n,device=dev) if use_t else None); pred=(v*dsd_t).cpu().numpy()
    net.train(); return pred
def cos_mean(Yt,Yp):
    nz=np.linalg.norm(Yt,axis=1)>0
    return float(((Yt[nz]*Yp[nz]).sum(1)/(np.linalg.norm(Yt[nz],axis=1)*np.linalg.norm(Yp[nz],axis=1)+1e-12)).mean())

def train(matcher,sampling,use_t=True,steps=15000,bs=256,lr=1e-3,wd=1e-4,patience=15,tag='FM'):
    torch.manual_seed(SEED); net=VNet(G,C.shape[1],use_t=use_t).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=wd); net.train(); best=-2.0; best_state=None; bad=0
    for s in range(steps):
        if sampling in ('global','direct'): idx=tr_t[torch.randint(0,len(tr),(bs,),device=dev)]
        else:
            k=gkeys[np.random.choice(len(gkeys),p=gprob)]; idx=gidx[k][torch.randint(0,len(gidx[k]),(bs,),device=dev)]
        x0=X_t[idx]; c=C_t[idx]
        if sampling=='direct':
            pred=net(stdzx(x0),c,torch.zeros(len(idx),device=dev) if use_t else None); loss=((pred-DX_t[idx]/dsd_t)**2).mean()
        else:
            t,xt,ut=matcher.sample_location_and_conditional_flow(x0,x0+DX_t[idx]); loss=((net(stdzx(xt),c,t)-ut/dsd_t)**2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        if s%500==0 or s==steps-1:
            vc=cos_mean(DX[va],predict(net,use_t,va_t,len(va)))
            if vc>best: best=vc; bad=0; best_state={kk:vv.detach().clone() for kk,vv in net.state_dict().items()}
            else:
                bad+=1
                if bad>=patience: break
    net.load_state_dict(best_state); print(f'[{tag}] best val_cos {best:.3f}',flush=True)
    allidx=torch.arange(len(X_t),device=dev); full=predict(net,use_t,allidx,len(X_t))
    np.save(os.path.join(OUT,f'{tag}_pred.npy'), full.astype(np.float32)); print('saved',tag,full.shape,flush=True)
    return full

train(ExactOptimalTransportConditionalFlowMatcher(sigma=0.0),'cond',tag='otcfm')
train(None,'direct',use_t=False,tag='velreg')   # 1-step velocity regression reference
print('DONE')
