"""Train the competent VNet (linear skip + MLP, benchmark recipe) on an arbitrary delta-X target field
to test whether a stronger model than the lite net transfers the new sampler's condition-specificity.
Saves the predicted full-cell field to fields/<outtag>_pred.npy.
Usage: python train_vnet_on_target.py <target_npy> <outtag>"""
import sys, os, numpy as np, torch, torch.nn as nn, anndata as ad, scipy.sparse as sp
SEED=0; np.random.seed(SEED); torch.manual_seed(SEED)
dev='cuda' if torch.cuda.is_available() else 'cpu'
H5='/rds/user/wz369/hpc-work/GenDiff/data/integrated_mesc_group0_Nov7.h5ad'
PCOLS=['RA','Wnt','TgfB','Bmp','Fgf','Notch','Shh']
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,'fields'); os.makedirs(OUT,exist_ok=True)
tgt_path, outtag = sys.argv[1], sys.argv[2]

a=ad.read_h5ad(H5)
X=(a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)).astype(np.float32)
DX=np.load(tgt_path).astype(np.float32); assert DX.shape==X.shape
C=a.obs[PCOLS].to_numpy().astype(np.float32)
split=a.obs['split'].astype(str).to_numpy()
tr=np.where(split=='train')[0]; va=np.where(split=='val')[0]; G=X.shape[1]
have=tr[np.linalg.norm(DX[tr],axis=1)>0]          # train cells with a nonzero target
vahave=va[np.linalg.norm(DX[va],axis=1)>0] if (np.linalg.norm(DX[va],axis=1)>0).any() else have

class VNet(nn.Module):
    def __init__(s,g,cdim,hidden=256,p=0.1):
        super().__init__(); din=g+cdim
        s.skip=nn.Linear(din,g); s.mlp=nn.Sequential(nn.Linear(din,hidden),nn.SiLU(),nn.Dropout(p),nn.Linear(hidden,g))
    def forward(s,x,c): h=torch.cat([x,c],1); return s.skip(h)+s.mlp(h)

X_t=torch.tensor(X,device=dev); DX_t=torch.tensor(DX,device=dev); C_t=torch.tensor(C,device=dev)
xmu=torch.tensor(X[tr].mean(0),device=dev); xsd=torch.tensor(X[tr].std(0)+1e-6,device=dev)
dsd=torch.tensor((DX[have].std(0)+1e-6).astype(np.float32),device=dev)
def stdzx(x): return (x-xmu)/xsd
have_t=torch.tensor(have,device=dev); va_t=torch.tensor(vahave,device=dev)
def predict(net,idx):
    net.eval()
    with torch.no_grad(): v=net(stdzx(X_t[idx]),C_t[idx])*dsd
    net.train(); return v.cpu().numpy()
def cosm(idx):
    Yt=DX[idx]; Yp=predict(net,torch.tensor(idx,device=dev)); nz=np.linalg.norm(Yt,axis=1)>0
    return float(((Yt[nz]*Yp[nz]).sum(1)/(np.linalg.norm(Yt[nz],axis=1)*np.linalg.norm(Yp[nz],axis=1)+1e-12)).mean())

net=VNet(G,C.shape[1]).to(dev); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-4); net.train()
best=-2; best_state=None; bad=0
for s in range(15000):
    idx=have_t[torch.randint(0,len(have),(256,),device=dev)]
    loss=((net(stdzx(X_t[idx]),C_t[idx])-DX_t[idx]/dsd)**2).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
    if s%500==0 or s==14999:
        vc=cosm(vahave)
        if vc>best: best=vc; bad=0; best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
        else:
            bad+=1
            if bad>=15: break
net.load_state_dict(best_state); print(f'[{outtag}] best val_cos {best:.3f}',flush=True)
allidx=torch.arange(len(X_t),device=dev)
np.save(os.path.join(OUT,f'{outtag}_pred.npy'), predict(net,allidx).astype(np.float32))
print('saved',outtag,flush=True)
