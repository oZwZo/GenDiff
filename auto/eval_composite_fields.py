"""Compute condition-specificity (composite delta-specificity) on each saved predicted velocity field.
This is the discriminating axis (the trajectory/direction metrics are reversion-confounded). Reuses
sweep.composite (Pearson + sign-agreement of the reversion-removed condition signature vs the
perturbed-minus-control effect on top-50 DEGs, and PDS argmax discrimination)."""
import os, numpy as np, csv
import sweep as S
FIELDS=os.path.join(S.HERE,'fields'); OUT=os.path.join(S.HERE,'composite_fields.csv')
TAGS=['geo_m15_r3','fermat_m15_r3','old_r100','otcfm','velreg','minusX']
rows=[]
for tag in TAGS:
    p=os.path.join(FIELDS,f'{tag}_pred.npy')
    if not os.path.exists(p): print('skip',tag); continue
    V=np.load(p).astype(np.float64)
    comp_all=S.composite(V,S.tr)            # on train cells (condition signatures)
    comp_te=S.composite(V,S.te)             # on held-out test cells
    rc=S.revcorr(V,S.tr)
    row=dict(tag=tag, revcorr=rc,
             tr_D_pear=comp_all['D_pear'], tr_D_dir=comp_all['D_dir'], tr_PDS=comp_all['PDS'],
             te_D_pear=comp_te['D_pear'], te_D_dir=comp_te['D_dir'], te_PDS=comp_te['PDS'])
    rows.append(row); print(tag, row, flush=True)
with open(OUT,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('saved',OUT)
