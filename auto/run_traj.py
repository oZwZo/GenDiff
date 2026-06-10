"""Run trajectory eval (CBDir/ICCoh + CellRank terminal recovery) on every predicted velocity field."""
import os, glob, traj_eval
HERE=os.path.dirname(os.path.abspath(__file__)); FIELDS=os.path.join(HERE,'fields')
ORDER=['geo_m15_r3','fermat_m15_r3','old_r100','otcfm','velreg','minusX']
for tag in ORDER:
    p=os.path.join(FIELDS,f'{tag}_pred.npy')
    if not os.path.exists(p): print('SKIP missing',tag,flush=True); continue
    print('=== traj_eval',tag,'===',flush=True)
    try: traj_eval.main(p,tag)
    except Exception as e: print('FAILED',tag,repr(e)[:300],flush=True)
print('ALL DONE')
