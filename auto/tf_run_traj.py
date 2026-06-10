"""Run TF-Atlas trajectory eval over all generated fields."""
import os, tf_traj
HERE=os.path.dirname(os.path.abspath(__file__)); F=os.path.join(HERE,'tf_fields')
for tag in ['geo_m15_r3','euclid_m15_r1','realvelo','minusX']:
    p=os.path.join(F,f'{tag}_pred.npy')
    if not os.path.exists(p): print('SKIP',tag,flush=True); continue
    print('=== tf_traj',tag,'===',flush=True)
    try: tf_traj.main(p,tag)
    except Exception as e: print('FAILED',tag,repr(e)[:300],flush=True)
print('ALL DONE')
