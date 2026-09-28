import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, numpy as np
from collections import defaultdict
sys.path.insert(0, str(paths.ANALYSIS))
from e2lib import ORDER, CELLS, load
for m in [x for x in ORDER if "OLMo" in x]:
    per_layer=defaultdict(list); arith_2614=[]; nev=0
    for c in CELLS:
        fam=c.split('/')[0]
        d=load(m,c); S=d["contribution"].astype(float)
        med=np.median(S,-1); mx=(S.max(-1)/np.maximum(med,1e-9)); am=S.argmax(-1)
        ev=mx>50
        if fam in("binding","relational"):
            for i in np.nonzero(ev.any(1))[0]:
                nev+=1
                for l in range(S.shape[1]):
                    per_layer[l].append((mx[i,l], am[i,l]))
        if fam=="arith":
            r=S[:,26,14]/np.maximum(med[:,26],1e-9); arith_2614+=list(r[mx[:,26]>50])
    print(m, "binding/relational event items:", nev)
    for l in sorted(per_layer):
        v=np.array([x[0] for x in per_layer[l]]); hs=np.bincount([x[1] for x in per_layer[l]]).argmax()
        if np.median(v)>20: print(f"   L{l}: median {np.median(v):.0f}x max {v.max():.0f}x modal head {hs}")
    a=np.array(arith_2614); print("  arith L26H14 events", len(a), "range %.0f-%.0f median %.0f p95 %.0f"%(a.min(),a.max(),np.median(a),np.percentile(a,95)) if len(a) else "")
