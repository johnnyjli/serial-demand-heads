import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, numpy as np
sys.path.insert(0, str(paths.ANALYSIS))
from e2lib import ORDER, CELLS, load, index
si = index("sdpa")
for m in ORDER:
    ag=[];ae=[];asd=[];cs=[];pooled=[]
    for c in CELLS:
        if (m,c) not in si: continue
        a,b=load(m,c,"eager"),load(m,c,"sdpa")
        x=a["pred_constrained"]==b["pred_constrained"]
        ag.append(x.mean()); pooled+=list(x); ae.append(a["correct"].mean()); asd.append(b["correct"].mean()); cs.append(c)
    print(f"{m:<28}{len(ag):>3} cellmean {np.mean(ag):.3f} pooled {np.mean(pooled):.3f} acc e {np.mean(ae):.3f} s {np.mean(asd):.3f}", cs if len(cs)==1 else "")
