import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, json, numpy as np
SP=str(paths.ANALYSIS)
sys.path.insert(0, SP); os.environ.pop("E2_CLEAN", None); import e2lib
from e2lib import CELLS
OL=["OLMo-2-1124-7B","OLMo-2-1124-7B-SFT","OLMo-2-1124-7B-DPO","OLMo-2-1124-7B-Instruct"]
d=e2lib.load(OL[0],"pointer/w0"); print("contribution shape", d["contribution"].shape, "L,H", d["L"], d["H"])
for m in OL:
    print("==", m)
    for c in CELLS:
        d=e2lib.load(m,c); S=d["contribution"].astype(float)
        R=S/np.maximum(np.median(S,-1,keepdims=True),1e-9)   # items x L x H
        r2614=R[:,26,14]
        flag=(R>50).any((1,2))
        # heads exceeding 50
        idx=np.argwhere(R>50); heads={}
        for it,l,h in idx: heads.setdefault((int(l),int(h)),[]).append(R[it,l,h])
        other={k:(len(v),round(float(max(v)))) for k,v in heads.items() if k!=(26,14)}
        big=(R>100).any((1,2))
        print(f" {c:15s} L26H14 min {r2614.min():5.1f} med {np.median(r2614):5.1f} max {r2614.max():6.1f} | >50 on {int((r2614>50).sum()):3d}/{len(r2614)} | any head>50 {int(flag.sum()):3d} | any>100 {int(big.sum()):2d} | maxR {R.max():6.1f} | argmax per item most common {np.bincount([int(np.argmax(R[i].ravel())) for i in range(len(R))]).argmax()//R.shape[2]},{np.bincount([int(np.argmax(R[i].ravel())) for i in range(len(R))]).argmax()%R.shape[2]}")
        if other: print("     other heads >50:", dict(sorted(other.items(), key=lambda x:-x[1][1])[:6]))
