import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, str(paths.ANALYSIS))
from e2lib import ORDER, CELLS, load
for m in [x for x in ORDER if "OLMo" in x]:
    r2614=[]; other=defaultdict(list); fams=Counter(); nitems=0; ndrop=0; dropfam=Counter(); nfam=Counter()
    for c in CELLS:
        d=load(m,c); S=d["contribution"].astype(float)
        med=np.median(S,-1); ratio=S/np.maximum(med[...,None],1e-9)
        mx=ratio.max(-1)  # item x layer
        ev=mx>50
        nitems+=len(S); drop=ev.any(1); ndrop+=drop.sum(); dropfam[c.split('/')[0]]+=drop.sum(); nfam[c.split('/')[0]]+=len(S)
        # ratio of L26H14 across all items
        r=ratio[:,26,14]
        r2614+=list(r[r>50])
        for i,l in zip(*np.nonzero(ev)):
            h=int(S[i,l].argmax())
            if (l,h)!=(26,14): other[(int(l),h,c.split('/')[0])].append(mx[i,l])
    r2614=np.array(r2614)
    print(m, "items", nitems, "dropped", ndrop, {f:f"{dropfam[f]}/{nfam[f]}" for f in nfam})
    if len(r2614): print("  L26H14 >50x events:", len(r2614), "range %.1f-%.1f, p90 %.1f"%(r2614.min(), r2614.max(), np.percentile(r2614,90)))
    agg=defaultdict(list)
    for (l,h,f),v in other.items(): agg[(l,h)]+=v
    top=sorted(agg.items(), key=lambda kv:-max(kv[1]))[:8]
    for (l,h),v in top: print(f"  L{l}H{h}: n={len(v)} max {max(v):.0f}x  fams {sorted(set(f for (ll,hh,f) in other if (ll,hh)==(l,h)))}")
