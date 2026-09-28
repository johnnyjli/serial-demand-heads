import json, numpy as np
from ilib import *
C = {r["model"]: r for r in json.load(open(f"{SP}/consist.json")) if r["stat"] == "q10"}
worst = 0
for m in ORDER:
    for f in TASKS:
        d = wavg(m, f); b, z = zexact(d["Y"], d["k"], d["body"], within(f))
        zc = np.array(C[m]["families"][f]["z"]); n = C[m]["families"][f]["n"]
        r = np.corrcoef(z, zc)[0, 1]; md = np.abs(z - zc).max(); bh = xrel(len(z)) >= .5
        worst = max(worst, md)
        if md > 0.3 or n != len(d["k"]): print(m, f, n, len(d["k"]), round(r, 4), round(md, 3), round(z[bh].mean() - zc[bh].mean(), 3))
print("max abs per-layer z diff", worst)
