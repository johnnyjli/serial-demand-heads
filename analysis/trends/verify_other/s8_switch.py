"""T3 selection check: every adjacent-decile pair; decile layer counts per model; post-trained vs base trends in the regions."""
import json, numpy as np
from vo_lib import *
Z = Zdec("q10"); P = Z.mean(1)
print("mean profile (task-avg, model-mean):", np.round(P.mean(0), 2))
for i in range(9):
    d = P[:, i] - P[:, i + 1]; ld = lin_mean(dict(zip(M, d)))
    print(f"  d{i}>d{i+1}: models {int((d>0).sum())}/17, lineages {sum(v>0 for v in ld.values())}/11, lineage mean {np.mean(list(ld.values())):+.2f}")
# layers per decile 4 and 5 per model
LZ = layer_z("q10")
for m in M:
    L = len(LZ[m]["pointer"]); x = np.arange(L) / (L - 1); d = np.minimum(np.floor(x * 10 + 1e-9).astype(int), 9)
    print(f"  {m:28s} L={L} layers in d4 {int((d==4).sum())} d5 {int((d==5).sum())}  d4 {P[M.index(m),4]:+.2f} d5 {P[M.index(m),5]:+.2f}")
