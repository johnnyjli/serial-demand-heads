import json, numpy as np
from ilib import *
np.set_printoptions(precision=2, suppress=True)
CJ = json.load(open(f"{SP}/consist.json"))
# n items per model x task (wording-matched, cleaned) and OLMo relative slope
C = {r["model"]: r for r in CJ if r["stat"] == "q10"}
for m in ["OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct"]:
    print(m, {f: C[m]["families"][f]["n"] for f in TASKS})
def reg(stat, key, region):
    Cs = {r["model"]: r for r in CJ if r["stat"] == stat}; out = {}
    for m in ORDER:
        v = []
        for f in TASKS:
            z = np.array(Cs[m]["families"][f][key]); x = xrel(len(z))
            sel = {"back": x >= .5, "50-70": (x >= .5) & (x < .7), "70-100": x >= .7, "front": x < .5}[region]; v.append(z[sel].mean())
        out[m] = np.array(v)
    return out
for stat in ["q10", "q25", "q50", "entropy", "top1"]:
    S = reg(stat, "z", "50-70"); D = np.array([S[p] - S[b] for b, p in PAIRS]); pm = D.mean(1)
    print(f"per-layer 50-70 {stat:8s} per pair {pm} LW {lw(pm):+.3f} lineages<0 {(lineages(pm)<0).sum()} pairs<0 {(pm<0).sum()} pxt<0 {(D<0).sum()}")
S = reg("q10", "z", "70-100"); print("per-layer 70-100 q10 per pair", np.array([S[p].mean() - S[b].mean() for b, p in PAIRS]))
S = reg("q10", "rel", "back"); D = np.array([S[p] - S[b] for b, p in PAIRS]); print("back rel per pair x task:\n", D)
# SD of PC1 among bases for scale
Zd = np.array([[dec(C[m]["families"][f]["z"]) for f in TASKS] for m in ORDER]); mu = Zd.mean((0, 1)); a = Zd.mean(1) - mu
U, s, Vt = np.linalg.svd(a - a.mean(0), full_matrices=False); sc = U * s
if Vt[0][5:].mean() > 0: sc[:, 0] *= -1
print("SD of PC1 across 11 bases", np.std([sc[ORDER.index(m), 0] for m in BASES], ddof=1))
# PC1 without post-trained models in the PCA (fit on bases, project posts)
bi = [ORDER.index(m) for m in BASES]; ab = a[bi]; cen = ab.mean(0)
U2, s2, Vt2 = np.linalg.svd(ab - cen, full_matrices=False); v1 = Vt2[0] * (-1 if Vt2[0][5:].mean() > 0 else 1)
proj = {m: (a[ORDER.index(m)] - cen) @ v1 for m in ORDER}
dp = np.array([proj[p] - proj[b] for b, p in PAIRS]); print("PC1 fit on bases only: loadings", v1, "change per pair", dp, "LW", lw(dp))
