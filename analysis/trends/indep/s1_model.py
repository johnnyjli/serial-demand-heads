import json, numpy as np
from ilib import *
np.set_printoptions(precision=3, suppress=True)
CJ = json.load(open(f"{SP}/consist.json"))
def C(stat): return {r["model"]: r for r in CJ if r["stat"] == stat}
def summ(stat, region, key="z"):
    Cs = C(stat); out = {}
    for m in ORDER:
        v = []
        for f in TASKS:
            z = np.array(Cs[m]["families"][f][key]); x = xrel(len(z))
            sel = {"back": x >= .5, "front": x < .5, "whole": np.ones(len(z), bool)}[region]
            v.append(z[sel].mean())
        out[m] = np.array(v)
    return out
res = {}
for stat in ["q10", "q25", "q50", "entropy", "top1"]:
    for region in ["back", "front", "whole"]:
        S = summ(stat, region)
        D = np.array([S[p] - S[b] for b, p in PAIRS])  # 6 x 4
        pm = D.mean(1)
        res[(stat, region)] = pm
        print(f"{stat:8s} {region:6s} per pair {pm}  LW {lw(pm):+.3f}  pairs<0 {(pm<0).sum()}/6 lineages<0 {(lineages(pm)<0).sum()}/4 lineage {lineages(pm)} pairxtask<0 {(D<0).sum()}/24")
        if stat == "q10" and region == "back":
            print("   per task LW:", dict(zip(TASKS, np.round(lw(D), 3))), " per pair x task:\n", D)
            print("   base models back z:", {m: round(S[m].mean(), 3) for m in BASES}, "SD(ddof1)", np.std([S[m].mean() for m in BASES], ddof=1), "SD(ddof0)", np.std([S[m].mean() for m in BASES]))
            print("   post models back z:", {p: round(S[p].mean(), 3) for b, p in PAIRS}, "bases", {b: round(S[b].mean(),3) for b in set(b for b,_ in PAIRS)})
# back relative slope
S = summ("q10", "back", "rel"); D = np.array([S[p] - S[b] for b, p in PAIRS]); pm = D.mean(1)
print("back rel %/step per pair", pm, "LW", lw(pm), "lineages<0", (lineages(pm) < 0).sum(), "pairxtask<0", (D < 0).sum())
# share of significantly spreading layers (pooled over 4 tasks)
Cs = C("q10")
def frac(m, sign, region="all"):
    n = s = 0
    for f in TASKS:
        z = np.array(Cs[m]["families"][f]["z"]); p = np.array(Cs[m]["families"][f]["p"]); x = xrel(len(z))
        sel = np.ones(len(z), bool) if region == "all" else (x < .5)
        s += ((p < .05) & (np.sign(z) == sign) & sel).sum(); n += sel.sum()
    return s / n
for b, p in PAIRS: print(f"frac sig spreading {b} {frac(b,-1):.3f} -> {p} {frac(p,-1):.3f}")
fc = np.array([frac(p, 1) - frac(b, 1) for b, p in PAIRS]); fcf = np.array([frac(p, 1, 'front') - frac(b, 1, 'front') for b, p in PAIRS])
print("frac sig conc change", fc, "LW", lw(fc), "front-only", fcf, "LW", lw(fcf))
