"""Item bootstrap of the post-minus-base back-half / region mean z (exact permutation z, wording-averaged items)."""
import json, numpy as np, sys
from ilib import *
np.set_printoptions(precision=3, suppress=True)
NB = 400
REG = {"back": lambda x: x >= .5, "front": lambda x: x < .5, "50-70": lambda x: (x >= .5) & (x < .7), "70-100": lambda x: x >= .7, "whole": lambda x: x >= 0}
def dec_reg(z):
    d = dec(z); return {"dback": d[5:].mean(), "dfront": d[:5].mean(), "d50-70": d[5:7].mean(), "d70-100": d[7:].mean()}
def regs(z):
    x = xrel(len(z)); out = {k: z[f(x)].mean() for k, f in REG.items()}; out.update(dec_reg(z)); return out
models = sorted(set(m for p in PAIRS for m in p))
rng = np.random.default_rng(7)
data = {(m, f): wavg(m, f) for m in models for f in TASKS}
def zr(d, idx):
    Y, k, body = d["Y"][idx], d["k"][idx], d["body"][idx]
    return regs(zexact(Y, k, body, within(f_of[id(d)]))[1])
f_of = {id(data[(m, f)]): f for (m, f) in data}
keys = list(regs(np.zeros(10)).keys())
point = {mf: regs(zexact(d["Y"], d["k"], d["body"], within(mf[1]))[1]) for mf, d in data.items()}
def pair_diff(vals):
    """vals[(m,f)][key] -> per pair (6) mean over tasks, for each key"""
    return {kk: np.array([np.mean([vals[(p, f)][kk] - vals[(b, f)][kk] for f in TASKS]) for b, p in PAIRS]) for kk in keys}
P0 = pair_diff(point)
for kk in keys: print(f"point {kk:8s} per pair {P0[kk]} LW {lw(P0[kk]):+.3f} lineages {lineages(P0[kk])}")
# unpaired bootstrap
B = {kk: [] for kk in keys}
for i in range(NB):
    vals = {}
    for mf, d in data.items():
        n = len(d["k"]); idx = rng.integers(0, n, n)
        vals[mf] = regs(zexact(d["Y"][idx], d["k"][idx], d["body"][idx], within(mf[1]))[1])
    pdv = pair_diff(vals)
    for kk in keys: B[kk].append(np.concatenate([pdv[kk], lineages(pdv[kk]), [lw(pdv[kk]), np.mean(lineages(pdv[kk])[:2])]]))
names = [f"pair{j}" for j in range(6)] + ["L3B", "L8B", "Qwen", "OLMo", "LW", "LW_llama_only"]
res = {}
for kk in keys:
    A = np.array(B[kk]); pt = np.concatenate([P0[kk], lineages(P0[kk]), [lw(P0[kk]), np.mean(lineages(P0[kk])[:2])]])
    lo, hi = np.percentile(A, 2.5, 0), np.percentile(A, 97.5, 0)
    res[kk] = {n: [float(pt[j]), float(lo[j]), float(hi[j])] for j, n in enumerate(names)}
    print(f"UNPAIRED {kk:8s} " + "  ".join(f"{n} {pt[j]:+.2f} [{lo[j]:+.2f},{hi[j]:+.2f}]" for j, n in enumerate(names) if j >= 6))
# LW without Qwen
for kk in ["back", "50-70", "front"]:
    A = np.array(B[kk]); v = A[:, [6, 7, 9]].mean(1); pt = np.mean(lineages(P0[kk])[[0, 1, 3]])
    print(f"LW without Qwen {kk}: {pt:+.3f} [{np.percentile(v,2.5):+.2f},{np.percentile(v,97.5):+.2f}]")
json.dump(res, open("s3_boot.json", "w"), indent=1)
