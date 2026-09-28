"""Design-aware bootstrap of post-minus-base region z: items resampled for pointer/arith, whole bodies (clusters)
for binding/relational.  (a) unpaired, primary item sets; (b) paired, items common to base and post."""
import json, numpy as np
from ilib import *
np.set_printoptions(precision=3, suppress=True)
NB = 400
def regs(z):
    x = xrel(len(z)); d = dec(z)
    return np.array([z[x >= .5].mean(), z[x < .5].mean(), z[(x >= .5) & (x < .7)].mean(), z[x >= .7].mean(), z.mean(), d[5:7].mean(), d[5:].mean()])
KEYS = ["back", "front", "50-70", "70-100", "whole", "d50-70", "dback"]
models = sorted(set(m for p in PAIRS for m in p))
data = {(m, f): wavg(m, f) for m in models for f in TASKS}
def resample_idx(d, f, rng):
    n = len(d["k"])
    if not within(f): return rng.integers(0, n, n), None
    bodies = np.unique(d["body"]); pick = rng.choice(bodies, len(bodies))
    parts = [np.nonzero(d["body"] == b)[0] for b in pick]
    return np.concatenate(parts), np.concatenate([np.full(len(p_), j) for j, p_ in enumerate(parts)])
def zreg(d, f, idx, groups=None):
    body = d["body"][idx] if groups is None else groups
    return regs(zexact(d["Y"][idx], d["k"][idx], body, within(f))[1])
out = {}
for mode in ["unpaired", "paired_common"]:
    rng = np.random.default_rng(11)
    if mode == "unpaired":
        pt = {mf: zreg(d, mf[1], np.arange(len(d["k"]))) for mf, d in data.items()}
        P0 = np.array([[pt[(p, f)] - pt[(b, f)] for f in TASKS] for b, p in PAIRS]).mean(1)   # 6 x K
        BS = []
        for _ in range(NB):
            v = {mf: zreg(d, mf[1], *resample_idx(d, mf[1], rng)) for mf, d in data.items()}
            BS.append(np.array([[v[(p, f)] - v[(b, f)] for f in TASKS] for b, p in PAIRS]).mean(1))
    else:
        P0 = []; BS = [[] for _ in range(NB)]
        for b, p in PAIRS:
            per_t = []; bs_t = [[] for _ in range(NB)]
            for f in TASKS:
                db, dp = data[(b, f)], data[(p, f)]
                common = sorted(set(db["keys"]) & set(dp["keys"]))
                ib = np.array([db["keys"].index(x) for x in common]); ip = np.array([dp["keys"].index(x) for x in common])
                sub = lambda d, i: {"Y": d["Y"][i], "k": d["k"][i], "body": d["body"][i], "keys": [d["keys"][j] for j in i]}
                sb, sp = sub(db, ib), sub(dp, ip)
                per_t.append(zreg(sp, f, np.arange(len(ib))) - zreg(sb, f, np.arange(len(ib))))
                for i in range(NB):
                    idx, g = resample_idx(sb, f, rng)       # same resampled keys for both models
                    bs_t[i].append(zreg(sp, f, idx, g) - zreg(sb, f, idx, g))
            P0.append(np.mean(per_t, 0))
            for i in range(NB): BS[i].append(np.mean(bs_t[i], 0))
        P0 = np.array(P0); BS = [np.array(x) for x in BS]
    BS = np.array(BS)   # NB x 6 x K
    res = {}
    for j, kk in enumerate(KEYS):
        def summ(fn):
            v = np.array([fn(B[:, j]) for B in BS]); return [float(fn(P0[:, j])), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        r = {"L3B": summ(lambda v: v[0]), "L8B": summ(lambda v: v[1]), "Qwen": summ(lambda v: v[2]), "OLMo": summ(lambda v: v[3:].mean()),
             "LW": summ(lambda v: lw(v)), "LW_noQwen": summ(lambda v: np.mean([v[0], v[1], v[3:].mean()]))}
        for q in range(6): r[f"pair{q}"] = summ(lambda v, q=q: v[q])
        res[kk] = r
        print(f"{mode:14s} {kk:7s} " + "  ".join(f"{n} {a:+.2f} [{lo:+.2f},{hi:+.2f}]" for n, (a, lo, hi) in r.items() if not n.startswith("pair")))
    out[mode] = res
json.dump(out, open("s3c_boot.json", "w"), indent=1)
