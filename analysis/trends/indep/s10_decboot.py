import numpy as np
from ilib import *
np.set_printoptions(precision=2, suppress=True)
NB = 300; rng = np.random.default_rng(5)
models = sorted(set(m for p in PAIRS for m in p)); data = {(m, f): wavg(m, f) for m in models for f in TASKS}
def ridx(d, f):
    n = len(d["k"])
    if not within(f):   # stratified within k (design has fixed counts per k)
        idx = np.concatenate([rng.choice(np.nonzero(d["k"] == kk)[0], (d["k"] == kk).sum()) for kk in np.unique(d["k"])]); return idx, d["body"][idx]
    bodies = np.unique(d["body"]); pick = rng.choice(bodies, len(bodies)); parts = [np.nonzero(d["body"] == b)[0] for b in pick]
    return np.concatenate(parts), np.concatenate([np.full(len(p), j) for j, p in enumerate(parts)])
def dz(d, f, idx=None, g=None):
    if idx is None: idx, g = np.arange(len(d["k"])), d["body"]
    return dec(zexact(d["Y"][idx], d["k"][idx], g, within(f))[1])
pt = {mf: dz(d, mf[1]) for mf, d in data.items()}
P0 = lw(np.array([np.mean([pt[(p, f)] - pt[(b, f)] for f in TASKS], 0) for b, p in PAIRS]))
BS = []
for _ in range(NB):
    v = {mf: dz(d, mf[1], *ridx(d, mf[1])) for mf, d in data.items()}
    BS.append(lw(np.array([np.mean([v[(p, f)] - v[(b, f)] for f in TASKS], 0) for b, p in PAIRS])))
BS = np.array(BS)
print("decile LW (exact z)", P0); print("CI lo", np.percentile(BS, 2.5, 0)); print("CI hi", np.percentile(BS, 97.5, 0))
for name, s_ in [("0-50", slice(0, 5)), ("50-70", slice(5, 7)), ("70-100", slice(7, 10)), ("50-100", slice(5, 10))]:
    v = BS[:, s_].mean(1); print(name, round(P0[s_].mean(), 3), np.percentile(v, [2.5, 97.5]).round(2))
