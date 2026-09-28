"""Sensitivity of the mass/floor results to the concentration statistic: q10, q25, q50 top-share and
concentration = -(effective number of heads)/H (exp entropy).  For each: raw back<0 / back<front, after log-mass
adjustment, and residual after the floor-model simulation (floor = per-head min; model-wide min; worst case)."""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
sys.path.insert(0, sys.argv[1] + "/vet3"); from mass import perm_idx, raw_z, adj_z, TASKS
from concurrent.futures import ProcessPoolExecutor
def conc(S, stat):
    P = S / np.clip(S.sum(-1, keepdims=True), 1e-12, None); H = S.shape[-1]
    if stat == "ent": return -np.exp(-(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0)).sum(-1)) / H
    q = {"q10": .1, "q25": .25, "q50": .5}[stat]; m = max(1, int(np.ceil(q * H)))
    return -np.sort(-P, -1)[..., :m].sum(-1)
STATS = ["q10", "q25", "q50", "ent"]
def job(model):
    out = []; cells = {c: load(model, c) for c in CELLS}
    Sg = np.min(np.stack([d["contribution"].astype(float).min(0) for d in cells.values()]), 0)
    for c, d in cells.items():
        fam = c.split("/")[0]
        if fam not in TASKS: continue
        S = d["contribution"].astype(float); k = d["k"].astype(float); n, L, H = S.shape
        body = np.array([int(t["id"].split("/")[-1]) for t in d["items"]])
        P = perm_idx(k, body, fam, np.random.default_rng(0)); x = np.arange(L) / (L - 1); back = x >= .5
        M = S.sum(-1); lM = np.log(M); mn = S.min(0)
        m25 = max(1, int(np.ceil(.25 * H))); Tm = np.zeros((L, H), bool); np.put_along_axis(Tm, np.argsort(-S.mean(0), -1)[:, :m25], True, -1)
        fl = {"min": mn, "global": Sg, "worst": np.where(Tm, 0.0, mn)}
        rs = np.random.default_rng(1); perms = [rs.permutation(n) for _ in range(40)]
        r = {"model": model, "cell": c, "fam": fam}
        for st in STATS:
            Y = conc(S, st); bY, zY = raw_z(Y, k, P); sd = bY / zY
            r[st + "_raw"] = [float(zY[~back].mean()), float(zY[back].mean())]
            _, za = adj_z(Y, k, P, lambda l: lM[:, l:l+1]); r[st + "_adj"] = [float(za[~back].mean()), float(za[back].mean())]
            for fn, f in fl.items():
                u = np.clip(S - f[None], 0, None); U = u.sum(-1, keepdims=True); C = u / np.clip(U, 1e-12, None)
                bp = np.mean([raw_z(conc(f[None] + U * C[pi], st), k, P[:1])[0] for pi in perms], 0)
                zr = (bY - bp) / sd; r[f"{st}_sim_{fn}"] = [float(zr[~back].mean()), float(zr[back].mean())]
                r[f"{st}_sim_{fn}_ratio"] = float(bp[back].sum() / bY[back].sum())
        out.append(r)
    return out
if __name__ == "__main__":
    with ProcessPoolExecutor(10) as ex: R = [r for rr in ex.map(job, ORDER) for r in rr]
    json.dump(R, open(sys.argv[2], "w"))
    for st in STATS:
        s = f"{st:4s}"
        for v in ["raw", "adj", "sim_min", "sim_global", "sim_worst"]:
            f = np.array([r[f"{st}_{v}"][0] for r in R]); b = np.array([r[f"{st}_{v}"][1] for r in R])
            s += f" | {v} {np.mean(b<0):.2f}/{np.mean(b<f):.2f} ({b.mean():+.2f})"
        sp = [r for r in R if r[f"{st}_raw"][1] < 0]
        s += " | floor-predicted fraction (median, spreading cells): " + " ".join(f"{fn} {np.median([r[f'{st}_sim_{fn}_ratio'] for r in sp]):.2f}" for fn in ["min", "global", "worst"])
        print(s)
