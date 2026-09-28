"""Task 1 sensitivity: is the correct-vs-incorrect difference at the same k an effect of which digit is being
written?  Before computing d, remove from each layer's top-10% share the additive effect of the gold digit
('gold') or of the model's predicted digit ('pred'), fitted jointly with depth dummies over the cell's items.
Same d, pooling and within-depth null as cw.py (B = 2000).  Writes cw_sens.json."""
import json, zlib, numpy as np, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
from clib import *
from cw import depth_stats
B, MINC = 2000, 8
def resid_on(Y, k, lab):
    X = np.c_[(k[:, None] == np.unique(k)[None]).astype(float), (lab[:, None] == np.arange(10)[None]).astype(float)]
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]; kd = (k[:, None] == np.unique(k)[None]).astype(float)
    # keep the depth means (they do not affect within-depth d); remove only the digit effects
    return Y - X[:, kd.shape[1]:] @ beta[kd.shape[1]:]
def one(key):
    m, f = key; D = {w: load_cell(m, f, w) for w in ("w0", "w1")}; k = D["w0"]["k"]; n = len(k); L = D["w0"]["L"]
    rng = np.random.default_rng(zlib.crc32(f"{m}|{f}|sens".encode()))
    perm = np.tile(np.arange(n), (B + 1, 1))
    for kk in np.unique(k):
        ii = np.nonzero(k == kk)[0]
        for b in range(1, B + 1): perm[b, ii] = ii[rng.permutation(len(ii))]
    out = []
    for w in ("w0", "w1"):
        d = D[w]; ok = d["correct"]
        depths = [kk for kk in np.unique(k) if ok[k == kk].sum() >= MINC and (~ok[k == kk]).sum() >= MINC]
        if not depths: continue
        r = {"model": m, "fam": f, "w": w, "L": L}
        for var, Y in (("none", d["y10"]), ("gold", resid_on(d["y10"], k, d["gold"])), ("pred", resid_on(d["y10"], k, d["pred"]))):
            num, W = {}, {}
            for kk in depths:
                ii = np.nonzero(k == kk)[0]; dd, _, wt = depth_stats(Y[ii], ok[perm[:, ii]]); num[kk] = dd; W[kk] = wt
            wt = np.array([W[x] for x in depths]); dall = sum(W[x] * num[x] for x in depths) / wt.sum()
            obs, null = dall[0], dall[1:]; A = np.abs(null)
            p = (np.sum(A >= np.abs(obs) - 1e-12, 0) + 1) / (B + 1)
            srt = np.sort(A, 0); pnull = (B - np.stack([np.searchsorted(srt[:, l], A[:, l], side="left") for l in range(L)], 1) + 1) / (B + 1)
            r[var] = {"d": obs.tolist(), "p": p.tolist(), "null_sig_counts": (pnull < .05).sum(1).tolist(),
                      "null_mean_absd": A.mean(1).tolist(), "null_mean_d": null.mean(1).tolist()}
        out.append(r)
    return out
if __name__ == "__main__":
    keys = [(m, f) for m in ORDER for f in FAMS]
    with ProcessPoolExecutor(10) as ex: R = [r for rr in ex.map(one, keys) for r in rr]
    json.dump(R, open("cw_sens.json", "w"))
    for g, fs in {**{f: [f] for f in FAMS}, "serial": TASKS}.items():
        rr = [r for r in R if r["fam"] in fs]; line = f"{g:12s}"
        for var in ("none", "gold", "pred"):
            d = np.concatenate([r[var]["d"] for r in rr]); p = np.concatenate([r[var]["p"] for r in rr])
            nc = np.sum([r[var]["null_sig_counts"] for r in rr], 0); na = np.mean([np.array(r[var]["null_mean_absd"]) * r["L"] for r in rr], 0) * len(rr) / len(d)
            line += f" | {var}: mean d {d.mean():+.3f} mean|d| {np.abs(d).mean():.3f} (null {na.mean():.3f}) sig {100 * (p < .05).mean():.1f}% (null {100 * nc.mean() / len(d):.1f}%)"
        print(line)
