"""Task 3 (direct test): does the slope of the top-10% share on k differ between correct and incorrect items?
Per model x task x wording cell and layer: y = depth dummies (+ prompt-body dummies for binding / relational)
+ g*c + h*c*(k - mean k), with c = strict correctness.  h (pp per step) is the difference between the k-slope of
correct items and that of incorrect items; g the level difference at mean k.  Null: correctness shuffled within
depth (B = 1000; shuffles shared by the two wordings; independent across models).  Reported per task family:
back-half (relative depth >= 0.5) mean h averaged over cells, with its null, against the all-items back-half slope.
Writes interact.json."""
import json, zlib, numpy as np, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
from clib import *
B = 1000
def one(key):
    m, f = key; D = {w: load_cell(m, f, w) for w in ("w0", "w1")}; k = D["w0"]["k"]; n = len(k); x = D["w0"]["x"]
    dums = [(k[:, None] == np.unique(k)[None]).astype(float)]
    if f in WITHIN: dums.append((D["w0"]["body"][:, None] == np.unique(D["w0"]["body"])[None]).astype(float)[:, 1:])
    Dm = np.hstack(dums); Mproj = np.eye(n) - Dm @ np.linalg.pinv(Dm)
    rng = np.random.default_rng(zlib.crc32(f"{m}|{f}|inter".encode()))
    perm = np.tile(np.arange(n), (B + 1, 1))
    for kk in np.unique(k):
        ii = np.nonzero(k == kk)[0]
        for b in range(1, B + 1): perm[b, ii] = ii[rng.permutation(len(ii))]
    kc = k - k.mean(); out = []
    for w in ("w0", "w1"):
        Y = D[w]["y10"]; ok = D[w]["correct"].astype(float)
        if ok.sum() < 10 or (1 - ok).sum() < 10: continue
        Yr = Mproj @ Y; H = np.empty((B + 1, Y.shape[1])); G = np.empty((B + 1, Y.shape[1]))
        for b in range(B + 1):
            c = ok[perm[b]]; X = Mproj @ np.c_[c, c * kc]
            coef = np.linalg.lstsq(X, Yr, rcond=None)[0]; G[b], H[b] = coef[0], coef[1]
        sl = kc @ (Y - Y.mean(0)) / (kc @ kc)
        out.append({"model": m, "fam": f, "w": w, "n_correct": int(ok.sum()), "n_wrong": int((1 - ok).sum()),
                    "back_h": (100 * H[:, x >= .5].mean(1)).astype(np.float32), "back_g": (100 * G[:, x >= .5].mean(1)).astype(np.float32),
                    "h_layers": (100 * H[0]).tolist(),
                    "p_h_layers": ((np.sum(np.abs(H[1:]) >= np.abs(H[0]) - 1e-15, 0) + 1) / (B + 1)).tolist(),
                    "back_slope_all_pp": float(100 * sl[x >= .5].mean())})
    return out
if __name__ == "__main__":
    keys = [(m, f) for m in ORDER for f in FAMS]
    with ProcessPoolExecutor(10) as ex: R = [r for rr in ex.map(one, keys) for r in rr]
    res = {}
    for g, fs in {**{f: [f] for f in FAMS}, "serial": TASKS}.items():
        rr = [r for r in R if r["fam"] in fs]
        bh_ = np.mean([r["back_h"] for r in rr], 0); bg = np.mean([r["back_g"] for r in rr], 0)
        p_cells = np.array([(np.sum(np.abs(r["back_h"][1:]) >= abs(r["back_h"][0])) + 1) / B for r in rr])
        pl = np.concatenate([r["p_h_layers"] for r in rr])
        res[g] = {"n_cells": len(rr), "back_mean_h_pp": float(bh_[0]), "null_sd": float(bh_[1:].std()),
                  "p": float((np.sum(np.abs(bh_[1:]) >= abs(bh_[0])) + 1) / B),
                  "back_mean_g_pp": float(bg[0]), "p_g": float((np.sum(np.abs(bg[1:]) >= abs(bg[0])) + 1) / B),
                  "back_mean_slope_all_pp": float(np.mean([r["back_slope_all_pp"] for r in rr])),
                  "cells_back_h_p_lt_05": int((p_cells < .05).sum()), "cells_back_h_BH": int(bh(p_cells).sum()),
                  "frac_cells_back_h_pos": float(np.mean([r["back_h"][0] > 0 for r in rr])),
                  "frac_layers_h_p_lt_05": float((pl < .05).mean()), "n_layers": int(len(pl))}
        v = res[g]
        print(f"{g:12s} cells {v['n_cells']:3d} back h {v['back_mean_h_pp']:+.3f} pp/step (null sd {v['null_sd']:.3f}, p {v['p']:.4f}) vs all-items slope {v['back_mean_slope_all_pp']:+.3f}; "
              f"back g {v['back_mean_g_pp']:+.3f}pp (p {v['p_g']:.4f}); cells p<.05 {v['cells_back_h_p_lt_05']} (BH {v['cells_back_h_BH']}); h>0 in {v['frac_cells_back_h_pos']:.2f}; layers p<.05 {100*v['frac_layers_h_p_lt_05']:.1f}%")
    res["cells"] = [{k: (v if not isinstance(v, np.ndarray) else float(v[0])) for k, v in r.items() if k not in ("h_layers", "p_h_layers")} for r in R]
    json.dump(res, open("interact.json", "w"), indent=1)
