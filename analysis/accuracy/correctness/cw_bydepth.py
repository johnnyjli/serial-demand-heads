"""Task 1 by depth: Cohen's d (correct - incorrect) of the top-10% share at each depth separately, per model x task x
wording cell, for depths with >= MINC correct and >= MINC incorrect items (MINC = 5 here, so that k = 1 enters in
more cells; also reported with MINC = 8).  Summaries per task family and depth: mean d over back-half layers
(relative depth >= 0.5) and over all layers, averaged over cells, with a null from shuffling correctness within
the depth (B = 2000; shuffles shared by the two wordings of a model x task, independent across models).
Writes cw_bydepth.json."""
import json, zlib, numpy as np, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
from clib import *
from cw import depth_stats
B = 2000
def one(key):
    m, f = key; D = {w: load_cell(m, f, w) for w in ("w0", "w1")}; k = D["w0"]["k"]; n = len(k); x = D["w0"]["x"]
    rng = np.random.default_rng(zlib.crc32(f"{m}|{f}|bydepth".encode()))
    out = []
    for kk in np.unique(k):
        ii = np.nonzero(k == kk)[0]
        P = np.stack([np.arange(len(ii))] + [rng.permutation(len(ii)) for _ in range(B)])      # shared by wordings
        for w in ("w0", "w1"):
            ok = D[w]["correct"][ii]; nc, nw = int(ok.sum()), int((~ok).sum())
            if min(nc, nw) < 5: continue
            dd, diff, _ = depth_stats(D[w]["y10"][ii], ok[P])
            out.append({"model": m, "fam": f, "w": w, "k": int(kk), "n_correct": nc, "n_wrong": nw,
                        "back_d": dd[:, x >= .5].mean(1).astype(np.float32), "all_d": dd.mean(1).astype(np.float32),
                        "back_diff_pp": float(100 * diff[0, x >= .5].mean())})
    return out
if __name__ == "__main__":
    keys = [(m, f) for m in ORDER for f in FAMS]
    with ProcessPoolExecutor(10) as ex: R = [r for rr in ex.map(one, keys) for r in rr]
    res = {}
    for minc in (5, 8):
        for f in FAMS:
            for kk in sorted({r["k"] for r in R if r["fam"] == f}):
                rr = [r for r in R if r["fam"] == f and r["k"] == kk and min(r["n_correct"], r["n_wrong"]) >= minc]
                if not rr: continue
                bd = np.mean([r["back_d"] for r in rr], 0); ad = np.mean([r["all_d"] for r in rr], 0)
                pos = np.mean([r["back_d"][0] > 0 for r in rr])
                res[f"{f}|k={kk}|minc={minc}"] = {"n_cells": len(rr), "n_models": len({r["model"] for r in rr}),
                    "back_mean_d": float(bd[0]), "back_null_sd": float(bd[1:].std()), "p_back": float((np.sum(np.abs(bd[1:]) >= abs(bd[0])) + 1) / B),
                    "all_mean_d": float(ad[0]), "p_all": float((np.sum(np.abs(ad[1:]) >= abs(ad[0])) + 1) / B),
                    "frac_cells_back_d_pos": float(pos), "mean_back_diff_pp": float(np.mean([r["back_diff_pp"] for r in rr]))}
    json.dump(res, open("cw_bydepth.json", "w"), indent=1)
    for key, v in res.items():
        if key.endswith("minc=5"):
            print(f"{key:28s} cells {v['n_cells']:3d} ({v['n_models']:2d} models) back d {v['back_mean_d']:+.3f} (null sd {v['back_null_sd']:.3f}, p {v['p_back']:.4f}) "
                  f"all d {v['all_mean_d']:+.3f} (p {v['p_all']:.4f}) cells back d>0 {v['frac_cells_back_d_pos']:.2f} back diff {v['mean_back_diff_pp']:+.3f}pp")
