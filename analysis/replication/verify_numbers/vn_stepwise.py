"""Verify stepwise-vs-direct numbers (stepwise-discussion item)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, itertools, numpy as np
from scipy.stats import binomtest
from vn_lib import *
OUT = {}
ITEMS = json.load(open(str(paths.E2B / "items.json")))
ASET = {it["id"]: [int(a) for a in it["answer_set"]] for f in T for it in ITEMS[f]}
def back_layer(z, L): return float(np.nanmean(z[rel_depth(L) >= .5]))

# (1) answer uncertainty at the measured position (admissible digits, measured items after the OLMo exclusion)
unc = {}
for mode in ("direct", "stepwise"):
    mp, en, slopes = [], [], []
    for m in M8:
        for t in T:
            d = CACHE[(m, mode, t)]; _, _, _, keep = cell(m, mode, t)
            e_c, k_c = [], []
            for i in np.nonzero(keep)[0]:
                x = d["items"][i]; a = ASET[x["id"]]; lg = d["DL"][i][a].astype(float)
                p = np.exp(lg - lg.max()); p /= p.sum(); h = float(-(p[p > 0] * np.log(p[p > 0])).sum())
                mp.append(p.max()); en.append(h); e_c.append(h); k_c.append(x["k"])
            k_c = np.array(k_c, float); e_c = np.array(e_c); kc = k_c - k_c.mean(); slopes.append(kc @ (e_c - e_c.mean()) / (kc @ kc))
    unc[mode] = {"mean_maxp": float(np.mean(mp)), "mean_entropy": float(np.mean(en)), "median_slope": float(np.median(slopes)),
                 "n_pos_slope": int((np.array(slopes) > 0).sum()), "n_cells": len(slopes)}
print("uncertainty", unc); OUT["uncertainty"] = unc

# (2) 50-80% decile mean per mode and statistic; t-CI across models
res = {}
for stat in ("q10", "q25", "neff"):
    for mode in ("direct", "stepwise"):
        Z = Zprof(M8, mode, stat)
        for tag, models in (("M8", M8), ("M6", M6), ("clean4", CLEAN4)):
            ix = [M8.index(m) for m in models]; pm = Z[ix].mean(1)
            res[(stat, mode, tag)] = {"50_80": tci(pm[:, 5:8].mean(1)), "80_100": tci(pm[:, 8:10].mean(1)), "d0": tci(pm[:, 0]), "d4": tci(pm[:, 4]), "mu": np.round(Z[ix].mean((0, 1)), 2).tolist()}
for k_, v in res.items():
    if k_[0] == "q10" or k_[2] == "M8": print(k_, {a: np.round(b, 2).tolist() for a, b in v.items()})
OUT["deciles"] = {"|".join(k): v for k, v in res.items()}

# (3) cross-mode signatures, per-model profile correlation
Zd = Zprof(M8, "direct"); Zs = Zprof(M8, "stepwise")
ad = Zd.mean(1) - Zd.mean((0, 1)); as_ = Zs.mean(1) - Zs.mean((0, 1))
per = [r_(ad[i], as_[i]) for i in range(8)]
OUT["cross_mode_sig"] = {"per_model": dict(zip(M8, np.round(per, 3).tolist())), "mean": float(np.mean(per)), "pooled": r_(ad, as_),
                         "mean_profile_r": float(np.mean([r_(Zd[i].mean(0), Zs[i].mean(0)) for i in range(8)])), "mu_r": r_(Zd.mean((0, 1)), Zs.mean((0, 1)))}
print("cross-mode", OUT["cross_mode_sig"])

# (4) item-split reliability of model profiles (task-averaged), 4 random k-stratified splits
def halves_rel(mode, seed):
    rng = np.random.default_rng(40_000 + seed); rs = []; cs = []
    for m in M8:
        p1, p2 = [], []
        for t in T:
            Y, k, _, _ = cell(m, mode, t); sel = np.zeros(len(k), bool)
            for kk in np.unique(k):
                ix = np.nonzero(k == kk)[0]; sel[rng.choice(ix, len(ix) // 2, replace=False)] = True
            a = dec_prof(zprof(Y[sel], k[sel])[2]); b = dec_prof(zprof(Y[~sel], k[~sel])[2]); p1.append(a); p2.append(b); cs.append(r_(a, b))
        rs.append(r_(np.mean(p1, 0), np.mean(p2, 0)))
    return np.mean(rs), np.mean(cs)
for mode in ("direct", "stepwise"):
    v = [halves_rel(mode, s) for s in range(4)]
    OUT[f"itemsplit_{mode}"] = {"model_profile_r": float(np.mean([a for a, _ in v])), "cell_profile_r": float(np.mean([b for _, b in v]))}
    print("item split", mode, OUT[f"itemsplit_{mode}"])

# (5) back-half mean z per cell (layer-based and decile-based), within-cell difference, clean-4 counts
cells_ = []
for m in M8:
    L = CACHE[(m, "direct", "pointer")]["L"]
    for t in T:
        zd = zlayers(m, "direct", t); zs = zlayers(m, "stepwise", t)
        cells_.append({"m": m, "t": t, "d_layer": back_layer(zd, L), "s_layer": back_layer(zs, L), "d_dec": float(dec_prof(zd)[5:].mean()), "s_dec": float(dec_prof(zs)[5:].mean())})
for key in ("layer", "dec"):
    hi = sum(c[f"s_{key}"] > c[f"d_{key}"] for c in cells_)
    OUT[f"within_diff_{key}"] = {"stepwise_higher": hi, "p_sign": binomtest(hi, 32, .5).pvalue,
                                 "neg_direct": sum(c[f"d_{key}"] < 0 for c in cells_), "neg_step": sum(c[f"s_{key}"] < 0 for c in cells_),
                                 "clean4_neg_direct": sum(c[f"d_{key}"] < 0 for c in cells_ if c["m"] in CLEAN4), "clean4_neg_step": sum(c[f"s_{key}"] < 0 for c in cells_ if c["m"] in CLEAN4)}
    print("back-half", key, OUT[f"within_diff_{key}"])
# OLMo stepwise cells with and without exclusion
ol = {}
for m in ("OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct"):
    for t in T:
        ol[f"{m}|{t}"] = [back_layer(zlayers(m, "stepwise", t), 32), back_layer(zlayers(m, "stepwise", t, excl=False), 32),
                          float(dec_prof(zlayers(m, "stepwise", t))[5:].mean()), float(dec_prof(zlayers(m, "stepwise", t, excl=False))[5:].mean())]
print("OLMo stepwise back z (layer excl, layer noexcl, dec excl, dec noexcl):", {k: np.round(v, 2).tolist() for k, v in ol.items()})
OUT["olmo_step"] = ol
# (6) collinearity of k and context length in stepwise cells (measured items after exclusion)
rr = []; perfect = []
for m in M8:
    for t in T:
        _, k, its, _ = cell(m, "stepwise", t); n = np.array([x["n_tokens_measured"] for x in its], float)
        r = r_(k, n); rr.append(r)
        # perfectly collinear: n is a deterministic (linear) function of k
        if all(len(set(n[k == kk])) == 1 for kk in np.unique(k)) and abs(r) > .999: perfect.append(f"{m}|{t}")
OUT["collinearity"] = {"median_r": float(np.median(rr)), "min": float(np.min(rr)), "max": float(np.max(rr)), "perfect": perfect}
print("collinearity", OUT["collinearity"])
json.dump(OUT, open(f"{HERE}/vn_stepwise.json", "w"), indent=1, default=float)
