"""Task 3b: does correctness change the k-response?  Per-layer z of the top-10% share on k (the paper's statistic
and permutation null, exact SD) recomputed on subsets of items, and compared with all items:
  incorrect      instances answered wrongly in both wordings (wording-averaged statistic, as in the paper)
  incorrect_pw   per wording, items wrong in that wording; z per wording, then averaged over wordings
  k>=2           all items with k >= 2
  correct        instances answered correctly in both wordings (only where >= 2 depths have >= 8 such items)
Estimator: pointer / arithmetic: slope over items, k shuffled across items.  Binding / relational: k shuffled within
prompt body; on subsets the slope is computed within body (body-centred), which equals the paper's slope on the full,
balanced design ('plain' = the paper's pooled slope is reported as a sensitivity).
Reference for chance: 'matched' random subsets with the same number of items per depth as the incorrect (or correct)
subset, drawn from all items (R = 200), put through the same pipeline.
Comparisons: model signatures a_m (decile profiles, two-way decomposition as in model_diff.py) vs all items; shared
profile mu; per-cell decile-profile and layer-profile correlations; back-half slope (pp per step).  Writes subsets.json."""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
from clib import *
sys.path.insert(0, OLD)
import model_diff; model_diff.SP = OLD
from model_diff import ss, perm_model, split_half
R_MATCH = 200; MIN_PER_DEPTH = 8
rng = np.random.default_rng(12345)
DATA = {}
for m in ORDER:
    for t in TASKS:
        P = load_pair(m, t); c0, c1 = load_cell(m, t, "w0"), load_cell(m, t, "w1")
        DATA[(m, t)] = {"Y": P["y10"], "k": P["k"], "body": P["body"], "c0": P["c0"], "c1": P["c1"], "L": P["L"], "x": P["x"],
                        "Yw": {"w0": c0["y10"], "w1": c1["y10"]}}
def z_of(D, t, sel, centre=True, Y=None):
    Y = D["Y"] if Y is None else Y
    return zprof(Y[sel], D["k"][sel], D["body"][sel], t in WITHIN, centre and t in WITHIN)
def feasible(D, sel):
    k = D["k"][sel]; return sum((k == kk).sum() >= MIN_PER_DEPTH for kk in np.unique(k)) >= 2
def matched(D, sel, rng):
    """random subset of all items with the same count per depth as sel"""
    k = D["k"]; out = np.zeros(len(k), bool)
    for kk in np.unique(k):
        n = int((sel & (k == kk)).sum()); ii = np.nonzero(k == kk)[0]
        if n: out[rng.choice(ii, n, replace=False)] = True
    return out
def sig_stats(Z, Zref):
    mu, a, b, e = decompose(Z); mu0, a0, _, _ = decompose(Zref)
    return {"r_signatures": float(np.corrcoef(a.ravel(), a0.ravel())[0, 1]),
            "per_model_r": [float(np.corrcoef(a[i], a0[i])[0, 1]) for i in range(len(a))],
            "r_mu": float(np.corrcoef(mu, mu0)[0, 1]), "mu": mu.tolist(), "mu_back_mean": float(mu[5:].mean())}
# ---- all items (paper estimator) ----
ALL = {key: z_of(D, key[1], np.ones(len(D["k"]), bool), centre=False) for key, D in DATA.items()}
ALLc = {key: z_of(D, key[1], np.ones(len(D["k"]), bool), centre=True) for key, D in DATA.items()}
def Zmat(zd): return np.array([[dec_prof(zd[(m, t)][2]) for t in TASKS] for m in ORDER])
Z_all = Zmat(ALL); out = {"all_items": {"n_items": {f"{m}|{t}": int(len(DATA[(m, t)]["k"])) for m, t in DATA},
                                        "r_signatures_centred_vs_paper": sig_stats(Zmat(ALLc), Z_all)["r_signatures"]}}
def back(D, v): return float(np.nanmean(v[D["x"] >= .5]))
def compare(name, SEL, centre=True, perwording=None, full=True):
    """SEL: dict key -> boolean mask (or None if infeasible)."""
    Z = {}; cells = {}
    for key, D in DATA.items():
        sel = SEL[key]
        if sel is None or not feasible(D, sel): cells[f"{key[0]}|{key[1]}"] = {"feasible": False, "n": int(0 if sel is None else sel.sum())}; continue
        if perwording is None: b, sd, z = z_of(D, key[1], sel, centre)
        else:
            zz = []; bb = []
            for w in ("w0", "w1"):
                s = perwording[key][w]
                if feasible(D, s): r_ = z_of(D, key[1], s, centre, D["Yw"][w]); zz.append(r_[2]); bb.append(r_[0])
            if not zz: cells[f"{key[0]}|{key[1]}"] = {"feasible": False}; continue
            z = np.mean(zz, 0); b = np.mean(bb, 0)
        Z[key] = (b, None, z); z0 = ALL[key][2]; b0 = ALL[key][0]
        cells[f"{key[0]}|{key[1]}"] = {"feasible": True, "n": int(sel.sum()), "n_by_k": {int(kk): int((sel & (D["k"] == kk)).sum()) for kk in np.unique(D["k"])},
                                     "r_layers": float(np.corrcoef(z, z0)[0, 1]), "r_deciles": float(np.corrcoef(dec_prof(z), dec_prof(z0))[0, 1]),
                                     "back_z": back(D, z), "back_z_all": back(D, z0), "back_b_pp": 100 * back(D, b), "back_b_pp_all": 100 * back(D, b0)}
    o = {"n_feasible": len(Z), "n_cells": len(DATA), "cells": cells}
    fz = {t: [c for kk, c in cells.items() if kk.split("|")[1] == t and c["feasible"]] for t in TASKS}
    def bt(v):
        if not v: return {"n_feasible": 0}
        g = lambda f: float(np.mean([c[f] for c in v]))
        return {"n_feasible": len(v), "median_r_layers": float(np.median([c["r_layers"] for c in v])),
                "median_r_deciles": float(np.median([c["r_deciles"] for c in v])), "mean_back_b_pp": g("back_b_pp"),
                "mean_back_b_pp_all": g("back_b_pp_all"), "mean_back_z": g("back_z"), "mean_back_z_all": g("back_z_all"),
                "frac_back_b_same_sign": float(np.mean([np.sign(c["back_b_pp"]) == np.sign(c["back_b_pp_all"]) for c in v]))}
    o["by_task"] = {t: bt(fz[t]) for t in TASKS}
    if len(Z) == len(DATA):
        Zs = Zmat(Z); o.update(sig_stats(Zs, Z_all)); s = ss(Zs); ns = s["model"] + s["task"] + s["resid"]
        o["model_share"] = s["model"] / ns
        if full:
            _, nm, p = perm_model(Zs, 2000); sh = split_half(Zs)
            o.update({"p_model": p, "split_half": sh[0], "p_split": sh[2]})
    return o, Z
res = {}
SELS = {"incorrect": {key: ~D["c0"] & ~D["c1"] for key, D in DATA.items()},
        "k>=2": {key: D["k"] >= 2 for key, D in DATA.items()},
        "correct": {key: D["c0"] & D["c1"] for key, D in DATA.items()}}
for name, S in SELS.items():
    res[name], _ = compare(name, S, True)
    res[name + "|plain"], _ = compare(name, S, False)
PW = {key: {"w0": ~DATA[key]["c0"], "w1": ~DATA[key]["c1"]} for key in DATA}
res["incorrect_pw"], _ = compare("incorrect_pw", {key: ~D["c0"] | ~D["c1"] for key, D in DATA.items()}, True, PW)
PWall = {key: {"w0": np.ones(len(D["k"]), bool), "w1": np.ones(len(D["k"]), bool)} for key, D in DATA.items()}
res["all_pw (reference for incorrect_pw)"], _ = compare("all_pw", {key: np.ones(len(D["k"]), bool) for key, D in DATA.items()}, True, PWall)
# ---- matched random subsets ----
for name in ("incorrect", "correct"):
    S = SELS[name]; draws = []
    for r in range(R_MATCH):
        M = {key: (matched(D, S[key], rng) if feasible(D, S[key]) else None) for key, D in DATA.items()}
        o, _ = compare(name + "_matched", M, True, full=False)
        d = {"by_task": {t: {kk: o["by_task"][t].get(kk) for kk in ("median_r_layers", "median_r_deciles", "mean_back_b_pp", "mean_back_z")} for t in TASKS},
             "cells_r_layers": {kk: c["r_layers"] for kk, c in o["cells"].items() if c["feasible"]}}
        if "r_signatures" in o: d.update({"r_signatures": o["r_signatures"], "r_mu": o["r_mu"], "model_share": o["model_share"], "per_model_r_median": float(np.median(o["per_model_r"]))})
        draws.append(d)
    summ = {}
    if "r_signatures" in draws[0]:
        for kk in ("r_signatures", "r_mu", "model_share", "per_model_r_median"):
            v = np.array([d[kk] for d in draws]); summ[kk] = {"mean": float(v.mean()), "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5)),
                                                              "frac_below_observed": float((v < res[name][kk if kk != "per_model_r_median" else "r_signatures"]).mean()) if kk != "per_model_r_median" else float((v < np.median(res[name]["per_model_r"])).mean())}
    for t in TASKS:
        for kk in ("median_r_layers", "median_r_deciles", "mean_back_b_pp", "mean_back_z"):
            v = np.array([d["by_task"][t][kk] for d in draws if d["by_task"][t].get(kk) is not None])
            if len(v):
                ob = res[name]["by_task"][t][kk]
                summ[f"{t}|{kk}"] = {"mean": float(v.mean()), "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5)),
                                     "observed": ob, "p_two_sided": float(min(1, 2 * min((np.sum(v <= ob) + 1) / (len(v) + 1), (np.sum(v >= ob) + 1) / (len(v) + 1))))}
    # per cell: is the incorrect-only layer profile further from all items than matched subsets? (one-sided rank)
    pc = {}
    for kk, c in res[name]["cells"].items():
        if not c["feasible"]: continue
        v = np.array([d["cells_r_layers"][kk] for d in draws if kk in d["cells_r_layers"]])
        pc[kk] = float((np.sum(v <= c["r_layers"]) + 1) / (len(v) + 1))
    summ["cells_p_r_layers_lower_than_matched"] = pc
    summ["n_cells_p_lt_05"] = int(sum(v < .05 for v in pc.values())); summ["n_cells_BH"] = int(bh(np.array(list(pc.values()))).sum())
    res[name + "|matched_random"] = summ
out.update(res)
json.dump(out, open("subsets.json", "w"), indent=1)
for name in ("incorrect", "incorrect|plain", "incorrect_pw", "all_pw (reference for incorrect_pw)", "k>=2", "k>=2|plain", "correct"):
    o = out[name]
    line = f"{name:36s} feasible {o['n_feasible']}/{o['n_cells']}"
    if "r_signatures" in o: line += f" | r_sig {o['r_signatures']:.3f} (per-model median {np.median(o['per_model_r']):.2f}) r_mu {o['r_mu']:.3f} mu_back {o['mu_back_mean']:+.2f} | model share {o['model_share']:.2f} p {o['p_model']:.4f} split {o['split_half']:.2f}"
    print(line)
    for t in TASKS:
        b = o["by_task"][t]
        if b["n_feasible"]: print(f"     {t:11s} n {b['n_feasible']:2d} r_layers {b['median_r_layers']:.2f} r_dec {b['median_r_deciles']:.2f} back b {b['mean_back_b_pp']:+.3f} (all {b['mean_back_b_pp_all']:+.3f}) back z {b['mean_back_z']:+.2f} (all {b['mean_back_z_all']:+.2f}) same sign {b['frac_back_b_same_sign']:.2f}")
for name in ("incorrect", "correct"):
    s = out[name + "|matched_random"]; print(name, "matched random:", {k: v for k, v in s.items() if k != "cells_p_r_layers_lower_than_matched"})
print("all-items centred vs paper signatures r:", out["all_items"]["r_signatures_centred_vs_paper"])
print("correct-only feasible cells:", [k for k, c in out["correct"]["cells"].items() if c["feasible"]])
