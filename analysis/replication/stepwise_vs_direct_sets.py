"""Stepwise-vs-direct summaries for further model sets, from stepwise_vs_direct.json cells:
all8, no_qwen7 (drop the eager-degraded Qwen2.5-7B pair), no_olmo (OLMo-2 stepwise cells depend on the outlier
exclusion, whose flagged items sit mostly at k=1), clean4 (neither), and all8 with OLMo-2 un-excluded.
Per set, mode, subset, statistic: counts of back-half spreading cells, median back z, model-mean decile profile
with t-CI across models, and the paired stepwise-minus-direct difference (sign test; bootstrap CI over models)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json
import numpy as np
from scipy import stats as sst
SPD = str(paths.ANALYSIS / "replication")
C = json.load(open(f"{SPD}/stepwise_vs_direct.json"))["cells"]
M = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
     "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
F = ["pointer", "arith", "binding", "relational"]
Q7 = {"Qwen2.5-7B", "Qwen2.5-7B-Instruct"}
SETS = {"all8": (M, False), "all8_olmo_noexcl": (M, True), "no_qwen7": ([m for m in M if m not in Q7], False),
        "no_olmo": ([m for m in M if not m.startswith("OLMo")], False),
        "clean4": ([m for m in M if m not in Q7 and not m.startswith("OLMo")], False)}


def get(stat, mode, sub, m, f, noexcl):
    s = "noexcl" if (noexcl and m.startswith("OLMo") and sub == "all") else sub
    return C[f"{stat}|{mode}|{s}|{m}|{f}"]


def tci(x):
    x = np.asarray(x, float); h = sst.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return [float(x.mean() - h), float(x.mean() + h)]


out = {}
for stat in ("q10", "q25", "neff"):
    for sname, (ms, ne) in SETS.items():
        for sub in ("all", "correct"):
            for mode in ("direct", "stepwise"):
                rr = [get(stat, mode, sub, m, f, ne) for m in ms for f in F]
                rr = [r for r in rr if r.get("ok")]
                b = np.array([r["back"] for r in rr])
                prof = np.array([np.nanmean([get(stat, mode, sub, m, f, ne)["dec"] for f in F], 0) for m in ms])
                out[f"{stat}|{sname}|{sub}|{mode}"] = {
                    "n_cells": len(rr), "n_back_neg": int((b < 0).sum()), "n_sig_spread": int(sum(r["p_back_lo"] < .05 for r in rr)),
                    "n_sig_conc": int(sum(r["p_back_hi"] < .05 for r in rr)), "median_back": float(np.median(b)),
                    "dec_mean": prof.mean(0).tolist(), "dec_ci": [tci(prof[:, j]) for j in range(10)],
                    "mean_40_50": float(prof[:, 4].mean()), "ci_40_50": tci(prof[:, 4]),
                    "mean_50_80": float(prof[:, 5:8].mean()), "ci_50_80": tci(prof[:, 5:8].mean(1)),
                    "mean_80_100": float(prof[:, 8:].mean()), "ci_80_100": tci(prof[:, 8:].mean(1)),
                    "mean_0_10": float(prof[:, 0].mean()), "ci_0_10": tci(prof[:, 0]),
                    "n_models_dec4_gt_dec5": int((prof[:, 4] > prof[:, 5]).sum()), "n_models": len(ms)}
            # paired
            dl, pm = [], {}
            for m in ms:
                v = []
                for f in F:
                    a, b_ = get(stat, "direct", sub, m, f, ne), get(stat, "stepwise", sub, m, f, ne)
                    if a.get("ok") and b_.get("ok"): v.append(b_["back"] - a["back"])
                dl += v; pm[m] = float(np.mean(v))
            dl = np.array(dl); nn, npp = int((dl < 0).sum()), int((dl > 0).sum()); a = np.array(list(pm.values()))
            rng = np.random.default_rng(0); bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(10000)]
            out[f"{stat}|{sname}|{sub}|paired"] = {"n": len(dl), "step_lower": nn, "step_higher": npp,
                                                   "sign_p": float(sst.binomtest(nn, nn + npp).pvalue), "mean": float(dl.mean()),
                                                   "boot_ci_models": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                                   "per_model": pm}
json.dump(out, open(f"{SPD}/stepwise_vs_direct_sets.json", "w"), indent=1)
for stat in ("q10", "q25", "neff"):
    for sname in SETS:
        for sub in ("all", "correct"):
            for mode in ("direct", "stepwise"):
                s = out[f"{stat}|{sname}|{sub}|{mode}"]
                print(f"{stat} {sname:17s} {sub:7s} {mode:8s} back<0 {s['n_back_neg']:2d}/{s['n_cells']} sig- {s['n_sig_spread']:2d} sig+ {s['n_sig_conc']:2d} med {s['median_back']:+.2f} | "
                      f"0-10 {s['mean_0_10']:+.2f} [{s['ci_0_10'][0]:+.2f},{s['ci_0_10'][1]:+.2f}] 40-50 {s['mean_40_50']:+.2f} [{s['ci_40_50'][0]:+.2f},{s['ci_40_50'][1]:+.2f}] "
                      f"50-80 {s['mean_50_80']:+.2f} [{s['ci_50_80'][0]:+.2f},{s['ci_50_80'][1]:+.2f}] 80-100 {s['mean_80_100']:+.2f} [{s['ci_80_100'][0]:+.2f},{s['ci_80_100'][1]:+.2f}] d4>d5 {s['n_models_dec4_gt_dec5']}/{s['n_models']}")
            p = out[f"{stat}|{sname}|{sub}|paired"]
            print(f"      paired step-direct: lower {p['step_lower']} higher {p['step_higher']} sign p {p['sign_p']:.3f} mean {p['mean']:+.2f} CI [{p['boot_ci_models'][0]:+.2f},{p['boot_ci_models'][1]:+.2f}]")
