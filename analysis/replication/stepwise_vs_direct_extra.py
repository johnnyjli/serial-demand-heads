"""Supplementary checks for the E2b stepwise-vs-direct contrast (top-10% share unless stated).
(a) item-split reliability of decile profiles within each mode (4 random halves stratified by k), so the
    cross-mode correlation can be read against it;  (b) OLMo-2 without the outlier exclusion;
(c) paired difference without OLMo-2-Instruct, and median over models;  (d) where the stepwise k effect sits
    once length is removed (decile profiles of the length analyses);  (e) answer uncertainty at the measured
    position;  (f) E2b direct against the paper's E2 (same 8 models);  (g) well-powered correct-only subsets;
(h) models with high stepwise accuracy (Qwen2.5-14B, Mistral-Small-24B, Llama pair)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json
import numpy as np
from scipy import stats as sst
sys.path.insert(0, str(paths.ANALYSIS / "replication"))
import stepwise_vs_direct_lib as E

SPD = str(paths.ANALYSIS / "replication")
OLD = str(paths.ANALYSIS)
J = json.load(open(f"{SPD}/stepwise_vs_direct.json")); C = J["cells"]
LJ = json.load(open(f"{SPD}/stepwise_vs_direct_length.json"))
D = E.load(exclude=True)
out = {}

# (a) item-split reliability of per-cell decile profiles and of model profiles (mean over families)
rng = np.random.default_rng(7)
rel = {}
for mode in E.MODES:
    cell_r, model_r = [], []
    for rep in range(4):
        prof = {h: {} for h in (0, 1)}
        for m in E.MODELS:
            for f in E.FAMS:
                d = D[(m, mode, f)]; k = d["k"]; half = np.zeros(len(k), int)
                for v in np.unique(k):
                    i = np.nonzero(k == v)[0]; rng.shuffle(i); half[i[: len(i) // 2]] = 1
                for h in (0, 1):
                    s = half == h
                    prof[h][(m, f)] = E.dec_prof(E.zprof(d["q10"][s], k[s])[2])
                cell_r.append(np.corrcoef(prof[0][(m, f)], prof[1][(m, f)])[0, 1])
        for m in E.MODELS:
            a = np.mean([prof[0][(m, f)] for f in E.FAMS], 0); b = np.mean([prof[1][(m, f)] for f in E.FAMS], 0)
            model_r.append(np.corrcoef(a, b)[0, 1])
    rel[mode] = {"cell_profile_r": float(np.mean(cell_r)), "model_profile_r": float(np.mean(model_r))}
out["item_split_reliability"] = rel

# (b) OLMo-2 without exclusion
ol = {}
for m in ("OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct"):
    for f in E.FAMS:
        a, b = C[f"q10|direct|noexcl|{m}|{f}"], C[f"q10|stepwise|noexcl|{m}|{f}"]
        ol[f"{m}|{f}"] = {"direct_back": a["back"], "direct_p_lo": a["p_back_lo"], "direct_p_hi": a["p_back_hi"], "direct_n": a["n"],
                          "step_back": b["back"], "step_p_lo": b["p_back_lo"], "step_p_hi": b["p_back_hi"], "step_n": b["n"],
                          "excl_direct_back": C[f"q10|direct|all|{m}|{f}"]["back"], "excl_step_back": C[f"q10|stepwise|all|{m}|{f}"]["back"]}
out["olmo_noexcl"] = ol


def cellsum(mode, sub, models, noexcl_olmo=False):
    bs, sl, sh = [], [], []
    for m in models:
        for f in E.FAMS:
            s = "noexcl" if (noexcl_olmo and m.startswith("OLMo")) else sub
            r = C[f"q10|{mode}|{s}|{m}|{f}"]; bs.append(r["back"]); sl.append(r["p_back_lo"] < .05); sh.append(r["p_back_hi"] < .05)
    bs = np.array(bs)
    return {"n_cells": len(bs), "n_back_neg": int((bs < 0).sum()), "n_sig_spread": int(sum(sl)), "n_sig_conc": int(sum(sh)), "median_back": float(np.median(bs))}


out["summary_olmo_noexcl"] = {mode: cellsum(mode, "all", E.MODELS, True) for mode in E.MODES}

# (c) paired difference variants
def paired(models, sub="all", noexcl=False, key="back"):
    pm = {}; dl = []
    for m in models:
        v = []
        for f in E.FAMS:
            s = "noexcl" if (noexcl and m.startswith("OLMo")) else sub
            a, b = C[f"q10|direct|{s}|{m}|{f}"], C[f"q10|stepwise|{s}|{m}|{f}"]
            if a.get("ok") and b.get("ok"): v.append(b[key] - a[key]); dl.append(b[key] - a[key])
        pm[m] = float(np.mean(v))
    dl = np.array(dl); nn, npp = int((dl < 0).sum()), int((dl > 0).sum())
    rngb = np.random.default_rng(0); a = np.array(list(pm.values()))
    bs = [a[rngb.integers(0, len(a), len(a))].mean() for _ in range(10000)]
    return {"n": len(dl), "step_lower": nn, "step_higher": npp, "sign_p": float(sst.binomtest(nn, nn + npp).pvalue), "mean": float(dl.mean()),
            "median_over_models": float(np.median(a)), "boot_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_models_step_higher": int((a > 0).sum()), "n_models": len(a)}


noOI = [m for m in E.MODELS if m != "OLMo-2-1124-7B-Instruct"]
noQ7 = [m for m in E.MODELS if m not in E.QWEN7]
out["paired_variants"] = {"all8": paired(E.MODELS), "no_olmo_instruct": paired(noOI), "no_qwen7": paired(noQ7),
                          "no_qwen7_no_olmo_instruct": paired([m for m in noQ7 if m != "OLMo-2-1124-7B-Instruct"]),
                          "all8_olmo_noexcl": paired(E.MODELS, noexcl=True), "no_qwen7_olmo_noexcl": paired(noQ7, noexcl=True)}

# (d) decile profiles of the stepwise length analyses (mean over cells where computed) vs unadjusted
dp = {}
for part in ("strat_exact", "len_within_k", "partial_k_given_len"):
    for sub in ("all",):
        ps, pu, cells = [], [], []
        for m in E.MODELS:
            for f in E.FAMS:
                o = LJ["cells"].get(f"q10|{m}|{f}|{sub}")
                x = o[part]
                if "dec" in x and x.get("ok", True):
                    ps.append(x["dec"]); pu.append(C[f"q10|stepwise|all|{m}|{f}"]["dec"]); cells.append(f"{m}|{f}")
        dp[f"{part}|{sub}"] = {"n_cells": len(ps), "mean_dec": np.nanmean(ps, 0).tolist(), "unadjusted_same_cells": np.nanmean(pu, 0).tolist()}
out["length_decile_profiles"] = dp
# cells where the partial k|length analysis has real leverage (>=10% of k variance left after length)
lev = []
for m in E.MODELS:
    for f in E.FAMS:
        o = LJ["cells"][f"q10|{m}|{f}|all"]
        c = o["collinearity"]; p = o["partial_k_given_len"]; se = o["strat_exact"]
        if c["frac_k_var_unexpl_linear"] >= .10:
            lev.append({"cell": f"{m}|{f}", "frac_k_unexpl": c["frac_k_var_unexpl_linear"], "unadj": o["unadjusted"]["back"],
                        "unadj_sig": "spread" if o["unadjusted"]["p_back_lo"] < .05 else ("conc" if o["unadjusted"]["p_back_hi"] < .05 else ""),
                        "partial": p["back"], "partial_sig": "spread" if p["p_back_lo"] < .05 else ("conc" if p["p_back_hi"] < .05 else ""),
                        "strat_exact": se.get("back"), "strat_n_inf": se["n_inf"]})
out["leverage_cells"] = lev

# (e) answer uncertainty at the measured position
ent = {}
for mode in E.MODES:
    for sub in ("all", "correct"):
        v = [x for kk, x in J["answer_uncertainty"].items() if kk.endswith(f"|{mode}|{sub}")]
        ent[f"{mode}|{sub}"] = {"mean_entropy_nats": float(np.mean([x["mean_entropy"] for x in v])), "mean_pmax": float(np.mean([x["mean_pmax"] for x in v])),
                                "median_entropy_slope_per_k": float(np.nanmedian([x["entropy_slope_per_k"] for x in v])),
                                "n_cells_entropy_rises": int(sum(x["entropy_slope_per_k"] > 0 for x in v)), "n_cells": len(v)}
out["answer_uncertainty"] = ent

# (f) E2b direct vs E2 (paper's consist.json, q10, wording-averaged; E2 used the within-body null for binding/relational)
E2 = {r["model"]: r for r in json.load(open(f"{OLD}/consist.json")) if r["stat"] == "q10"}
e2b_back, e2_back, prof_r = [], [], []
P2, Pb = [], []
for m in E.MODELS:
    pa, pb = [], []
    for f in E.FAMS:
        z2 = np.array(E2[m]["families"][f]["z"]); L = len(z2); x = np.arange(L) / (L - 1)
        e2_back.append(z2[x >= .5].mean()); e2b_back.append(C[f"q10|direct|all|{m}|{f}"]["back"])
        d2 = E.dec_prof(z2); db = np.array(C[f"q10|direct|all|{m}|{f}"]["dec"]); prof_r.append(np.corrcoef(d2, db)[0, 1])
        pa.append(d2); pb.append(db)
    P2.append(np.mean(pa, 0)); Pb.append(np.mean(pb, 0))
P2, Pb = np.array(P2), np.array(Pb)
A2, Ab = P2 - P2.mean(0), Pb - Pb.mean(0)
Ps = np.array([np.mean([C[f"q10|stepwise|all|{m}|{f}"]["dec"] for f in E.FAMS], 0) for m in E.MODELS]); As = Ps - Ps.mean(0)
out["e2_vs_e2b_direct"] = {"r_back_over_32_cells": float(np.corrcoef(e2_back, e2b_back)[0, 1]),
                           "n_same_sign_back": int(sum(np.sign(a) == np.sign(b) for a, b in zip(e2_back, e2b_back))),
                           "e2_n_back_neg": int(sum(v < 0 for v in e2_back)), "mean_cell_profile_r": float(np.mean(prof_r)),
                           "model_profile_r": dict(zip(E.MODELS, [float(np.corrcoef(P2[i], Pb[i])[0, 1]) for i in range(8)])),
                           "signature_r": dict(zip(E.MODELS, [float(np.corrcoef(A2[i], Ab[i])[0, 1]) for i in range(8)])),
                           "pooled_signature_r": float(np.corrcoef(A2.ravel(), Ab.ravel())[0, 1]),
                           "shared_profile_r": float(np.corrcoef(P2.mean(0), Pb.mean(0))[0, 1]),
                           "e2_vs_stepwise_pooled_signature_r": float(np.corrcoef(A2.ravel(), As.ravel())[0, 1]),
                           "e2_vs_stepwise_shared_profile_r": float(np.corrcoef(P2.mean(0), Ps.mean(0))[0, 1])}

# (g) correct-only cells powered in both modes (>= 5 and >= 10 correct items at every depth)
pw = {}
for thr in (5, 10):
    rows = []
    for m in E.MODELS:
        for f in E.FAMS:
            a, b = C[f"q10|direct|correct|{m}|{f}"], C[f"q10|stepwise|correct|{m}|{f}"]
            if min(a["n_per_k"].values()) >= thr and min(b["n_per_k"].values()) >= thr:
                rows.append({"cell": f"{m}|{f}", "direct": a["back"], "direct_sig": "spread" if a["p_back_lo"] < .05 else ("conc" if a["p_back_hi"] < .05 else ""),
                             "step": b["back"], "step_sig": "spread" if b["p_back_lo"] < .05 else ("conc" if b["p_back_hi"] < .05 else ""),
                             "direct_slope_pp": a["back_slope_pp"], "step_slope_pp": b["back_slope_pp"], "n_direct": a["n"], "n_step": b["n"]})
    pw[thr] = rows
out["correct_powered_both"] = pw
# stepwise-correct cells with >= 10 correct at every depth (stepwise only)
st10 = []
for m in E.MODELS:
    for f in E.FAMS:
        b = C[f"q10|stepwise|correct|{m}|{f}"]
        if min(b["n_per_k"].values()) >= 10: st10.append((m, f, b["back"], b["p_back_lo"] < .05, b["p_back_hi"] < .05, E.dec_prof(np.array(b["z"]))))
out["stepwise_correct_ge10"] = {"n_cells": len(st10), "cells": [f"{m}|{f}" for m, f, *_ in st10], "n_back_neg": int(sum(x[2] < 0 for x in st10)),
                                "n_sig_spread": int(sum(x[3] for x in st10)), "n_sig_conc": int(sum(x[4] for x in st10)),
                                "median_back": float(np.median([x[2] for x in st10])), "mean_dec": np.nanmean([x[5] for x in st10], 0).tolist()}

# (h) high-stepwise-accuracy models
HI = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-14B", "Mistral-Small-24B-Base-2501"]
hh = {}
for mode in E.MODES:
    for sub in ("all", "correct"):
        bs = [C[f"q10|{mode}|{sub}|{m}|{f}"] for m in HI for f in E.FAMS]
        prof = np.array([np.mean([C[f"q10|{mode}|{sub}|{m}|{f}"]["dec"] for f in E.FAMS], 0) for m in HI])
        hh[f"{mode}|{sub}"] = {"n_back_neg": int(sum(r["back"] < 0 for r in bs)), "n_sig_spread": int(sum(r["p_back_lo"] < .05 for r in bs)),
                               "n_sig_conc": int(sum(r["p_back_hi"] < .05 for r in bs)), "median_back": float(np.median([r["back"] for r in bs])),
                               "mean_dec": prof.mean(0).tolist()}
out["high_accuracy_models"] = hh
out["high_accuracy_paired"] = paired(HI)
out["high_accuracy_paired_correct"] = paired(HI, sub="correct")

json.dump(out, open(f"{SPD}/stepwise_vs_direct_extra.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("olmo_noexcl", "correct_powered_both", "leverage_cells")}, indent=1)[:9000])
