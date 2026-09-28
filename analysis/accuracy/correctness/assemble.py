"""Collect the correctness analyses into correctness.json (run after cw.py, cw_sens.py, cw_bydepth.py, decode.py,
decode_parts.py, altexp_check.py, subsets.py, kge2_ref.py, interact.py, setup_checks.py)."""
import json, numpy as np
from clib import *
import cw_summary, decode_summary
J = lambda f: json.load(open(f))
out = {"meta": {
    "statistic": "top-10% head share per layer at the answer position (share of the ceil(0.1 H) strongest heads' output norms)",
    "correctness": "strict = correct_constrained AND first generated digit correct (correct_strict in the captures; verified identical in all 313 files)",
    "data": "E2 main run; 17 models x {pointer k1-5, arith k1-6, binding k1-6, relational k1-6, fixed_digit control} x {w0, w1}; 42 test items per depth and wording",
    "models": ORDER, "attention": "eager, except Qwen2.5-7B-Instruct (SDPA)",
    "exclusion": "OLMo-2 items with one head > 50x its layer's median head norm in either wording are dropped (only OLMo-2 cells affected)",
    "z_reproduction": "per-layer z from the exact permutation SD reproduces consist.json (q10) at r >= 0.9996 in all 68 model x task cells (check_z.py)"}}
# ---------------- Task 1 ----------------
t1 = cw_summary.summarise()
R = [r for r in J("cw.json") if "d" in r]; N = np.load("cw_null.npz"); ex = {}
for g, fs in {**{f: [f] for f in FAMS}, "serial (4 tasks)": TASKS}.items():
    rr = [r for r in R if r["fam"] in fs]
    d = np.concatenate([r["d"] for r in rr]); nd = np.concatenate([N[f"{r['model']}|{r['fam']}|{r['w']}|d"] for r in rr], 1).astype(float)
    x = np.concatenate([np.arange(r["L"]) / (r["L"] - 1) for r in rr])
    kd = np.concatenate([r["k_d_per_step"] for r in rr])
    kr = np.concatenate([np.array(r["k_d_per_step"]) * (max(map(int, r["acc_by_k"])) - min(map(int, r["acc_by_k"]))) for r in rr])
    ex[g] = {"rms_d": float(np.sqrt((d ** 2).mean())), "null_rms_d": float(np.sqrt((nd ** 2).mean())),
             "excess_rms_d (sd of d beyond sampling noise)": float(np.sqrt(max(0, (d ** 2).mean() - (nd ** 2).mean()))),
             "k_effect_d_per_step_mean_abs": float(np.abs(kd).mean()), "k_effect_d_over_k_range_mean_abs": float(np.abs(kr).mean()),
             "k_effect_d_over_k_range_back_half_mean": float(kr[x >= .5].mean())}
S = J("cw_sens.json"); sens = {}
for g, fs in {**{f: [f] for f in FAMS}, "serial (4 tasks)": TASKS}.items():
    rr = [r for r in S if r["fam"] in fs]; sens[g] = {}
    for var in ("none", "gold", "pred"):
        d = np.concatenate([r[var]["d"] for r in rr]); p = np.concatenate([r[var]["p"] for r in rr]); nc = np.sum([r[var]["null_sig_counts"] for r in rr], 0)
        sens[g][var] = {"mean_d": float(d.mean()), "mean_abs_d": float(np.abs(d).mean()), "frac_sig": float((p < .05).mean()), "null_frac_sig": float(nc.mean() / len(d))}
out["task1_cohens_d"] = {"method": "per model x task x wording cell and layer: Cohen's d (correct - incorrect) of the top-10% share within each depth, pooled over depths with >= 8 correct and >= 8 incorrect items (weights 1/(1/n_c+1/n_w)); null: correctness shuffled within depth, B = 5000, shuffles shared by the two wordings",
                         "by_family": {g: v for g, v in t1.items() if isinstance(v, dict) and "mean_d" in v},
                         "cell_level": {k: t1[k] for k in ("n_cells_total", "cells_p_n_sig_lt05", "cells_BH_n_sig", "cells_p_mean_d_lt05", "cells_BH_mean_d", "cells_without_usable_depth")},
                         "cells_with_p_lt_05": [c for c in t1["cells"] if c["p_n_sig"] < .05 or c["p_mean_d"] < .05],
                         "size_vs_k_effect": ex,
                         "sensitivity_remove_digit_identity": {"method": "top-10% share residualised on gold digit (or predicted digit) one-hot + depth dummies before d; B = 2000", "by_family": sens},
                         "by_depth": J("cw_bydepth.json")}
# ---------------- Task 2 ----------------
d2 = decode_summary.summarise(); parts = J("decode_parts.json")["summary"]
old = json.load(open(f"{OLD}/decode2.json")); ctx = {}
for t in FAMS:
    rr = [r for r in old if r["cell"].startswith(t) and "activity" in r]
    ctx[t] = {"n": len(rr), "all_head_log_norms_PCA20": float(np.median([r["activity"] for r in rr])), "prompt": float(np.median([r["prompt features"]["auc"] for r in rr])),
              "prompt+activity": float(np.median([r["prompt+activity"] for r in rr]))}
out["task2_decoding"] = {"method": "grouped 5-fold CV (StratifiedGroupKFold, prompt body as group, strata depth x correctness), StandardScaler + L2 logistic regression C=0.5; features residualised on k; depths with >= 8 correct and >= 8 incorrect, cells with >= 40 such items; NPERM = 100 (p >= 1/101, so BH across all 125+ cells cannot reach significance; within-family BH reported)",
                         "by_family": {g: v for g, v in d2.items() if isinstance(v, dict)}, "n_cells_run": d2["n_cells_run"], "cells_not_run": d2["cells_not_run"],
                         "prompt_feature_split_median_auc": parts,
                         "context_earlier_decode2_all_head_norms (not re-verified; no OLMo exclusion, Q7I eager)": ctx}
# ---------------- Task 3 ----------------
A = J("altexp_check.json")
out["task3a_adjusted_rows"] = {
    "what_they_compute": ("vet3/altexp.py: for every model x task x wording cell (Qwen2.5-7B-Instruct under EAGER, items flagged in that wording dropped), "
                          "each layer's top-10% share is regressed by OLS on k plus item-level covariates: answer entropy (entropy of the softmax over the "
                          "admissible digits at the answer position) and, for the second row, strict correctness (0/1). For binding and relational all "
                          "variables are demeaned within prompt body (body fixed effects). The response is the OLS t-statistic of the k coefficient (not the "
                          "permutation z). model_diff4.py averages these t over layers within relative-depth deciles, then over the two wordings, giving "
                          "Z[17 models, 4 tasks, 10 deciles]; it decomposes Z as in model_diff.py and reports model share, split-half, lineage/other r, and r = "
                          "correlation of the flattened 17x10 signature matrix with the primary signatures (consist.json q10: permutation z of wording-averaged "
                          "items, Q7I SDPA)."),
    "confirmed_values": {"r_entropy": A["adjusted for answer entropy"]["vs_primary"]["r_with_ref"], "r_entropy_and_correctness": A["adjusted for entropy and correctness"]["vs_primary"]["r_with_ref"],
                         "reproduction_of_altexp_layers.csv_max_abs_t_diff": max(A[k]["max_abs_t_diff"] for k in A if k.startswith("reproduces"))},
    "decomposition_of_r": {"same route, unadjusted (OLS t, per wording, body FE)": A["unadjusted, same route (OLS t, per wording)"]["vs_primary"]["r_with_ref"],
                           "correctness only": {"r_primary": A["adjusted for correctness only"]["vs_primary"]["r_with_ref"], "r_with_same_route_unadjusted": A["adjusted for correctness only"]["r_with_unadjusted_route"]},
                           "entropy": {"r_with_same_route_unadjusted": A["adjusted for answer entropy"]["r_with_unadjusted_route"]},
                           "entropy + correctness": {"r_with_same_route_unadjusted": A["adjusted for entropy and correctness"]["r_with_unadjusted_route"]},
                           "with Q7I from SDPA": {k.split("|")[1]: v for k, v in A.items() if k.startswith("Q7I")}},
    "shared_profile_back_half_mean_t": {k: A[k]["mu_back_mean"] for k in ("unadjusted, same route (OLS t, per wording)", "adjusted for correctness only", "adjusted for answer entropy", "adjusted for entropy and correctness")},
    "coefficients": A["coefficients"]}
Sb = J("subsets.json"); K2 = J("kge2_ref.json"); I = J("interact.json")
def pick(o): return {k: o[k] for k in ("n_feasible", "n_cells", "r_signatures", "r_mu", "mu_back_mean", "model_share", "p_model", "split_half", "p_split", "by_task") if k in o} | ({"per_model_r_median": float(np.median(o["per_model_r"]))} if "per_model_r" in o else {})
out["task3b_subsets"] = {"method": open("subsets.py").read().split('"""')[1],
                         "all_items_reference": {"r_signatures_centred_vs_paper": Sb["all_items"]["r_signatures_centred_vs_paper"]},
                         "incorrect_both_wordings": pick(Sb["incorrect"]), "incorrect_both_wordings_plain_estimator": pick(Sb["incorrect|plain"]),
                         "incorrect_per_wording": pick(Sb["incorrect_pw"]), "all_per_wording_reference": pick(Sb["all_pw (reference for incorrect_pw)"]),
                         "incorrect_matched_random": {k: v for k, v in Sb["incorrect|matched_random"].items() if k != "cells_p_r_layers_lower_than_matched"},
                         "incorrect_cells_less_similar_than_matched_p_lt_05": sorted([k for k, v in Sb["incorrect|matched_random"]["cells_p_r_layers_lower_than_matched"].items() if v < .05]),
                         "k>=2": pick(Sb["k>=2"]), "k>=2_reference_random_drop_same_n": {k: v for k, v in K2.items() if k.startswith("random")},
                         "correct_both_wordings": pick(Sb["correct"]), "correct_matched_random": {k: v for k, v in Sb["correct|matched_random"].items() if k != "cells_p_r_layers_lower_than_matched"},
                         "correct_only_feasible_cells": [k for k, c in Sb["correct"]["cells"].items() if c["feasible"]],
                         "correct_only_feasible_per_wording": K2["correct_only_feasible_per_wording"],
                         "items_per_cell": {t: {"incorrect_both_median": float(np.median([c["n"] for k, c in Sb["incorrect"]["cells"].items() if k.split("|")[1] == t])),
                                                "correct_both_median": float(np.median([c["n"] for k, c in Sb["correct"]["cells"].items() if k.split("|")[1] == t])),
                                                "all_median": float(np.median([v for k, v in Sb["all_items"]["n_items"].items() if k.split("|")[1] == t]))} for t in TASKS}}
out["task3c_interaction"] = {"method": open("interact.py").read().split('"""')[1], "by_family": {k: v for k, v in I.items() if k != "cells"}}
out["setup_checks"] = J("setup_checks.json")
json.dump(out, open("correctness.json", "w"), indent=1, default=float)
print("written correctness.json", len(json.dumps(out, default=float)) // 1000, "kB")
