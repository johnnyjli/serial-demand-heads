"""Extras for the E2b accuracy/behaviour task: Qwen2.5-7B pair kernel diagnostics, correct-but-inexact traces,
one-step share bootstrap over models, stepwise context length.  Output e2b_accuracy_extras.json."""
import sys, os, re, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util
spec = importlib.util.spec_from_file_location("ab", os.path.join(HERE, "e2b_accuracy_behaviour.py"))
ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)   # re-runs the main analysis (deterministic)
MODELS, FAMS, KR, R, ITEM, tq_items = ab.MODELS, ab.FAMS, ab.KR, ab.R, ab.ITEM, ab.tq_items
OUT = {}

# 1. Qwen2.5-7B pair: anomalies by k, accuracy on clean traces, versus other models
an = {}
for m in MODELS:
    rec = [r for f in FAMS for r in tq_items[(m, f)]]
    by_k = {k: float(np.mean([r["anomalous"] for r in rec if r["k"] == k])) for k in range(1, 7)}
    clean = [r for r in rec if not r["anomalous"]]
    clean_k3 = [r for r in clean if r["k"] >= 3]
    an[m] = {"anomalous_rate": float(np.mean([r["anomalous"] for r in rec])), "anomalous_by_k": by_k,
             "acc_all": float(np.mean([r["correct"] for r in rec])),
             "acc_clean_items": float(np.mean([r["correct"] for r in clean])), "n_clean": len(clean),
             "acc_clean_k>=3": float(np.mean([r["correct"] for r in clean_k3])),
             "exact_clean": float(np.mean([r["exact"] for r in clean])),
             "acc_anomalous_items": float(np.mean([r["correct"] for r in rec if r["anomalous"]])) if any(r["anomalous"] for r in rec) else None}
OUT["stepwise_anomalies"] = an

# E2 kernel comparison for Qwen2.5-7B-Instruct, k=1 and k>=3 (serial families, strict, wording-averaged)
q = {}
for m in ("Qwen2.5-7B-Instruct",):
    for attn in ("eager", "sdpa"):
        acc = {f: ab.e2_acc_by_k(m, f, attn) for f in FAMS}
        q[f"E2_{attn}"] = {"k=1": float(np.mean([acc[f][0] for f in FAMS])),
                           "k>=3": float(np.mean([np.mean(acc[f][2:]) for f in FAMS])),
                           "overall": float(np.mean([np.mean(acc[f]) for f in FAMS]))}
for m in ab.Q7:
    q[f"{m}_E2b"] = {src: ab.per[m][src] for src in ("E2b_direct_strict", "E2b_stepwise_written")}
# Qwen2.5-7B base: only pointer/w0 under both kernels
d_e, d_s = ab.e2_load("Qwen2.5-7B", "pointer", "w0", "eager"), ab.e2_load("Qwen2.5-7B", "pointer", "w0", "sdpa")
q["Qwen2.5-7B_pointer_w0"] = {"eager_by_k": [float(d_e["strict"][d_e["k"] == k].mean()) for k in range(1, 6)],
                              "sdpa_by_k": [float(d_s["strict"][d_s["k"] == k].mean()) for k in range(1, 6)]}
OUT["q7_kernel"] = q

# 2. correct final answers with an inexact trace
ci = {}
for sub, ms in ab.SUBSETS.items():
    ci[sub] = {}
    for f in FAMS:
        rr = [r for m in ms for r in tq_items[(m, f)] if r["correct"]]
        ci[sub][f] = {"n_correct": len(rr), "inexact_share": float(np.mean([not r["exact"] for r in rr])),
                      "inexact_overshoot": float(np.mean([(not r["exact"]) and r["n_hops"] > r["gold_hops"] for r in rr])),
                      "inexact_undershoot": float(np.mean([(not r["exact"]) and r["n_hops"] < r["gold_hops"] for r in rr]))}
OUT["correct_with_inexact_trace"] = ci

# 3. one-step share at k=2 (pred_constrained), per model, E2b direct vs E2 (paper kernels), bootstrap over models
os_ = {}
for f in ("pointer", "arith"):
    e2b_pm, e2_pm, st_pm = [], [], []
    for m in MODELS:
        h = [ab.hops_done(f, ITEM[x["id"]]["text_direct"], x["pred_constrained"]) for x in R[(m, "direct", f)]["items"] if x["k"] == 2]
        e2b_pm.append(np.mean([v == 1 for v in h]))
        h = [ab.hops_done(f, ITEM[x["id"]]["text_direct"], x.get("pred_gen")) for x in R[(m, "stepwise", f)]["items"] if x["k"] == 2]
        st_pm.append(np.mean([v == 1 for v in h]))
        hh = []
        for w in ("w0", "w1"):
            d = ab.e2_load(m, f, w, ab.e2_attn(m))
            hh += [ab.hops_done(f, it["text"], it["pred_constrained"]) for it in d["items"] if it["k"] == 2]
        e2_pm.append(np.mean([v == 1 for v in hh]))
    os_[f] = {"models": MODELS, "E2b_direct": [float(v) for v in e2b_pm], "E2": [float(v) for v in e2_pm],
              "E2b_stepwise_written": [float(v) for v in st_pm],
              "E2b_direct_mean_ci": ab.boot(e2b_pm), "E2_mean_ci": ab.boot(e2_pm),
              "diff_ci": ab.boot(np.array(e2b_pm) - np.array(e2_pm)),
              "stepwise_mean_ci": ab.boot(st_pm)}
OUT["one_step_k2"] = os_

# 4. stepwise context length at the measured position
nt = {}
for f in FAMS:
    vals = {k: [r["ntok"] for m in MODELS for r in tq_items[(m, f)] if r["k"] == k and r["ntok"]] for k in KR[f]}
    nt[f] = {k: [int(np.min(v)), float(np.median(v)), int(np.max(v))] for k, v in vals.items()}
# direct-mode prompt length is not stored; approximate from the E2b direct texts is not tokenised here
OUT["stepwise_ntokens_min_median_max"] = nt

json.dump(ab.conv(OUT), open(os.path.join(HERE, "e2b_accuracy_extras.json"), "w"), indent=1)
print(json.dumps(ab.conv(OUT), indent=1)[:6000])
