"""Summaries of decode.json (Task 2) by task family."""
import json, numpy as np
from clib import *
def summarise():
    R = json.load(open("decode.json")); out = {}
    for g, fs in {**{f: [f] for f in FAMS}, "serial (4 tasks)": TASKS}.items():
        rr = [r for r in R if r["fam"] in fs and "activity" in r]
        if not rr: continue
        o = {"n_cells": len(rr), "n_cells_total": sum(r["fam"] in fs for r in R), "n_items_used": int(sum(r["n_used"] for r in rr))}
        for name in ("activity", "prompt"):
            a = np.array([r[name]["auc"] for r in rr]); p = np.array([r[name]["p"] for r in rr]); nm = np.array([r[name]["null_mean"] for r in rr])
            o[name] = {"median_auc": float(np.median(a)), "mean_auc": float(a.mean()), "iqr": [float(np.percentile(a, 25)), float(np.percentile(a, 75))],
                       "median_null_mean": float(np.median(nm)), "n_p_lt_05": int((p < .05).sum()), "n_BH": int(bh(p).sum()),
                       "expected_by_chance": 0.05 * len(rr)}
        a = np.array([r["prompt+activity"]["auc"] for r in rr]); pi = np.array([r["prompt+activity"]["p_increment"] for r in rr])
        pa = np.array([r["prompt"]["auc"] for r in rr])
        o["prompt+activity"] = {"median_auc": float(np.median(a)), "mean_auc": float(a.mean()), "median_gain_over_prompt": float(np.median(a - pa)),
                                "mean_gain_over_prompt": float((a - pa).mean()), "n_increment_p_lt_05": int((pi < .05).sum()), "n_increment_BH": int(bh(pi).sum())}
        gs = [r["gold_shown_only_auc"] for r in rr if r.get("gold_shown_only_auc") is not None]
        o["gold_shown_only_auc_median"] = float(np.median(gs)) if gs else None
        allr = [r for r in R if r["fam"] in fs]
        o["setup_gold_is_worked_example_answer"] = {
            "frac_items": float(np.mean([r["frac_gold_shown"] for r in allr])),
            "acc_when_shown": float(np.nanmean([r["acc_gold_shown"] if r["acc_gold_shown"] is not None else np.nan for r in allr])),
            "acc_when_not_shown": float(np.nanmean([r["acc_gold_notshown"] if r["acc_gold_notshown"] is not None else np.nan for r in allr])),
            "frac_wrong_predictions_that_are_a_worked_example_answer": float(np.nanmean([r["frac_wrong_pred_shown"] if r["frac_wrong_pred_shown"] is not None else np.nan for r in allr]))}
        out[g] = o
    out["cells_activity_BH"] = [f"{r['model']}|{r['fam']}|{r['w']} auc {r['activity']['auc']:.3f} p {r['activity']['p']:.3f}" for r, s in
                                zip([r for r in R if "activity" in r], bh(np.array([r["activity"]["p"] for r in R if "activity" in r]))) if s]
    out["cells_increment_BH"] = [f"{r['model']}|{r['fam']}|{r['w']}" for r, s in
                                 zip([r for r in R if "activity" in r], bh(np.array([r["prompt+activity"]["p_increment"] for r in R if "activity" in r]))) if s]
    out["n_cells_run"] = sum("activity" in r for r in R); out["cells_not_run"] = [f"{r['model']}|{r['fam']}|{r['w']}" for r in R if "activity" not in r]
    return out
if __name__ == "__main__":
    o = summarise()
    for g, v in o.items():
        if isinstance(v, dict): print(g, json.dumps(v, indent=None))
    print("activity BH cells:", o["cells_activity_BH"]); print("increment BH cells:", o["cells_increment_BH"]); print("not run:", o["cells_not_run"])
    json.dump(o, open("decode_summary.json", "w"), indent=1)
