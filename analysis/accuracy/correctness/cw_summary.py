"""Summaries of cw.json (Task 1) by task family, with the within-depth correctness-shuffle null."""
import json, numpy as np
from clib import *
def summarise():
    R = json.load(open("cw.json")); N = np.load("cw_null.npz")
    R = [r for r in R if "d" in r]; out = {}
    groups = {**{f: [f] for f in FAMS}, "serial (4 tasks)": TASKS}
    for g, fs in groups.items():
        rr = [r for r in R if r["fam"] in fs]
        if not rr: continue
        d = np.concatenate([r["d"] for r in rr]); p = np.concatenate([r["p"] for r in rr])
        x = np.concatenate([np.arange(r["L"]) / (r["L"] - 1) for r in rr])
        nd = np.concatenate([N[f"{r['model']}|{r['fam']}|{r['w']}|d"] for r in rr], 1).astype(float)   # [2000, layers]
        ns = np.concatenate([N[f"{r['model']}|{r['fam']}|{r['w']}|sig"] for r in rr], 1)
        nsig = int((p < .05).sum()); null_cnt = ns.sum(1)
        bh_cell = int(sum(bh(np.array(r["p"])).sum() for r in rr)); bh_pool = int(bh(p).sum())
        diff = np.concatenate([r["diff_pp"] for r in rr]); ks = np.concatenate([r["k_slope_pp"] for r in rr])
        kd = np.concatenate([r["k_d_per_step"] for r in rr])
        def stat(v, nv):
            return {"obs": float(v), "null_mean": float(nv.mean()), "null_95": [float(np.percentile(nv, 2.5)), float(np.percentile(nv, 97.5))],
                    "p_two_sided": float((np.sum(np.abs(nv - nv.mean()) >= abs(v - nv.mean())) + 1) / (len(nv) + 1))}
        o = {"n_cells": len(rr), "n_models": len({r["model"] for r in rr}), "n_layers": int(len(d)),
             "n_correct_used": int(sum(r["n_correct_used"] for r in rr)), "n_wrong_used": int(sum(r["n_wrong_used"] for r in rr)),
             "mean_d": stat(d.mean(), nd.mean(1)), "mean_abs_d": stat(np.abs(d).mean(), np.abs(nd).mean(1)),
             "frac_absd_gt_0.2": stat((np.abs(d) > .2).mean(), (np.abs(nd) > .2).mean(1)),
             "frac_absd_gt_0.5": stat((np.abs(d) > .5).mean(), (np.abs(nd) > .5).mean(1)),
             "median_abs_d": float(np.median(np.abs(d))), "max_abs_d": float(np.abs(d).max()),
             "n_sig_uncorrected": nsig, "frac_sig_uncorrected": nsig / len(d), "null_n_sig_mean": float(null_cnt.mean()),
             "null_n_sig_95pct": float(np.percentile(null_cnt, 95)), "p_n_sig": float((np.sum(null_cnt >= nsig) + 1) / (len(null_cnt) + 1)),
             "n_sig_BH_within_cell": bh_cell, "n_sig_BH_pooled": bh_pool,
             "back_half_mean_d": stat(d[x >= .5].mean(), nd[:, x >= .5].mean(1)), "front_half_mean_d": stat(d[x < .5].mean(), nd[:, x < .5].mean(1)),
             "by_decile_mean_d": [float(d[np.minimum((x * 10).astype(int), 9) == i].mean()) for i in range(10)],
             "by_decile_null_sd": [float(nd[:, np.minimum((x * 10).astype(int), 9) == i].mean(1).std()) for i in range(10)],
             "mean_abs_diff_pp": float(np.abs(diff).mean()), "mean_diff_pp": float(diff.mean()),
             "mean_abs_k_slope_pp_per_step": float(np.abs(ks).mean()), "mean_k_slope_pp_per_step": float(ks.mean()),
             "back_half_mean_diff_pp": float(diff[x >= .5].mean()), "back_half_mean_k_slope_pp": float(ks[x >= .5].mean()),
             "mean_abs_k_d_per_step": float(np.abs(kd).mean())}
        for lab in ("above", "near"):
            v = [np.array(r[f"d_{lab}"]) for r in rr if f"d_{lab}" in r]
            if v: v = np.concatenate(v); o[f"depths_{lab}_chance"] = {"n_cells": sum(f"d_{lab}" in r for r in rr), "mean_d": float(v.mean()), "mean_abs_d": float(np.abs(v).mean()), "frac_absd_gt_0.2": float((np.abs(v) > .2).mean())}
        out[g] = o
    # per model x task: wording-averaged per-layer d, cell-level test of mean d and of the count of significant layers
    cells = []
    for m in ORDER:
        for f in FAMS:
            rr = [r for r in R if r["model"] == m and r["fam"] == f]
            if not rr: continue
            d = np.mean([r["d"] for r in rr], 0); nd = np.mean([N[f"{r['model']}|{r['fam']}|{r['w']}|d"] for r in rr], 0).astype(float)
            cnt = sum(int((np.array(r["p"]) < .05).sum()) for r in rr); nc = sum(N[f"{r['model']}|{r['fam']}|{r['w']}|sig"].sum(1) for r in rr)
            x = np.arange(rr[0]["L"]) / (rr[0]["L"] - 1)
            cells.append({"model": m, "fam": f, "wordings": [r["w"] for r in rr], "mean_d": float(d.mean()),
                          "p_mean_d": float((np.sum(np.abs(nd.mean(1)) >= abs(d.mean())) + 1) / (len(nd) + 1)),
                          "back_mean_d": float(d[x >= .5].mean()),
                          "p_back_mean_d": float((np.sum(np.abs(nd[:, x >= .5].mean(1)) >= abs(d[x >= .5].mean())) + 1) / (len(nd) + 1)),
                          "mean_abs_d": float(np.abs(d).mean()), "n_sig": cnt, "n_layers": int(len(rr) * rr[0]["L"]),
                          "null_n_sig_mean": float(nc.mean()), "p_n_sig": float((np.sum(nc >= cnt) + 1) / (len(nc) + 1)),
                          "acc": float(np.mean([r["acc"] for r in rr])), "depths": [r["depths"] for r in rr]})
    pc = np.array([c["p_n_sig"] for c in cells]); pm = np.array([c["p_mean_d"] for c in cells])
    for c, a, b in zip(cells, bh(pc), bh(pm)): c["BH_n_sig"] = bool(a); c["BH_mean_d"] = bool(b)
    out["cells"] = cells
    out["n_cells_total"] = len(cells)
    out["cells_p_n_sig_lt05"] = int((pc < .05).sum()); out["cells_p_mean_d_lt05"] = int((pm < .05).sum())
    out["cells_BH_n_sig"] = int(bh(pc).sum()); out["cells_BH_mean_d"] = int(bh(pm).sum())
    out["cells_without_usable_depth"] = [f"{r['model']}|{r['fam']}|{r['w']}" for r in json.load(open("cw.json")) if "d" not in r]
    return out
if __name__ == "__main__":
    o = summarise()
    for g, v in o.items():
        if not isinstance(v, dict) or "mean_d" not in v: continue
        print(f"{g:18s} cells {v['n_cells']:3d} layers {v['n_layers']:5d} | mean d {v['mean_d']['obs']:+.3f} (null 95% {v['mean_d']['null_95'][0]:+.3f},{v['mean_d']['null_95'][1]:+.3f}) "
              f"| mean|d| {v['mean_abs_d']['obs']:.3f} (null {v['mean_abs_d']['null_mean']:.3f}) | >0.2 {v['frac_absd_gt_0.2']['obs']:.3f} (null {v['frac_absd_gt_0.2']['null_mean']:.3f}) "
              f"| >0.5 {v['frac_absd_gt_0.5']['obs']:.3f} (null {v['frac_absd_gt_0.5']['null_mean']:.3f}) | sig {v['n_sig_uncorrected']} = {100*v['frac_sig_uncorrected']:.1f}% (null {v['null_n_sig_mean']:.1f}, 95pct {v['null_n_sig_95pct']:.0f}, p {v['p_n_sig']:.4f}) "
              f"| BH cell {v['n_sig_BH_within_cell']} pooled {v['n_sig_BH_pooled']} | back d {v['back_half_mean_d']['obs']:+.3f} p {v['back_half_mean_d']['p_two_sided']:.3f} "
              f"| |diff| {v['mean_abs_diff_pp']:.3f}pp vs |k slope| {v['mean_abs_k_slope_pp_per_step']:.3f}pp/step")
        for lab in ("above", "near"):
            if f"depths_{lab}_chance" in v: print(f"      {lab}-chance depths: {v[f'depths_{lab}_chance']}")
        print("      decile mean d:", np.round(v["by_decile_mean_d"], 3), "null sd", np.round(v["by_decile_null_sd"], 3))
    print("cells:", o["n_cells_total"], "p_n_sig<.05:", o["cells_p_n_sig_lt05"], "BH:", o["cells_BH_n_sig"], "| p_mean_d<.05:", o["cells_p_mean_d_lt05"], "BH:", o["cells_BH_mean_d"])
    for c in o["cells"]:
        if c["p_n_sig"] < .05 or c["p_mean_d"] < .05:
            print(f"   {c['model']:28s} {c['fam']:11s} acc {c['acc']:.2f} mean d {c['mean_d']:+.3f} (p {c['p_mean_d']:.4f}{' BH' if c['BH_mean_d'] else ''}) back {c['back_mean_d']:+.3f} (p {c['p_back_mean_d']:.4f}) "
                  f"sig {c['n_sig']}/{c['n_layers']} (null {c['null_n_sig_mean']:.1f}, p {c['p_n_sig']:.4f}{' BH' if c['BH_n_sig'] else ''}) mean|d| {c['mean_abs_d']:.3f}")
    print("no usable depth:", o["cells_without_usable_depth"])
    json.dump(o, open("cw_summary.json", "w"), indent=1)
