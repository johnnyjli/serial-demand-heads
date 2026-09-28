import json, sys, numpy as np
R = json.load(open(sys.argv[1]))
T = ["pointer", "arith", "binding", "relational"]
def fr(key, rows):
    f = np.array([r[key][0] for r in rows]); b = np.array([r[key][1] for r in rows])
    return (b < 0).mean(), (b < f).mean(), b.mean(), f.mean(), np.median(b)
def line(key, label=None):
    s = f"{(label or key):28s}"
    a = fr(key, R); s += f" ALL back<0 {a[0]:.2f} back<front {a[1]:.2f} backz {a[2]:+.2f} frontz {a[3]:+.2f} |"
    for t in T:
        rr = [r for r in R if r["fam"] == t]; a = fr(key, rr); s += f" {t[:4]} {a[0]:.2f}/{a[1]:.2f}"
    print(s)
print("n cells", len(R))
for k in ["raw", "logM", "adj_logM", "adj_invM", "adj_cubic", "adj_int",
          "floor_min50", "floor_min100", "floor_p5", "floor_global",
          "sim_min50_resid", "sim_min100_resid", "sim_p5_resid", "sim_global_resid", "sim_uniform_resid",
          "sim_worst_min_resid", "sim_worst_p5_resid", "fixedshare", "logT", "logR"]:
    line(k)
sp = [r for r in R if r["raw"][1] < 0]
print("\ncells with raw back<0:", len(sp))
def pooled(key, rows):   # pooled ratio: sum over cells of adjusted back slope / sum raw back slope
    return np.sum([r[key] * r["bY_back"] for r in rows]) / np.sum([r["bY_back"] for r in rows])
for k in ["adj_logM", "adj_invM", "adj_cubic", "adj_int", "floor_min50", "floor_min100", "floor_p5", "floor_global"]:
    v = np.array([r[k + "_ratio_back"] for r in sp])
    print(f"{k:14s} remaining fraction of back-half slope: median {np.median(v):.2f} IQR [{np.percentile(v,25):.2f},{np.percentile(v,75):.2f}] pooled {pooled(k+'_ratio_back', sp):.2f}",
          " by fam:", " ".join(f"{t[:4]} {np.median([r[k+'_ratio_back'] for r in sp if r['fam']==t]):.2f}" for t in T))
for k in ["adj_logM", "adj_int"]:
    v = np.array([r[k + "_ratio_contrast"] for r in R])
    print(f"{k:14s} remaining fraction of back-minus-front slope contrast: median {np.median(v):.2f}")
print()
for k in ["sim_min50", "sim_min100", "sim_p5", "sim_global", "sim_uniform", "sim_worst_min", "sim_worst_p5"]:
    v = np.array([r[k + "_pred_ratio_back"] for r in sp]); ff = np.array([r[k + "_floorfrac"] for r in R])
    print(f"{k:14s} floor = {np.median(ff):.2f} of layer mass (median); floor model predicts fraction of back-half slope: median {np.median(v):.2f} IQR [{np.percentile(v,25):.2f},{np.percentile(v,75):.2f}] pooled {pooled(k+'_pred_ratio_back', sp):.2f}",
          " by fam:", " ".join(f"{t[:4]} {np.median([r[k+'_pred_ratio_back'] for r in sp if r['fam']==t]):.2f}" for t in T))
for k in ["floor_min50", "floor_min100", "floor_p5", "floor_global"]:
    print(k, "floor frac median", round(np.median([r[k + "_floorfrac"] for r in R]), 2))
print("\n(3) cross-sectional (within k [+ body FE]):")
v = np.array([r["xs_r_back_mean"] for r in R]); print(" mean within-k corr(share, logM) over back layers: median", round(np.median(v), 3), "frac>0", round((v > 0).mean(), 2))
v = np.array([r["xs_frac_pos_back"] for r in R]); print(" frac of back layers with positive within-k slope: median", round(np.median(v), 2))
v = np.array([r["xs_pred_ratio_back"] for r in sp]); print(" predicted k-slope via mass / observed (spreading cells): median", round(np.median(v), 2), "IQR", np.percentile(v, [25, 75]).round(2), "pooled", round(pooled("xs_pred_ratio_back", sp), 2))
for t in T: print("   ", t, round(np.median([r["xs_pred_ratio_back"] for r in sp if r["fam"] == t]), 2))
print("\n(4) head level")
v = np.array([r["logM"][1] for r in R]); print(" back logM z<0 frac", round((v < 0).mean(), 2), "mean", round(v.mean(), 2))
for key in ["dlogT_back", "dlogR_back", "dlogM_back"]:
    v = np.array([r[key] for r in R]); print(f" {key} median {np.median(v)*100:+.2f}%/k  frac<0 {(v<0).mean():.2f}")
v = np.array([r["dlogT_back"] - r["dlogR_back"] for r in R]); print(" dlogT - dlogR <0 frac", round((v < 0).mean(), 2))
for key in ["frac_heads_rising_back", "frac_heads_falling_back", "frac_heads_rising_in_massfall", "frac_rest_rising_in_massfall",
            "frac_back_layers_rest_rises", "frac_back_layers_top_falls_rest_rises", "floorfit_R2_back"]:
    v = np.array([r[key] for r in R if r[key] is not None]); print(f" {key}: median {np.median(v):.3f} mean {v.mean():.3f} (n={len(v)})")
# floor-fit share explanation pooled over layers (fixed-set share)
o = np.concatenate([r["obs_dP_layers_back"] for r in R]); e = np.concatenate([r["floorfit_dP_layers_back"] for r in R])
print(" fixed-set share: back layers with obs spreading:", round((o < 0).mean(), 2), " pooled floor-fit explained fraction (over spreading layers):",
      round(e[o < 0].sum() / o[o < 0].sum(), 2), " all layers:", round(e.sum() / o.sum(), 2))
for t in T:
    o = np.concatenate([r["obs_dP_layers_back"] for r in R if r["fam"] == t]); e = np.concatenate([r["floorfit_dP_layers_back"] for r in R if r["fam"] == t])
    print("   ", t, "spreading layers", round((o < 0).mean(), 2), "floorfit explained", round(e[o < 0].sum() / o[o < 0].sum(), 2), "all", round(e.sum() / o.sum(), 2))
v = np.array([r["floor_pred_ratio_back"] for r in R if r["obs_dP_back"] < 0]); print(" fixed-set, floor=min_h prediction / obs (cells with fixed-set spreading):", round(np.median(v), 2), "n", len(v))
print(" top term / rest term (fixed-set share slope, back sum), medians:",
      np.median([r["topterm_back"] for r in R]), np.median([r["restterm_back"] for r in R]))
# per model
print("\nper model (back<0 frac: raw / adj_logM / sim_min100_resid / sim_worst_min_resid):")
for m in dict.fromkeys(r["model"] for r in R):
    rr = [r for r in R if r["model"] == m]
    print(f"  {m:28s}", " ".join(f"{np.mean([r[k][1] < 0 for r in rr]):.2f}" for k in ["raw", "adj_logM", "sim_min100_resid", "sim_worst_min_resid"]),
          " floor(min) frac", round(np.median([r["floor_min100_floorfrac"] for r in rr]), 2),
          " rising heads", round(np.mean([r["frac_heads_rising_back"] for r in rr]), 3))
