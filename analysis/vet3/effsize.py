"""(4) Effect size in interpretable units vs baseline differences.  Uses layertab.csv (primary: eager, clean)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '.'); from v2lib import *
T = pd.read_csv("layertab.csv"); T = T[T.task.isin(TASKS)]
rng = np.random.default_rng(0)
B = T[T.x >= .5].groupby(["model", "task", "w"]).agg(b_pp=("b_pp", "mean"), neff_b=("neff_b", "mean"), base_pp=("base_pp", "mean"),
      base_k1=("base_k1_pp", "mean"), item_sd=("item_sd_pp", "mean"), neff_base=("neff_base", "mean"), H=("H", "first"),
      krange=("kmax", "first"), kmin=("kmin", "first")).reset_index()
F = T[T.x < .5].groupby(["model", "task", "w"]).agg(b_pp=("b_pp", "mean"), neff_b=("neff_b", "mean")).reset_index()
B["span"] = B.krange - B.kmin; B["total_pp"] = B.b_pp * B.span; B["total_neff"] = B.neff_b * B.span
B["neff_pctH"] = 100 * B.neff_b / B.H; B["d_per_step"] = B.b_pp / B.item_sd; B["bf_pp"] = B.b_pp.values - F.b_pp.values
def clboot(v, g, n=5000):
    gs = np.unique(g); idx = {x: np.nonzero(g == x)[0] for x in gs}; s = []
    for _ in range(n):
        pick = rng.choice(gs, len(gs)); s.append(np.mean(np.concatenate([v[idx[x]] for x in pick])))
    return [float(np.quantile(s, .025)), float(np.quantile(s, .975))]
out = {}
for label, sub in {"all4": B, "binding+relational": B[B.task.isin(["binding", "relational"])], **{t: B[B.task == t] for t in TASKS}}.items():
    r = {}
    for col in ["b_pp", "bf_pp", "total_pp", "neff_b", "total_neff", "neff_pctH", "d_per_step"]:
        v = sub[col].values; r[col] = {"mean": float(v.mean()), "median": float(np.median(v)), "q25": float(np.quantile(v, .25)), "q75": float(np.quantile(v, .75)),
                                        "ci_model_boot": clboot(v, sub.model.values)}
    out[label] = r
    print(f"{label:18s} back-half k-slope: mean {r['b_pp']['mean']:+.3f} pp/step (median {r['b_pp']['median']:+.3f}, IQR [{r['b_pp']['q25']:+.3f},{r['b_pp']['q75']:+.3f}], CI {np.round(r['b_pp']['ci_model_boot'],3)}); back-front {r['bf_pp']['mean']:+.3f}; over k range {r['total_pp']['mean']:+.2f} pp; "
          f"eff heads {r['neff_b']['mean']:+.3f}/step ({r['neff_pctH']['mean']:+.2f}% of H), over range {r['total_neff']['mean']:+.2f} heads; std per step {r['d_per_step']['mean']:+.3f} item-SD")
# baselines: back-half top-25% share by model (averaged over tasks, wordings), and in effective heads
M = B.groupby("model").agg(base=("base_pp", "mean"), neff=("neff_base", "mean"), H=("H", "first"), total=("total_pp", "mean"))
print("\nper-model back-half mean top-25% share (pp):", M.base.round(1).to_dict())
print("between-model SD of back-half share: %.2f pp, range %.1f-%.1f pp" % (M.base.std(), M.base.min(), M.base.max()))
print("between-model SD of back-half eff. heads as %% of H: %.2f%%" % (100 * (M.neff / M.H)).std())
wt = B.groupby(["model"]).base_pp.std().mean(); print("mean within-model between-cell (task x wording) SD of back-half share: %.2f pp" % wt)
ww = B.pivot_table(index=["model", "task"], columns="w", values="base_pp"); print("mean |w0-w1| difference in back-half share: %.2f pp" % (ww.w0 - ww.w1).abs().mean())
L = T[T.x >= .5].groupby(["model", "task", "w"]).base_pp.std().mean(); print("mean between-layer SD of share within back half: %.2f pp" % L)
print("mean within-layer item SD of share: %.2f pp" % B.item_sd.mean())
print("mean total change over k range (back half): %.2f pp; ratio to between-model SD: %.3f" % (B.total_pp.mean(), abs(B.total_pp.mean()) / M.base.std()))
out["baselines"] = {"between_model_sd_pp": float(M.base.std()), "model_range_pp": [float(M.base.min()), float(M.base.max())],
                    "within_model_between_cell_sd_pp": float(wt), "mean_abs_w0_w1_pp": float((ww.w0 - ww.w1).abs().mean()),
                    "between_layer_sd_back_pp": float(L), "item_sd_pp": float(B.item_sd.mean()),
                    "between_model_sd_neff_pctH": float((100 * (M.neff / M.H)).std()), "mean_neff_back": float(M.neff.mean()), "mean_H": float(M.H.mean())}
# most extreme layers: the largest per-layer spreading slopes
print("\nlayer-level b_pp quantiles (back half):", np.round(np.quantile(T[T.x >= .5].b_pp, [.01, .05, .25, .5, .75, .95, .99]), 3))
print("layer-level b_pp quantiles (front half):", np.round(np.quantile(T[T.x < .5].b_pp, [.01, .05, .25, .5, .75, .95, .99]), 3))
json.dump(out, open("effsize.json", "w"), indent=1)
