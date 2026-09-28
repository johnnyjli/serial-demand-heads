"""Decile profile of per-layer z (mean, fraction < 0) with model-cluster bootstrap CIs; median-based and
layer-fraction versions of the back-half prevalence; front-half tests; per-model grand means."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, '.'); from v2lib import *
T = pd.read_csv("layertab.csv"); rng = np.random.default_rng(1)
T["dec"] = np.minimum((T.x * 10).astype(int), 9)
out = {}
for label, tasks in {"all4": TASKS, "binding+relational": ["binding", "relational"], "pointer": ["pointer"], "arith": ["arith"], "fixed_digit": ["fixed_digit"]}.items():
    S = T[T.task.isin(tasks)]
    # cell-level decile means, then average across cells; bootstrap over models
    C = S.groupby(["model", "task", "w", "dec"]).z.mean().unstack("dec")
    models = C.index.get_level_values(0).values; ms = np.unique(models)
    mean = C.mean().values
    boots = []
    for _ in range(3000):
        pick = rng.choice(ms, len(ms)); boots.append(np.concatenate([C.values[models == m] for m in pick]).mean(0))
    lo, hi = np.quantile(boots, [.025, .975], axis=0)
    fneg = (C < 0).mean().values
    out[label] = {"mean_z": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "frac_cells_neg": fneg.tolist()}
    print(f"{label:18s} decile mean z: " + " ".join(f"{m:+.2f}[{a:+.2f},{b:+.2f}]" for m, a, b in zip(mean, lo, hi)))
    print(f"{'':18s} frac cells <0 : " + " ".join(f"{v:.2f}" for v in fneg))
# median-based and layer-fraction prevalence (all4)
S = T[T.task.isin(TASKS)]
g = S.groupby(["model", "task", "w"])
med_b = g.apply(lambda d: d.z[d.x >= .5].median()); med_f = g.apply(lambda d: d.z[d.x < .5].median())
fr_b = g.apply(lambda d: (d.z[d.x >= .5] < 0).mean()); fr_f = g.apply(lambda d: (d.z[d.x < .5] < 0).mean())
print("\nmedian-based: back median z<0 in %.2f of cells; back median < front median in %.2f" % ((med_b < 0).mean(), (med_b < med_f).mean()))
print("layer-fraction: >50%% of back-half layers z<0 in %.2f of cells; mean frac back layers z<0 = %.2f vs front %.2f" % ((fr_b > .5).mean(), fr_b.mean(), fr_f.mean()))
print("fraction of back-half layers individually significant spreading (z<-1.96): %.3f, concentrating (z>1.96): %.3f; front: %.3f / %.3f" % (
    (S[S.x >= .5].z < -1.96).mean(), (S[S.x >= .5].z > 1.96).mean(), (S[S.x < .5].z < -1.96).mean(), (S[S.x < .5].z > 1.96).mean()))
# per-model grand means (wording and task averaged)
cm = S.assign(back=S.x >= .5).groupby(["model", "task", "w", "back"]).z.mean().unstack("back")
pm = cm.groupby("model").mean(); pm.columns = ["front", "back"]; pm["diff"] = pm.back - pm.front
print("\nper-model mean z (front / back / back-front):"); print(pm.round(2).loc[ORDER].to_string())
nf = int((pm.front > 0).sum()); print("front>0 in %d/17 models (sign p two-sided %.3f); front mean across models %+.2f" % (nf, binomtest(nf, 17, .5).pvalue, pm.front.mean()))
pb = pm.assign(base=[BASE[m] for m in pm.index]).groupby("base").mean()
print("bases (11): back<0 %d, back<front %d, front>0 %d" % ((pb.back < 0).sum(), (pb["diff"] < 0).sum(), (pb.front > 0).sum()))
print(pb.round(2).to_string())
out["per_model"] = pm.to_dict(); json.dump(out, open("profile.json", "w"), indent=1)
