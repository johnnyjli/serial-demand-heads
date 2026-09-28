"""Robustness table for the model signatures: for each variant, model and task shares of the non-shared
variance, permutation p of the model effect, split-half r across tasks, correlation of the signatures with the
primary (top-10%) signatures, and mean signature correlation within lineage vs between lineages."""
import sys, json, numpy as np, pandas as pd
SP = sys.argv[1]; sys.path.insert(0, SP)
from model_diff import ORDER, T, LIN, decompose, ss, perm_model, split_half, from_consist, from_cells
a0 = decompose(from_consist("q10"))[1]
def row(Z, models=ORDER, ref=None):
    mu, a, b, e = decompose(Z); s = ss(Z); ns = s["model"] + s["task"] + s["resid"]
    o, nm, p = perm_model(Z, 2000); sh = split_half(Z); R = np.corrcoef(a)
    lin = np.mean([R[i, j] for i in range(len(models)) for j in range(i + 1, len(models)) if LIN[models[i]] == LIN[models[j]]])
    oth = np.mean([R[i, j] for i in range(len(models)) for j in range(i + 1, len(models)) if LIN[models[i]] != LIN[models[j]]])
    ref = a0 if ref is None else ref
    return {"model_share": s["model"] / ns, "task_share": s["task"] / ns, "p_model": p, "split_half": sh[0], "p_split": sh[2],
            "r_primary": float(np.corrcoef(a.ravel(), ref.ravel())[0, 1]), "lineage_r": float(lin), "other_r": float(oth), "n_models": len(models)}
out = {}
for st, lab in [("q10", "top-10% share (primary)"), ("q25", "top-25% share"), ("q50", "top-50% share"), ("entropy", "effective number of heads"), ("top1", "top-1 share")]:
    out[lab] = row(from_consist(st))
Zs, _ = from_cells("eager", "sink_head"); out["sink heads removed"] = row(Zs)
Zw, _ = from_cells("eager", "all"); out["per-wording z, averaged"] = row(Zw)
ms = [m for m in ORDER if from_cells("sdpa", "all", [m])[0] is not None]
Zsd, _ = from_cells("sdpa", "all", ms); Zse, _ = from_cells("eager", "all", ms)
ae = decompose(Zse)[1]; out[f"SDPA kernel ({len(ms)} models)"] = row(Zsd, ms, ref=ae)
A = pd.read_csv(f"{SP}/vet3/altexp_layers.csv"); A["dec"] = np.minimum((A.x * 10).astype(int), 9)
for fit, lab in [("k+ent", "adjusted for answer entropy"), ("k+ent+corr", "adjusted for entropy and correctness")]:
    S = A[A.fit == fit].groupby(["model", "task", "w", "dec"]).t.mean().groupby(["model", "task", "dec"]).mean()
    out[lab] = row(np.array([[[S[(m, t, d)] for d in range(10)] for t in T] for m in ORDER]))
json.dump(out, open(f"{SP}/model_diff4.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:40s} model {v['model_share']:.2f} task {v['task_share']:.2f} p {v['p_model']:.4f} split {v['split_half']:.2f} (p {v['p_split']:.4f}) r_prim {v['r_primary']:.2f} lineage {v['lineage_r']:.2f} other {v['other_r']:+.2f}")
