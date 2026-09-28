import json, numpy as np
from scipy import stats
from ilib import *
np.set_printoptions(precision=3, suppress=True)
def demean(v, body, wi):
    v = np.asarray(v, float)
    if not wi: return v - v.mean(0)
    out = v.copy()
    for g in np.unique(body): i = body == g; out[i] = v[i] - v[i].mean(0)
    return out
def ols_t(Y, X):
    n, p = X.shape; XtX = np.linalg.pinv(X.T @ X); B = XtX @ X.T @ Y; E = Y - X @ B
    s2 = (E ** 2).sum(0) / (n - p - 1); se = np.sqrt(np.outer(np.diag(XtX), s2)); return B / se
def cellfit(m, f, w, attn, stat="y10"):
    d = raw(m, f, w, attn); kp = d["keep"]
    Y = d[stat][kp]; k = d["k"][kp].astype(float); body = d["body"][kp]; e = d["ent"][kp]; c = d["correct"][kp].astype(float)
    wi = within(f); x = xrel(Y.shape[1])
    Yd = demean(Y, body, wi); kd = demean(k[:, None], body, wi); ed = demean(e[:, None], body, wi); cd = demean(c[:, None], body, wi)
    out = {}
    for name, X in [("k", np.hstack([kd])), ("k+ent", np.hstack([kd, ed])), ("k+ent+corr", np.hstack([kd, ed, cd]))]:
        if np.linalg.matrix_rank(X) < X.shape[1]: out[name] = np.nan; continue
        t = ols_t(Yd, X)[0]; out[name] = t[x >= .5].mean()
    kc = k - k.mean(); out["ent_slope"] = kc @ (e - e.mean()) / (kc @ kc)
    out["acc"] = d["correct"][kp].mean(); out["acc_k1"] = d["correct"][kp][k == 1].mean()
    return out
def model_feats(m, attn, tasks=TASKS):
    R = [cellfit(m, f, w, attn) for f in tasks for w in ("w0", "w1")]
    return {key: float(np.nanmean([r[key] for r in R])) for key in R[0]}
res = {}
for label, attn_of in [("primary", prim_attn), ("all-eager", lambda m: "eager")]:
    F = {m: model_feats(m, attn_of(m)) for m in sorted(set(x for p in PAIRS for x in p))}
    for key in ["acc", "acc_k1", "ent_slope", "k", "k+ent", "k+ent+corr"]:
        dv = np.array([F[p][key] - F[b][key] for b, p in PAIRS])
        print(f"{label:9s} {key:11s} change per pair {dv}  LW {lw(dv):+.3f}  lineages<0 {(lineages(dv)<0).sum()} pairs>0 {(dv>0).sum()}")
    res[label] = F
    Fc = {m: model_feats(m, attn_of(m), ["fixed_digit"]) for m in sorted(set(x for p in PAIRS for x in p))}
    for key in ["k", "k+ent", "k+ent+corr", "acc", "acc_k1", "ent_slope"]:
        dv = np.array([Fc[p][key] - Fc[b][key] for b, p in PAIRS])
        print(f"   control {label:9s} {key:11s} change per pair {dv}  LW {lw(dv):+.3f}  lineages<0 {(lineages(dv)<0).sum()}")
# between-base regression: back-half z (consist q10) on acc_k1
CJ = {r["model"]: r for r in json.load(open(f"{SP}/consist.json")) if r["stat"] == "q10"}
def backz(m): return np.mean([np.array(CJ[m]["families"][f]["z"])[xrel(CJ[m]["L"]) >= .5].mean() for f in TASKS])
a1 = np.array([model_feats(m, prim_attn(m))["acc_k1"] for m in BASES]); bz = np.array([backz(m) for m in BASES])
lr = stats.linregress(a1, bz); sp = stats.spearmanr(a1, bz)
print(f"between bases: slope {lr.slope:+.3f} r {lr.rvalue:+.3f} p {lr.pvalue:.3f} spearman {sp.correlation:+.3f} (p {sp.pvalue:.3f})")
F = res["primary"]; da1 = np.array([F[p]["acc_k1"] - F[b]["acc_k1"] for b, p in PAIRS]); pred = lr.slope * da1
obs = np.array([backz(p) - backz(b) for b, p in PAIRS])
print("predicted change per pair", pred, "LW", lw(pred), "observed LW", lw(obs), "residual", lw(obs) - lw(pred))
# also overall accuracy predictor
aa = np.array([res["primary"].get(m, None) and 0 for m in []])
acc_all = np.array([model_feats(m, prim_attn(m))["acc"] for m in BASES]); lr2 = stats.linregress(acc_all, bz)
print(f"between bases, overall acc: slope {lr2.slope:+.3f} r {lr2.rvalue:+.3f} p {lr2.pvalue:.3f}")
print("bases acc_k1", dict(zip(BASES, np.round(a1, 3))), "back z", dict(zip(BASES, np.round(bz, 2))))
# model_diff2 / model_diff3 comparison
M2 = json.load(open(f"{SP}/model_diff2.json"))["features"]
print("model_diff2 acc_k1 change", [round(M2["acc_k1"][p] - M2["acc_k1"][b], 3) for b, p in PAIRS], "ent_slope", [round(M2["ent_slope"][p] - M2["ent_slope"][b], 3) for b, p in PAIRS])
M3 = json.load(open(f"{SP}/model_diff3.json"))
for fit in ["k", "k+ent", "k+ent+corr"]:
    dv = np.array([M3[fit]["back_mean_t"][p] - M3[fit]["back_mean_t"][b] for b, p in PAIRS]); print("model_diff3 decile back t", fit, dv, "LW", lw(dv))
