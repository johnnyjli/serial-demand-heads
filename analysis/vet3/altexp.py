"""(3) Alternative explanations: answer-set entropy and correctness as item-level covariates of the k-slope,
layer by layer (body fixed effects for binding/relational).  Reports back-half prevalence and depth trend of the
adjusted k-slope, and the covariates' own back-half signs.  Writes altexp.json and altexp_layers.csv."""
import sys, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '.'); from v2lib import *
import statsmodels.formula.api as smf
def demean(v, body, within):
    if not within: return v - v.mean(0)
    out = v.astype(float).copy()
    for g in np.unique(body):
        i = body == g; out[i] = v[i] - v[i].mean(0)
    return out
def ols_t(Y, X):
    """Y [n, L], X [n, p] (already demeaned); returns coef [p, L], t [p, L]"""
    n, p = X.shape; XtX = np.linalg.pinv(X.T @ X); B = XtX @ X.T @ Y; E = Y - X @ B
    dof = n - p - 1; s2 = (E ** 2).sum(0) / dof; se = np.sqrt(np.outer(np.diag(XtX), s2)); return B, B / se
rows = []; cellsum = []
for m, f, w, a in cells("eager", TASKS + ["fixed_digit"]):
    d = load(m, f, w, a, True); body = d["body"]; wi = d["within"]; x = d["x"]
    Y = demean(d["y25"], body, wi); k = demean(d["k"][:, None], body, wi)
    e = demean(d["ent"][:, None], body, wi); c = demean(d["correct"].astype(float)[:, None], body, wi)
    lg = demean(np.log(np.clip(d["pgold"], 1e-6, 1))[:, None], body, wi)
    fits = {"k": np.hstack([k]), "k+ent": np.hstack([k, e]), "k+corr": np.hstack([k, c]), "k+ent+corr": np.hstack([k, e, c]),
            "k+ent+corr+logpgold": np.hstack([k, e, c, lg])}
    res = {}
    for name, X in fits.items():
        if np.linalg.matrix_rank(X) < X.shape[1]: res[name] = None; continue
        B, t = ols_t(Y, X); res[name] = (B, t)
    rk_e = np.corrcoef(k[:, 0], e[:, 0])[0, 1]; rk_c = np.corrcoef(k[:, 0], c[:, 0])[0, 1] if c.std() > 0 else np.nan
    cs = {"model": m, "task": f, "w": w, "r_k_ent": rk_e, "r_k_corr": rk_c, "acc": float(d["correct"].mean()), "mean_ent": float(d["ent"].mean()),
          "max_ent": float(np.log(int(d["nans"])))}
    for name, v in res.items():
        if v is None: continue
        B, t = v
        cs[f"{name}|back_t"] = float(t[0, x >= .5].mean()); cs[f"{name}|front_t"] = float(t[0, x < .5].mean())
        cs[f"{name}|back_bpp"] = float(100 * B[0, x >= .5].mean()); cs[f"{name}|front_bpp"] = float(100 * B[0, x < .5].mean())
        if "ent" in name:
            cs[f"{name}|ent_back_t"] = float(t[1, x >= .5].mean()); cs[f"{name}|ent_front_t"] = float(t[1, x < .5].mean())
        for l in range(d["L"]):
            rows.append({"model": m, "task": f, "w": w, "cell": f"{m}|{f}|{w}", "x": x[l], "fit": name, "b_pp": 100 * B[0, l], "t": t[0, l]})
    cellsum.append(cs)
CS = pd.DataFrame(cellsum); Lt = pd.DataFrame(rows); Lt.to_csv("altexp_layers.csv", index=False)
out = {}
for label, T in {"all4": TASKS, "no_arith": ["pointer", "binding", "relational"], "binding+relational": ["binding", "relational"],
                 **{t: [t] for t in TASKS + ["fixed_digit"]}}.items():
    c = CS[CS.task.isin(T)]; r = {"n": len(c), "mean_r_k_ent": float(c.r_k_ent.mean()), "mean_r_k_corr": float(c.r_k_corr.mean()), "acc": float(c.acc.mean())}
    for name in ["k", "k+ent", "k+corr", "k+ent+corr", "k+ent+corr+logpgold"]:
        col = f"{name}|back_t"
        if col not in c: continue
        cc = c.dropna(subset=[col])
        r[name] = {"n": len(cc), "back<0": float((cc[col] < 0).mean()), "back<front": float((cc[col] < cc[f"{name}|front_t"]).mean()),
                   "mean_back_t": float(cc[col].mean()), "mean_back_bpp": float(cc[f"{name}|back_bpp"].mean()), "mean_front_bpp": float(cc[f"{name}|front_bpp"].mean())}
        if "ent" in name:
            r[name]["ent_back_t_mean"] = float(cc[f"{name}|ent_back_t"].mean()); r[name]["ent_back<0"] = float((cc[f"{name}|ent_back_t"] < 0).mean())
            r[name]["ent_back<front"] = float((cc[f"{name}|ent_back_t"] < cc[f"{name}|ent_front_t"]).mean())
        sub = Lt[Lt.task.isin(T) & (Lt.fit == name)]
        fe = smf.ols("b_pp ~ x + C(cell)", sub).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.model)[0]})
        r[name]["depth_slope_bpp"] = float(fe.params["x"]); r[name]["depth_ci"] = [float(v) for v in fe.conf_int().loc["x"]]
    out[label] = r
    print(f"{label:20s} n={r['n']:3d} r(k,ent)={r['mean_r_k_ent']:+.2f} r(k,corr)={r['mean_r_k_corr']:+.2f} acc={r['acc']:.2f}")
    for name in ["k", "k+ent", "k+corr", "k+ent+corr", "k+ent+corr+logpgold"]:
        if name not in r: continue
        s = r[name]
        extra = f" | ent coef back t {s['ent_back_t_mean']:+.2f}, ent back<0 {s['ent_back<0']:.2f}, ent back<front {s['ent_back<front']:.2f}" if "ent_back_t_mean" in s else ""
        print(f"    {name:20s} back<0 {s['back<0']:.2f} back<front {s['back<front']:.2f} mean back t {s['mean_back_t']:+.2f} back/front b {s['mean_back_bpp']:+.3f}/{s['mean_front_bpp']:+.3f} pp/step depth-slope {s['depth_slope_bpp']:+.3f} [{s['depth_ci'][0]:+.3f},{s['depth_ci'][1]:+.3f}]{extra}")
json.dump(out, open("altexp.json", "w"), indent=1)
CS.to_csv("altexp_cells.csv", index=False)
