"""(1) Continuous depth moderator: layer-level k-slopes (pp of top-25% share per step) regressed on relative depth.
Equivalent to the depth x k interaction of an item-level model with cell x layer intercepts.
Cluster-robust (by model; by base model) and wild-cluster bootstrap CIs; MixedLM with random intercept+depth slope by model
and random intercepts per cell; piecewise and decile forms."""
import sys, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import statsmodels.formula.api as smf, statsmodels.api as sm
sys.path.insert(0, '.'); from v2lib import *
T = pd.read_csv("layertab.csv")
out = {}
def wild(df, yv, xv, cl, B=4999, seed=0):
    """wild cluster bootstrap-t (Webb 6-point) for coefficient on xv in yv ~ xv + cell FE (via within-cell demeaning), restricted null."""
    rng = np.random.default_rng(seed)
    y = df[yv] - df.groupby("cell")[yv].transform("mean"); x = df[xv] - df.groupby("cell")[xv].transform("mean")
    y, x = y.values, x.values; g = pd.factorize(df[cl])[0]; G = g.max() + 1
    beta = (x @ y) / (x @ x); e = y - beta * x
    def se(x, e):
        s = np.bincount(g, weights=x * e, minlength=G); V = (s ** 2).sum() / (x @ x) ** 2
        return np.sqrt(V * G / (G - 1))
    t0 = beta / se(x, e)
    e0 = y  # restricted residuals under beta=0 (FE already partialled)
    W = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
    ts = np.empty(B); bs = np.empty(B)
    for i in range(B):
        v = W[rng.integers(6, size=G)][g]; ys = e0 * v
        bb = (x @ ys) / (x @ x); ts[i] = bb / se(x, ys - bb * x); bs[i] = bb
    p = (np.sum(np.abs(ts) >= abs(t0)) + 1) / (B + 1)
    # CI by inverting bootstrap-t with unrestricted residuals
    ts2 = np.empty(B)
    for i in range(B):
        v = W[rng.integers(6, size=G)][g]; ys = beta * x + e * v
        bb = (x @ ys) / (x @ x); ts2[i] = (bb - beta) / se(x, ys - bb * x)
    s0 = se(x, e); lo, hi = beta - np.quantile(ts2, .975) * s0, beta - np.quantile(ts2, .025) * s0
    return dict(beta=float(beta), se_cr1=float(s0), t=float(t0), p_wild=float(p), ci_wild=[float(lo), float(hi)], G=int(G))
for label, sub in {"all4": T[T.task.isin(TASKS)], "no_arith": T[T.task.isin(["pointer", "binding", "relational"])],
                   "binding+relational": T[T.task.isin(["binding", "relational"])],
                   **{t: T[T.task == t] for t in TASKS + ["fixed_digit"]},
                   "all4_x<0.9": T[T.task.isin(TASKS) & (T.x < .9)]}.items():
    r = {}
    for yv in ("b_pp", "z", "neff_b"):
        fe = smf.ols(f"{yv} ~ x + C(cell)", sub).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.model)[0]})
        fb = smf.ols(f"{yv} ~ x + C(cell)", sub).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.base_model)[0]})
        r[yv] = {"slope_on_depth": float(fe.params["x"]), "ci_cluster_model": [float(v) for v in fe.conf_int().loc["x"]], "p_cluster_model": float(fe.pvalues["x"]),
                 "ci_cluster_base": [float(v) for v in fb.conf_int().loc["x"]], "p_cluster_base": float(fb.pvalues["x"]),
                 "wild_model": wild(sub, yv, "x", "model"), "wild_base": wild(sub, yv, "x", "base_model")}
        # piecewise: back-half indicator
        sub2 = sub.assign(back=(sub.x >= .5).astype(float))
        pw = smf.ols(f"{yv} ~ back + C(cell)", sub2).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub2.model)[0]})
        r[yv]["back_minus_front"] = float(pw.params["back"]); r[yv]["back_minus_front_ci"] = [float(v) for v in pw.conf_int().loc["back"]]
        # predicted slope at depth .25 / .75 from a pooled (no FE) OLS, cluster by model
        po = smf.ols(f"{yv} ~ x", sub).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.model)[0]})
        r[yv]["pooled_intercept"] = float(po.params["Intercept"]); r[yv]["pooled_slope"] = float(po.params["x"])
        r[yv]["pred_at_.25"] = float(po.params["Intercept"] + .25 * po.params["x"]); r[yv]["pred_at_.75"] = float(po.params["Intercept"] + .75 * po.params["x"])
        # mixed model: random intercept + depth slope by model, variance component for cell
        try:
            mm = smf.mixedlm(f"{yv} ~ x + C(task) + C(w)", sub, groups=sub["model"], re_formula="~x",
                             vc_formula={"cell": "0 + C(cell)"}).fit(reml=True, method="lbfgs")
            r[yv]["mixed_x"] = float(mm.params["x"]); r[yv]["mixed_ci"] = [float(v) for v in mm.conf_int().loc["x"]]; r[yv]["mixed_p"] = float(mm.pvalues["x"])
            r[yv]["mixed_re_x_sd"] = float(np.sqrt(max(mm.cov_re.loc["x", "x"], 0)))
        except Exception as ex:
            r[yv]["mixed_err"] = str(ex)[:200]
    out[label] = r
    b = r["b_pp"]; z = r["z"]; n = r["neff_b"]
    print(f"{label:20s} b_pp~depth {b['slope_on_depth']:+.3f} CI(model) [{b['ci_cluster_model'][0]:+.3f},{b['ci_cluster_model'][1]:+.3f}] wild p {b['wild_model']['p_wild']:.4f} wildCI [{b['wild_model']['ci_wild'][0]:+.3f},{b['wild_model']['ci_wild'][1]:+.3f}] base-wild p {b['wild_base']['p_wild']:.4f} | mixed {b.get('mixed_x',np.nan):+.3f} [{b.get('mixed_ci',[np.nan,np.nan])[0]:+.3f},{b.get('mixed_ci',[np.nan,np.nan])[1]:+.3f}] | pred .25/.75 {b['pred_at_.25']:+.3f}/{b['pred_at_.75']:+.3f} | back-front {b['back_minus_front']:+.3f} [{b['back_minus_front_ci'][0]:+.3f},{b['back_minus_front_ci'][1]:+.3f}]")
    print(f"{'':20s} z~depth {z['slope_on_depth']:+.2f} CI [{z['ci_cluster_model'][0]:+.2f},{z['ci_cluster_model'][1]:+.2f}] wild p {z['wild_model']['p_wild']:.4f} base-wild p {z['wild_base']['p_wild']:.4f} mixed {z.get('mixed_x',np.nan):+.2f} [{z.get('mixed_ci',[np.nan,np.nan])[0]:+.2f},{z.get('mixed_ci',[np.nan,np.nan])[1]:+.2f}] | neff_b~depth {n['slope_on_depth']:+.3f} [{n['ci_cluster_model'][0]:+.3f},{n['ci_cluster_model'][1]:+.3f}] pred .25/.75 {n['pred_at_.25']:+.3f}/{n['pred_at_.75']:+.3f}")
json.dump(out, open("regress.json", "w"), indent=1)
