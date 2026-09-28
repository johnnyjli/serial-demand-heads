import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from e2lib import *
from concurrent.futures import ProcessPoolExecutor
def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    n, p = X.shape; s2 = r @ r / (n - p); cov = s2 * np.linalg.inv(X.T @ X)
    return b, np.sqrt(np.diag(cov))
def one(key):
    m, c = key
    d = load(m, c); S = d["contribution"].astype(float); L = d["L"]
    y = top_share(S)[:, window(L)].mean(1); k = d["k"].astype(float); ok = d["correct"].astype(bool)
    gp = d["gold_prob"].astype(float)
    ks = sorted(set(k)); k0 = ks[0]
    r = {"model": m, "cell": c, "mean": float(y.mean())}
    r["slope_all"], r["p_all"] = perm_p(y, k, 1000)
    # correctness-controlled: y ~ 1 + k + correct
    X = np.c_[np.ones_like(k), k, ok.astype(float)]; b, se = ols(X, y)
    r["slope_ctrl"], r["t_ctrl"], r["corr_coef"], r["t_corr"] = b[1], b[1]/se[1], b[2], b[2]/se[2]
    # within-k correct-minus-wrong (stratified mean over depths with >=5 of each)
    diffs, w = [], []
    for kk in ks:
        a, bb = y[(k==kk)&ok], y[(k==kk)&~ok]
        if len(a) >= 5 and len(bb) >= 5: diffs.append(a.mean()-bb.mean()); w.append(min(len(a), len(bb)))
    r["cw_diff"] = float(np.average(diffs, weights=w)) if diffs else None
    # wrong items only
    sel = ~ok
    r["slope_wrong"], r["p_wrong"] = perm_p(y[sel], k[sel], 1000) if len(set(k[sel])) > 2 else (None, None)
    # k >= 2 only, and k1->k2 step
    sel = k >= ks[1]
    r["slope_k2plus"], r["p_k2plus"] = perm_p(y[sel], k[sel], 1000)
    r["step12"] = float(y[k==ks[1]].mean() - y[k==k0].mean())
    r["acc"] = [float(ok[k==kk].mean()) for kk in ks]
    r["y_by_k"] = [float(y[k==kk].mean()) for kk in ks]
    return r
if __name__ == "__main__":
    keys = [(m, c) for m in ORDER for c in CELLS]
    with ProcessPoolExecutor(10) as ex: R = list(ex.map(one, keys))
    json.dump(R, open(sys.argv[2], "w"))
