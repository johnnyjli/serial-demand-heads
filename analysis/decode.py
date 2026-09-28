"""Can correctness be decoded from head activity at fixed k?  (and, as a positive control, can depth?)"""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
from concurrent.futures import ProcessPoolExecutor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
NPERM, MINC = 30, 8
def oof_auc(X, y, strat, groups, seed=0):
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, strat, groups):
        clf = make_pipeline(StandardScaler(), PCA(min(20, len(tr) - 2), random_state=0), LogisticRegression(C=0.5, max_iter=2000))
        clf.fit(X[tr], y[tr]); p[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)
def resid(X, k):
    X = X.copy()
    for kk in np.unique(k): X[k == kk] -= X[k == kk].mean(0)
    return X
def one(key):
    m, c = key; d = load(m, c); S = d["contribution"].astype(float); L, H = d["L"], d["H"]
    k = d["k"]; ok = d["correct"].astype(int)
    grp = np.array([int(t["id"].split("/")[-1]) for t in d["items"]])   # item body: shared across k in binding/relational
    x = np.arange(L) / (L - 1)
    regions = {"all": np.ones(L, bool), "early 0-40%": x < .4, "mid 40-80%": (x >= .4) & (x <= .8), "late 80-100%": x > .8}
    use = np.isin(k, [kk for kk in np.unique(k) if min(ok[k == kk].sum(), (1 - ok[k == kk]).sum()) >= MINC])
    out = {"model": m, "cell": c, "n_used": int(use.sum()), "depths": sorted(set(k[use].tolist()))}
    rng = np.random.default_rng(0)
    views = {}
    for rn, rm in regions.items():
        views[("full", rn)] = np.log(S[:, rm] + 1e-6).reshape(len(k), -1)
    views[("pattern", "all")] = (S / S.sum(-1, keepdims=True)).reshape(len(k), -1)
    views[("pattern", "mid 40-80%")] = (S / S.sum(-1, keepdims=True))[:, regions["mid 40-80%"]].reshape(len(k), -1)
    if use.sum() >= 40 and len(out["depths"]) >= 1:
        ku, yu, gu = k[use], ok[use], grp[use]; strat = ku * 2 + yu
        for (view, rn), X in views.items():
            Xr = resid(X[use], ku)
            auc = oof_auc(Xr, yu, strat, gu)
            null = []
            for _ in range(NPERM):
                yp = yu.copy()
                for kk in np.unique(ku): idx = np.nonzero(ku == kk)[0]; yp[idx] = rng.permutation(yp[idx])
                null.append(oof_auc(Xr, yp, ku * 2 + yp, gu))
            out[f"{view}|{rn}"] = {"auc": auc, "null_mean": float(np.mean(null)), "null_95": float(np.percentile(null, 95)),
                                   "p": float((np.sum(np.array(null) >= auc) + 1) / (NPERM + 1))}
    # positive control: depth (low vs high half) from the mid-window pattern, all items
    yk = (k > np.median(k)).astype(int)
    out["ctrl_depth|pattern mid"] = oof_auc((S / S.sum(-1, keepdims=True))[:, regions["mid 40-80%"]].reshape(len(k), -1), yk, yk, grp)
    out["ctrl_depth|full all"] = oof_auc(np.log(S + 1e-6).reshape(len(k), -1), yk, yk, grp)
    return out
if __name__ == "__main__":
    keys = [(m, c) for m in ORDER for c in CELLS]
    with ProcessPoolExecutor(14) as ex: R = list(ex.map(one, keys))
    json.dump(R, open(sys.argv[2], "w"))
