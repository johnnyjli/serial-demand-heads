import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
from concurrent.futures import ProcessPoolExecutor
def ols_t(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; n, p = X.shape
    se = np.sqrt(np.diag(r @ r / (n - p) * np.linalg.inv(X.T @ X))); return b, b / se
def one(key):
    m, c = key; d = load(m, c); S = d["contribution"].astype(float); L = d["L"]; w = window(L)
    k = d["k"].astype(float)
    y = top_share(S)[:, w].mean(1)
    dl = d["digit_logits"].astype(float); mask = d["answer_mask"]
    z = np.where(mask, dl, -np.inf); p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    conf = p.max(1); ent = -(np.where(p > 0, p * np.log(p), 0)).sum(1) / np.log(mask.sum(1))
    # within-k correlation of y with confidence
    rs = [np.corrcoef(y[k == x], conf[k == x])[0, 1] for x in sorted(set(k)) if (k == x).sum() > 5 and conf[k == x].std() > 0]
    X0 = np.c_[np.ones_like(k), k]; b0, t0 = ols_t(X0, y)
    X1 = np.c_[np.ones_like(k), k, ent]; b1, t1 = ols_t(X1, y)
    # leader head identity: per window layer, the head with highest mean share at k=kmin
    SH = S[:, w] / S[:, w].sum(-1, keepdims=True)                   # [n, Lw, H]
    kmin, kmax = k.min(), k.max()
    lead = SH[k == kmin].mean(0).argmax(-1)                          # [Lw]
    lead_share = np.take_along_axis(SH, lead[None, :, None], -1)[..., 0].mean(1)   # per item
    b2, t2 = ols_t(X0, lead_share)
    # is the per-item top head the k=1 leader?
    same = (SH.argmax(-1) == lead[None]).mean(1)
    # entropy slope on k
    be, te = ols_t(X0, ent)
    return {"model": m, "cell": c, "slope": b0[1], "t": t0[1], "slope_ctrl_ent": b1[1], "t_ctrl_ent": t1[1], "coef_ent": b1[2], "t_ent": t1[2],
            "r_within_conf": float(np.nanmean(rs)), "lead_slope": b2[1], "lead_t": t2[1], "same_leader_k1": float(same[k == kmin].mean()),
            "same_leader_kmax": float(same[k == kmax].mean()), "ent_slope_t": te[1], "mean": float(y.mean())}
if __name__ == "__main__":
    keys = [(m, c) for m in ORDER for c in CELLS]
    with ProcessPoolExecutor(10) as ex: R = list(ex.map(one, keys))
    json.dump(R, open(sys.argv[2], "w"))
