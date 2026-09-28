import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
from concurrent.futures import ProcessPoolExecutor
def one(key):
    m, c = key; d = load(m, c); S = d["contribution"].astype(float); L, H = d["L"], d["H"]; w = window(L)
    k = d["k"].astype(float); kc = k - k.mean(); ss = kc @ kc
    P = S[:, w] / S[:, w].sum(-1, keepdims=True)                      # [n, Lw, H]
    base = P[k == k.min()].mean(0)                                     # [Lw, H]
    rank = np.argsort(np.argsort(-base, -1), -1)                       # baseline rank of each head (0 = leader)
    slope = np.einsum("n,nlh->lh", kc, P - P.mean(0)) / ss            # share slope per head [Lw, H]
    x = (rank + 0.5) / H
    edges = [0, .05, .15, .3, .5, .75, 1.0]
    prof = [float(100 * slope[(x >= a) & (x < b)].sum() / len(range(slope.shape[0]))) for a, b in zip(edges, edges[1:])]
    # high-activation heads: share > 2/H (twice the even share) and > 3/H
    out = {"model": m, "cell": c, "H": H, "ident_profile": prof}
    for thr in (2, 3):
        hi = P > thr / H
        nh = hi.sum(-1).mean(1); sh = (P * hi).sum(-1).mean(1)
        for name, y in ((f"n_high{thr}", nh), (f"share_high{thr}", sh)):
            s, p = perm_p(y, k, 2000)
            out[name] = {"mean": float(y.mean()), "slope": s, "p": p,
                         "by_k": [float(y[k == kk].mean()) for kk in sorted(set(k))]}
    # heads newly entering the high group: share <= 2/H at k=1 on average but > 2/H at k_max
    bk, bK = P[k == k.min()].mean(0), P[k == k.max()].mean(0)
    out["enter"] = int(((bk <= 2 / H) & (bK > 2 / H)).sum()); out["leave"] = int(((bk > 2 / H) & (bK <= 2 / H)).sum())
    out["high_k1"] = int((bk > 2 / H).sum()); out["n_window_layers"] = int(bk.shape[0])
    return out
if __name__ == "__main__":
    keys = [(m, c) for m in ORDER for c in CELLS]
    with ProcessPoolExecutor(10) as ex: R = list(ex.map(one, keys))
    json.dump(R, open(sys.argv[2], "w"))
