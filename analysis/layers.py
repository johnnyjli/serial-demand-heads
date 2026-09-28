import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
from concurrent.futures import ProcessPoolExecutor
NB = 10
def one(key):
    m, c = key; d = load(m, c); S = d["contribution"].astype(float); L = d["L"]; k = d["k"].astype(float)
    TS = top_share(S); PR = pr_h(S); M = S.sum(-1)
    kc = k - k.mean(); ss = kc @ kc
    def rel(Y):  # per-layer slope as % of layer mean
        return 100 * (kc @ (Y - Y.mean(0)) / ss) / Y.mean(0)
    x = np.arange(L) / (L - 1); b = np.minimum((x * NB).astype(int), NB - 1)
    out = {}
    for n, Y in (("ts", TS), ("pr", PR), ("mass", M)):
        r = rel(Y); out[n] = [float(r[b == i].mean()) for i in range(NB)]
        out[n + "_layer"] = r.tolist()
    # mass share per layer (fraction of all-layer mass) and its slope
    MS = M / M.sum(1, keepdims=True); out["mass_share_slope"] = (100 * (kc @ (MS - MS.mean(0)) / ss)).tolist()
    out["mass_share"] = MS.mean(0).tolist()
    if "write_norm" in d:
        W = d["write_norm"].astype(float); W = np.where(np.isfinite(W), W, S)
        r = rel(top_share(W)); out["ts_post"] = [float(r[b == i].mean()) for i in range(NB)]
    return {"model": m, "cell": c, "L": L, **out}
if __name__ == "__main__":
    keys = [(m, c) for m in ORDER for c in CELLS]
    with ProcessPoolExecutor(10) as ex: R = list(ex.map(one, keys))
    json.dump(R, open(sys.argv[2], "w"))
