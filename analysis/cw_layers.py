import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
from concurrent.futures import ProcessPoolExecutor
def d_within_k(Y, k, ok, minc=8):
    """Cohen's d (correct - wrong) per column, pooled over depths with >= minc of each."""
    num, den, w = 0, 0, 0
    for kk in np.unique(k):
        a, b = Y[(k == kk) & ok], Y[(k == kk) & ~ok]
        if len(a) < minc or len(b) < minc: continue
        sd = np.sqrt((a.var(0, ddof=1) * (len(a) - 1) + b.var(0, ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))
        ww = 1 / (1 / len(a) + 1 / len(b))
        num = num + ww * (a.mean(0) - b.mean(0)) / np.maximum(sd, 1e-12); w += ww
    return None if w == 0 else num / w
def one(key):
    m, c = key; d = load(m, c); S = d["contribution"].astype(float); L = d["L"]; k = d["k"]; ok = d["correct"].astype(bool)
    P = S / S.sum(-1, keepdims=True)
    ent = np.exp(-(np.where(P > 0, P * np.log(P), 0)).sum(-1)) / S.shape[-1]
    out = {"model": m, "cell": c, "L": L}
    for n, Y in (("eff_heads", ent), ("top1", P.max(-1)), ("log_mass", np.log(S.sum(-1)))):
        r = d_within_k(Y, k, ok); out[n] = None if r is None else r.tolist()
    return out
if __name__ == "__main__":
    keys = [(m, c) for m in ORDER for c in CELLS]
    with ProcessPoolExecutor(2) as ex: R = list(ex.map(one, keys))
    json.dump(R, open(sys.argv[2], "w"))
