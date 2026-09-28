import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
def one(key):
    m, c = key; d = load(m, c); S = d["contribution"].astype(float); L = d["L"]; k = d["k"]; w = window(L)
    med = np.median(S, -1); mx = S.max(-1); ratio = mx / np.maximum(med, 1e-9)
    ev = ratio > 50                                # item x layer events
    items = ev.any(1); items_w = ev[:, w].any(1)
    heads = Counter()
    for i, l in zip(*np.nonzero(ev)): heads[(int(l), int(S[i, l].argmax()))] += 1
    y = top_share(S)[:, w].mean(1)
    kk = k.astype(float)
    out = {"model": m, "cell": c, "n": len(k), "items_any": int(items.sum()), "items_window": int(items_w.sum()),
           "by_k": [int(items_w[k == x].sum()) for x in sorted(set(k))], "heads": [[list(h), n] for h, n in heads.most_common(3)],
           "slope_all": slope(y, kk), "mean": float(y.mean())}
    if items_w.any() and (~items_w).sum() > 20:
        out["slope_clean"], out["p_clean"] = perm_p(y[~items_w], kk[~items_w], 1000)
    out["p_all"] = perm_p(y, kk, 1000)[1]
    # robust: slope of per-item median over window layers
    ym = np.median(top_share(S)[:, w], 1); out["slope_med"], out["p_med"] = perm_p(ym, kk, 1000)
    return out
if __name__ == "__main__":
    keys = [(m, c) for m in ORDER for c in CELLS]
    with ProcessPoolExecutor(10) as ex: R = list(ex.map(one, keys))
    json.dump(R, open(sys.argv[2], "w"))
