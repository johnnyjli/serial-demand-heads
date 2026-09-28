import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
from concurrent.futures import ProcessPoolExecutor
def rel(y, k, n=1000):
    s, p = perm_p(y, k, n); return 100 * s / np.nanmean(y), p
def one(key):
    attn, m, c = key
    try: d = load(m, c, attn)
    except KeyError: return None
    S = d["contribution"].astype(float); L = d["L"]; w = window(L); k = d["k"].astype(float)
    TS, PR = top_share(S), pr_h(S)
    out = {"attn": attn, "model": m, "cell": c}
    out["ts_rel"], out["ts_p"] = rel(TS[:, w].mean(1), k)
    if attn == "eager":
        out["pr_rel"], out["pr_p"] = rel(PR[:, w].mean(1), k)
        if "pos0_attn" in d:
            Sm = np.where(d["pos0_attn"] > 0.9, np.nan, S)
            y = top_share(Sm)[:, w].mean(1)       # masked (sink) heads are NaN, treated as absent
            out["sink_rel"], out["sink_p"] = rel(y, k); out["sink_frac"] = float((d["pos0_attn"][:, w] > 0.9).mean())
        k1 = k == k.min()
        out["ts_layer_k1"] = TS[k1].mean(0).tolist()
        M = S.sum(-1); out["mass_share_k1"] = (M[k1] / M[k1].sum(1, keepdims=True)).mean(0).tolist()
        out["H"] = d["H"]
        if "write_norm" in d:
            W = d["write_norm"].astype(float); W = np.where(np.isfinite(W), W, S)
            out["post_rel"], out["post_p"] = rel(top_share(W)[:, w].mean(1), k)
    return out
if __name__ == "__main__":
    keys = [(a, m, c) for a in ("eager", "sdpa") for m in ORDER for c in CELLS]
    with ProcessPoolExecutor(10) as ex: R = [r for r in ex.map(one, keys) if r]
    json.dump(R, open(sys.argv[2], "w")); print(len(R))
