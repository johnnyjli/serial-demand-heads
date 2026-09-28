import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
from concurrent.futures import ProcessPoolExecutor
PAIRS = [("Llama-3.2-3B", "Llama-3.2-3B-Instruct"), ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"),
         ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT"),
         ("OLMo-2-1124-7B", "OLMo-2-1124-7B-DPO"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct")]
def ystat(d, mode):
    S = d["contribution"].astype(float); L = d["L"]; w = window(L)
    if mode == "nosink" and "pos0_attn" in d:
        mask = d["pos0_attn"] > 0.9
        S = np.where(mask, np.nan, S)
        TS = top_share(S)          # masked heads are NaN, treated as absent
    else:
        TS = top_share(S)
    TS = TS[:, w]
    if mode == "median": return np.nanmedian(TS, 1)
    return np.nanmean(TS, 1)
def one(args):
    a, b, c = args
    A, B = load(a, c), load(b, c)
    ia, ib = [t["id"] for t in A["items"]], [t["id"] for t in B["items"]]
    common = sorted(set(ia) & set(ib), key=ia.index)
    def sub(X, ids):
        pos = {i: j for j, i in enumerate(ids)}; sel = np.array([pos[i] for i in common])
        n = len(ids)
        for key, v in list(X.items()):
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n: X[key] = v[sel]
        X["items"] = [X["items"][j] for j in sel]
    sub(A, ia); sub(B, ib)
    k = A["k"].astype(float); out = {"base": a, "post": b, "cell": c}
    for mode in ("mean", "median", "nosink"):
        ya, yb = ystat(A, mode), ystat(B, mode)
        # relative (to each model's mean) per-item values so levels don't matter
        ra, rb = ya / np.nanmean(ya), yb / np.nanmean(yb)
        diff = rb - ra
        for sub, sel in (("all", np.ones(len(k), bool)), ("k2plus", k >= sorted(set(k))[1]),
                         ("both_wrong", ~A["correct"] & ~B["correct"])):
            if len(set(k[sel])) < 2: continue
            s, p = perm_p(diff[sel], k[sel], 1000)
            out[f"{mode}_{sub}"] = (100 * s, p)
    return out
if __name__ == "__main__":
    jobs = [(a, b, c) for a, b in PAIRS for c in CELLS]
    with ProcessPoolExecutor(10) as ex: R = list(ex.map(one, jobs))
    json.dump(R, open(sys.argv[2], "w"))
