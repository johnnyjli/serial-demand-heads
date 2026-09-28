"""Task 1: per-layer Cohen's d of the top-10% share, correct minus incorrect at the same k, for every
model x task x wording cell; pooled over depths with >= 8 correct and >= 8 incorrect items (weights
1/(1/n_c + 1/n_w), as in cw_layers.py).  Items: instances flagged by the OLMo-2 exclusion rule in either wording
are dropped; Qwen2.5-7B-Instruct from SDPA.

Null: correctness shuffled within depth (B = 5000).  The shuffle permutes instances within depth and is shared by
the two wordings of a model x task (they have the same instances), so wording-averaged and family-level
summaries have a valid null; draws are independent across models and tasks.
Also: the raw difference (percentage points of share), the k effect per step in the same units, and d
restricted to depths whose accuracy is clearly above chance vs near chance.  Writes cw.json."""
import json, zlib, numpy as np, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import binomtest
from clib import *
B, MINC = 5000, 8
def depth_stats(Y, C):
    """Y [n, L] items at one depth; C [B, n] boolean labels (correct).  Returns d [B, L], diff [B, L], weight."""
    Yc = Y - Y.mean(0); na = C[0].sum(); nb = C.shape[1] - na
    Sa = C.astype(float) @ Yc; Qa = C.astype(float) @ (Yc ** 2); S = Yc.sum(0); Q = (Yc ** 2).sum(0)
    ma = Sa / na; mb = (S - Sa) / nb
    ssa = Qa - na * ma ** 2; ssb = (Q - Qa) - nb * mb ** 2
    sd = np.sqrt(np.maximum(ssa + ssb, 0) / (na + nb - 2))
    return (ma - mb) / np.maximum(sd, 1e-12), ma - mb, 1 / (1 / na + 1 / nb)
def one(key):
    m, f = key; D = {w: load_cell(m, f, w) for w in ("w0", "w1")}
    k = D["w0"]["k"]; assert (k == D["w1"]["k"]).all(); L = D["w0"]["L"]; n = len(k)
    rng = np.random.default_rng(zlib.crc32(f"{m}|{f}".encode()))
    # shared within-depth permutations of instances: perm[b] maps position -> instance
    perm = np.tile(np.arange(n), (B + 1, 1))
    for kk in np.unique(k):
        ii = np.nonzero(k == kk)[0]
        for b in range(1, B + 1): perm[b, ii] = ii[rng.permutation(len(ii))]
    out = []
    for w in ("w0", "w1"):
        d = D[w]; Y, ok = d["y10"], d["correct"]; chance = 1 / np.median(d["nans"])
        acc_k = {int(kk): float(ok[k == kk].mean()) for kk in np.unique(k)}
        depths = [kk for kk in np.unique(k) if ok[k == kk].sum() >= MINC and (~ok[k == kk]).sum() >= MINC]
        kc = k - k.mean(); slope = kc @ (Y - Y.mean(0)) / (kc @ kc)
        wsd = np.sqrt(sum(((Y[k == kk] - Y[k == kk].mean(0)) ** 2).sum(0) for kk in np.unique(k)) / (n - len(np.unique(k))))
        r = {"model": m, "fam": f, "w": w, "attn": d["attn"], "L": L, "n": int(n), "n_drop": d["n_drop"], "chance": float(chance),
             "acc": float(ok.mean()), "acc_by_k": acc_k, "depths": [int(x) for x in depths],
             "n_correct_used": int(sum(ok[k == kk].sum() for kk in depths)), "n_wrong_used": int(sum((~ok[k == kk]).sum() for kk in depths)),
             "k_slope_pp": (100 * slope).tolist(), "k_d_per_step": (slope / wsd).tolist(), "within_sd_pp": (100 * wsd).tolist()}
        if not depths: out.append(r); continue
        above = [kk for kk in depths if acc_k[int(kk)] - chance >= .15 and
                 binomtest(int(ok[k == kk].sum()), int((k == kk).sum()), chance, "greater").pvalue < .01]
        num = {}; W = {}
        for kk in depths:
            ii = np.nonzero(k == kk)[0]; C = ok[perm[:, ii]]           # labels follow the permuted instances
            dd, df, wt = depth_stats(Y[ii], C); num[kk] = (dd, df); W[kk] = wt
        def pool(ds):
            wt = np.array([W[x] for x in ds]); return (sum(W[x] * num[x][0] for x in ds) / wt.sum(), sum(W[x] * num[x][1] for x in ds) / wt.sum())
        dall, diff = pool(depths)                                     # [B+1, L]; row 0 = observed
        obs, null = dall[0], dall[1:]
        A = np.abs(null); p = (np.sum(A >= np.abs(obs) - 1e-12, 0) + 1) / (B + 1)
        srt = np.sort(A, 0)
        pnull = (B - np.stack([np.searchsorted(srt[:, l], A[:, l], side="left") for l in range(L)], 1) + 1) / (B + 1)
        r.update({"depths_above_chance": [int(x) for x in above], "depths_near_chance": [int(x) for x in depths if x not in above],
                  "d": obs.tolist(), "diff_pp": (100 * diff[0]).tolist(), "p": p.tolist(), "null_sd": null.std(0).tolist(),
                  "null_d": null[:2000].astype(np.float32), "null_sig": (pnull < .05)[:2000]})
        for lab, ds in (("above", above), ("near", [x for x in depths if x not in above])):
            if ds: r[f"d_{lab}"] = pool(ds)[0][0].tolist()
        out.append(r)
    return out
if __name__ == "__main__":
    keys = [(m, f) for m in ORDER for f in FAMS]
    with ProcessPoolExecutor(10) as ex: R = [r for rr in ex.map(one, keys) for r in rr]
    np.savez_compressed("cw_null.npz", **{f"{r['model']}|{r['fam']}|{r['w']}|d": r.pop("null_d") for r in R if "null_d" in r},
                        **{f"{r['model']}|{r['fam']}|{r['w']}|sig": r.pop("null_sig") for r in R if "null_sig" in r})
    json.dump(R, open("cw.json", "w")); print(len(R), sum("d" in r for r in R))
