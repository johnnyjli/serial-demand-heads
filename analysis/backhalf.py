"""Robustness of the cross-task depth regularity: per model x cell (single wording), per-layer slope of
top-q share on k (z vs a permutation null), then (a) back-half mean z < 0, (b) back < front,
(c) threshold-free: Spearman correlation of z with relative depth.  Variants: eager / sdpa, sink heads removed."""
import sys, os, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
from scipy.stats import spearmanr
from concurrent.futures import ProcessPoolExecutor
B = 1000
def topq(S, q=float(os.environ.get("VQ", ".25")), keep=None):
    """Top-q head share.  keep [L, H] (bool): heads to keep in each layer (a fixed set per cell);
    q is applied to the heads kept, so masking sink heads does not change the group's relative size."""
    if keep is None:
        P = S / S.sum(-1, keepdims=True); m = max(1, int(np.ceil(q * S.shape[-1])))
        return -np.sort(-P, -1)[..., :m].sum(-1)
    out = np.empty(S.shape[:2])
    for l in range(S.shape[1]):
        Sl = S[:, l, keep[l]]; P = Sl / Sl.sum(-1, keepdims=True); m = max(1, int(np.ceil(q * Sl.shape[-1])))
        out[:, l] = -np.sort(-P, -1)[:, :m].sum(-1)
    return out
def zprof(Y, k, body, fam, rng):
    kc = k - k.mean(); obs = kc @ (Y - Y.mean(0)) / (kc @ kc); nl = []
    groups = [np.nonzero(body == g)[0] for g in np.unique(body)] if fam in ("binding", "relational") else None
    for _ in range(B):
        if groups:
            kk = k.copy()
            for g in groups: kk[g] = k[g][rng.permutation(len(g))]
        else: kk = k[rng.permutation(len(k))]
        kc2 = kk - kk.mean(); nl.append(kc2 @ (Y - Y.mean(0)) / (kc2 @ kc2))
    return obs / np.std(nl, 0)
def one(args):
    m, c, attn, variant = args
    try: d = load(m, c, attn)
    except KeyError: return None
    S = d["contribution"].astype(float); k = d["k"].astype(float); L = d["L"]
    body = np.array([int(t["id"].split("/")[-1]) for t in d["items"]])
    keep = None
    if variant == "nosink" and "pos0_attn" in d:
        # a fixed set per cell: heads whose MEAN attention to position 0 exceeds 0.9 are dropped;
        # every layer keeps at least 4 heads
        a0 = d["pos0_attn"].mean(0); keep = a0 <= 0.9
        for l in range(keep.shape[0]):
            if keep[l].sum() < 4: keep[l, np.argsort(a0[l])[:4]] = True
    seed = abs(hash((m, c, attn, variant))) % (2 ** 32)          # per-cell seed
    z = zprof(topq(S, keep=keep), k, body, c.split("/")[0], np.random.default_rng(seed))
    x = np.arange(L) / (L - 1)
    return {"model": m, "cell": c, "attn": attn, "variant": variant, "front": float(z[x < .5].mean()), "back": float(z[x >= .5].mean()),
            "rho": float(spearmanr(x, z).correlation)}
if __name__ == "__main__":
    T = [c for c in CELLS if not c.startswith("fixed")]
    jobs = [(m, c, "eager", "all") for m in ORDER for c in T] + [(m, c, "eager", "nosink") for m in ORDER for c in T] + \
           [(m, c, "sdpa", "all") for m in ORDER for c in T]
    with ProcessPoolExecutor(12) as ex: R = [r for r in ex.map(one, jobs) if r]
    json.dump(R, open(sys.argv[2], "w")); print(len(R))
