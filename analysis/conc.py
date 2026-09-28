"""Concentration beyond the top head: Lorenz-curve change and a family of metrics."""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
from concurrent.futures import ProcessPoolExecutor
NPERM = 2000
# direction: +1 = larger value means more concentrated, -1 = larger means more spread, 0 = not directional
METRICS = {"top1": 1, "top2": 1, "top4": 1, "top8": 1, "top10pct": 1, "top25pct": 1, "gini": 1,
           "share_high": 1, "eff_entropy": -1, "pr": -1, "n50": -1, "n80": -1, "n_high": 0}
def per_item(S, w):
    P = S[:, w] / S[:, w].sum(-1, keepdims=True)                  # [n, Lw, H] shares
    n, Lw, H = P.shape
    Ps = -np.sort(-P, -1)                                          # descending
    C = np.cumsum(Ps, -1)                                          # top-r share, r = 1..H
    asc = Ps[..., ::-1]; i = np.arange(1, H + 1)
    ent = -(np.where(P > 0, P * np.log(P), 0)).sum(-1)
    hi = P > 2.0 / H
    m = {"top1": C[..., 0], "top2": C[..., min(1, H - 1)], "top4": C[..., min(3, H - 1)], "top8": C[..., min(7, H - 1)],
         "top10pct": C[..., max(1, int(np.ceil(0.10 * H))) - 1], "top25pct": C[..., max(1, int(np.ceil(0.25 * H))) - 1],
         "gini": (asc * (2 * i - H - 1)).sum(-1) / H, "eff_entropy": np.exp(ent) / H, "pr": 1 / (P ** 2).sum(-1) / H,
         "n50": ((C < 0.5).sum(-1) + 1) / H, "n80": ((C < 0.8).sum(-1) + 1) / H,
         "n_high": hi.sum(-1) / H, "share_high": (P * hi).sum(-1)}
    return {k: v.mean(1) for k, v in m.items()}, C.mean(1), Ps.mean(1)   # window-averaged per item
def joint(Y, k, rng):
    """OLS slopes of each column of Y on k and permutation p (one shuffle for all columns)."""
    kc = k - k.mean(); ss = kc @ kc; Yc = Y - Y.mean(0)
    obs = kc @ Yc / ss
    nl = np.array([rng.permutation(kc) @ Yc / ss for _ in range(NPERM)])
    p = (np.sum(np.abs(nl) >= np.abs(obs) - 1e-15, 0) + 1) / (NPERM + 1)
    return obs, p
def bh(p, q=0.05):
    p = np.asarray(p); n = len(p); o = np.argsort(p); th = q * np.arange(1, n + 1) / n
    ok = p[o] <= th; kmax = np.nonzero(ok)[0].max() + 1 if ok.any() else 0
    out = np.zeros(n, bool); out[o[:kmax]] = True; return out
def one(key):
    m, c = key; d = load(m, c); S = d["contribution"].astype(float); L, H = d["L"], d["H"]; w = window(L)
    k = d["k"].astype(float); rng = np.random.default_rng(0)
    M, C, Ps = per_item(S, w)
    names = list(METRICS)
    Y = np.stack([M[n] for n in names], 1)
    s, p = joint(Y, k, rng)
    out = {"model": m, "cell": c, "H": H, "n_dropped": d.get("n_dropped", 0)}
    for j, n in enumerate(names):
        mu = Y[:, j].mean()
        out[n] = {"mean": float(mu), "slope": float(s[j]), "rel": float(100 * s[j] / mu), "p": float(p[j])}
    # Lorenz: slope of top-r share for r = 1..H-1 (r = H is always 1)
    sc, pc = joint(C[:, :-1], k, rng)
    out["lorenz_slope"] = (100 * sc).tolist()        # percentage points of layer activity per step
    out["lorenz_p"] = pc.tolist()
    sig = bh(pc)
    pos, neg = sig & (sc > 0), sig & (sc < 0)
    if pos.any() and neg.any():
        first = np.nonzero(sig)[0][0]
        out["lorenz_verdict"] = "leader→group" if sc[first] < 0 else "leader↑, rest evens"
    elif pos.any(): out["lorenz_verdict"] = "concentrate"
    elif neg.any(): out["lorenz_verdict"] = "spread"
    else: out["lorenz_verdict"] = "flat"
    # rank-resolved share change (not cumulative)
    sr, pr_ = joint(Ps, k, rng); out["rank_slope"] = (100 * sr).tolist()
    # identity: the k=1 top-m heads per window layer, their combined share
    P = S[:, w] / S[:, w].sum(-1, keepdims=True)
    base = P[k == k.min()].mean(0)                                 # [Lw, H]
    order = np.argsort(-base, -1)
    ident = {}
    for mm in (1, 2, 4, 8):
        mm = min(mm, H)
        idx = order[:, :mm]
        y = np.take_along_axis(P, idx[None], -1).sum(-1).mean(1)
        s1, p1 = joint(y[:, None], k, rng); ident[f"k1top{mm}"] = {"rel": float(100 * s1[0] / y.mean()), "p": float(p1[0])}
    out["identity"] = ident
    return out
if __name__ == "__main__":
    keys = [(m, c) for m in ORDER for c in CELLS]
    with ProcessPoolExecutor(10) as ex: R = list(ex.map(one, keys))
    json.dump(R, open(sys.argv[2], "w"))
