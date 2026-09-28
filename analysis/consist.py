"""Cross-task consistency of layer-level spreading / concentrating.

Statistic (concentration index, + = concentrating): top-q head share = summed share of the
ceil(q*H) strongest heads in the layer (q = 0.25 primary; 0.10, 0.50 sensitivity), or the
negated effective number of heads (exp entropy / H).  Per item, averaged over the two wordings
(same instances).  Per model x task family: per-layer OLS slope on k, relative to the layer
mean; permutation null shuffles k across items (pointer, arithmetic) or within prompt body
(binding, relational), one shuffle for all layers.  A layer is consistent if it is
significant (p < .05) in the same direction in >= 3 of the 4 families and significant in the
opposite direction in none.  The null count of consistent layers combines independent
per-family null draws.
"""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
from concurrent.futures import ProcessPoolExecutor
TASKS = ["pointer", "arith", "binding", "relational"]
B = 2000
ATTN = {"Qwen2.5-7B-Instruct": "sdpa"}          # eager bf16 is numerically degraded for this model
def conc_index(S, stat):
    P = S / S.sum(-1, keepdims=True); H = S.shape[-1]
    if stat == "entropy":
        return -np.exp(-(np.where(P > 0, P * np.log(P), 0)).sum(-1)) / H
    if stat == "top1": return P.max(-1)
    q = {"q10": .10, "q25": .25, "q50": .50}[stat]; m = max(1, int(np.ceil(q * H)))
    return -np.sort(-P, -1)[..., :m].sum(-1)
def family_data(model, fam, stat, attn):
    a, b = load(model, fam + "/w0", attn), load(model, fam + "/w1", attn)
    ka = {tuple(t["id"].split("/")[-2:]): j for j, t in enumerate(a["items"])}
    kb = {tuple(t["id"].split("/")[-2:]): j for j, t in enumerate(b["items"])}
    keys = sorted(set(ka) & set(kb), key=lambda x: (int(x[0]), int(x[1])))
    ia, ib = np.array([ka[x] for x in keys]), np.array([kb[x] for x in keys])
    Y = 0.5 * (conc_index(a["contribution"].astype(float)[ia], stat) + conc_index(b["contribution"].astype(float)[ib], stat))
    k = np.array([int(x[0]) for x in keys], float); body = np.array([int(x[1]) for x in keys])
    return Y, k, body
def perms(k, body, fam, rng):
    n = len(k); P = np.empty((B, n))
    if fam in ("binding", "relational"):
        groups = [np.nonzero(body == g)[0] for g in np.unique(body)]
        for i in range(B):
            kk = k.copy()
            for g in groups: kk[g] = k[g][rng.permutation(len(g))]
            P[i] = kk
    else:
        for i in range(B): P[i] = k[rng.permutation(n)]
    return P
def slopes(Kmat, Y):
    Kc = Kmat - Kmat.mean(-1, keepdims=True); Yc = Y - Y.mean(0)
    return (Kc @ Yc) / (Kc ** 2).sum(-1, keepdims=True)
def one(args):
    model, stat = args; attn = ATTN.get(model, "eager")
    rng = np.random.default_rng(0); fams = {}
    for f in TASKS:
        Y, k, body = family_data(model, f, stat, attn)
        mu = np.abs(Y.mean(0))
        obs = slopes(k[None], Y)[0]; null = slopes(perms(k, body, f, rng), Y)
        p = (np.sum(np.abs(null) >= np.abs(obs) - 1e-15, 0) + 1) / (B + 1)
        # p-values of each null draw (for the null count of consistent layers)
        A = np.abs(null); srt = np.sort(A, 0)
        pnull = (B - np.array([np.searchsorted(srt[:, l], A[:, l], side="left") for l in range(A.shape[1])]).T + 1) / (B + 1)
        L = Y.shape[1]; w = window(L); yw = Y[:, w].mean(1)
        ow = slopes(k[None], yw[:, None])[0, 0]; nw = slopes(perms(k, body, f, rng), yw[:, None])[:, 0]
        fams[f] = {"rel": 100 * obs / mu, "p": p, "sign": np.sign(obs), "null_sign": np.sign(null), "pnull": pnull,
                   "z": obs / null.std(0), "win_rel": float(100 * ow / abs(yw.mean())),
                   "win_p": float((np.sum(np.abs(nw) >= abs(ow) - 1e-15) + 1) / (B + 1)), "n": len(k)}
    L = len(fams[TASKS[0]]["rel"])
    def consistent(sig_sign):                      # sig_sign: [F, L] in {-1, 0, +1}
        pos, neg = (sig_sign > 0).sum(0), (sig_sign < 0).sum(0)
        return ((pos >= 3) & (neg == 0)).astype(int) - ((neg >= 3) & (pos == 0)).astype(int)
    ss = np.stack([fams[f]["sign"] * (fams[f]["p"] < .05) for f in TASKS])
    cons = consistent(ss)
    null_cnt = np.array([[((c := consistent(np.stack([fams[f]["null_sign"][b] * (fams[f]["pnull"][b] < .05) for f in TASKS]))) > 0).sum(),
                          (c < 0).sum()] for b in range(B)])
    n_conc, n_spread = int((cons > 0).sum()), int((cons < 0).sum())
    # cross-task similarity of the per-layer profile: mean pairwise correlation of z-profiles
    Z = np.stack([fams[f]["z"] for f in TASKS]); iu = np.triu_indices(len(TASKS), 1)
    r_obs = np.corrcoef(Z)[iu].mean()
    # null for profile correlation: rebuild z from null draws (independent across families)
    rz = []
    for b in range(0, B, 4):
        Zn = np.stack([np.where(True, fams[f]["null_sign"][b], 0) * 0 for f in TASKS])  # placeholder, replaced below
    return {"model": model, "stat": stat, "attn": attn, "L": L,
            "families": {f: {"rel": fams[f]["rel"].tolist(), "p": fams[f]["p"].tolist(), "z": fams[f]["z"].tolist(),
                             "win_rel": fams[f]["win_rel"], "win_p": fams[f]["win_p"], "n": fams[f]["n"],
                             "n_sig_spread": int(((fams[f]["sign"] < 0) & (fams[f]["p"] < .05)).sum()),
                             "n_sig_conc": int(((fams[f]["sign"] > 0) & (fams[f]["p"] < .05)).sum())} for f in TASKS},
            "consistent": cons.tolist(), "n_conc": n_conc, "n_spread": n_spread,
            "null_conc_mean": float(null_cnt[:, 0].mean()), "null_spread_mean": float(null_cnt[:, 1].mean()),
            "p_conc": float((np.sum(null_cnt[:, 0] >= n_conc) + 1) / (B + 1)),
            "p_spread": float((np.sum(null_cnt[:, 1] >= n_spread) + 1) / (B + 1)),
            "profile_r": float(r_obs)}
if __name__ == "__main__":
    stats = sys.argv[3].split(",") if len(sys.argv) > 3 else ["q25"]
    jobs = [(m, s) for s in stats for m in ORDER]
    with ProcessPoolExecutor(12) as ex: R = list(ex.map(one, jobs))
    json.dump(R, open(sys.argv[2], "w")); print(len(R))
