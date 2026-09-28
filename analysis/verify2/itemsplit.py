"""Item-split reliability of q10 depth profiles and signatures (reviewer noise-ceiling check).
Splits instance/body ids into two disjoint halves, recomputes per-layer permutation z (as consist.py) on each half,
builds decile profiles Z1, Z2 (17x4x10), and reports reliability of each decomposition component and the
cross-half lineage similarity (base on one half, post-trained on the other)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, numpy as np, itertools
SP = str(paths.ANALYSIS)
sys.path.insert(0, SP)
from e2lib import load, ORDER
T = ["pointer", "arith", "binding", "relational"]
ATTN = {"Qwen2.5-7B-Instruct": "sdpa"}
LIN = {"Llama-3.2-3B-Instruct": "Llama-3.2-3B", "Llama-3.1-8B-Instruct": "Llama-3.1-8B", "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
       "OLMo-2-1124-7B-SFT": "OLMo-2-1124-7B", "OLMo-2-1124-7B-DPO": "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct": "OLMo-2-1124-7B"}
LIN = {m: LIN.get(m, m) for m in ORDER}
B = 400
def q10(S):
    P = S / S.sum(-1, keepdims=True); H = S.shape[-1]; m = max(1, int(np.ceil(.1 * H)))
    return -np.sort(-P, -1)[..., :m].sum(-1)
def fam_data(model, fam):
    attn = ATTN.get(model, "eager")
    a, b = load(model, fam + "/w0", attn), load(model, fam + "/w1", attn)
    ka = {tuple(t["id"].split("/")[-2:]): j for j, t in enumerate(a["items"])}
    kb = {tuple(t["id"].split("/")[-2:]): j for j, t in enumerate(b["items"])}
    keys = sorted(set(ka) & set(kb), key=lambda x: (int(x[0]), int(x[1])))
    ia, ib = np.array([ka[x] for x in keys]), np.array([kb[x] for x in keys])
    Y = 0.5 * (q10(a["contribution"].astype(float)[ia]) + q10(b["contribution"].astype(float)[ib]))
    k = np.array([int(x[0]) for x in keys], float); body = np.array([int(x[1]) for x in keys])
    return Y, k, body
def zprof(Y, k, body, fam, rng):
    n = len(k); P = np.empty((B, n))
    if fam in ("binding", "relational"):
        groups = [np.nonzero(body == g)[0] for g in np.unique(body)]
        for i in range(B):
            kk = k.copy()
            for g in groups: kk[g] = k[g][rng.permutation(len(g))]
            P[i] = kk
    else:
        for i in range(B): P[i] = k[rng.permutation(n)]
    def sl(K):
        Kc = K - K.mean(-1, keepdims=True); Yc = Y - Y.mean(0); return (Kc @ Yc) / (Kc ** 2).sum(-1, keepdims=True)
    obs = sl(k[None])[0]; null = sl(P)
    return obs / null.std(0)
def dec(z):
    L = len(z); x = np.arange(L) / (L - 1); d = np.minimum((x * 10).astype(int), 9)
    return np.array([z[d == i].mean() for i in range(10)])
def decompose(Z):
    mu = Z.mean((0, 1)); a = Z.mean(1) - mu; b = Z.mean(0) - mu; e = Z - mu - a[:, None] - b[None]; return mu, a, b, e
if __name__ == "__main__":
    NSPLIT = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    data = {(m, t): fam_data(m, t) for m in ORDER for t in T}
    # sanity: full-data profile vs consist.json
    C = {r["model"]: r for r in json.load(open(f"{SP}/consist.json")) if r["stat"] == "q10"}
    rng = np.random.default_rng(1)
    chk = [np.corrcoef(zprof(*data[(m, t)], t, rng), C[m]["families"][t]["z"])[0, 1] for m in ORDER[:3] for t in T]
    print("sanity r(full z, consist z):", np.round(chk, 3))
    res = []
    for s in range(NSPLIT):
        rs = np.random.default_rng(100 + s); Z1 = np.zeros((17, 4, 10)); Z2 = np.zeros((17, 4, 10))
        for j, t in enumerate(T):
            bodies = np.unique(data[(ORDER[0], t)][2]); perm = rs.permutation(bodies); h1 = set(perm[: len(bodies) // 2].tolist())
            for i, m in enumerate(ORDER):
                Y, k, body = data[(m, t)]; sel = np.array([b in h1 for b in body])
                Z1[i, j] = dec(zprof(Y[sel], k[sel], body[sel], t, rs)); Z2[i, j] = dec(zprof(Y[~sel], k[~sel], body[~sel], t, rs))
        mu1, a1, b1, e1 = decompose(Z1); mu2, a2, b2, e2 = decompose(Z2)
        r_cell = np.mean([np.corrcoef(Z1[i, j], Z2[i, j])[0, 1] for i in range(17) for j in range(4)])
        r_a = np.corrcoef(a1.ravel(), a2.ravel())[0, 1]; r_e = np.corrcoef(e1.ravel(), e2.ravel())[0, 1]
        r_b = np.corrcoef(b1.ravel(), b2.ravel())[0, 1]; r_mu = np.corrcoef(mu1, mu2)[0, 1]
        r_a_pm = [np.corrcoef(a1[i], a2[i])[0, 1] for i in range(17)]
        # lineage: same-half vs cross-half similarity of signatures
        same, cross, oth_same, oth_cross, self_cross = [], [], [], [], []
        for i, jj in itertools.combinations(range(17), 2):
            s_ = 0.5 * (np.corrcoef(a1[i], a1[jj])[0, 1] + np.corrcoef(a2[i], a2[jj])[0, 1])
            c_ = 0.5 * (np.corrcoef(a1[i], a2[jj])[0, 1] + np.corrcoef(a2[i], a1[jj])[0, 1])
            if LIN[ORDER[i]] == LIN[ORDER[jj]]: same.append(s_); cross.append(c_)
            else: oth_same.append(s_); oth_cross.append(c_)
        # residual similarity within lineage (model x task interaction), same half
        e_lin = [np.corrcoef(e1[i].ravel(), e1[jj].ravel())[0, 1] for i, jj in itertools.combinations(range(17), 2) if LIN[ORDER[i]] == LIN[ORDER[jj]]]
        e_oth = [np.corrcoef(e1[i].ravel(), e1[jj].ravel())[0, 1] for i, jj in itertools.combinations(range(17), 2) if LIN[ORDER[i]] != LIN[ORDER[jj]]]
        res.append(dict(r_cell=r_cell, r_a=r_a, r_e=r_e, r_b=r_b, r_mu=r_mu, r_a_pm=r_a_pm, lin_same=np.mean(same), lin_cross=np.mean(cross),
                        oth_same=np.mean(oth_same), oth_cross=np.mean(oth_cross), self_cross=np.mean(r_a_pm), e_lin=np.mean(e_lin), e_oth=np.mean(e_oth)))
        print(s, {k: (round(v, 3) if not isinstance(v, list) else None) for k, v in res[-1].items()})
    out = {k: float(np.mean([r[k] for r in res])) for k in res[0] if k != "r_a_pm"}
    out["r_a_per_model"] = dict(zip(ORDER, np.mean([r["r_a_pm"] for r in res], 0).tolist()))
    sb = lambda r: 2 * r / (1 + r)
    out["spearman_brown"] = {k: sb(out[k]) for k in ("r_cell", "r_a", "r_e", "r_b")}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{SP}/verify2/itemsplit.json", "w"), indent=1)
