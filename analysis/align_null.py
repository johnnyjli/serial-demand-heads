"""Are the SAME layers consistent across tasks beyond chance alignment?  Null: shift each family's
per-layer pattern by an independent random circular offset (keeps each family's number and
clustering of significant layers, destroys depth alignment); also an independent layer permutation."""
import sys, json, numpy as np
sys.path.insert(0, sys.argv[1]); from e2lib import ORDER
T = ["pointer", "arith", "binding", "relational"]; NB = 5000
R = json.load(open(sys.argv[2])); out = []
def consistent(M):
    pos, neg = (M > 0).sum(0), (M < 0).sum(0)
    return ((pos >= 3) & (neg == 0)).astype(int) - ((neg >= 3) & (pos == 0)).astype(int)
for r in R:
    L = r["L"]; rng = np.random.default_rng(0)
    M = np.stack([np.sign(r["families"][f]["rel"]) * (np.array(r["families"][f]["p"]) < .05) for f in T])
    Z = np.stack([np.array(r["families"][f]["z"]) for f in T]); iu = np.triu_indices(4, 1)
    c = consistent(M); ns, nc = int((c < 0).sum()), int((c > 0).sum()); rz = np.corrcoef(Z)[iu].mean()
    res = {"model": r["model"], "stat": r["stat"], "L": L, "n_spread": ns, "n_conc": nc, "profile_r": float(rz),
           "sig_frac_spread": float((M < 0).mean()), "sig_frac_conc": float((M > 0).mean())}
    for name, draw in (("shift", lambda: np.stack([np.roll(M[i], rng.integers(L)) for i in range(4)])),
                       ("perm", lambda: np.stack([M[i][rng.permutation(L)] for i in range(4)]))):
        cnt = np.array([[(cc := consistent(draw()))[...].__lt__(0).sum(), (cc > 0).sum()] for _ in range(NB)])
        res[f"{name}_spread_mean"] = float(cnt[:, 0].mean()); res[f"{name}_conc_mean"] = float(cnt[:, 1].mean())
        res[f"{name}_p_spread"] = float((np.sum(cnt[:, 0] >= ns) + 1) / (NB + 1)); res[f"{name}_p_conc"] = float((np.sum(cnt[:, 1] >= nc) + 1) / (NB + 1))
    rn = []
    for _ in range(2000):
        Zs = np.stack([np.roll(Z[i], rng.integers(L)) for i in range(4)]); rn.append(np.corrcoef(Zs)[iu].mean())
    res["profile_null_mean"] = float(np.mean(rn)); res["profile_p"] = float((np.sum(np.array(rn) >= rz) + 1) / (len(rn) + 1))
    out.append(res)
json.dump(out, open(sys.argv[3], "w"))
