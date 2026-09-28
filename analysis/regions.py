import sys, json, numpy as np
sys.path.insert(0, sys.argv[1]); from e2lib import ORDER
T = ["pointer", "arith", "binding", "relational"]; NB = 5000
R = [r for r in json.load(open(sys.argv[2])) if r["stat"] == sys.argv[3]]
out = {"prevalence": [], "deciles": []}
for r in R:
    L = r["L"]; x = np.arange(L) / (L - 1); dec = np.minimum((x * 10).astype(int), 9)
    M = np.stack([np.sign(r["families"][f]["rel"]) * (np.array(r["families"][f]["p"]) < .05) for f in T])
    for i, f in enumerate(T):
        out["prevalence"].append({"model": r["model"], "family": f, "spread": float((M[i] < 0).mean()), "conc": float((M[i] > 0).mean()),
                                  "n_spread": int((M[i] < 0).sum()), "n_conc": int((M[i] > 0).sum())})
    # decile-level: fraction of spreading minus concentrating layers per decile, per family
    D = np.array([[((M[i][dec == d] < 0).mean() - (M[i][dec == d] > 0).mean()) for d in range(10)] for i in range(4)])
    # D = share spreading - share concentrating: D > 0 net spreading, D < 0 net concentrating
    agree_s = (D > 0).all(0); agree_c = (D < 0).all(0)
    rng = np.random.default_rng(0); ns, nc = [], []
    for _ in range(NB):
        Ms = np.stack([np.roll(M[i], rng.integers(L)) for i in range(4)])
        Ds = np.array([[((Ms[i][dec == d] < 0).mean() - (Ms[i][dec == d] > 0).mean()) for d in range(10)] for i in range(4)])
        ns.append((Ds > 0).all(0).sum()); nc.append((Ds < 0).all(0).sum())
    out["deciles"].append({"model": r["model"], "D": D.tolist(), "agree_spread": agree_s.astype(int).tolist(), "agree_conc": agree_c.astype(int).tolist(),
                           "n_agree_spread": int(agree_s.sum()), "n_agree_conc": int(agree_c.sum()),
                           "null_spread": float(np.mean(ns)), "null_conc": float(np.mean(nc)),
                           "p_spread": float((np.sum(np.array(ns) >= agree_s.sum()) + 1) / (NB + 1)),
                           "p_conc": float((np.sum(np.array(nc) >= agree_c.sum()) + 1) / (NB + 1))})
json.dump(out, open(sys.argv[4], "w"))
