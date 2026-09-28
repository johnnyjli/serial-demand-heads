import json, numpy as np
C = [r for r in json.load(open("consist.json")) if r["stat"] == "q10"]
T = ["pointer", "arith", "binding", "relational"]; Zd = []
for r in C:
    L = r["L"]; x = np.arange(L) / (L - 1); dec = np.minimum((x * 10).astype(int), 9)
    Z = np.stack([r["families"][f]["z"] for f in T]).mean(0); Zd.append([Z[dec == d].mean() for d in range(10)])
Zd = np.array(Zd); print(np.round(Zd.mean(0), 2))
for seed in range(6):
    rng = np.random.default_rng(seed)
    b = np.array([Zd[rng.integers(0, 17, 17)].mean(0) for _ in range(4000)]); lo, hi = np.quantile(b, [.025, .975], 0)
    print(seed, "40-50", round(lo[4], 2), round(hi[4], 2), "90-100 hi", round(hi[9], 3))
from scipy import stats
m = Zd[:, 9]; print("t-CI 90-100", m.mean() + np.array([-1, 1]) * stats.t.ppf(.975, 16) * m.std(ddof=1) / 4.1231, "t-CI 40-50", Zd[:,4].mean() + np.array([-1, 1]) * stats.t.ppf(.975, 16) * Zd[:,4].std(ddof=1) / 17**.5)
