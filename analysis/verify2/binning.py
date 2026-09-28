import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, itertools, numpy as np
SP = str(paths.ANALYSIS)
sys.argv = [sys.argv[0], SP]; sys.path.insert(0, SP)
from model_diff import ORDER, T, LIN, decompose, ss
rng = np.random.default_rng(3)
C = {r["model"]: r for r in json.load(open(f"{SP}/consist.json")) if r["stat"] == "q10"}
def prof(z, nd, mode):
    z = np.asarray(z, float); L = len(z); x = np.arange(L) / (L - 1)
    if mode == "bin":
        d = np.minimum((x * nd).astype(int), nd - 1); return np.array([z[d == i].mean() for i in range(nd)])
    g = np.linspace(0, 1, nd); return np.interp(g, x, z)
def run(nd, mode):
    Z = np.array([[prof(C[m]["families"][t]["z"], nd, mode) for t in T] for m in ORDER])
    mu, a, b, e = decompose(Z); S = ss(Z); ns = S["model"] + S["task"] + S["resid"]
    null = np.mean([ss(np.stack([Z[rng.permutation(17), j] for j in range(4)], 1))["model"] for _ in range(1000)])
    sh = []
    for Sx in [(0, 1), (0, 2), (0, 3)]:
        Sc = [j for j in range(4) if j not in Sx]; A1 = Z[:, list(Sx)].mean(1); A1 -= A1.mean(0); A2 = Z[:, Sc].mean(1); A2 -= A2.mean(0)
        sh.append(np.corrcoef(A1.ravel(), A2.ravel())[0, 1])
    R = np.corrcoef(a)
    lin = np.mean([R[i, j] for i, j in itertools.combinations(range(17), 2) if LIN[ORDER[i]] == LIN[ORDER[j]]])
    oth = np.mean([R[i, j] for i, j in itertools.combinations(range(17), 2) if LIN[ORDER[i]] != LIN[ORDER[j]]])
    print(f"{mode:6s} nd={nd:2d}: shared {S['shared']/sum(S.values()):.2f} model {S['model']/ns:.2f} (null {null/ns:.2f}) task {S['task']/ns:.2f} split-half {np.mean(sh):.2f} {np.round(sh,2)} lineage {lin:.2f} other {oth:.2f}")
for nd in (5, 8, 10): run(nd, "bin")
for nd in (10, 20, 40): run(nd, "interp")
