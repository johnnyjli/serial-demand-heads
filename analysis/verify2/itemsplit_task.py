import sys, numpy as np
sys.argv=[sys.argv[0]]
from itemsplit import *
data = {(m, t): fam_data(m, t) for m in ORDER for t in T}
acc = {t: [] for t in T}; ra=[]; ae=[]
for s in range(3):
    rs = np.random.default_rng(200 + s); Z1 = np.zeros((17, 4, 10)); Z2 = np.zeros((17, 4, 10))
    for j, t in enumerate(T):
        bodies = np.unique(data[(ORDER[0], t)][2]); perm = rs.permutation(bodies); h1 = set(perm[: len(bodies) // 2].tolist())
        for i, m in enumerate(ORDER):
            Y, k, body = data[(m, t)]; sel = np.array([b in h1 for b in body])
            Z1[i, j] = dec(zprof(Y[sel], k[sel], body[sel], t, rs)); Z2[i, j] = dec(zprof(Y[~sel], k[~sel], body[~sel], t, rs))
        acc[t].append(np.mean([np.corrcoef(Z1[i, j], Z2[i, j])[0, 1] for i in range(17)]))
    # cross-task split-half using disjoint items: tasks (0,1) half1 vs tasks (2,3) half2 etc.
    a1=decompose(Z1)[1]; a2=decompose(Z2)[1]; ra.append(np.corrcoef(a1.ravel(),a2.ravel())[0,1])
print({t: round(float(np.mean(v)), 3) for t, v in acc.items()}, 'r_a', np.round(ra,3))
