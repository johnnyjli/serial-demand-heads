import sys, numpy as np
sys.argv=[sys.argv[0]]
import itemsplit as I
T3 = ["pointer", "binding", "relational"]
data = {(m, t): I.fam_data(m, t) for m in I.ORDER for t in T3}
out=[]
for s in range(3):
    rs = np.random.default_rng(300 + s); Z1 = np.zeros((17, 3, 10)); Z2 = np.zeros((17, 3, 10))
    for j, t in enumerate(T3):
        bodies = np.unique(data[(I.ORDER[0], t)][2]); perm = rs.permutation(bodies); h1 = set(perm[: len(bodies) // 2].tolist())
        for i, m in enumerate(I.ORDER):
            Y, k, body = data[(m, t)]; sel = np.array([b in h1 for b in body])
            Z1[i, j] = I.dec(I.zprof(Y[sel], k[sel], body[sel], t, rs)); Z2[i, j] = I.dec(I.zprof(Y[~sel], k[~sel], body[~sel], t, rs))
    _, a1, _, e1 = I.decompose(Z1); _, a2, _, e2 = I.decompose(Z2)
    # cross-task consistency within the full data for the same 3 tasks
    out.append((np.corrcoef(a1.ravel(), a2.ravel())[0,1], np.corrcoef(e1.ravel(), e2.ravel())[0,1]))
print('3 tasks (no arith): r_a, r_e per split', np.round(out,3), 'mean', np.round(np.mean(out,0),3))
