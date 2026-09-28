"""Monte Carlo stability of item-split reliabilities (20 random k-stratified splits instead of 4)."""
import json, numpy as np
from vn_lib import *
def sb(r): return 2 * r / (1 + r)
NS = 20; out = {}
for mode in ("direct", "stepwise"):
    mp, cellr, pert, dev, sig = [], [], {t: [] for t in T}, {t: [] for t in T}, []
    for s in range(NS):
        rng = np.random.default_rng(50_000 + s); Z1 = np.zeros((8, 4, 10)); Z2 = np.zeros((8, 4, 10))
        for i, m in enumerate(M8):
            for j, t in enumerate(T):
                Y, k, _, _ = cell(m, mode, t); sel = np.zeros(len(k), bool)
                for kk in np.unique(k):
                    ix = np.nonzero(k == kk)[0]; sel[rng.choice(ix, len(ix) // 2, replace=False)] = True
                Z1[i, j] = dec_prof(zprof(Y[sel], k[sel])[2]); Z2[i, j] = dec_prof(zprof(Y[~sel], k[~sel])[2])
        mp.append(np.mean([r_(Z1[i].mean(0), Z2[i].mean(0)) for i in range(8)]))
        cr = np.array([[r_(Z1[i, j], Z2[i, j]) for j in range(4)] for i in range(8)]); cellr.append(cr.mean())
        for j, t in enumerate(T):
            pert[t].append(cr[:, j].mean()); dev[t].append(r_(Z1[:, j] - Z1[:, j].mean(0), Z2[:, j] - Z2[:, j].mean(0)))
        sig.append(r_(Z1.mean(1) - Z1.mean((0, 1)), Z2.mean(1) - Z2.mean((0, 1))))
    out[mode] = {"model_profile_r": [float(np.mean(mp)), float(np.std(mp))], "cell_r": [float(np.mean(cellr)), float(np.std(cellr))],
                 "per_task_half": {t: [float(np.mean(v)), float(np.std(v))] for t, v in pert.items()},
                 "dev_rel_sb": {t: float(sb(np.mean(v))) for t, v in dev.items()}, "dev_rel_sb_sd_of_4split_mean": {t: float(np.std([sb(np.mean(np.random.default_rng(q).choice(v, 4, replace=False))) for q in range(200)])) for t, v in dev.items()},
                 "sig_half": [float(np.mean(sig)), float(np.std(sig))]}
print(json.dumps(out, indent=1, default=lambda x: round(x, 3))); json.dump(out, open("vn_split_stability.json", "w"), indent=1)
