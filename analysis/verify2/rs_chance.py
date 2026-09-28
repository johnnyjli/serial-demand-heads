import sys, json, numpy as np, pandas as pd
SP = sys.argv[1]; sys.path.insert(0, SP)
import model_diff as md
from model_diff import ORDER, T, decompose, ss, perm_model, from_consist, from_cells
md.rng = np.random.default_rng(1)
def chance(Z, NB=1000):
    s = ss(Z); ns = s["model"] + s["task"] + s["resid"]
    o, nm, p = perm_model(Z, NB)
    return o / ns, nm / ns, p
rows = {}
for st in ["q10", "q25", "q50", "entropy", "top1"]:
    rows[st] = chance(from_consist(st))
rows["sink"] = chance(from_cells("eager", "sink_head")[0])
rows["perword"] = chance(from_cells("eager", "all")[0])
ms = [m for m in ORDER if from_cells("sdpa", "all", [m])[0] is not None]
print("sdpa models", len(ms), [m for m in ORDER if m not in ms])
rows["sdpa"] = chance(from_cells("sdpa", "all", ms)[0])
A = pd.read_csv(f"{SP}/vet3/altexp_layers.csv"); A["dec"] = np.minimum((A.x * 10).astype(int), 9)
for fit in ["k+ent", "k+ent+corr"]:
    S = A[A.fit == fit].groupby(["model", "task", "w", "dec"]).t.mean().groupby(["model", "task", "dec"]).mean()
    rows[fit] = chance(np.array([[[S[(m, t, d)] for d in range(10)] for t in T] for m in ORDER]))
for k, v in rows.items(): print(f"{k:12s} model share {v[0]:.3f} chance {v[1]:.3f} p {v[2]:.4f}")
