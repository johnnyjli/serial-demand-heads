"""Reference for the k>=2 comparison in subsets.py: drop the same number of items as the k = 1 depth holds, at random
across all depths (R = 100), and compare model signatures and per-cell layer profiles with all items (same
estimators as subsets.py).  Also: per-wording feasibility of a correct-only analysis (>= 2 depths with >= 8 items
correct in that wording).  Writes kge2_ref.json."""
import json, numpy as np, warnings
warnings.filterwarnings("ignore")
from clib import *
rng = np.random.default_rng(7)
DATA = {}
for m in ORDER:
    for t in TASKS:
        P = load_pair(m, t); DATA[(m, t)] = P
ALL = {key: zprof(P["y10"], P["k"], P["body"], key[1] in WITHIN, False)[2] for key, P in DATA.items()}
Z0 = np.array([[dec_prof(ALL[(m, t)]) for t in TASKS] for m in ORDER]); a0 = decompose(Z0)[1]
rs, cell_r = [], {t: [] for t in TASKS}
for r in range(100):
    Z = {}
    for key, P in DATA.items():
        n = len(P["k"]); drop = rng.choice(n, int((P["k"] == 1).sum()), replace=False); sel = np.ones(n, bool); sel[drop] = False
        Z[key] = zprof(P["y10"][sel], P["k"][sel], P["body"][sel], key[1] in WITHIN, key[1] in WITHIN)[2]
        cell_r[key[1]].append(np.corrcoef(Z[key], ALL[key])[0, 1])
    a = decompose(np.array([[dec_prof(Z[(m, t)]) for t in TASKS] for m in ORDER]))[1]; rs.append(np.corrcoef(a.ravel(), a0.ravel())[0, 1])
out = {"random_drop_r_signatures": {"mean": float(np.mean(rs)), "p2.5": float(np.percentile(rs, 2.5)), "p97.5": float(np.percentile(rs, 97.5))},
       "random_drop_median_cell_r_layers": {t: float(np.median(v)) for t, v in cell_r.items()}}
feas = {}
for m in ORDER:
    for t in TASKS:
        for w in ("w0", "w1"):
            d = load_cell(m, t, w); k = d["k"]; c = d["correct"]
            feas[f"{m}|{t}|{w}"] = bool(sum((c & (k == kk)).sum() >= 8 for kk in np.unique(k)) >= 2)
out["correct_only_feasible_per_wording"] = {"n": int(sum(feas.values())), "of": len(feas),
                                            "by_task": {t: int(sum(v for kk, v in feas.items() if kk.split("|")[1] == t)) for t in TASKS}}
print(json.dumps(out, indent=1)); json.dump(out, open("kge2_ref.json", "w"), indent=1)
