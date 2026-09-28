"""Which prompt features carry the correctness signal at fixed depth?  Same CV as decode.py (no nulls):
gold-digit one-hot alone vs the family-specific structural features alone.  Writes decode_parts.json."""
import json, numpy as np, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode import *
def one(key):
    m, f, w = key; d = load_cell(m, f, w); k, ok = d["k"], d["correct"].astype(int)
    depths = [kk for kk in np.unique(k) if min(ok[k == kk].sum(), (1 - ok[k == kk]).sum()) >= MINC]
    use = np.isin(k, depths)
    if use.sum() < 40: return None
    ku, yu, gu = k[use], ok[use], d["body"][use]; strat = ku * 2 + yu
    F = np.array([prompt_feats(f, w, t, str(g)) for t, g in zip(d["text"][use], d["gold"][use])], float)
    o = {"model": m, "fam": f, "w": w}
    for name, X in (("gold_onehot", F[:, :10]), ("structure", F[:, 10:])):
        X = resid(X, ku); X = X[:, X.std(0) > 1e-9]; o[name] = oof(X, yu, strat, gu) if X.shape[1] else None
    return o
if __name__ == "__main__":
    keys = [(m, f, w) for m in ORDER for f in FAMS for w in ("w0", "w1")]
    with ProcessPoolExecutor(12) as ex: R = [r for r in ex.map(one, keys) if r]
    out = {}
    for g, fs in {**{f: [f] for f in FAMS}, "serial": TASKS}.items():
        rr = [r for r in R if r["fam"] in fs]
        out[g] = {"n_cells": len(rr), **{n: float(np.median([r[n] for r in rr if r[n] is not None])) for n in ("gold_onehot", "structure")}}
        print(g, out[g])
    json.dump({"summary": out, "cells": R}, open("decode_parts.json", "w"), indent=1)
