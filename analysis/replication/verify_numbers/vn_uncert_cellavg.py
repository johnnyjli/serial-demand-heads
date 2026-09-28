"""Answer uncertainty at the measured position, averaged over cells (each cell weighted once) vs pooled over items."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np
from vn_lib import *
ITEMS = json.load(open(str(paths.E2B / "items.json")))
ASET = {it["id"]: [int(a) for a in it["answer_set"]] for f in T for it in ITEMS[f]}
out = {}
for mode in ("direct", "stepwise"):
    for excl in (True, False):
        cm, ce = [], []
        for m in M8:
            for t in T:
                d = CACHE[(m, mode, t)]; _, _, _, keep = cell(m, mode, t, excl=excl); mp, en = [], []
                for i in np.nonzero(keep)[0]:
                    lg = d["DL"][i][ASET[d["items"][i]["id"]]].astype(float); p = np.exp(lg - lg.max()); p /= p.sum()
                    mp.append(p.max()); en.append(-(p[p > 0] * np.log(p[p > 0])).sum())
                cm.append(np.mean(mp)); ce.append(np.mean(en))
        out[f"{mode}|excl={excl}"] = {"cellavg_maxp": float(np.mean(cm)), "cellavg_entropy": float(np.mean(ce))}
print(json.dumps(out, indent=1)); json.dump(out, open("vn_uncert_cellavg.json", "w"), indent=1)
