"""Checks: (1) y10 equals the earlier vet3 cache; (2) zprof on wording-averaged items reproduces consist.json z (q10)."""
import json, numpy as np
from clib import *
V = OLD + "/vet3/cache"
mx = 0
for m in ORDER:
    for f in FAMS:
        for w in ("w0", "w1"):
            a = raw(m, f, w); b = np.load(f"{V}/{m}__{f}__{w}__{attn_of(m)}.npz")
            mx = max(mx, np.abs(a["y10"] - b["y10"]).max()); assert (a["correct"] == b["correct"]).all() and (a["keep"] == b["keep"]).all()
print("max |y10 - vet3 cache|", mx)
C = {r["model"]: r for r in json.load(open(f"{OLD}/consist.json")) if r["stat"] == "q10"}
rs, ratios, nmis = [], [], []
for m in ORDER:
    for f in TASKS:
        P = load_pair(m, f)
        _, _, z = zprof(P["Y"], P["k"], P["body"], f in WITHIN)
        zc = np.array(C[m]["families"][f]["z"])
        rs.append(np.corrcoef(z, zc)[0, 1]); ratios.append(np.median(z / zc))
        if C[m]["families"][f]["n"] != len(P["k"]): nmis.append((m, f, C[m]["families"][f]["n"], len(P["k"])))
print("r(z exact-SD, consist z) min/median", np.min(rs), np.median(rs), "ratio median/min/max", np.median(ratios), np.min(ratios), np.max(ratios))
print("n mismatches", nmis)
