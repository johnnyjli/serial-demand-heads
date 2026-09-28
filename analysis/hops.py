import sys, re, json, numpy as np
sys.path.insert(0, sys.argv[1])
from e2lib import ORDER, load
from solve import blocks
def hops_done(fam, w, t):
    b = blocks(t["text"])[-1]; pred = int(t["pred_constrained"])
    if fam == "pointer":
        tbl = dict((int(a), int(c)) for a, c in re.findall(r"(\d) (?:goes|leads) to (\d)\.", b))
        s = int(re.search(r"(?:Start at|From) (\d)", b).group(1)); path = [s]
        for _ in range(9): path.append(tbl[path[-1]])
        return path.index(pred) if pred in path else None
    if fam == "arith":
        if w == "w0": d, mod, s = map(int, re.search(r"add (\d) and keep the remainder after dividing by (\d+)\. Start at (\d)", b).groups())
        else: s, d, mod = map(int, re.search(r"x = (\d)\. Repeat \d+ times: x = \(x \+ (\d)\) mod (\d+)", b).groups())
        if pred >= mod: return None
        inv = pow(d, -1, mod); return ((pred - s) * inv) % mod
out = {}
for m in ORDER:
    for fam in ("pointer", "arith"):
        for w in ("w0", "w1"):
            d = load(m, f"{fam}/{w}")
            for t in d["items"]:
                out.setdefault(m, {}).setdefault(fam, {}).setdefault(str(t["k"]), []).append(hops_done(fam, w, t))
json.dump(out, open(sys.argv[2], "w"))
