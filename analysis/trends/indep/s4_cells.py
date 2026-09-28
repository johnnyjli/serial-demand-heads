import json, numpy as np
from ilib import *
np.set_printoptions(precision=2, suppress=True)
R = json.load(open(f"{SP}/vet3/cells.json"))
D = {}
for r in R: D[(r["attn"], r["variant"], r["model"], r["fam"], r["w"])] = np.array(r["z"], float)
def get(attn, var, m, f):
    a = D.get((attn, var, m, f, "w0")); b = D.get((attn, var, m, f, "w1"))
    if a is None and var == "noclean": a, b = D.get((attn, "all", m, f, "w0")), D.get((attn, "all", m, f, "w1"))   # noclean only stored for OLMo
    if a is None or b is None: return None
    return 0.5 * (a + b)
def region(z, reg):
    x = xrel(len(z)); d = dec(z)
    return {"back": z[x >= .5].mean(), "front": z[x < .5].mean(), "50-70": z[(x >= .5) & (x < .7)].mean(), "d50-70": d[5:7].mean(), "70-100": z[x >= .7].mean()}[reg]
def pairdiff(attn_of, var, reg, tasks=TASKS):
    out = []
    for b, p in PAIRS:
        v = []
        for f in tasks:
            zb, zp = get(attn_of(b), var, b, f), get(attn_of(p), var, p, f)
            if zb is None or zp is None: v.append(np.nan); continue
            v.append(region(zp, reg) - region(zb, reg))
        out.append(np.array(v))
    return np.array(out)
print("noclean models:", sorted(set(r["model"] for r in R if r["variant"] == "noclean")))
for label, attn_of, var in [("eager all (Qwen-I eager)", lambda m: "eager", "all"),
                            ("eager sink_head", lambda m: "eager", "sink_head"),
                            ("eager sink_item", lambda m: "eager", "sink_item"),
                            ("eager noclean", lambda m: "eager", "noclean")]:
    for reg in ["back", "50-70", "d50-70", "front"]:
        X = pairdiff(attn_of, var, reg); pm = X.mean(1)
        print(f"{label:26s} {reg:6s} per pair {pm} LW {lw(pm):+.3f} lineages<0 {(lineages(pm)<0).sum()} pairs<0 {(pm<0).sum()}")
# mixed: Qwen-I SDPA from cells, others eager, per-wording z averaged
for var in ["all", "sink_head", "sink_item"]:
    X = pairdiff(lambda m: "sdpa" if m == "Qwen2.5-7B-Instruct" else "eager", var, "back"); pm = X.mean(1)
    print(f"cells {var} with Qwen-I sdpa (Qwen-I sdpa is only 'all' variant?) back per pair {pm} LW {lw(pm):+.3f}")
# SDPA vs eager for L3B pair all tasks
for attn in ["eager", "sdpa"]:
    X = pairdiff(lambda m: attn, "all", "back"); print("L3B pair", attn, "back per task", X[0], "mean", X[0].mean())
    X = pairdiff(lambda m: attn, "all", "back", ["fixed_digit"]); print("L3B pair", attn, "fixed_digit back", X[0])
# pointer/w0 kernel check, per-wording z (w0 only)
for b, p in PAIRS:
    r = {}
    for attn in ["eager", "sdpa"]:
        zb, zp = D.get((attn, "all", b, "pointer", "w0")), D.get((attn, "all", p, "pointer", "w0"))
        if zb is not None and zp is not None: r[attn] = region(zp, "back") - region(zb, "back")
        r[f"{attn}_b"] = None if zb is None else region(zb, "back"); r[f"{attn}_p"] = None if zp is None else region(zp, "back")
    print("pointer/w0", p, {k: (round(v, 2) if v is not None else None) for k, v in r.items()})
print("sink fractions (eager all), mean over cells:", {m: round(np.mean([r['sink_frac'] for r in R if r['model']==m and r['variant']=='all' and r['attn']=='eager']),3) for m in sorted(set(x for p in PAIRS for x in p))})
