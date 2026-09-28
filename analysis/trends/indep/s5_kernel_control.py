import json, numpy as np
from ilib import *
np.set_printoptions(precision=2, suppress=True)
def back(m, f, attn=None, clean=True, reg="back"):
    d = wavg(m, f, attn=attn, clean=clean)
    if d is None: return np.nan
    z = zexact(d["Y"], d["k"], d["body"], within(f))[1]; x = xrel(len(z))
    return {"back": z[x >= .5].mean(), "front": z[x < .5].mean(), "50-70": z[(x >= .5) & (x < .7)].mean()}[reg]
# Qwen pair
for attn in ["eager", "sdpa"]:
    v = [back("Qwen2.5-7B-Instruct", f, attn) for f in TASKS]; print("Qwen-I", attn, np.round(v, 2), "mean", np.mean(v))
vb = [back("Qwen2.5-7B", f, "eager") for f in TASKS]; print("Qwen base eager", np.round(vb, 2), np.mean(vb))
# primary with Qwen both eager
def pairs_back(attn_of, clean_of=lambda m: True, tasks=TASKS, reg="back"):
    return np.array([np.mean([back(p, f, attn_of(p), clean_of(p), reg) - back(b, f, attn_of(b), clean_of(b), reg) for f in tasks]) for b, p in PAIRS])
pm = pairs_back(lambda m: prim_attn(m)); print("primary exact back", pm, "LW", lw(pm))
pm = pairs_back(lambda m: "eager"); print("all eager exact back", pm, "LW", lw(pm), "pairs<0", (pm<0).sum())
pm = pairs_back(lambda m: prim_attn(m), lambda m: False); print("OLMo exclusion off (primary kernel)", pm, "LW", lw(pm), "pairs<0", (pm<0).sum())
pm = pairs_back(lambda m: prim_attn(m), lambda m: False, reg="50-70"); print("OLMo exclusion off 50-70", pm, "LW", lw(pm))
# n dropped per OLMo cell
for m in ["OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct"]:
    s = []
    for f in TASKS + ["fixed_digit"]:
        for w in ("w0", "w1"):
            r = raw(m, f, w, "eager"); s.append(f"{f[:4]}/{w}:{int((~r['keep']).sum())}/{len(r['keep'])}")
    print(m, " ".join(s))
# ---- numeral control (fixed_digit) ----
fd = {m: back(m, "fixed_digit") for m in ORDER}
rank = sorted(ORDER, key=lambda m: fd[m])
print("fixed_digit back z (primary kernel):"); [print(f"   {i+1:2d} {m:28s} {fd[m]:+.2f}") for i, m in enumerate(rank)]
print("bases mean", np.mean([fd[m] for m in BASES]), ">=0:", sum(fd[m] >= 0 for m in BASES), "/", len(BASES))
dfd = np.array([fd[p] - fd[b] for b, p in PAIRS]); print("fixed_digit change per pair", dfd, "LW", lw(dfd), "pairs<0", (dfd<0).sum())
ser = pairs_back(lambda m: prim_attn(m)); print("serial minus control", ser - dfd, "LW", lw(ser - dfd), ">0:", (ser - dfd > 0).sum())
fe = {m: back(m, "fixed_digit", "eager") for m in ["Qwen2.5-7B", "Qwen2.5-7B-Instruct"]}; print("Qwen fixed_digit both eager: change", fe["Qwen2.5-7B-Instruct"] - fe["Qwen2.5-7B"], fe)
for attn in ["eager", "sdpa"]:
    print("L3B pair fixed_digit", attn, back("Llama-3.2-3B-Instruct", "fixed_digit", attn) - back("Llama-3.2-3B", "fixed_digit", attn))
fdw = {}
for m in ORDER:
    fdw[m] = back(m, "fixed_digit", reg="whole") if False else None
# per-wording (not averaged) fixed_digit to check wording consistency
for b, p in PAIRS:
    out = []
    for w in ("w0", "w1"):
        rs = []
        for m in (b, p):
            r = raw(m, "fixed_digit", w, prim_attn(m)); kp = r["keep"]
            z = zexact(r["y10"][kp], r["k"][kp].astype(float), r["body"][kp], False)[1]; rs.append(z[xrel(len(z)) >= .5].mean())
        out.append(rs[1] - rs[0])
    print("fixed_digit per wording change", p, np.round(out, 2))
