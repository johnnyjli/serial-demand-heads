"""Same-item base vs post comparison recomputed from vet3/cache (own code): per wording cell, items common to both
models (keep masks applied), per-item mean top-10% share over the 40-80% window, divided by the model's mean;
slope of (post - base) on k in %/step; permutation p (k shuffled across items, 2000 draws; also within-body variant)."""
import json, sys, numpy as np
from ilib import *
def win(L, lo=.4, hi=.8): return slice(int(lo * (L - 1)), int(hi * (L - 1)) + 1)
def perm(y, k, body, within_body, n=2000, seed=0):
    rng = np.random.default_rng(seed); kc = k - k.mean(); yc = y - y.mean(); o = kc @ yc / (kc @ kc)
    if within_body:
        groups = [np.nonzero(body == g)[0] for g in np.unique(body)]
        nl = []
        for _ in range(n):
            kk = k.copy()
            for g in groups: kk[g] = k[g][rng.permutation(len(g))]
            kk = kk - kk.mean(); nl.append(kk @ yc / (kk @ kk))
        nl = np.array(nl)
    else:
        nl = np.array([rng.permutation(kc) @ yc / (kc @ kc) for _ in range(n)])
    return 100 * o, (np.sum(np.abs(nl) >= abs(o) - 1e-15) + 1) / (n + 1)
def cell(b, p, f, w, attn_of, stat="y10", clean=True):
    A, B = raw(b, f, w, attn_of(b)), raw(p, f, w, attn_of(p))
    if A is None or B is None: return None
    def km(d): kp = d["keep"] if clean else np.ones(len(d["k"]), bool); return {(int(k), int(bd)): j for j, (k, bd, q) in enumerate(zip(d["k"], d["body"], kp)) if q}
    ka, kb = km(A), km(B); common = sorted(set(ka) & set(kb)); ia = np.array([ka[x] for x in common]); ib = np.array([kb[x] for x in common])
    L = A[stat].shape[1]; W = win(L)
    ya = np.nanmean(A[stat][ia][:, W], 1); yb = np.nanmean(B[stat][ib][:, W], 1)
    d = yb / np.nanmean(yb) - ya / np.nanmean(ya)
    k = np.array([x[0] for x in common], float); body = np.array([x[1] for x in common])
    ca, cb = A["correct"][ia].astype(bool), B["correct"][ib].astype(bool)
    out = {}
    for sub, sel in [("all", np.ones(len(k), bool)), ("k2plus", k >= 2), ("both_wrong", ~ca & ~cb)]:
        out[sub] = perm(d[sel], k[sel], body[sel], False)
        out[sub + "_wb"] = perm(d[sel], k[sel], body[sel], within(f)) if within(f) else out[sub]
    return out
def summarize(label, attn_of, stat="y10", tasks=TASKS):
    rows = []
    for b, p in PAIRS:
        for f in tasks:
            for w in ("w0", "w1"):
                c = cell(b, p, f, w, attn_of, stat)
                if c is not None: rows.append((p, f, w, c))
    for sub in ["all", "k2plus", "both_wrong", "all_wb"]:
        v = np.array([r[3][sub][0] for r in rows]); pv = np.array([r[3][sub][1] for r in rows])
        per = np.array([np.mean([r[3][sub][0] for r in rows if r[0] == p]) for b, p in PAIRS])
        print(f"{label:28s} {sub:11s} n={len(v)} neg {(v<0).sum()} signeg {((v<0)&(pv<.05)).sum()} sigpos {((v>0)&(pv<.05)).sum()} mean {v.mean():+.3f} LW {lw(per):+.3f}  per pair {np.round(per,2)}")
    return rows
r1 = summarize("eager (as pairs_q10)", lambda m: "eager")
r2 = summarize("primary kernel (Qwen-I SDPA)", prim_attn)
r3 = summarize("control eager", lambda m: "eager", tasks=["fixed_digit"])
r4 = summarize("control primary", prim_attn, tasks=["fixed_digit"])
r5 = summarize("nosink primary", prim_attn, stat="y10ns")
r6 = summarize("control nosink primary", prim_attn, stat="y10ns", tasks=["fixed_digit"])
# Llama-3.2-3B pair all-SDPA
rows = [(f, w, cell("Llama-3.2-3B", "Llama-3.2-3B-Instruct", f, w, lambda m: "sdpa")) for f in TASKS for w in ("w0", "w1")]
v = np.array([r[2]["all"][0] for r in rows]); pv = np.array([r[2]["all"][1] for r in rows]); print("L3B all-SDPA serial:", (v<0).sum(), ((v<0)&(pv<.05)).sum(), ((v>0)&(pv<.05)).sum())
# compare with pairs_q10.json exactly
R = {(r["post"], r["cell"]): r for r in json.load(open(f"{SP}/pairs_q10.json"))}
err = max(abs(R[(p, f"{f}/{w}")]["mean_all"][0] - c["all"][0]) for p, f, w, c in r1); print("max |slope diff| vs pairs_q10.json (eager):", err)
