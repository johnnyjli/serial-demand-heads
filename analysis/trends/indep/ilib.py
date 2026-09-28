"""Independent verification library (own code; reads vet3/cache npz and consist.json only)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, json, numpy as np
SP = str(paths.ANALYSIS)
CACHE = f"{SP}/vet3/cache"
ORDER = ["Llama-3.2-1B","Llama-3.2-3B","Llama-3.2-3B-Instruct","Llama-3.1-8B","Llama-3.1-8B-Instruct",
         "Qwen2.5-0.5B","Qwen2.5-1.5B","Qwen2.5-3B","Qwen2.5-7B","Qwen2.5-7B-Instruct","Qwen2.5-14B",
         "OLMo-2-1124-7B","OLMo-2-1124-7B-SFT","OLMo-2-1124-7B-DPO","OLMo-2-1124-7B-Instruct",
         "Mistral-7B-v0.1","Mistral-Small-24B-Base-2501"]
TASKS = ["pointer","arith","binding","relational"]
PAIRS = [("Llama-3.2-3B","Llama-3.2-3B-Instruct"),("Llama-3.1-8B","Llama-3.1-8B-Instruct"),("Qwen2.5-7B","Qwen2.5-7B-Instruct"),
         ("OLMo-2-1124-7B","OLMo-2-1124-7B-SFT"),("OLMo-2-1124-7B","OLMo-2-1124-7B-DPO"),("OLMo-2-1124-7B","OLMo-2-1124-7B-Instruct")]
LINEAGE = [0,1,2,3,3,3]   # pair index -> lineage
BASES = [m for m in ORDER if not any(m == p for _, p in PAIRS)]
def prim_attn(m): return "sdpa" if m == "Qwen2.5-7B-Instruct" else "eager"
def raw(m, f, w, attn):
    fn = f"{CACHE}/{m}__{f}__{w}__{attn}.npz"
    if not os.path.exists(fn): return None
    z = np.load(fn); return {n: z[n] for n in z.files}
def wavg(m, f, attn=None, stat="y10", clean=True, extra=()):
    """wording-averaged per-item statistic on items present (and kept) in both wordings, keyed by (k, body)"""
    attn = attn or prim_attn(m)
    a, b = raw(m, f, "w0", attn), raw(m, f, "w1", attn)
    if a is None or b is None: return None
    def keys(d):
        kp = d["keep"] if clean else np.ones(len(d["k"]), bool)
        return {(int(k), int(bd)): j for j, (k, bd, kk) in enumerate(zip(d["k"], d["body"], kp)) if kk}
    ka, kb = keys(a), keys(b); common = sorted(set(ka) & set(kb))
    ia = np.array([ka[x] for x in common]); ib = np.array([kb[x] for x in common])
    out = {"Y": 0.5 * (a[stat][ia] + b[stat][ib]), "k": np.array([x[0] for x in common], float),
           "body": np.array([x[1] for x in common]), "keys": common, "L": a[stat].shape[1]}
    for e in extra: out[e] = (a[e][ia].astype(float), b[e][ib].astype(float))
    return out
def zexact(Y, k, body, within):
    """OLS slope per column and exact permutation-null SD (k permuted across items, or within body)."""
    Y = np.asarray(Y, float); k = np.asarray(k, float)
    kc = k - k.mean(); Skk = kc @ kc; Yc = Y - Y.mean(0); b = kc @ Yc / Skk
    if not within:
        n = len(k); var = (Yc ** 2).sum(0) / (n - 1) / Skk
    else:
        var = np.zeros(Y.shape[1])
        for g in np.unique(body):
            i = body == g; ng = i.sum()
            if ng < 2: continue
            kg = k[i] - k[i].mean(); Yg = Y[i] - Y[i].mean(0)
            var += (kg @ kg) * (Yg ** 2).sum(0) / (ng - 1)
        var = var / Skk ** 2
    return b, b / np.sqrt(var)
def within(f): return f in ("binding", "relational")
def xrel(L): return np.arange(L) / (L - 1)
def dec(z):
    z = np.asarray(z, float); L = len(z); x = xrel(L); d = np.minimum((x * 10).astype(int), 9)
    return np.array([z[d == i].mean() for i in range(10)])
def lw(v):
    """lineage-weighted mean over 6 pairs (OLMo pairs averaged first)"""
    v = np.asarray(v, float); return float(np.mean([v[0], v[1], v[2], v[3:6].mean(0)] if v.ndim == 1 else None)) if v.ndim == 1 else np.mean([v[0], v[1], v[2], v[3:6].mean(0)], 0)
def lineages(v):
    v = np.asarray(v, float); return np.array([v[0], v[1], v[2], v[3:6].mean(0)])
