"""Independent verifier library (own code). Reads consist.json, vet3/cells.json, vet3/cache/*.npz, model_diff2.json (features only)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, json, itertools, numpy as np
from scipy import stats
SP = str(paths.ANALYSIS)
OUT = f"{SP}/trends/verify_other"
CACHE = f"{SP}/vet3/cache"
M = ["Llama-3.2-1B","Llama-3.2-3B","Llama-3.2-3B-Instruct","Llama-3.1-8B","Llama-3.1-8B-Instruct",
     "Qwen2.5-0.5B","Qwen2.5-1.5B","Qwen2.5-3B","Qwen2.5-7B","Qwen2.5-7B-Instruct","Qwen2.5-14B",
     "OLMo-2-1124-7B","OLMo-2-1124-7B-SFT","OLMo-2-1124-7B-DPO","OLMo-2-1124-7B-Instruct",
     "Mistral-7B-v0.1","Mistral-Small-24B-Base-2501"]
TK = ["pointer","arith","binding","relational"]
POST2BASE = {"Llama-3.2-3B-Instruct":"Llama-3.2-3B","Llama-3.1-8B-Instruct":"Llama-3.1-8B","Qwen2.5-7B-Instruct":"Qwen2.5-7B",
             "OLMo-2-1124-7B-SFT":"OLMo-2-1124-7B","OLMo-2-1124-7B-DPO":"OLMo-2-1124-7B","OLMo-2-1124-7B-Instruct":"OLMo-2-1124-7B"}
LINEAGE = {m: POST2BASE.get(m, m) for m in M}
BASES = [m for m in M if LINEAGE[m] == m]
LINS = BASES
def fam(m):
    return "Llama" if m.startswith("Llama") else "Qwen" if m.startswith("Qwen") else "OLMo" if m.startswith("OLMo") else "Mistral"
# parameter counts (billions; published totals)
PARAMS = {"Llama-3.2-1B":1.24,"Llama-3.2-3B":3.21,"Llama-3.1-8B":8.03,"Qwen2.5-0.5B":0.49,"Qwen2.5-1.5B":1.54,"Qwen2.5-3B":3.09,
          "Qwen2.5-7B":7.61,"Qwen2.5-14B":14.7,"OLMo-2-1124-7B":7.30,"Mistral-7B-v0.1":7.24,"Mistral-Small-24B-Base-2501":23.6}
for m in M: PARAMS.setdefault(m, PARAMS[LINEAGE[m]])
def prim_attn(m): return "sdpa" if m == "Qwen2.5-7B-Instruct" else "eager"

def deciles(z):
    z = np.asarray(z, float); L = len(z); x = np.arange(L) / (L - 1)
    d = np.minimum(np.floor(x * 10 + 1e-9).astype(int), 9)
    return np.array([np.nanmean(z[d == i]) for i in range(10)])

_C = None
def consist():
    global _C
    if _C is None: _C = json.load(open(f"{SP}/consist.json"))
    return _C
def layer_z(stat="q10", key="z"):
    """dict m -> t -> per-layer array from consist.json"""
    out = {}
    for r in consist():
        if r["stat"] == stat: out[r["model"]] = {t: np.array(r["families"][t][key], float) for t in TK}
    return out
def Zdec(stat="q10", key="z"):
    LZ = layer_z(stat, key); return np.array([[deciles(LZ[m][t]) for t in TK] for m in M])

# ---- cache rebuild (exact permutation z) ----
def npz(m, f, w, attn):
    fn = f"{CACHE}/{m}__{f}__{w}__{attn}.npz"
    if not os.path.exists(fn): return None
    z = np.load(fn); return {n: z[n] for n in z.files}
def wavg_items(m, f, attn, stat="y10", clean=True, fields=("correct",)):
    a, b = npz(m, f, "w0", attn), npz(m, f, "w1", attn)
    if a is None or b is None: return None
    def km(d):
        kp = d["keep"] if clean else np.ones(len(d["k"]), bool)
        return {(int(k), int(bb)): j for j, (k, bb, q) in enumerate(zip(d["k"], d["body"], kp)) if q}
    ka, kb = km(a), km(b); com = sorted(set(ka) & set(kb))
    ia = np.array([ka[c] for c in com]); ib = np.array([kb[c] for c in com])
    Y = 0.5 * (a[stat][ia] + b[stat][ib])
    return {"Y": Y, "k": np.array([c[0] for c in com], float), "body": np.array([c[1] for c in com])}
def exact_z(Y, k, body=None, within=False):
    Y = np.asarray(Y, float); k = np.asarray(k, float); kc = k - k.mean(); Skk = kc @ kc
    Yc = Y - Y.mean(0); b = kc @ Yc / Skk
    if not within:
        var = (Yc ** 2).sum(0) / (len(k) - 1) / Skk
    else:
        v = np.zeros(Y.shape[1])
        for g in np.unique(body):
            i = body == g
            if i.sum() < 2: continue
            kg = k[i] - k[i].mean(); Yg = Y[i] - Y[i].mean(0)
            v += (kg @ kg) * (Yg ** 2).sum(0) / (i.sum() - 1)
        var = v / Skk ** 2
    return b / np.sqrt(var), 100 * b / np.abs(Y.mean(0))
def rebuild_z(m, f, attn=None, stat="y10", clean=True):
    attn = attn or prim_attn(m)
    d = wavg_items(m, f, attn, stat, clean)
    if d is None: return None
    return exact_z(d["Y"], d["k"], d["body"], f in ("binding", "relational"))
def accuracy(m, f, attn=None, clean=True, k1=False):
    attn = attn or prim_attn(m); v = []
    for w in ("w0", "w1"):
        d = npz(m, f, w, attn)
        kp = d["keep"] if clean else np.ones(len(d["k"]), bool)
        c = d["correct"][kp]; kk = d["k"][kp]
        v.append(c[kk == 1].mean() if k1 else c.mean())
    return float(np.mean(v))

# ---- stats helpers ----
def binom2(k, n): return float(stats.binomtest(k, n, 0.5).pvalue)
def lin_mean(vals):
    """vals: dict model->value; returns dict lineage->mean over its models"""
    return {L: float(np.mean([vals[m] for m in M if LINEAGE[m] == L])) for L in LINS}
def boot_ci(x, B=20000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
def count(vals, pred):
    return int(sum(pred(v) for v in vals))
def spear(x, y):
    r = stats.spearmanr(x, y); return float(r.correlation), float(r.pvalue)
def partial_spear(x, y, *zs):
    """Spearman partial correlation of x,y given z (rank then residualise); p from t with n-3 df"""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Z = np.column_stack([np.ones(len(x))] + [stats.rankdata(np.asarray(zz, float)) for zz in zs])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1]); df = len(x) - 2 - (Z.shape[1] - 1)
    t = r * np.sqrt(df / (1 - r ** 2)); p = float(2 * stats.t.sf(abs(t), df))
    return r, p
