import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import glob, json, os, numpy as np
ROOT = str(paths.E2_SWEEP)
ORDER = ["Llama-3.2-1B","Llama-3.2-3B","Llama-3.2-3B-Instruct","Llama-3.1-8B","Llama-3.1-8B-Instruct",
         "Qwen2.5-0.5B","Qwen2.5-1.5B","Qwen2.5-3B","Qwen2.5-7B","Qwen2.5-7B-Instruct","Qwen2.5-14B",
         "OLMo-2-1124-7B","OLMo-2-1124-7B-SFT","OLMo-2-1124-7B-DPO","OLMo-2-1124-7B-Instruct",
         "Mistral-7B-v0.1","Mistral-Small-24B-Base-2501"]
FAMS = ["pointer","arith","binding","relational","fixed_digit"]
CELLS = [f"{f}/{w}" for f in FAMS for w in ("w0","w1")]
_idx = None
def index(attn="eager"):
    out = {}
    for f in glob.glob(f"{ROOT}/bf16_{attn}/*.npz"):
        b = os.path.basename(f); model = b.split("__")[0].split("_", 1)[1]
        suite = b.split("__")[1]; fam = suite.split("_w")[0]; w = "w" + suite.split("_w")[1][0]
        out[(model, f"{fam}/{w}")] = f
    return out
def load(model, cell, attn="eager"):
    f = index(attn)[(model, cell)]
    z = np.load(f, allow_pickle=False); m = json.loads(str(z["meta"]))
    d = {n: z[n] for n in z.files if n != "meta"}
    d["meta"] = m; d["items"] = m["items"]; d["model"] = model; d["cell"] = cell
    d["correct"] = d["correct_strict"] if "correct_strict" in d else d["correct_constrained"]
    d["L"], d["H"] = m["n_layers"], m["n_heads"]
    if os.environ.get("E2_CLEAN"):
        # drop items where one head outputs > 50x its layer's median head (OLMo-2 only)
        S = d["contribution"].astype(float)
        keep = ~((S.max(-1) / np.maximum(np.median(S, -1), 1e-9)) > 50).any(1)
        n = len(keep); d["n_dropped"] = int((~keep).sum())
        for key, v in list(d.items()):
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n: d[key] = v[keep]
        d["items"] = [t for t, kk in zip(d["items"], keep) if kk]
    return d
def window(L, lo=.4, hi=.8): return slice(int(lo*(L-1)), int(hi*(L-1))+1)
# The concentration statistic.  E2_STAT = "top1" (default, the old top-head share) or "q25" /
# "q10" / "q50": the summed share of the ceil(q*H) strongest heads.  NaN entries (masked heads)
# are treated as absent.
STAT = os.environ.get("E2_STAT", "top1")
STAT_LABEL = {"top1": "top-head share", "q10": "top-10% head share", "q25": "top-25% head share",
              "q50": "top-50% head share"}[STAT]
def top_share(S):
    S = np.where(np.isfinite(S), S, 0.0)
    tot = np.clip(S.sum(-1), 1e-12, None)
    if STAT == "top1": return S.max(-1) / tot
    H = S.shape[-1]; m = max(1, int(np.ceil({"q10": .10, "q25": .25, "q50": .50}[STAT] * H)))
    return -np.sort(-S, -1)[..., :m].sum(-1) / tot
def pr_h(S):
    s, s2 = S.sum(-1), (S*S).sum(-1); return s*s/np.maximum(s2, 1e-30)/S.shape[-1]
def slope(y, k):
    k = np.asarray(k, float); kc = k - k.mean(); return float(kc @ (y - y.mean()) / (kc @ kc))
def perm_p(y, k, n=2000, seed=0):
    rng = np.random.default_rng(seed); k = np.asarray(k, float); kc = k - k.mean(); ss = kc @ kc
    yc = y - y.mean(); o = kc @ yc / ss
    nl = np.array([rng.permutation(kc) @ yc / ss for _ in range(n)])
    return float(o), float((np.sum(np.abs(nl) >= abs(o) - 1e-15) + 1) / (n + 1))
