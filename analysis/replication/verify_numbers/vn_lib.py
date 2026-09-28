"""Verifier library (independent code; reuses only the paper's zprof / dec_prof / decompose / ss / perm_model / split_half)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, json, pickle, itertools, math, numpy as np
from scipy.stats import t as tdist, spearmanr
SP8 = str(paths.ANALYSIS)
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["VPRIM"] = "y10"
sys.path.insert(0, SP8); sys.path.insert(0, f"{SP8}/vet3")
from v2lib import zprof, load as vload            # paper library
import model_diff as md                            # paper library
from model_diff import dec_prof, decompose, ss, perm_model, split_half

T = ["pointer", "arith", "binding", "relational"]
M8 = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
      "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
M6 = [m for m in M8 if not m.startswith("Qwen2.5-7B")]
CLEAN4 = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-14B", "Mistral-Small-24B-Base-2501"]
LIN = {"Llama-3.1-8B": "L8", "Llama-3.1-8B-Instruct": "L8", "Qwen2.5-7B": "Q7", "Qwen2.5-7B-Instruct": "Q7",
       "Qwen2.5-14B": "Q14", "OLMo-2-1124-7B": "O7", "OLMo-2-1124-7B-Instruct": "O7", "Mistral-Small-24B-Base-2501": "M24"}
PAIRS = [("Llama-3.1-8B", "Llama-3.1-8B-Instruct"), ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct")]
KR = {"pointer": range(1, 6), "arith": range(1, 7), "binding": range(1, 7), "relational": range(1, 7)}

CACHE = pickle.load(open(f"{HERE}/vn_cache.pkl", "rb"))

def cell(m, mode, fam, stat="q10", excl=True):
    """E2b: statistic [items, L], k, items, keep mask (measured and, for OLMo-2 if excl, not flagged)."""
    d = CACHE[(m, mode, fam)]
    keep = d["measured"].copy()
    if excl and m.startswith("OLMo"): keep &= ~d["flag"]
    k = np.array([x["k"] for x in d["items"]], float)
    return d[stat][keep], k[keep], [x for x, kp in zip(d["items"], keep) if kp], keep

def zlayers(m, mode, fam, stat="q10", excl=True):
    Y, k, _, _ = cell(m, mode, fam, stat, excl)
    if stat == "neff": Y = -Y                      # + = concentrating
    return zprof(Y, k, within=False)[2]

def Zprof(models, mode="direct", stat="q10", excl=True):
    return np.array([[dec_prof(zlayers(m, mode, t, stat, excl)) for t in T] for m in models])

CQ = {r["model"]: r for r in json.load(open(f"{SP8}/consist.json")) if r["stat"] == "q10"}
def e2_z(m, t, key="z"): return np.array(CQ[m]["families"][t][key], float)
def Zprof_e2(models, key="z"): return np.array([[dec_prof(e2_z(m, t, key)) for t in T] for m in models])

CELLS = json.load(open(f"{SP8}/vet3/cells.json"))
def e2_cells_z(m, t, w, attn, variant="all"):
    for r in CELLS:
        if r["model"] == m and r["fam"] == t and r["w"] == w and r["attn"] == attn and r["variant"] == variant: return np.array(r["z"], float)
    return None

def r_(x, y): return float(np.corrcoef(np.ravel(x), np.ravel(y))[0, 1])
def tci(v):
    v = np.asarray(v, float); n = len(v); h = tdist.ppf(.975, n - 1) * v.std(ddof=1) / math.sqrt(n)
    return (float(v.mean()), float(v.mean() - h), float(v.mean() + h))
def exact_perm_corr(x, y, spearman=False, two_sided=True):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if spearman:
        from scipy.stats import rankdata
        x = rankdata(x); y = rankdata(y)
    obs = r_(x, y); P = np.array(list(itertools.permutations(range(len(x)))))
    yp = y[P]; xc = x - x.mean(); ypc = yp - yp.mean(1, keepdims=True)
    null = (ypc @ xc) / (np.sqrt((ypc ** 2).sum(1)) * np.sqrt((xc ** 2).sum()))
    p = np.mean(np.abs(null) >= abs(obs) - 1e-12) if two_sided else np.mean(null >= obs - 1e-12)
    return obs, float(p)
def rel_depth(L): return np.arange(L) / (L - 1)
