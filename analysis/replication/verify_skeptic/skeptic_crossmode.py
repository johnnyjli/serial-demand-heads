"""Cross-mode (stepwise vs direct) signature agreement in subsets, and item-split reliability of stepwise
signatures (not just model profiles).  Signatures centred on the models in each subset."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, json, numpy as np
SP8 = str(paths.ANALYSIS)
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP8); sys.path.insert(0, f"{SP8}/vet3"); sys.path.insert(0, str(paths.REPLICATION))
from v2lib import zprof
from model_diff import dec_prof, decompose
import e2b
T = ["pointer", "arith", "binding", "relational"]
M8 = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
      "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
def q10(C):
    C = C.astype(np.float64); P = C / C.sum(-1, keepdims=True); H = C.shape[-1]; m = max(1, int(np.ceil(0.1 * H)))
    return -np.sort(-P, -1)[..., :m].sum(-1)
def r_(x, y): return float(np.corrcoef(np.ravel(x), np.ravel(y))[0, 1])
runs = e2b.load_results(str(paths.E2B))
RB = {(r["model"].split("/")[-1], r["family"], r["mode"]): r for r in runs}
def data(m, t, mode):
    r = RB[(m, t, mode)]; C = r["C"]; k = np.array([x["k"] for x in r["items"]], float)
    ok = ~np.isnan(C).any((1, 2))
    if m.startswith("OLMo"):
        ok &= ~((np.nan_to_num(C).max(-1) / np.maximum(np.nanmedian(C, -1), 1e-9)) > 50).any(1)
    return q10(C[ok]), k[ok]
Y = {(m, t, md): data(m, t, md) for m in M8 for t in T for md in ("direct", "stepwise")}
def Zof(mode, sel=None):
    Z = []
    for m in M8:
        row = []
        for t in T:
            y, k = Y[(m, t, mode)]
            if sel is not None: y, k = y[sel[(m, t, mode)]], k[sel[(m, t, mode)]]
            row.append(dec_prof(zprof(y, k, within=False)[2]))
        Z.append(row)
    return np.array(Z)
Zd, Zs = Zof("direct"), Zof("stepwise")
SUB = {"M8": M8, "no_Qwen7B": [m for m in M8 if not m.startswith("Qwen2.5-7B")],
       "no_OLMo": [m for m in M8 if not m.startswith("OLMo")],
       "clean4": ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-14B", "Mistral-Small-24B-Base-2501"]}
out = {"cross_mode": {}}
for nm, ms in SUB.items():
    ii = [M8.index(m) for m in ms]
    ad, as_ = decompose(Zd[ii])[1], decompose(Zs[ii])[1]
    out["cross_mode"][nm] = {"pooled": r_(ad, as_), "per_model": {m: round(r_(ad[j], as_[j]), 3) for j, m in enumerate(ms)},
                             "mean": float(np.mean([r_(ad[j], as_[j]) for j in range(len(ms))]))}
# item-split reliability of signatures (4 random splits stratified by k)
rng = np.random.default_rng(0); rel = {"direct": [], "stepwise": []}
for s in range(4):
    for mode in ("direct", "stepwise"):
        A, B = {}, {}
        for m in M8:
            for t in T:
                y, k = Y[(m, t, mode)]; idx = np.zeros(len(k), bool)
                for kk in np.unique(k):
                    w = np.where(k == kk)[0]; rng.shuffle(w); idx[w[: len(w) // 2]] = True
                A[(m, t, mode)] = idx; B[(m, t, mode)] = ~idx
        ZA, ZB = Zof(mode, A), Zof(mode, B)
        rel[mode].append(r_(decompose(ZA)[1], decompose(ZB)[1]))
out["signature_item_split_half_r"] = {k: [float(np.mean(v)), [round(x, 3) for x in v]] for k, v in rel.items()}
json.dump(out, open(os.path.join(OUT, "skeptic_crossmode.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
