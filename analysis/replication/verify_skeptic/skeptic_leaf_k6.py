"""E2b binding/relational: at k=6 the queried entity is a chain leaf (mentioned once in the body; twice at k<=5),
the surface confound E2's generator avoided with chains one link longer than max k.  Does dropping k=6 change
(a) E2b vs E2 per-cell agreement for binding/relational, (b) the binding-relational deviation agreement within E2b,
(c) signature agreement with E2 (and with E2-w0 / E2-w1)?"""
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
runs = [r for r in e2b.load_results(str(paths.E2B)) if r["mode"] == "direct"]
RB = {(r["model"].split("/")[-1], r["family"]): r for r in runs}
def build(kmax_chain):
    Z = []
    for m in M8:
        row = []
        for t in T:
            r = RB[(m, t)]; C = r["C"]; k = np.array([x["k"] for x in r["items"]], float)
            keep = np.ones(len(k), bool)
            if m.startswith("OLMo"): keep = ~((C.max(-1) / np.maximum(np.median(C, -1), 1e-9)) > 50).any(1)
            if t in ("binding", "relational"): keep &= k <= kmax_chain
            row.append(dec_prof(zprof(q10(C[keep]), k[keep], within=False)[2]))
        Z.append(row)
    return np.array(Z)
CQ = {r["model"]: r for r in json.load(open(f"{SP8}/consist.json")) if r["stat"] == "q10"}
Za = np.array([[dec_prof(np.array(CQ[m]["families"][t]["z"])) for t in T] for m in M8])
CC = json.load(open(f"{SP8}/vet3/cells.json"))
def e2w(w):
    D = {(r["model"], r["fam"]): np.array(r["z"], float) for r in CC if r["variant"] == "all" and r["fam"] in T and r["w"] == w
         and r["attn"] == ("sdpa" if r["model"] == "Qwen2.5-7B-Instruct" else "eager")}
    return np.array([[dec_prof(D[(m, t)]) for t in T] for m in M8])
W0, W1 = e2w("w0"), e2w("w1")
def dev_pairs(Z):
    d = Z - Z.mean(0, keepdims=True)
    return {"binding-relational": r_(d[:, 2], d[:, 3]), "pointer-arith": r_(d[:, 0], d[:, 1]),
            "across_mean": float(np.mean([r_(d[:, i], d[:, j]) for i in (0, 1) for j in (2, 3)]))}
out = {}
for tag, kmax in (("all_k", 6), ("k<=5_chain", 5)):
    Zb = build(kmax)
    o = {}
    for nm, Zr in (("E2avg", Za), ("E2w0", W0), ("E2w1", W1)):
        aA, aB = decompose(Zr)[1], decompose(Zb)[1]
        o[nm] = {"cell_binding": float(np.mean([r_(Zr[i, 2], Zb[i, 2]) for i in range(8)])),
                 "cell_relational": float(np.mean([r_(Zr[i, 3], Zb[i, 3]) for i in range(8)])),
                 "sig_pooled": r_(aA, aB), "sig_own_mean": float(np.mean([r_(aA[i], aB[i]) for i in range(8)]))}
    o["E2b_task_pairs"] = dev_pairs(Zb)
    o["E2b_back_half_mean_binding_relational"] = [float(Zb[:, 2, 5:].mean()), float(Zb[:, 3, 5:].mean())]
    out[tag] = o
out["E2_task_pairs"] = dev_pairs(Za)
out["E2w0_task_pairs"] = dev_pairs(W0); out["E2w1_task_pairs"] = dev_pairs(W1)
json.dump(out, open(os.path.join(OUT, "skeptic_leaf_k6.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
