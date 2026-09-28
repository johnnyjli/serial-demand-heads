"""Skeptical checks on the proposed E2b additions.
A. E2b shares E2's wording w0 (pointer/arith identical templates; binding/relational w0 + question line).
   Does E2b agree more with E2-w0 than with E2-w1?  (signatures, per-cell profiles)
B. Argmin-decile agreement between runs vs a shared-profile baseline (other models' argmin).
C. Task-pair agreement of model deviations: bootstrap over models, E2 vs E2b.
D. Post-training: k=1 accuracy differences (McNemar on the identical E2b items); pair-level agreement across runs.
E. Lineage-weighted shared profile CIs (as the paper's paragraph uses lineage weighting).
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, json, itertools, math, numpy as np
from scipy.stats import binomtest, t as tdist
SP8 = str(paths.ANALYSIS)
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP8); sys.path.insert(0, f"{SP8}/vet3"); sys.path.insert(0, str(paths.REPLICATION))
from v2lib import zprof
from model_diff import dec_prof, decompose
import e2b
T = ["pointer", "arith", "binding", "relational"]
M8 = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
      "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
LIN = {"Llama-3.1-8B": "L8", "Llama-3.1-8B-Instruct": "L8", "Qwen2.5-7B": "Q7", "Qwen2.5-7B-Instruct": "Q7", "Qwen2.5-14B": "Q14",
       "OLMo-2-1124-7B": "O7", "OLMo-2-1124-7B-Instruct": "O7", "Mistral-Small-24B-Base-2501": "M24"}
def q10(C):
    C = C.astype(np.float64); P = C / C.sum(-1, keepdims=True); H = C.shape[-1]; m = max(1, int(np.ceil(0.1 * H)))
    return -np.sort(-P, -1)[..., :m].sum(-1)
def r_(x, y): return float(np.corrcoef(np.ravel(x), np.ravel(y))[0, 1])

runs = e2b.load_results(str(paths.E2B))
RB = {(r["model"].split("/")[-1], r["family"], r["mode"]): r for r in runs}
E2B = {}; ACC = {}
for m in M8:
    for t in T:
        r = RB[(m, t, "direct")]; C = r["C"]; k = np.array([x["k"] for x in r["items"]], float)
        keep = np.ones(len(k), bool)
        if m.startswith("OLMo"): keep = ~((C.max(-1) / np.maximum(np.median(C, -1), 1e-9)) > 50).any(1)
        E2B[(m, t)] = zprof(q10(C[keep]), k[keep], within=False)[2]
        ACC[(m, t)] = {x["id"]: (x["k"], bool(x["correct"])) for x in r["items"]}
Zb = np.array([[dec_prof(E2B[(m, t)]) for t in T] for m in M8])
out = {}

# ---------------- A. wording match
CC = json.load(open(f"{SP8}/vet3/cells.json"))
def e2w(w, q7i):
    D = {(r["model"], r["fam"]): np.array(r["z"], float) for r in CC if r["variant"] == "all" and r["fam"] in T and r["w"] == w
         and r["attn"] == (q7i if r["model"] == "Qwen2.5-7B-Instruct" else "eager")}
    return np.array([[dec_prof(D[(m, t)]) for t in T] for m in M8])
CQ = {r["model"]: r for r in json.load(open(f"{SP8}/consist.json")) if r["stat"] == "q10"}
Zavg = np.array([[dec_prof(np.array(CQ[m]["families"][t]["z"])) for t in T] for m in M8])
def compare(Za, Zc, models_idx=None):
    idx = list(range(len(M8))) if models_idx is None else models_idx
    A = Za[idx]; B = Zc[idx]
    aA, aB = decompose(A)[1], decompose(B)[1]
    own = [r_(aA[i], aB[i]) for i in range(len(idx))]
    X = np.array([[r_(aA[i], aB[j]) for j in range(len(idx))] for i in range(len(idx))])
    cell = [r_(A[i, j], B[i, j]) for i in range(len(idx)) for j in range(4)]
    dA = A - A.mean(0, keepdims=True); dB = B - B.mean(0, keepdims=True)
    per_task_cell = {t: float(np.mean([r_(A[i, j], B[i, j]) for i in range(len(idx))])) for j, t in enumerate(T)}
    return {"sig_pooled": r_(aA, aB), "sig_own_mean": float(np.mean(own)), "sig_own": [round(x, 3) for x in own],
            "n_own_best": int(sum(np.argmax(X[i]) == i for i in range(len(idx)))),
            "cell_mean": float(np.mean(cell)), "cell_per_task": per_task_cell, "deviation_pooled": r_(dA, dB)}
res = {}
for q7i in ("sdpa", "eager"):
    W0, W1 = e2w("w0", q7i), e2w("w1", q7i)
    for nm, idx in (("M8", None), ("M6", [i for i, m in enumerate(M8) if not m.startswith("Qwen2.5-7B")])):
        res[f"{q7i}|{nm}"] = {"E2b_vs_w0": compare(Zb, W0, idx), "E2b_vs_w1": compare(Zb, W1, idx), "w0_vs_w1": compare(W0, W1, idx),
                               "E2b_vs_avg(consist)": compare(Zb, Zavg, idx) if q7i == "sdpa" else None}
out["A_wording_match"] = res

# ---------------- B. argmin decile baseline
am_b = Zb.mean(1).argmin(1); am_a = Zavg.mean(1).argmin(1)
same = int((am_b == am_a).sum())
off = [int(am_b[i] == am_a[j]) for i in range(8) for j in range(8) if i != j]
p0 = float(np.mean(off))
# permutation: shuffle model labels of E2 argmins
P = np.array(list(itertools.permutations(range(8))))
null = (am_b[None, :] == am_a[P]).sum(1)
out["B_argmin"] = {"E2b_argmin": am_b.tolist(), "E2_argmin": am_a.tolist(), "n_same": same, "offdiag_match_rate": p0,
                   "expected_by_label_perm": float(null.mean()), "exact_perm_p": float(np.mean(null >= same))}

# ---------------- C. task-pair agreement bootstrap over models
def pair_stats(Z):
    d = Z - Z.mean(0, keepdims=True)
    pr = {}
    for (i, a), (j, b) in itertools.combinations(enumerate(T), 2): pr[f"{a}-{b}"] = r_(d[:, i], d[:, j])
    across = np.mean([pr["pointer-binding"], pr["pointer-relational"], pr["arith-binding"], pr["arith-relational"]])
    return pr["binding-relational"], pr["pointer-arith"], float(across)
rng = np.random.default_rng(0); B = 5000
bs = {"E2": [], "E2b": [], "diff_BR": []}
for _ in range(B):
    ii = rng.integers(0, 8, 8)
    if len(set(ii)) < 4: continue
    sa = pair_stats(Zavg[ii]); sb = pair_stats(Zb[ii])
    bs["E2"].append(sa[0] - sa[2]); bs["E2b"].append(sb[0] - sb[2]); bs["diff_BR"].append(sa[0] - sb[0])
ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
out["C_task_pairs"] = {"E2 (BR, PA, across)": pair_stats(Zavg), "E2b (BR, PA, across)": pair_stats(Zb),
                       "E2 BR-minus-across CI": ci(bs["E2"]), "E2b BR-minus-across CI": ci(bs["E2b"]),
                       "BR E2 minus E2b CI": ci(bs["diff_BR"]), "n_boot": len(bs["E2"])}

# ---------------- D. post-training k=1 accuracy (identical items), McNemar
PAIRS = [("Llama-3.1-8B", "Llama-3.1-8B-Instruct"), ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct")]
dres = {}
for b_, p_ in PAIRS:
    g = l = nb = 0; ab = ap = 0
    for t in T:
        for iid, (k, cb) in ACC[(b_, t)].items():
            if k != 1: continue
            cp = ACC[(p_, t)][iid][1]; nb += 1; ab += cb; ap += cp
            g += (cp and not cb); l += (cb and not cp)
    dres[f"{b_}->{p_}"] = {"n_k1": nb, "acc_base": ab / nb, "acc_post": ap / nb, "gained": g, "lost": l,
                           "mcnemar_p": float(binomtest(g, g + l, .5).pvalue) if g + l else 1.0}
R = json.load(open(os.path.join(OUT, "..", "e2b_direct_replication.json")))
out["D_posttraining"] = {"k1_accuracy": dres}

# ---------------- E. lineage-weighted CIs at deciles 4..8 (5 lineages), E2b and E2 same models
def lin_prof(Z):
    G = {}
    for i, m in enumerate(M8): G.setdefault(LIN[m], []).append(i)
    return np.array([Z[G[g]].mean((0, 1)) for g in G])
def tci(v):
    v = np.asarray(v, float); n = len(v); h = tdist.ppf(.975, n - 1) * v.std(ddof=1) / math.sqrt(n); return [round(float(v.mean() - h), 2), round(float(v.mean() + h), 2)]
e = {}
for nm, Z in (("E2b", Zb), ("E2", Zavg)):
    ZL = lin_prof(Z)
    e[nm] = {f"d{d}": {"mean": round(float(ZL[:, d].mean()), 2), "ci": tci(ZL[:, d])} for d in range(4, 10)}
    e[nm]["d5-7 mean"] = {"mean": round(float(ZL[:, 5:8].mean()), 2), "ci": tci(ZL[:, 5:8].mean(1))}
out["E_lineage_weighted"] = e

json.dump(out, open(os.path.join(OUT, "skeptic_checks.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
