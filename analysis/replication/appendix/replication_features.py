"""Recompute the per-feature replication checks that main-text clauses may cite (same pipeline as replication_numbers.py):
final-layer family pattern, 50-70% signature, projection on the paper's 17-model PC1, back-half mean z, lineage-weighted
shared profile, post-training change, task-pair agreement.  Writes replication_features.json."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys
sys.dont_write_bytecode = True
import os, json, itertools, math
import numpy as np
from scipy.stats import t as tdist, spearmanr
SP8 = str(paths.ANALYSIS)
OUT = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], SP8]
sys.path.insert(0, SP8); sys.path.insert(0, f"{SP8}/vet3"); sys.path.insert(0, str(paths.REPLICATION))
from v2lib import zprof
from model_diff import dec_prof, decompose
import e2b
T = ["pointer", "arith", "binding", "relational"]
M8 = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
      "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
LIN = ["L8", "L8", "Q7", "Q7", "Q14", "O7", "O7", "M24"]
PAIRS = [(0, 1), (2, 3), (5, 6)]
PERMS = np.array(list(itertools.permutations(range(8))))
def q10(C):
    C = C.astype(np.float64); P = C / C.sum(-1, keepdims=True); m = max(1, int(np.ceil(0.1 * C.shape[-1])))
    return -np.sort(-P, -1)[..., :m].sum(-1)
def r_(x, y): return float(np.corrcoef(np.ravel(x), np.ravel(y))[0, 1])
def exact_pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); obs = r_(x, y)
    null = np.array([r_(x, y[p]) for p in PERMS]); return obs, float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
def tci(v):
    v = np.asarray(v, float); h = tdist.ppf(.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)); return [float(v.mean() - h), float(v.mean() + h)]

runs = [r for r in e2b.load_results(str(paths.E2B)) if r["mode"] == "direct"]
RB = {(r["model"].split("/")[-1], r["family"]): r for r in runs}
zb, za = {}, {}
CQ = {r["model"]: r for r in json.load(open(f"{SP8}/consist.json")) if r["stat"] == "q10"}
for m in M8:
    for t in T:
        r = RB[(m, t)]; C = r["C"]; k = np.array([x["k"] for x in r["items"]], float)
        keep = ~((C.max(-1) / np.maximum(np.median(C, -1), 1e-9)) > 50).any(1) if m.startswith("OLMo") else np.ones(len(k), bool)
        zb[(m, t)] = zprof(q10(C[keep]), k[keep], within=False)[2]
        za[(m, t)] = np.array(CQ[m]["families"][t]["z"], float)
Zb = np.array([[dec_prof(zb[(m, t)]) for t in T] for m in M8]); Za = np.array([[dec_prof(za[(m, t)]) for t in T] for m in M8])
res = {}
# final layer: task-mean z of the last layer; Qwen2.5 negative, others positive?
for nm, Dz in (("replication", zb), ("main", za)):
    fl = {m: float(np.mean([Dz[(m, t)][-1] for t in T])) for m in M8}
    res[f"final_layer_{nm}"] = {m: round(v, 2) for m, v in fl.items()}
    res[f"final_layer_{nm}_matches_family"] = all((v < 0) == m.startswith("Qwen") for m, v in fl.items())
isq = np.array([m.startswith("Qwen") for m in M8])
res["final_layer_label_perm_p"] = float(np.mean([np.array_equal(isq[p], isq) for p in PERMS]))   # 3!5!/8! = 1/56
# 50-70% signature and back half, across runs
s57b, s57a = Zb[:, :, 5:7].mean((1, 2)), Za[:, :, 5:7].mean((1, 2))
res["z50_70"] = {"main": dict(zip(M8, s57a.round(2).tolist())), "replication": dict(zip(M8, s57b.round(2).tolist())),
                 "pearson_exact": exact_pearson(s57a, s57b), "spearman": float(spearmanr(s57a, s57b).correlation)}
bb, ba = Zb[:, :, 5:].mean((1, 2)), Za[:, :, 5:].mean((1, 2))
res["back_half_dec"] = {"main": dict(zip(M8, ba.round(2).tolist())), "replication": dict(zip(M8, bb.round(2).tolist())),
                        "pearson_exact": exact_pearson(ba, bb)}
MDJ = json.load(open(f"{SP8}/model_diff.json")); v1 = np.array(MDJ["pca"]["loadings"][0])
paper = np.array([MDJ["pca"]["scores"][m][0] for m in M8]); pB = decompose(Zb)[1] @ v1
res["paper_pc1_projection"] = {"replication": dict(zip(M8, pB.round(2).tolist())), "paper_scores": dict(zip(M8, paper.round(2).tolist())),
                               "pearson_exact": exact_pearson(paper, pB), "spearman": float(spearmanr(paper, pB).correlation),
                               "post_further_than_base": [bool(pB[j] > pB[i]) for i, j in PAIRS],
                               "loadings_r_with_50_70": None}
# lineage-weighted shared profile at 40-80%
def lin(Z):
    g = {}
    for i, l in enumerate(LIN): g.setdefault(l, []).append(i)
    return np.array([Z[idx].mean((0, 1)) for idx in g.values()])
for nm, Z in (("main", Za), ("replication", Zb)):
    ZL = lin(Z)
    res[f"lineage_weighted_{nm}"] = {"d4": [float(ZL[:, 4].mean()), tci(ZL[:, 4])], "d5": [float(ZL[:, 5].mean()), tci(ZL[:, 5])],
                                     "d6": [float(ZL[:, 6].mean()), tci(ZL[:, 6])], "d7": [float(ZL[:, 7].mean()), tci(ZL[:, 7])],
                                     "d5_7": [float(ZL[:, 5:8].mean()), tci(ZL[:, 5:8].mean(1))]}
    per = Z.mean(1); back = Z[:, :, 5:].mean((1, 2)); front = Z[:, :, :5].mean((1, 2))
    res[f"shared_{nm}"] = {"back_lt0": int((back < 0).sum()), "back_lt_front": int((back < front).sum()),
                           "d4_gt_d5": int((per[:, 4] > per[:, 5]).sum())}
res["mu_r"] = r_(Za.mean((0, 1)), Zb.mean((0, 1)))
# post-training: change in back-half mean z over layers with l/(L-1) >= 0.5, averaged over the four tasks
def backlay(Dz, m):
    L = len(Dz[(m, T[0])]); x = np.arange(L) / (L - 1); return float(np.mean([Dz[(m, t)][x >= .5].mean() for t in T]))
res["posttraining_layer_based"] = {f"{M8[i]} -> {M8[j]}": {"main": round(backlay(za, M8[j]) - backlay(za, M8[i]), 3),
                                                            "replication": round(backlay(zb, M8[j]) - backlay(zb, M8[i]), 3)} for i, j in PAIRS}
# task-pair agreement of within-task model deviations
def pairs(Z):
    d = Z - Z.mean(0, keepdims=True); pr = {f"{T[a]}-{T[b]}": r_(d[:, a], d[:, b]) for a, b in itertools.combinations(range(4), 2)}
    across = np.mean([pr["pointer-binding"], pr["pointer-relational"], pr["arith-binding"], pr["arith-relational"]])
    return {"binding-relational": pr["binding-relational"], "pointer-arith": pr["pointer-arith"], "across_mean": float(across)}
res["task_pairs"] = {"main": pairs(Za), "replication": pairs(Zb)}
# per-task back-half mean z (replication; one null for all tasks)
res["per_task_back_replication"] = {t: [float(Zb[:, j, 5:].mean()), tci(Zb[:, j, 5:].mean(1)), int((Zb[:, j, 5:].mean(1) < 0).sum())] for j, t in enumerate(T)}
json.dump(res, open(f"{OUT}/replication_features.json", "w"), indent=1)
print(json.dumps(res, indent=1))
