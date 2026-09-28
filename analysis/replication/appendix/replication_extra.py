"""Extra anchors for main-text clauses: strongest-spreading decile, Qwen2.5-14B, back-half slope per step, accuracy.
Writes replication_extra.json."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys
sys.dont_write_bytecode = True
import os, json, itertools, math
import numpy as np
from scipy.stats import t as tdist
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
def q10(C):
    C = C.astype(np.float64); P = C / C.sum(-1, keepdims=True); m = max(1, int(np.ceil(0.1 * C.shape[-1])))
    return -np.sort(-P, -1)[..., :m].sum(-1)
def tci(v):
    v = np.asarray(v, float); h = tdist.ppf(.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)); return [float(v.mean() - h), float(v.mean() + h)]
allr = e2b.load_results(str(paths.E2B))
RB = {(r["model"].split("/")[-1], r["family"], r["mode"]): r for r in allr}
CQ = {r["model"]: r for r in json.load(open(f"{SP8}/consist.json")) if r["stat"] == "q10"}
Zb, Za, slope = np.zeros((8, 4, 10)), np.zeros((8, 4, 10)), np.zeros((8, 4))
for i, m in enumerate(M8):
    for j, t in enumerate(T):
        r = RB[(m, t, "direct")]; C = r["C"]; k = np.array([x["k"] for x in r["items"]], float)
        keep = ~((C.max(-1) / np.maximum(np.median(C, -1), 1e-9)) > 50).any(1) if m.startswith("OLMo") else np.ones(len(k), bool)
        b, sd, z = zprof(q10(C[keep]), k[keep], within=False)
        Zb[i, j] = dec_prof(z); Za[i, j] = dec_prof(np.array(CQ[m]["families"][t]["z"]))
        L = len(z); x = np.arange(L) / (L - 1); slope[i, j] = 100 * b[x >= .5].mean()
res = {}
amb, ama = Zb.mean(1).argmin(1), Za.mean(1).argmin(1)
P = np.array(list(itertools.permutations(range(8)))); null = (amb[None, :] == ama[P]).sum(1)
res["strongest_spreading_decile"] = {"replication": amb.tolist(), "main": ama.tolist(), "n_same": int((amb == ama).sum()),
                                     "expected_label_perm": float(null.mean()), "exact_p": float(np.mean(null >= (amb == ama).sum()))}
aB = decompose(Zb)[1]; aA = decompose(Za)[1]
res["Q14_vs_Q7_signature_r"] = {"replication": float(np.corrcoef(aB[4], aB[2])[0, 1]), "main_same8": float(np.corrcoef(aA[4], aA[2])[0, 1])}
per_model = slope.mean(1)
res["back_half_share_slope_pp_per_step"] = {"mean": float(per_model.mean()), "t_ci": tci(per_model), "per_model": dict(zip(M8, per_model.round(3).tolist()))}
acc = {}
for mode in ("direct", "stepwise"):
    pm_all, pm_k3 = [], []
    for m in M8:
        a_all, a_k3 = [], []
        for t in T:
            its = RB[(m, t, mode)]["items"]; ks = sorted({x["k"] for x in its})
            byk = {k: np.mean([x["correct"] for x in its if x["k"] == k]) for k in ks}
            a_all.append(np.mean(list(byk.values()))); a_k3.append(np.mean([v for k, v in byk.items() if k >= 3]))
        pm_all.append(np.mean(a_all)); pm_k3.append(np.mean(a_k3))
    acc[mode] = {"overall": float(np.mean(pm_all)), "k_ge_3": float(np.mean(pm_k3)), "per_model_k_ge_3": dict(zip(M8, np.round(pm_k3, 3).tolist()))}
res["accuracy_written_answer"] = acc
# arithmetic k=1, direct
res["arith_k1_direct"] = float(np.mean([np.mean([x["correct"] for x in RB[(m, "arith", "direct")]["items"] if x["k"] == 1]) for m in M8]))
json.dump(res, open(f"{OUT}/replication_extra.json", "w"), indent=1)
print(json.dumps(res, indent=1))
