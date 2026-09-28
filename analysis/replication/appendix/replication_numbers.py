"""Recompute the numbers of the replication appendix table from raw data, with the paper pipeline.

E2b (replication): direct mode, top-10% head share per item and layer, OLMo-2 exclusion (any head > 50x its
layer's median), v2lib.zprof with k shuffled across items, model_diff.dec_prof deciles.
Main run: consist.json stat q10 (wording-averaged items; Qwen2.5-7B-Instruct under SDPA); per-wording profiles
from vet3/cells.json (variant 'all', exact per-wording z; Qwen2.5-7B-Instruct under SDPA, others eager).
Signatures a_m = model_diff.decompose(Z)[1], centred on the eight models.
Writes replication_numbers.json (and the 8x8 matrices the figure uses).
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys
sys.dont_write_bytecode = True                       # never write __pycache__ into read-only folders
import os, json, itertools, math
import numpy as np
SP8 = str(paths.ANALYSIS)
OUT = os.path.dirname(os.path.abspath(__file__))
os.environ["VPRIM"] = "y10"
sys.argv = [sys.argv[0], SP8]
sys.path.insert(0, SP8); sys.path.insert(0, f"{SP8}/vet3"); sys.path.insert(0, str(paths.REPLICATION))
from v2lib import zprof
import model_diff as md
from model_diff import dec_prof, decompose, ss, perm_model, split_half, task_consistency
import e2b

T = ["pointer", "arith", "binding", "relational"]
M8 = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
      "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
LIN = {"Llama-3.1-8B": "L8", "Llama-3.1-8B-Instruct": "L8", "Qwen2.5-7B": "Q7", "Qwen2.5-7B-Instruct": "Q7", "Qwen2.5-14B": "Q14",
       "OLMo-2-1124-7B": "O7", "OLMo-2-1124-7B-Instruct": "O7", "Mistral-Small-24B-Base-2501": "M24"}
PAIRS = [(0, 1), (2, 3), (5, 6)]
PERMS = np.array(list(itertools.permutations(range(8))))


def q10(C):
    C = C.astype(np.float64); P = C / C.sum(-1, keepdims=True); m = max(1, int(np.ceil(0.1 * C.shape[-1])))
    return -np.sort(-P, -1)[..., :m].sum(-1)


def r_(x, y): return float(np.corrcoef(np.ravel(x), np.ravel(y))[0, 1])


# ── profiles ──────────────────────────────────────────────────────────────────
runs = [r for r in e2b.load_results(str(paths.E2B)) if r["mode"] == "direct"]
RB = {(r["model"].split("/")[-1], r["family"]): r for r in runs}
Zb = np.zeros((8, 4, 10)); nkeep = {}
for i, m in enumerate(M8):
    for j, t in enumerate(T):
        r = RB[(m, t)]; C = r["C"]; k = np.array([x["k"] for x in r["items"]], float)
        keep = np.ones(len(k), bool)
        if m.startswith("OLMo"): keep = ~((C.max(-1) / np.maximum(np.median(C, -1), 1e-9)) > 50).any(1)
        Zb[i, j] = dec_prof(zprof(q10(C[keep]), k[keep], within=False)[2]); nkeep[f"{m}|{t}"] = int(keep.sum())
CQ = {r["model"]: r for r in json.load(open(f"{SP8}/consist.json")) if r["stat"] == "q10"}
Za = np.array([[dec_prof(np.array(CQ[m]["families"][t]["z"])) for t in T] for m in M8])
CC = json.load(open(f"{SP8}/vet3/cells.json"))
def wprof(w, q7i="sdpa"):
    D = {(r["model"], r["fam"]): np.array(r["z"], float) for r in CC if r["variant"] == "all" and r["fam"] in T and r["w"] == w
         and r["attn"] == (q7i if r["model"] == "Qwen2.5-7B-Instruct" else "eager")}
    return np.array([[dec_prof(D[(m, t)]) for t in T] for m in M8])
W0, W1 = wprof("w0"), wprof("w1")
Za_eager = Za.copy()                                     # Qwen2.5-7B-Instruct from its eager capture (mean of wordings)
D = {(r["fam"], r["w"]): np.array(r["z"], float) for r in CC if r["variant"] == "all" and r["model"] == "Qwen2.5-7B-Instruct" and r["attn"] == "eager"}
for j, t in enumerate(T): Za_eager[3, j] = dec_prof(0.5 * (D[(t, "w0")] + D[(t, "w1")]))


def agreement(ZB, ZA):
    """replication signatures (rows) against main-run signatures (columns), both centred on the eight models"""
    aB, aA = decompose(ZB)[1], decompose(ZA)[1]
    X = np.array([[r_(aB[i], aA[j]) for j in range(8)] for i in range(8)])
    own = np.diag(X); other = X[~np.eye(8, dtype=bool)]
    diag_null = np.array([X[np.arange(8), p].mean() for p in PERMS])           # relabel the main-run models
    pooled_null = np.array([r_(aB, aA[p]) for p in PERMS])
    pooled = r_(aB, aA)
    return {"X": X.round(4).tolist(), "own": own.round(4).tolist(), "mean_own": float(own.mean()), "mean_other": float(other.mean()),
            "pooled": pooled, "n_own_best": int(sum(np.argmax(X[i]) == i for i in range(8))),
            "row_best": [M8[int(np.argmax(X[i]))] for i in range(8)],
            "n_own_gt_mean_others": int(sum(X[i, i] > np.delete(X[i], i).mean() for i in range(8))),
            "exact_p_mean_own": float(np.mean(diag_null >= own.mean() - 1e-12)),
            "exact_p_pooled": float(np.mean(pooled_null >= pooled - 1e-12)), "n_perms": len(PERMS)}, aB, aA


def within(Z):
    md.rng = np.random.default_rng(0)
    S = ss(Z); tot = sum(S.values()); ns = tot - S["shared"]
    obs, nullm, p = perm_model(Z)
    sh = split_half(Z)
    a = decompose(Z)[1]; R = np.corrcoef(a)
    lin = [float(R[i, j]) for i, j in PAIRS]
    diff = [float(R[i, j]) for i in range(8) for j in range(i + 1, 8) if LIN[M8[i]] != LIN[M8[j]]]
    # exact lineage test: relabel models, mean r over the three lineage pairs
    nullL = np.array([np.mean([R[p[i], p[j]] for i, j in PAIRS]) for p in PERMS])
    return {"model_share": S["model"] / ns, "model_share_chance": nullm / ns, "p_model": p,
            "task_share": S["task"] / ns, "resid_share": S["resid"] / ns, "shared_share": S["shared"] / tot,
            "split_half": sh[0], "split_half_null": sh[1], "p_split_half": sh[2],
            "lineage_pairs": dict(zip(["Llama-3.1-8B", "Qwen2.5-7B", "OLMo-2-7B"], lin)), "lineage_mean": float(np.mean(lin)),
            "different_run_mean": float(np.mean(diff)), "n_different_run_pairs": len(diff), "centring_baseline": -1 / 7,
            "lineage_exact_p": float(np.mean(nullL >= np.mean(lin) - 1e-12)),
            "same_model_cross_task_r": float(np.mean(task_consistency(Z)[0])), "diff_model_same_task_r": task_consistency(Z)[2]}


res = {"models": M8, "n_items_kept": nkeep}
res["vs_main_averaged"], aB, aA = agreement(Zb, Za)
res["vs_main_w0"], _, _ = agreement(Zb, W0)
res["vs_main_w1"], _, _ = agreement(Zb, W1)
res["main_w0_vs_w1"], _, _ = agreement(W0, W1)
res["vs_main_averaged_q7i_eager"], _, _ = agreement(Zb, Za_eager)
res["within_main"] = within(Za)
res["within_replication"] = within(Zb)
res["mu_r"] = r_(Za.mean((0, 1)), Zb.mean((0, 1)))
cell = np.array([[r_(Za[i, j], Zb[i, j]) for j in range(4)] for i in range(8)])
res["cell_r_mean"] = float(cell.mean()); res["cell_r_range"] = [float(cell.min()), float(cell.max())]
res["cell_r_other_model_same_task"] = float(np.mean([r_(Za[i, j], Zb[i2, j]) for j in range(4) for i in range(8) for i2 in range(8) if i2 != i]))
res["a_replication"] = aB.round(4).tolist(); res["a_main"] = aA.round(4).tolist()
json.dump(res, open(f"{OUT}/replication_numbers.json", "w"), indent=1)

f = lambda x: f"{x:.3f}"
for key in ("vs_main_averaged", "vs_main_w0", "vs_main_w1", "main_w0_vs_w1", "vs_main_averaged_q7i_eager"):
    d = res[key]
    print(f"{key:28s} pooled {f(d['pooled'])}  own mean {f(d['mean_own'])}  other {f(d['mean_other'])}  own best {d['n_own_best']}/8  "
          f"own>others {d['n_own_gt_mean_others']}/8  p_own {d['exact_p_mean_own']:.2e}  p_pooled {d['exact_p_pooled']:.2e}")
    print(" " * 30 + "own: " + " ".join(f"{x:.3f}" for x in d["own"]))
for key in ("within_main", "within_replication"):
    d = res[key]
    print(f"{key:20s} share {f(d['model_share'])} (chance {f(d['model_share_chance'])}, p {d['p_model']:.4f})  split-half {f(d['split_half'])} "
          f"(null {d['split_half_null']:.3f}, p {d['p_split_half']:.4f})  lineage {d['lineage_pairs']} mean {f(d['lineage_mean'])}  "
          f"other {f(d['different_run_mean'])} ({d['n_different_run_pairs']} pairs)  lineage exact p {d['lineage_exact_p']:.4f}  "
          f"same-model cross-task {f(d['same_model_cross_task_r'])} vs diff-model same-task {f(d['diff_model_same_task_r'])}")
print("mu r", f(res["mu_r"]), " cell r mean", f(res["cell_r_mean"]), res["cell_r_range"], " other-model", f(res["cell_r_other_model_same_task"]))
print("Llama-3.1-8B row:", res["vs_main_averaged"]["X"][0])
