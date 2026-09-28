"""Does E2b DIRECT mode (fresh items, one wording, 8 worked examples covering all depths, 30 items per depth)
replicate the paper's E2 findings for the 8 models run in both?  Primary statistic: top-10% head share.

Pipeline (reused from the paper): per-layer OLS slope of the statistic on k standardised by its exact
permutation-null SD (v2lib.zprof; k shuffled across items for every family, because E2b binding/relational
bodies are not shared across k), decile profiles l/(L-1) (model_diff.dec_prof), decomposition
Z = mu + a_m + b_t + e (model_diff.decompose / ss / perm_model / split_half / task_consistency).
OLMo-2: drop items where any head in any layer exceeds 50x its layer's median head norm.
E2 profiles: consist.json, stat q10 (wording-averaged items; Qwen2.5-7B-Instruct under SDPA), plus an
eager-kernel variant for Qwen2.5-7B-Instruct from vet3/cells.json (per-wording exact z, averaged).
Everything is run for the 8 models and again without Qwen2.5-7B and Qwen2.5-7B-Instruct."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, json, itertools, math, numpy as np
from scipy.stats import spearmanr, t as tdist
SP8 = str(paths.ANALYSIS)
OUT = os.path.dirname(os.path.abspath(__file__))
os.environ["VPRIM"] = "y10"
sys.path.insert(0, SP8); sys.path.insert(0, f"{SP8}/vet3"); sys.path.insert(0, str(paths.REPLICATION))
from v2lib import zprof, load as vload
import model_diff as md
from model_diff import dec_prof, decompose, ss, perm_model, split_half, task_consistency
import e2b
md.rng = np.random.default_rng(0)
T = ["pointer", "arith", "binding", "relational"]
M8 = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
      "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
M6 = [m for m in M8 if not m.startswith("Qwen2.5-7B")]
LIN = {"Llama-3.1-8B": "L8", "Llama-3.1-8B-Instruct": "L8", "Qwen2.5-7B": "Q7", "Qwen2.5-7B-Instruct": "Q7", "Qwen2.5-14B": "Q14",
       "OLMo-2-1124-7B": "O7", "OLMo-2-1124-7B-Instruct": "O7", "Mistral-Small-24B-Base-2501": "M24"}
PAIRS = [("Llama-3.1-8B", "Llama-3.1-8B-Instruct"), ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct")]
SHORT = {"Llama-3.1-8B": "L8B", "Llama-3.1-8B-Instruct": "L8B-I", "Qwen2.5-7B": "Q7B", "Qwen2.5-7B-Instruct": "Q7B-I", "Qwen2.5-14B": "Q14B",
         "OLMo-2-1124-7B": "O7B", "OLMo-2-1124-7B-Instruct": "O7B-I", "Mistral-Small-24B-Base-2501": "M24B"}
ND = 10

def q10(C):
    C = C.astype(np.float64); P = C / C.sum(-1, keepdims=True); H = C.shape[-1]; m = max(1, int(np.ceil(0.1 * H)))
    return -np.sort(-P, -1)[..., :m].sum(-1)                     # [items, L]

# ---------------------------------------------------------------- E2b direct
runs = [r for r in e2b.load_results(str(paths.E2B)) if r["mode"] == "direct"]
RB = {(r["model"].split("/")[-1], r["family"]): r for r in runs}
E2B, info = {}, {}
for m in M8:
    for t in T:
        r = RB[(m, t)]; C = r["C"]; k = np.array([x["k"] for x in r["items"]], float)
        keep = np.ones(len(k), bool)
        if m.startswith("OLMo"):
            keep = ~((C.max(-1) / np.maximum(np.median(C, -1), 1e-9)) > 50).any(1)
        Y = q10(C[keep]); kk = k[keep]
        b, sd, z = zprof(Y, kk, within=False)
        rel = 100 * b / np.abs(Y.mean(0))
        its = [x for x, kp in zip(r["items"], keep) if kp]
        E2B[(m, t)] = {"z": z, "rel": rel, "Y": Y, "k": kk, "ids": [x["id"] for x in its], "L": r["L"], "H": r["H"],
                       "strict": np.array([x["correct"] and x["correct_constrained"] for x in r["items"]]),
                       "written": np.array([x["correct"] for x in r["items"]]), "kall": k}
        info[(m, t)] = {"n": int(keep.sum()), "dropped": int((~keep).sum())}
def prof(D, key="z", models=M8): return np.array([[dec_prof(D[(m, t)][key]) for t in T] for m in models])

# ---------------------------------------------------------------- E2 profiles
CQ = {r["model"]: r for r in json.load(open(f"{SP8}/consist.json")) if r["stat"] == "q10"}
E2 = {(m, t): {"z": np.array(CQ[m]["families"][t]["z"]), "rel": np.array(CQ[m]["families"][t]["rel"])} for m in M8 for t in T}
CE = {(r["model"], r["fam"], r["w"]): np.array(r["z"], float) for r in json.load(open(f"{SP8}/vet3/cells.json"))
      if r["attn"] == "eager" and r["variant"] == "all" and r["fam"] in T}
E2eag = dict(E2)                                                # Qwen2.5-7B-Instruct replaced by its eager-kernel profile
for t in T: E2eag[("Qwen2.5-7B-Instruct", t)] = {"z": 0.5 * (CE[("Qwen2.5-7B-Instruct", t, "w0")] + CE[("Qwen2.5-7B-Instruct", t, "w1")])}
MDJ = json.load(open(f"{SP8}/model_diff.json")); MD2 = json.load(open(f"{SP8}/model_diff2.json"))["features"]

# ---------------------------------------------------------------- helpers
def r_(x, y): return float(np.corrcoef(np.ravel(x), np.ravel(y))[0, 1])
def tci(v):
    v = np.asarray(v, float); n = len(v); h = tdist.ppf(.975, n - 1) * v.std(ddof=1) / math.sqrt(n); return [float(v.mean() - h), float(v.mean() + h)]
def exact_perm_spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    obs = spearmanr(x, y).correlation; P = np.array(list(itertools.permutations(range(len(x)))))
    rp = ry[P]; rxc = rx - rx.mean(); rpc = rp - rp.mean(1, keepdims=True)
    null = (rpc @ rxc) / (np.sqrt((rpc ** 2).sum(1)) * np.sqrt((rxc ** 2).sum()))
    return float(obs), float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
def exact_perm_pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); obs = r_(x, y); P = np.array(list(itertools.permutations(range(len(x)))))
    yp = y[P]; xc = x - x.mean(); ypc = yp - yp.mean(1, keepdims=True)
    null = (ypc @ xc) / (np.sqrt((ypc ** 2).sum(1)) * np.sqrt((xc ** 2).sum()))
    return float(obs), float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
def lineage_groups(models):
    g = {}
    for m in models: g.setdefault(LIN[m], []).append(m)
    return g

def analyze(models, Zb, Za, Rb=None, Ra=None, label=""):
    """Zb: E2b profiles [M,4,10]; Za: E2 profiles on the same models.  Rb/Ra: relative-slope profiles."""
    n = len(models); res = {"models": models}
    # (1) per-cell agreement
    cell = np.array([[r_(Za[i, j], Zb[i, j]) for j in range(4)] for i in range(n)])
    other = [r_(Za[i, j], Zb[i2, j]) for j in range(4) for i in range(n) for i2 in range(n) if i2 != i]
    dA = Za - Za.mean(0, keepdims=True); dB = Zb - Zb.mean(0, keepdims=True)       # model deviation within each task (a + e)
    eA, eB = decompose(Za)[3], decompose(Zb)[3]
    res["1_cells"] = {"per_cell_r_mean": float(cell.mean()), "per_cell_r_min": float(cell.min()), "per_cell_r_max": float(cell.max()),
                      "per_cell_r_median": float(np.median(cell)), "n_cells_r_gt0": int((cell > 0).sum()), "n_cells": int(cell.size),
                      "per_cell": {m: dict(zip(T, cell[i].round(3).tolist())) for i, m in enumerate(models)},
                      "per_model_mean": {m: float(cell[i].mean()) for i, m in enumerate(models)},
                      "per_task_mean": {t: float(cell[:, j].mean()) for j, t in enumerate(T)},
                      "pooled_r_all_values": r_(Za, Zb),
                      "baseline_diff_model_same_task_r": float(np.mean(other)),
                      "within_task_deviation_pooled_r": r_(dA, dB),
                      "within_task_deviation_per_task_r": {t: r_(dA[:, j], dB[:, j]) for j, t in enumerate(T)},
                      "interaction_e_pooled_r": r_(eA, eB)}
    # (2) shared profile
    out2 = {}
    for nm, Z in (("E2", Za), ("E2b", Zb)):
        mu = Z.mean((0, 1)); G = lineage_groups(models); ZL = np.array([Z[[models.index(x) for x in G[g]]].mean((0, 1)) for g in G])
        per_model = Z.mean(1)                                              # [M,10] task-averaged profile
        back = Z[:, :, 5:].mean((1, 2)); front = Z[:, :, :5].mean((1, 2))
        rng = np.random.default_rng(1); bt = np.array([ZL[rng.integers(0, len(ZL), len(ZL))].mean(0) for _ in range(5000)])
        out2[nm] = {"mu_models": mu.round(3).tolist(), "mu_lineages": ZL.mean(0).round(3).tolist(),
                    "lineage_boot_ci": np.quantile(bt, [.025, .975], 0).round(3).tolist(),
                    "model_t_ci_d4": tci(per_model[:, 4]), "model_t_ci_d5_7": [tci(per_model[:, d]) for d in (5, 6, 7)],
                    "lineage_t_ci_d4": tci(ZL[:, 4]), "lineage_t_ci_d5_7": [tci(ZL[:, d]) for d in (5, 6, 7)],
                    "d4_per_task_mean": {t: float(Z[:, j, 4].mean()) for j, t in enumerate(T)},
                    "d4_positive_models": int((per_model[:, 4] > 0).sum()),
                    "d5_7_mean_per_model": {m: float(per_model[i, 5:8].mean()) for i, m in enumerate(models)},
                    "d5_7_negative_models": int((per_model[:, 5:8].mean(1) < 0).sum()),
                    "back_lt_front_models": int((back < front).sum()), "back_lt0_models": int((back < 0).sum()),
                    "back_lt_front_lineages": int(sum(Z[[models.index(x) for x in G[g]]][:, :, 5:].mean() < Z[[models.index(x) for x in G[g]]][:, :, :5].mean() for g in G)),
                    "n_lineages": len(G),
                    "d4_gt_d5_models": int((per_model[:, 4] > per_model[:, 5]).sum()),
                    "d4pos_d5neg_models": int(((per_model[:, 4] > 0) & (per_model[:, 5] < 0)).sum()),
                    "back_mean_z": {m: float(back[i]) for i, m in enumerate(models)}, "front_mean_z": {m: float(front[i]) for i, m in enumerate(models)},
                    "back_mean_z_per_task": {t: float(Z[:, j, 5:].mean()) for j, t in enumerate(T)},
                    "back_lt0_per_task": {t: int((Z[:, j, 5:].mean(1) < 0).sum()) for j, t in enumerate(T)}}
    ba = np.array(list(out2["E2"]["back_mean_z"].values())); bb = np.array(list(out2["E2b"]["back_mean_z"].values()))
    out2["back_mean_z_r_pearson_exactp"] = exact_perm_pearson(ba, bb); out2["back_mean_z_r_spearman_exactp"] = exact_perm_spearman(ba, bb)
    out2["mu_r_E2_vs_E2b"] = r_(out2["E2"]["mu_models"], out2["E2b"]["mu_models"])
    if Rb is not None:
        rba = Ra[:, :, 5:].mean((1, 2)); rbb = Rb[:, :, 5:].mean((1, 2))
        out2["rel_slope"] = {"mu_E2": Ra.mean((0, 1)).round(4).tolist(), "mu_E2b": Rb.mean((0, 1)).round(4).tolist(),
                             "back_mean_rel_E2": dict(zip(models, rba.round(4).tolist())), "back_mean_rel_E2b": dict(zip(models, rbb.round(4).tolist())),
                             "back_rel_r_pearson_exactp": exact_perm_pearson(rba, rbb), "back_rel_r_spearman_exactp": exact_perm_spearman(rba, rbb)}
    res["2_shared"] = out2
    # (3) signatures across datasets
    aA, aB = decompose(Za)[1], decompose(Zb)[1]
    X = np.array([[r_(aB[i], aA[j]) for j in range(n)] for i in range(n)])        # rows: E2b model, cols: E2 model
    own = np.diag(X); best = X.argmax(1)
    P = np.array(list(itertools.permutations(range(n)))); null = X[np.arange(n)[None], P].mean(1)
    pooled_null = np.array([r_(aB, aA[p]) for p in P])
    res["3_signatures"] = {"per_model_r_own": {m: float(own[i]) for i, m in enumerate(models)}, "mean_own_r": float(own.mean()),
                           "mean_other_r": float(X[~np.eye(n, dtype=bool)].mean()),
                           "pooled_r": r_(aA, aB), "pooled_r_exact_perm_p": float(np.mean(pooled_null >= r_(aA, aB) - 1e-12)),
                           "n_best_match_self": int((best == np.arange(n)).sum()),
                           "best_match": {m: models[best[i]] for i, m in enumerate(models)},
                           "n_best_match_same_lineage": int(sum(LIN[models[best[i]]] == LIN[m] for i, m in enumerate(models))),
                           "rank_of_own": {m: int((X[i] > X[i, i]).sum() + 1) for i, m in enumerate(models)},
                           "n_own_gt_mean_others": int(sum(X[i, i] > np.mean(np.delete(X[i], i)) for i in range(n))),
                           "exact_perm_p_mean_diag": float(np.mean(null >= own.mean() - 1e-12)), "perm_null_mean_diag": float(null.mean()),
                           "n_perms": int(len(P)), "X": X.round(3).tolist(),
                           "a_E2": {m: aA[i].round(3).tolist() for i, m in enumerate(models)}, "a_E2b": {m: aB[i].round(3).tolist() for i, m in enumerate(models)}}
    # (4) within-dataset structure, E2b and E2 (same models) side by side
    out4 = {}
    for nm, Z in (("E2b", Zb), ("E2", Za)):
        md.rng = np.random.default_rng(0)
        S = ss(Z); tot = sum(S.values()); ns_ = tot - S["shared"]
        o, nm_, p = perm_model(Z, 5000); sh = split_half(Z); within, between, stb = task_consistency(Z)
        dev = Z - Z.mean(0, keepdims=True)
        pair = {f"{T[i]}-{T[j]}": r_(dev[:, i], dev[:, j]) for i, j in itertools.combinations(range(4), 2)}
        cross = [pair[k_] for k_ in ("pointer-binding", "pointer-relational", "arith-binding", "arith-relational")]
        spl = {}
        for Sx in [(0, 1), (0, 2), (0, 3)]:
            Sc = tuple(j for j in range(4) if j not in Sx)
            A1 = Z[:, Sx].mean(1); A1 = A1 - A1.mean(0); A2 = Z[:, Sc].mean(1); A2 = A2 - A2.mean(0)
            spl["+".join(T[j] for j in Sx) + " vs " + "+".join(T[j] for j in Sc)] = r_(A1, A2)
        a = decompose(Z)[1]; R = np.corrcoef(a)
        lin_pairs = {f"{SHORT[x]}/{SHORT[y]}": float(R[models.index(x), models.index(y)]) for x, y in PAIRS if x in models and y in models}
        diff = [R[i, j] for i, j in itertools.combinations(range(n), 2) if LIN[models[i]] != LIN[models[j]]]
        def lstat(labs): return np.mean([R[i, j] for i, j in itertools.combinations(range(n), 2) if LIN[labs[i]] == LIN[labs[j]]])
        lo_ = lstat(models); nl = np.array([lstat([models[q] for q in pp]) for pp in P])
        # PCA
        U, s_, Vt = np.linalg.svd(a - a.mean(0), full_matrices=False); ev = s_ ** 2 / (s_ ** 2).sum(); sc = U * s_
        b5070 = a[:, 5:7].mean(1)
        if np.corrcoef(sc[:, 0], b5070)[0, 1] > 0: sc[:, 0] *= -1; Vt[0] *= -1           # orient: high PC1 = more spreading at 50-70%
        out4[nm] = {"ss_frac": {k_: float(v / tot) for k_, v in S.items()},
                    "ss_frac_nonshared": {k_: float(S[k_] / ns_) for k_ in ("model", "task", "resid")},
                    "model_share_chance": float(nm_ / ns_), "p_model": p, "model_ss_ratio_to_null": float(o / nm_),
                    "split_half_r": sh[0], "split_half_null": sh[1], "p_split_half": sh[2], "splits": spl,
                    "split_half_per_model": dict(zip(models, np.round(sh[3], 3).tolist())),
                    "within_model_cross_task_r": dict(zip(models, np.round(within, 3).tolist())), "within_model_cross_task_r_mean": float(np.mean(within)),
                    "between_model_same_task_r": stb, "between_model_cross_task_r": between,
                    "task_pairs": pair, "binding-relational": pair["binding-relational"], "pointer-arith": pair["pointer-arith"],
                    "across_types_mean": float(np.mean(cross)), "across_types_range": [float(min(cross)), float(max(cross))],
                    "lineage_pairs": lin_pairs, "lineage_mean": float(lo_), "different_run_mean": float(np.mean(diff)), "n_different_run_pairs": len(diff),
                    "centring_baseline": -1 / (n - 1), "lineage_exact_perm_p": float(np.mean(nl >= lo_ - 1e-12)),
                    "pca_explained": ev[:4].round(3).tolist(), "pc1_loadings": Vt[0].round(3).tolist(), "pc1_scores": dict(zip(models, sc[:, 0].round(3).tolist())),
                    "pc1_r_with_sig_50_70": r_(sc[:, 0], b5070)}
    out4["pc1_scores_r_E2_vs_E2b"] = exact_perm_pearson(list(out4["E2"]["pc1_scores"].values()), list(out4["E2b"]["pc1_scores"].values()))
    res["4_within"] = out4
    return res

# ---------------------------------------------------------------- run both model sets
RES = {"info": {f"{m}|{t}": v for (m, t), v in info.items()}}
for tag, models in (("M8", M8), ("M6_noQwen7B", M6)):
    Zb = prof(E2B, "z", models); Za = prof(E2, "z", models); Zae = prof(E2eag, "z", models)
    Rb = prof(E2B, "rel", models); Ra = prof(E2, "rel", models)
    RES[tag] = analyze(models, Zb, Za, Rb, Ra, tag)
    if "Qwen2.5-7B-Instruct" in models:
        RES[tag + "_E2_Q7I_eager"] = analyze(models, Zb, Zae, None, None, tag + " (E2 Q7I eager)")

# ---------------------------------------------------------------- (5) post-training
def backz_dec(D, m): return float(np.mean([dec_prof(D[(m, t)]["z"])[5:].mean() for t in T]))
def backz_lay(D, m, L): x = np.arange(L) / (L - 1); return float(np.mean([D[(m, t)]["z"][x >= .5].mean() for t in T]))
def backz_dec_task(D, m): return {t: float(dec_prof(D[(m, t)]["z"])[5:].mean()) for t in T}
post = {}
for b_, p_ in PAIRS:
    L = E2B[(b_, T[0])]["L"]
    d = {"E2b_dec": backz_dec(E2B, p_) - backz_dec(E2B, b_), "E2_dec": backz_dec(E2, p_) - backz_dec(E2, b_),
         "E2b_layer": backz_lay(E2B, p_, L) - backz_lay(E2B, b_, L), "E2_layer": backz_lay(E2, p_, L) - backz_lay(E2, b_, L),
         "E2b_per_task": {t: backz_dec_task(E2B, p_)[t] - backz_dec_task(E2B, b_)[t] for t in T},
         "E2_per_task": {t: backz_dec_task(E2, p_)[t] - backz_dec_task(E2, b_)[t] for t in T},
         "E2b_base_post": [backz_dec(E2B, b_), backz_dec(E2B, p_)], "E2_base_post": [backz_dec(E2, b_), backz_dec(E2, p_)]}
    if b_ == "Qwen2.5-7B":
        d["E2_eager_both_dec"] = backz_dec(E2eag, p_) - backz_dec(E2eag, b_); d["E2_eager_both_layer"] = backz_lay(E2eag, p_, L) - backz_lay(E2eag, b_, L)
    d["sign_agree_dec"] = bool(np.sign(d["E2b_dec"]) == np.sign(d["E2_dec"]))
    # same-item analysis (E2b items are identical across models): slope on k of (post - base) mean top-10% share at 40-80% depth
    sl, nulls = [], []
    rng = np.random.default_rng(3)
    for t in T:
        A, B = E2B[(b_, t)], E2B[(p_, t)]; common = sorted(set(A["ids"]) & set(B["ids"]))
        ia = [A["ids"].index(i) for i in common]; ib = [B["ids"].index(i) for i in common]
        w = e2b.window(L); dlt = 100 * (B["Y"][ib][:, w].mean(1) - A["Y"][ia][:, w].mean(1)); k = A["k"][ia]
        kc = k - k.mean(); sl.append(kc @ (dlt - dlt.mean()) / (kc @ kc))
        nulls.append(np.array([(rng.permutation(kc) @ (dlt - dlt.mean())) / (kc @ kc) for _ in range(5000)]))
    obs = float(np.mean(sl)); nm_ = np.mean(nulls, 0)
    d["same_item_window_slope_pp_per_step"] = {"per_task": dict(zip(T, np.round(sl, 4).tolist())), "mean": obs,
                                               "perm_p_two_sided": float((np.sum(np.abs(nm_) >= abs(obs)) + 1) / 5001)}
    post[f"{b_} -> {p_}"] = d
RES["5_posttraining"] = post

# ---------------------------------------------------------------- (6) main-axis correlates (low power)
acc = {}
for m in M8:
    acc[m] = {"strict_mean": float(np.mean([E2B[(m, t)]["strict"].mean() for t in T])),
              "strict_k1": float(np.mean([E2B[(m, t)]["strict"][E2B[(m, t)]["kall"] == 1].mean() for t in T])),
              "written_mean": float(np.mean([E2B[(m, t)]["written"].mean() for t in T])),
              "written_k1": float(np.mean([E2B[(m, t)]["written"][E2B[(m, t)]["kall"] == 1].mean() for t in T])),
              "H": int(E2B[(m, T[0])]["H"]), "E2_acc": MD2["acc"][m], "E2_acc_k1": MD2["acc_k1"][m]}
RES["accuracy"] = acc
six = {}
for tag, models in (("M8", M8), ("M6_noQwen7B", M6)):
    back_b = [RES[tag]["2_shared"]["E2b"]["back_mean_z"][m] for m in models]; back_a = [RES[tag]["2_shared"]["E2"]["back_mean_z"][m] for m in models]
    pc_b = [RES[tag]["4_within"]["E2b"]["pc1_scores"][m] for m in models]; pc_a = [RES[tag]["4_within"]["E2"]["pc1_scores"][m] for m in models]
    s = {}
    for f in ["H", "strict_mean", "strict_k1", "written_mean", "E2_acc", "E2_acc_k1"]:
        v = [acc[m][f] for m in models]
        s[f] = {"E2b_back": exact_perm_spearman(v, back_b), "E2b_pc1": exact_perm_spearman(v, pc_b),
                "E2_back": exact_perm_spearman(v, back_a), "E2_pc1": exact_perm_spearman(v, pc_a)}
    s["E2b_pc1_vs_back"] = exact_perm_spearman(pc_b, back_b)
    s["post_further_on_pc1_E2b"] = {f"{SHORT[b_]}->{SHORT[p_]}": bool(RES[tag]["4_within"]["E2b"]["pc1_scores"][p_] > RES[tag]["4_within"]["E2b"]["pc1_scores"][b_])
                                   for b_, p_ in PAIRS if b_ in models}
    s["post_further_on_pc1_E2"] = {f"{SHORT[b_]}->{SHORT[p_]}": bool(RES[tag]["4_within"]["E2"]["pc1_scores"][p_] > RES[tag]["4_within"]["E2"]["pc1_scores"][b_])
                                  for b_, p_ in PAIRS if b_ in models}
    if "Qwen2.5-7B" in models:
        s["Q14_further_than_Q7_on_pc1"] = {"E2b": bool(RES[tag]["4_within"]["E2b"]["pc1_scores"]["Qwen2.5-14B"] > RES[tag]["4_within"]["E2b"]["pc1_scores"]["Qwen2.5-7B"]),
                                           "E2": bool(RES[tag]["4_within"]["E2"]["pc1_scores"]["Qwen2.5-14B"] > RES[tag]["4_within"]["E2"]["pc1_scores"]["Qwen2.5-7B"])}
    six[tag] = s
RES["6_correlates"] = six
RES["E2_17model_pc1_scores"] = {m: MDJ["pca"]["scores"][m][0] for m in M8}

# (6b) the paper's main axis: project signatures onto the 17-model E2 PC1 loadings (shift-invariant across centring sets)
v1 = np.array(MDJ["pca"]["loadings"][0]); ax = {}
for tag, models in (("M8", M8), ("M6_noQwen7B", M6)):
    aB = decompose(prof(E2B, "z", models))[1]; aA = decompose(prof(E2, "z", models))[1]
    pB = aB @ v1; pA = aA @ v1; paper = np.array([MDJ["pca"]["scores"][m][0] for m in models])
    Zb = prof(E2B, "z", models); Za = prof(E2, "z", models)
    b57b = Zb[:, :, 5:7].mean((1, 2)); b57a = Za[:, :, 5:7].mean((1, 2))
    backb = Zb[:, :, 5:].mean((1, 2))
    d = {"E2b_proj": dict(zip(models, pB.round(3).tolist())), "E2_proj_r_with_paper_scores": r_(pA, paper),
         "E2b_proj_vs_paper_pc1_pearson": exact_perm_pearson(pB, paper), "E2b_proj_vs_paper_pc1_spearman": exact_perm_spearman(pB, paper),
         "z50_70_E2b": dict(zip(models, b57b.round(3).tolist())), "z50_70_E2": dict(zip(models, b57a.round(3).tolist())),
         "z50_70_pearson": exact_perm_pearson(b57a, b57b), "z50_70_spearman": exact_perm_spearman(b57a, b57b),
         "E2b_proj_vs_E2b_back_spearman": exact_perm_spearman(pB, backb),
         "post_further_on_paper_pc1_E2b": {f"{SHORT[b_]}->{SHORT[p_]}": bool(pB[models.index(p_)] > pB[models.index(b_)]) for b_, p_ in PAIRS if b_ in models},
         "post_further_on_paper_pc1_E2": {f"{SHORT[b_]}->{SHORT[p_]}": bool(paper[models.index(p_)] > paper[models.index(b_)]) for b_, p_ in PAIRS if b_ in models}}
    if "Qwen2.5-7B" in models: d["Q14_further_than_Q7_paper_pc1_E2b"] = bool(pB[models.index("Qwen2.5-14B")] > pB[models.index("Qwen2.5-7B")])
    for f in ["H", "strict_mean", "strict_k1", "E2_acc", "E2_acc_k1"]:
        v = [acc[m][f] for m in models]
        d[f"rho_{f}"] = {"E2b_proj": exact_perm_spearman(v, pB), "paper_pc1_same_models": exact_perm_spearman(v, paper)}
    ax[tag] = d
RES["6b_paper_axis"] = ax

# (5b) OLMo-2 post-training change without the outlier exclusion (E2b)
def backz_noex(m):
    vals = []
    for t in T:
        r = RB[(m, t)]; k = np.array([x["k"] for x in r["items"]], float); z = zprof(q10(r["C"]), k, within=False)[2]; vals.append(dec_prof(z)[5:].mean())
    return float(np.mean(vals))
RES["5b_olmo_no_exclusion"] = {"E2b_base": backz_noex("OLMo-2-1124-7B"), "E2b_post": backz_noex("OLMo-2-1124-7B-Instruct")}
RES["5b_olmo_no_exclusion"]["change"] = RES["5b_olmo_no_exclusion"]["E2b_post"] - RES["5b_olmo_no_exclusion"]["E2b_base"]
# (5c) post-training change in relative-slope units (% of layer mean per step), back-half deciles
def backrel(D, m): return float(np.mean([dec_prof(D[(m, t)]["rel"])[5:].mean() for t in T]))
RES["5c_posttraining_rel"] = {f"{b_} -> {p_}": {"E2b": backrel(E2B, p_) - backrel(E2B, b_), "E2": backrel(E2, p_) - backrel(E2, b_)} for b_, p_ in PAIRS}
json.dump(RES, open(f"{OUT}/e2b_direct_replication.json", "w"), indent=1, default=float)
print("saved", f"{OUT}/e2b_direct_replication.json")
