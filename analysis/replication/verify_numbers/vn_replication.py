"""Verify replication numbers: per-cell agreement, signatures E2 vs E2b, within-E2b structure, shared profile."""
import json, itertools, numpy as np
from vn_lib import *
OUT = {}

def sig_block(models, Za, Zb):
    n = len(models)
    aA = Za.mean(1) - Za.mean((0, 1)); aB = Zb.mean(1) - Zb.mean((0, 1))       # own implementation of signatures
    assert np.allclose(aA, decompose(Za)[1])
    X = np.array([[r_(aB[i], aA[j]) for j in range(n)] for i in range(n)])
    own = np.diag(X); off = X[~np.eye(n, dtype=bool)]
    best = X.argmax(1)
    P = list(itertools.permutations(range(n)))
    null_mean_own = np.array([np.mean([X[i, p[i]] for i in range(n)]) for p in P])
    null_pooled = np.array([r_(aB, aA[list(p)]) for p in P])
    pooled = r_(aA, aB)
    return {"own": dict(zip(models, np.round(own, 4).tolist())), "own_mean": float(own.mean()), "own_min": float(own.min()), "own_max": float(own.max()),
            "other_mean": float(off.mean()), "pooled": pooled,
            "own_gt_mean_others": int(sum(X[i, i] > np.delete(X[i], i).mean() for i in range(n))),
            "own_best": int((best == np.arange(n)).sum()),
            "best": {m: models[best[i]] for i, m in enumerate(models)},
            "best_same_lineage": int(sum(LIN[models[best[i]]] == LIN[m] for i, m in enumerate(models))),
            "p_exact_mean_own": float(np.mean(null_mean_own >= own.mean() - 1e-12)), "p_exact_pooled": float(np.mean(null_pooled >= pooled - 1e-12)),
            "n_perm": len(P), "X": np.round(X, 4).tolist()}

def cells_block(models, Za, Zb):
    n = len(models)
    C = np.array([[r_(Za[i, j], Zb[i, j]) for j in range(4)] for i in range(n)])
    other = [r_(Za[i, j], Zb[i2, j]) for j in range(4) for i in range(n) for i2 in range(n) if i2 != i]
    dA = Za - Za.mean(0, keepdims=True); dB = Zb - Zb.mean(0, keepdims=True)
    eA = decompose(Za)[3]; eB = decompose(Zb)[3]
    return {"mean": float(C.mean()), "median": float(np.median(C)), "min": float(C.min()), "max": float(C.max()),
            "n_pos": int((C > 0).sum()), "pooled": r_(Za, Zb), "other_same_task": float(np.mean(other)),
            "dev_pooled": r_(dA, dB), "e_pooled": r_(eA, eB),
            "per_cell": {m: dict(zip(T, np.round(C[i], 3).tolist())) for i, m in enumerate(models)},
            "per_task": dict(zip(T, np.round(C.mean(0), 3).tolist())), "per_model": dict(zip(models, np.round(C.mean(1), 3).tolist()))}

def within_block(models, Z, nperm=5000):
    md.rng = np.random.default_rng(0)
    S = ss(Z); tot = sum(S.values()); ns_ = tot - S["shared"]
    o, nullm, p = perm_model(Z, nperm)
    sh = split_half(Z)
    n = len(models)
    dev = Z - Z.mean(0, keepdims=True)
    pairs = {f"{T[i]}-{T[j]}": r_(dev[:, i], dev[:, j]) for i, j in itertools.combinations(range(4), 2)}
    cross = [pairs[x] for x in ("pointer-binding", "pointer-relational", "arith-binding", "arith-relational")]
    splits = {}
    for S_ in [(0, 1), (0, 2), (0, 3)]:
        Sc = [j for j in range(4) if j not in S_]
        A1 = Z[:, list(S_)].mean(1); A1 = A1 - A1.mean(0); A2 = Z[:, Sc].mean(1); A2 = A2 - A2.mean(0)
        splits["+".join(T[j] for j in S_)] = r_(A1, A2)
    a = Z.mean(1) - Z.mean((0, 1)); R = np.corrcoef(a)
    lin = {f"{x}/{y}": float(R[models.index(x), models.index(y)]) for x, y in PAIRS if x in models and y in models}
    diff = [R[i, j] for i, j in itertools.combinations(range(n), 2) if LIN[models[i]] != LIN[models[j]]]
    def lstat(labs): return np.mean([R[i, j] for i, j in itertools.combinations(range(n), 2) if LIN[labs[i]] == LIN[labs[j]]])
    lo = lstat(models); P = list(itertools.permutations(range(n)))
    nl = np.array([lstat([models[q] for q in p]) for p in P])
    # task consistency (same model different task vs different models same task)
    within = [np.mean([r_(Z[i, x], Z[i, y]) for x, y in itertools.combinations(range(4), 2)]) for i in range(n)]
    stb = np.mean([r_(Z[i, a_], Z[j, a_]) for i in range(n) for j in range(i + 1, n) for a_ in range(4)])
    return {"shared_frac": float(S["shared"] / tot), "model_share": float(S["model"] / ns_), "task_share": float(S["task"] / ns_), "resid_share": float(S["resid"] / ns_),
            "model_chance": float(nullm / ns_), "p_model": p, "split_half": sh[0], "split_half_null": sh[1], "p_split_half": sh[2],
            "pairs": pairs, "cross_mean": float(np.mean(cross)), "cross_range": [float(min(cross)), float(max(cross))], "splits": splits,
            "lineage_pairs": lin, "lineage_mean": float(lo), "other_mean": float(np.mean(diff)), "n_other": len(diff), "lineage_p_exact": float(np.mean(nl >= lo - 1e-12)),
            "task_consistency_mean": float(np.mean(within)), "task_consistency": dict(zip(models, np.round(within, 3).tolist())), "same_task_other_model": float(stb)}

def shared_block(models, Z):
    pm = Z.mean(1)   # [M,10]
    back = Z[:, :, 5:].mean((1, 2)); front = Z[:, :, :5].mean((1, 2))
    lins = {}
    for i, m in enumerate(models): lins.setdefault(LIN[m], []).append(i)
    ZL = np.array([Z[ix].mean((0, 1)) for ix in lins.values()])
    lin_back_lt_front = sum(Z[ix][:, :, 5:].mean() < Z[ix][:, :, :5].mean() for ix in lins.values())
    return {"mu": np.round(Z.mean((0, 1)), 3).tolist(), "mu_lineage": np.round(ZL.mean(0), 3).tolist(),
            "ci_50_80": tci(pm[:, 5:8].mean(1)), "ci_d4": tci(pm[:, 4]), "ci_d5": tci(pm[:, 5]), "ci_d6": tci(pm[:, 6]), "ci_d7": tci(pm[:, 7]),
            "d4_per_task": dict(zip(T, np.round(Z[:, :, 4].mean(0), 3).tolist())),
            "back_lt_front": int((back < front).sum()), "back_lt0": int((back < 0).sum()), "lin_back_lt_front": int(lin_back_lt_front), "n_lin": len(lins),
            "d4_gt_d5": int((pm[:, 4] > pm[:, 5]).sum()), "d4pos_d5neg": int(((pm[:, 4] > 0) & (pm[:, 5] < 0)).sum()),
            "back_mean": dict(zip(models, np.round(back, 3).tolist())), "front_mean": dict(zip(models, np.round(front, 3).tolist())),
            "grand_back": float(Z[:, :, 5:].mean()), "d50_80_per_model": dict(zip(models, np.round(pm[:, 5:8].mean(1), 3).tolist()))}

Zb8 = Zprof(M8); Za8 = Zprof_e2(M8)
# eager-kernel variant for Qwen2.5-7B-Instruct in E2 (cells.json per-wording exact z, averaged)
Zae8 = Za8.copy()
for j, t in enumerate(T):
    Zae8[3, j] = dec_prof(0.5 * (e2_cells_z("Qwen2.5-7B-Instruct", t, "w0", "eager") + e2_cells_z("Qwen2.5-7B-Instruct", t, "w1", "eager")))
for tag, models in (("M8", M8), ("M6", M6)):
    ix = [M8.index(m) for m in models]; Za, Zb = Za8[ix], Zb8[ix]
    OUT[tag] = {"cells": cells_block(models, Za, Zb), "sig": sig_block(models, Za, Zb),
                "within_E2b": within_block(models, Zb), "within_E2": within_block(models, Za),
                "shared_E2b": shared_block(models, Zb), "shared_E2": shared_block(models, Za),
                "r_mu": r_(Za.mean((0, 1)), Zb.mean((0, 1)))}
    # relative slope units
    Rb = np.array([[dec_prof(100 * zprof(cell(m, "direct", t)[0], cell(m, "direct", t)[1])[0] / np.abs(cell(m, "direct", t)[0].mean(0))) for t in T] for m in models])
    Ra = Zprof_e2(models, "rel")
    OUT[tag]["rel_mu_E2"] = np.round(Ra.mean((0, 1)), 3).tolist(); OUT[tag]["rel_mu_E2b"] = np.round(Rb.mean((0, 1)), 3).tolist()
OUT["M8_E2eagerQ7I"] = {"cells": cells_block(M8, Zae8, Zb8), "sig": sig_block(M8, Zae8, Zb8)}
json.dump(OUT, open(f"{HERE}/vn_replication.json", "w"), indent=1, default=float)

for tag in ("M8", "M6", "M8_E2eagerQ7I"):
    o = OUT[tag]; print("\n=====", tag)
    c = o["cells"]; print("cells: mean %.3f median %.3f min %.3f max %.3f npos %d pooled %.3f other %.3f dev %.3f e %.3f" % (c["mean"], c["median"], c["min"], c["max"], c["n_pos"], c["pooled"], c["other_same_task"], c["dev_pooled"], c["e_pooled"]))
    print(" per_task", c["per_task"], "\n per_model", c["per_model"])
    print(" worst cells", sorted([(v, m, t) for m, d in c["per_cell"].items() for t, v in d.items()])[:3])
    s = o["sig"]; print("sig: own", s["own"], "\n mean %.3f min %.3f max %.3f other %.3f pooled %.3f own>others %d own_best %d same_lin %d p_own %.2e p_pooled %.2e nperm %d" % (
        s["own_mean"], s["own_min"], s["own_max"], s["other_mean"], s["pooled"], s["own_gt_mean_others"], s["own_best"], s["best_same_lineage"], s["p_exact_mean_own"], s["p_exact_pooled"], s["n_perm"]))
    print(" best", s["best"]); print(" X row0", s["X"][0])
    if "within_E2b" not in o: continue
    for nm in ("within_E2b", "within_E2"):
        w = o[nm]; print(nm, "shared %.3f model %.3f (chance %.3f p %.4f) task %.3f resid %.3f split %.3f (null %.3f p %.4f)" % (w["shared_frac"], w["model_share"], w["model_chance"], w["p_model"], w["task_share"], w["resid_share"], w["split_half"], w["split_half_null"], w["p_split_half"]))
        print("  pairs", {k: round(v, 3) for k, v in w["pairs"].items()}, "cross mean %.3f range %s" % (w["cross_mean"], np.round(w["cross_range"], 3)))
        print("  splits", {k: round(v, 3) for k, v in w["splits"].items()})
        print("  lineage", {k: round(v, 3) for k, v in w["lineage_pairs"].items()}, "mean %.3f other %.3f (n %d) p %.4f" % (w["lineage_mean"], w["other_mean"], w["n_other"], w["lineage_p_exact"]))
        print("  task consistency mean %.3f vs same-task other model %.3f" % (w["task_consistency_mean"], w["same_task_other_model"]), w["task_consistency"])
    for nm in ("shared_E2", "shared_E2b"):
        sh = o[nm]; print(nm, "mu", sh["mu"]); print("  mu_lineage", sh["mu_lineage"])
        print("  50-80 %s d4 %s d5 %s d6 %s d7 %s" % tuple(np.round(sh[k], 2).tolist() for k in ("ci_50_80", "ci_d4", "ci_d5", "ci_d6", "ci_d7")))
        print("  d4 per task", sh["d4_per_task"], "back<front %d back<0 %d lin %d/%d d4>d5 %d d4+d5- %d grand back %.3f" % (sh["back_lt_front"], sh["back_lt0"], sh["lin_back_lt_front"], sh["n_lin"], sh["d4_gt_d5"], sh["d4pos_d5neg"], sh["grand_back"]))
        print("  back mean", sh["back_mean"]); print("  50-80 per model", sh["d50_80_per_model"])
    print("r_mu %.3f" % o["r_mu"]); print("rel mu E2", o["rel_mu_E2"]); print("rel mu E2b", o["rel_mu_E2b"])
