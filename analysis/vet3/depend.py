"""(2) Dependence: per-model and per-task sign tests on wording-averaged cells, lineage collapse, leave-one-lineage-out,
w0/w1 agreement.  Uses splits.json rows (primary variant and alternatives)."""
import sys, json, numpy as np
from scipy.stats import binomtest, pearsonr, wilcoxon
sys.path.insert(0, '.'); from v2lib import *
S = json.load(open("splits.json"))["rows"]
out = {}
def per_model_task(rows, key_back="back50", key_front="front50"):
    M = {}
    for r in rows:
        if r["task"] not in TASKS: continue
        M.setdefault((r["model"], r["task"]), []).append((r[key_back], r[key_front], r["rho"]))
    return {k: np.mean(v, 0) for k, v in M.items()}
for variant in ["primary(eager,clean,y25)", "q7i_sdpa", "nosink", "sdpa_all_available", "neg_eff_heads"]:
    for split in [("back50", "front50"), ("back40", "front40"), ("back60", "front60"), ("back3", "front3"), ("back50_no_last10", "front50")]:
        PMT = per_model_task(S[variant], *split)
        models = [m for m in ORDER if all((m, t) in PMT for t in TASKS)]
        spread = {m: sum(PMT[(m, t)][0] < 0 for t in TASKS) for m in models}
        bf = {m: sum(PMT[(m, t)][0] < PMT[(m, t)][1] for t in TASKS) for m in models}
        grand_b = {m: np.mean([PMT[(m, t)][0] for t in TASKS]) for m in models}
        grand_d = {m: np.mean([PMT[(m, t)][0] - PMT[(m, t)][1] for t in TASKS]) for m in models}
        n = len(models); c3 = sum(v >= 3 for v in spread.values()); c3bf = sum(v >= 3 for v in bf.values())
        r = {"n_models": n,
             "models_back_spreads_ge3of4": c3, "p_binom_vs_5/16": binomtest(c3, n, 5 / 16, alternative="greater").pvalue,
             "p_binom_vs_1/2": binomtest(c3, n, .5, alternative="greater").pvalue,
             "models_back<front_ge3of4": c3bf, "p_bf_vs_5/16": binomtest(c3bf, n, 5 / 16, alternative="greater").pvalue,
             "models_grand_back<0": sum(v < 0 for v in grand_b.values()), "p_grand_sign": binomtest(sum(v < 0 for v in grand_b.values()), n, .5, alternative="greater").pvalue,
             "models_grand_back<front": sum(v < 0 for v in grand_d.values()), "p_grand_bf_sign": binomtest(sum(v < 0 for v in grand_d.values()), n, .5, alternative="greater").pvalue,
             "wilcoxon_grand_back_p": float(wilcoxon(list(grand_b.values()), alternative="less").pvalue),
             "per_model_spread_count": spread, "per_model_bf_count": bf}
        # per task (17 models, wording-averaged)
        r["per_task"] = {t: {"back<0": int(sum(PMT[(m, t)][0] < 0 for m in models)), "back<front": int(sum(PMT[(m, t)][0] < PMT[(m, t)][1] for m in models)),
                             "p_sign_back": binomtest(int(sum(PMT[(m, t)][0] < 0 for m in models)), n, .5, alternative="greater").pvalue,
                             "p_sign_bf": binomtest(int(sum(PMT[(m, t)][0] < PMT[(m, t)][1] for m in models)), n, .5, alternative="greater").pvalue} for t in TASKS}
        # lineage: collapse to 11 base models and to 4 families
        for lvl, MAP in (("base", BASE), ("family", FAMILY)):
            gs = sorted(set(MAP[m] for m in models)); gb = [np.mean([grand_b[m] for m in models if MAP[m] == g]) for g in gs]
            gd = [np.mean([grand_d[m] for m in models if MAP[m] == g]) for g in gs]
            r[f"{lvl}_n"] = len(gs); r[f"{lvl}_back<0"] = int(sum(v < 0 for v in gb)); r[f"{lvl}_back<front"] = int(sum(v < 0 for v in gd))
            r[f"{lvl}_p_sign_back"] = binomtest(int(sum(v < 0 for v in gb)), len(gs), .5, alternative="greater").pvalue
            r[f"{lvl}_p_sign_bf"] = binomtest(int(sum(v < 0 for v in gd)), len(gs), .5, alternative="greater").pvalue
        # leave-one-family-out prevalence (cells) and model counts
        cellrows = [x for x in S[variant] if x["task"] in TASKS]
        lofo = {}
        for fam in ["Llama", "Qwen", "OLMo", "Mistral"]:
            R = [x for x in cellrows if FAMILY[x["model"]] != fam]
            b = np.array([x[split[0]] for x in R]); f = np.array([x[split[1]] for x in R])
            ms = [m for m in models if FAMILY[m] != fam]
            lofo[fam] = {"cells": len(R), "back<0": float((b < 0).mean()), "back<front": float((b < f).mean()),
                         "models_ge3of4": int(sum(spread[m] >= 3 for m in ms)), "n_models": len(ms)}
        r["leave_one_family_out"] = lofo
        out[f"{variant}|{split[0]}"] = r
        if variant.startswith("primary") or split[0] == "back50":
            print(f"{variant:26s} {split[0]:17s} models back<0 in>=3/4 tasks: {c3}/{n} (p vs 5/16 {r['p_binom_vs_5/16']:.2g}); back<front>=3/4: {c3bf}/{n} (p {r['p_bf_vs_5/16']:.2g}); grand back<0 {r['models_grand_back<0']}/{n} (p {r['p_grand_sign']:.2g}); grand back<front {r['models_grand_back<front']}/{n}; bases {r['base_back<0']}/{r['base_n']} (p {r['base_p_sign_back']:.2g}), families {r['family_back<0']}/{r['family_n']}")
            print("      per task back<0 / back<front (of 17):", {t: (v['back<0'], v['back<front']) for t, v in r["per_task"].items()})
            print("      LOFO cells back<0/back<front:", {k: (round(v['back<0'], 2), round(v['back<front'], 2), f"{v['models_ge3of4']}/{v['n_models']}") for k, v in lofo.items()})
            print("      per-model spread counts:", r["per_model_spread_count"])
# w0 vs w1 agreement of back-half z (primary)
R = S["primary(eager,clean,y25)"]; D = {(r["model"], r["task"], r["w"]): r for r in R}
for t in TASKS + ["all"]:
    keys = [(m, tt) for m in ORDER for tt in (TASKS if t == "all" else [t])]
    a = np.array([D[(m, tt, "w0")]["back50"] for m, tt in keys]); b = np.array([D[(m, tt, "w1")]["back50"] for m, tt in keys])
    a2 = np.array([D[(m, tt, "w0")]["back50"] - D[(m, tt, "w0")]["front50"] for m, tt in keys]); b2 = np.array([D[(m, tt, "w1")]["back50"] - D[(m, tt, "w1")]["front50"] for m, tt in keys])
    print(f"w0 vs w1 {t:10s} r(back z) = {pearsonr(a, b)[0]:+.2f}; sign agreement {np.mean((a < 0) == (b < 0)):.2f}; both<0 {np.mean((a < 0) & (b < 0)):.2f}; r(back-front) {pearsonr(a2, b2)[0]:+.2f}")
    out[f"w0w1_{t}"] = {"r_back": float(pearsonr(a, b)[0]), "sign_agree": float(np.mean((a < 0) == (b < 0))), "both_neg": float(np.mean((a < 0) & (b < 0)))}
json.dump(out, open("depend.json", "w"), indent=1, default=float)
