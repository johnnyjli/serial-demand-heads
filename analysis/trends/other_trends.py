"""Secondary cross-model trends (not post-training) in the top-10% head-share response to serial depth k.

Primary data: consist.json stat q10 (wording-averaged items, per-layer z with B=2000 permutation SD; Qwen2.5-7B-Instruct
under SDPA; OLMo-2 items with a head >50x the layer median excluded) -> decile profiles Z[m, t, d] (17 x 4 x 10).
Numeral control (fixed_digit) and per-task effect sizes are rebuilt from the vet3 cache with the same item matching
(wording average over items present in both wordings, keep-rule applied) and exact permutation SD (v2lib.zprof); this
rebuild is checked against consist.json on the four serial tasks.

Regions (relative depth x = l/(L-1), deciles d = floor(10x)): front = deciles 0-4, back = deciles 5-9 (x >= .5),
'40-50' = decile 4, '50-70' = deciles 5-6.  Region means are means of decile means (as in model_diff.py).
Lineages (11): a lineage's profile is the mean of its members' profiles (base + post-trained), as in model_diff5.py (e);
'bases' = the 11 base models alone.

Sections: A back-half spreading; B 40-50% concentration; C scale; D accuracy (+ heads per layer); E final layer and
first decile; F task ranking; G scan for other near-universal features; R robustness of the key counts.
Usage: python3 other_trends.py   (writes other_trends.json next to this file)"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, sys, json, itertools, numpy as np
from scipy.stats import spearmanr, binomtest, norm, t as tdist
SP = str(paths.ANALYSIS)
OUT = os.path.join(SP, "trends")
os.environ["VPRIM"] = "y10"
sys.path.insert(0, SP); sys.path.insert(0, f"{SP}/vet3")
import model_diff as MDm
MDm.SP = SP
from model_diff import ORDER, T, LIN, decompose, from_consist, from_cells, dec_prof
from v2lib import load as vload, zprof
rng = np.random.default_rng(0)
NB = 10000
T5 = T + ["fixed_digit"]
BASES = [m for m in ORDER if LIN[m] == m]
LINS = sorted(set(LIN.values()), key=lambda x: ORDER.index(x))
FAMOF = {m: ("Llama" if m.startswith("Llama") else "Qwen" if m.startswith("Qwen") else "OLMo" if m.startswith("OLMo") else "Mistral") for m in ORDER}
PARAMS = {"Llama-3.2-1B": 1.2, "Llama-3.2-3B": 3.2, "Llama-3.1-8B": 8.0, "Qwen2.5-0.5B": 0.5, "Qwen2.5-1.5B": 1.5, "Qwen2.5-3B": 3.1,
          "Qwen2.5-7B": 7.6, "Qwen2.5-14B": 14.7, "OLMo-2-1124-7B": 7.3, "Mistral-7B-v0.1": 7.2, "Mistral-Small-24B-Base-2501": 24.0}
for m in ORDER: PARAMS.setdefault(m, PARAMS[LIN[m]])
ATTN = {m: ("sdpa" if m == "Qwen2.5-7B-Instruct" else "eager") for m in ORDER}
out = {"meta": {"script": os.path.abspath(__file__), "primary": "consist.json q10 decile profiles", "n_boot": NB}}
def f2(x): return None if x is None else float(x)
def cnt(mask): mask = np.asarray(mask, bool); return int(mask.sum())
def binp(k, n): return float(binomtest(int(k), int(n), 0.5).pvalue)
def boot_ci(v, stat=np.mean):
    v = np.asarray(v, float); b = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(NB)])
    return [float(stat(v)), float(np.quantile(b, .025)), float(np.quantile(b, .975))]
def lin_mean(X):
    """X: array indexed by ORDER on axis 0 -> array over the 11 lineages (mean of members)"""
    return np.array([np.mean([X[ORDER.index(m)] for m in ORDER if LIN[m] == L_], 0) for L_ in LINS])
def base_sel(X): return np.array([X[ORDER.index(m)] for m in BASES])
def summary(v, pred):
    """v over 17 models; pred -> boolean. counts over 17 models, 11 lineages (lineage-mean), 11 base models"""
    v = np.asarray(v, float); vl = lin_mean(v); vb = base_sel(v)
    return {"models": cnt(pred(v)), "lineages": cnt(pred(vl)), "bases": cnt(pred(vb)),
            "p_models": binp(cnt(pred(v)), 17), "p_lineages": binp(cnt(pred(vl)), 11),
            "which_models_fail": [m for m, ok in zip(ORDER, pred(v)) if not ok], "which_lineages_fail": [L_ for L_, ok in zip(LINS, pred(vl)) if not ok]}

# ------------------------------------------------------------------ data
Z = from_consist("q10")                                     # 17 x 4 x 10 (primary)
C = {r["model"]: r for r in json.load(open(f"{SP}/consist.json")) if r["stat"] == "q10"}
zl = {(m, t): np.array(C[m]["families"][t]["z"]) for m in ORDER for t in T}          # per-layer z (primary)
pl = {(m, t): np.array(C[m]["families"][t]["p"]) for m in ORDER for t in T}
rl = {(m, t): np.array(C[m]["families"][t]["rel"]) for m in ORDER for t in T}        # % of layer mean per step
def cache_cell(m, f, attn):
    a, b = vload(m, f, "w0", attn, True), vload(m, f, "w1", attn, True)
    if a is None or b is None: return None
    ka = {(int(k), int(bd)): i for i, (k, bd) in enumerate(zip(a["k"], a["body"]))}
    kb = {(int(k), int(bd)): i for i, (k, bd) in enumerate(zip(b["k"], b["body"]))}
    keys = sorted(set(ka) & set(kb)); ia = np.array([ka[x] for x in keys]); ib = np.array([kb[x] for x in keys])
    Y = 0.5 * (a["y10"][ia] + b["y10"][ib]); k = np.array([x[0] for x in keys], float); body = np.array([x[1] for x in keys])
    sl, sd, z = zprof(Y, k, body, f in ("binding", "relational"))
    acc = 0.5 * (a["correct"].mean() + b["correct"].mean())
    return {"z": z, "rel": 100 * sl / np.abs(Y.mean(0)), "acc": float(acc), "n": len(keys)}
CC = {(m, f): cache_cell(m, f, ATTN[m]) for m in ORDER for f in T5}
# check the rebuild against consist.json
chk = [(np.corrcoef(CC[(m, t)]["z"], zl[(m, t)])[0, 1], np.abs(CC[(m, t)]["z"] - zl[(m, t)]).max(), np.abs(dec_prof(CC[(m, t)]["z"]) - Z[i, j]).max())
       for i, m in enumerate(ORDER) for j, t in enumerate(T)]
chk = np.array(chk)
out["rebuild_check"] = {"min_r_layer_z": float(chk[:, 0].min()), "max_abs_layer_z_diff": float(chk[:, 1].max()), "max_abs_decile_diff": float(chk[:, 2].max())}
print("rebuild vs consist.json: min r %.4f, max |dz| layer %.3f, decile %.3f" % tuple(out["rebuild_check"].values()))
Zfd = np.array([dec_prof(CC[(m, "fixed_digit")]["z"]) for m in ORDER])               # 17 x 10
Rel = np.array([[dec_prof(rl[(m, t)]) for t in T] for m in ORDER])                   # effect size: % of layer mean per step
Relfd = np.array([dec_prof(CC[(m, "fixed_digit")]["rel"]) for m in ORDER])
ACC = {m: {f: CC[(m, f)]["acc"] for f in T5} for m in ORDER}
MD2 = json.load(open(f"{SP}/model_diff2.json"))["features"]; MD = json.load(open(f"{SP}/model_diff.json"))
PC1 = np.array([MD["pca"]["scores"][m][0] for m in ORDER])

def regions(Zm):  # Zm [..., 10]
    return {"front": Zm[..., :5].mean(-1), "back": Zm[..., 5:].mean(-1), "d40_50": Zm[..., 4], "d30_40": Zm[..., 3],
            "d50_70": Zm[..., 5:7].mean(-1), "d70_100": Zm[..., 7:].mean(-1), "d0": Zm[..., 0], "d90_100": Zm[..., 9]}
RG = regions(Z)                      # each 17 x 4
RGa = {k: v.mean(1) for k, v in RG.items()}          # task-averaged, 17
RGfd = regions(Zfd)                  # 17

# ------------------------------------------------------------------ A. back-half spreading
A = {}
for j, t in enumerate(T + ["mean4"]):
    b = RG["back"][:, j] if j < 4 else RGa["back"]; f = RG["front"][:, j] if j < 4 else RGa["front"]
    A[t] = {"back_lt0": summary(b, lambda v: v < 0), "back_lt_front": summary(b - f, lambda v: v < 0),
            "mean_back_lineage_ci": boot_ci(lin_mean(b)), "mean_back_minus_front_lineage_ci": boot_ci(lin_mean(b - f)),
            "mean_back_models": float(b.mean())}
A["fixed_digit"] = {"back_lt0": summary(RGfd["back"], lambda v: v < 0), "back_lt_front": summary(RGfd["back"] - RGfd["front"], lambda v: v < 0),
                    "mean_back_lineage_ci": boot_ci(lin_mean(RGfd["back"])), "mean_back_minus_front_lineage_ci": boot_ci(lin_mean(RGfd["back"] - RGfd["front"])),
                    "mean_back_models": float(RGfd["back"].mean())}
dser = RGa["back"] - RGfd["back"]
A["serial_vs_control"] = {"serial_back_lt_control_back": summary(dser, lambda v: v < 0), "diff_lineage_ci": boot_ci(lin_mean(dser)),
                          "per_model": dict(zip(ORDER, np.round(dser, 3).tolist()))}
# how many of the 4 serial tasks have back<0, per model
A["n_tasks_back_lt0_per_model"] = dict(zip(ORDER, (RG["back"] < 0).sum(1).tolist()))
A["models_back_lt0_in_ge3_tasks"] = cnt((RG["back"] < 0).sum(1) >= 3)
A["lineages_back_lt0_in_ge3_tasks"] = cnt((lin_mean(RG["back"]) < 0).sum(1) >= 3)
out["A_backhalf"] = A
print("\nA. back half (decile 5-9 mean z < 0) / back < front: models of 17, lineages of 11, bases of 11")
for t in T + ["mean4", "fixed_digit"]:
    a_ = A[t]; print(f"  {t:12s} back<0 {a_['back_lt0']['models']:2d}/17 {a_['back_lt0']['lineages']:2d}/11 bases {a_['back_lt0']['bases']:2d} | back<front {a_['back_lt_front']['models']:2d}/17 {a_['back_lt_front']['lineages']:2d}/11"
                     f" | lineage mean back {a_['mean_back_lineage_ci'][0]:+.2f} [{a_['mean_back_lineage_ci'][1]:+.2f},{a_['mean_back_lineage_ci'][2]:+.2f}]  back-front {a_['mean_back_minus_front_lineage_ci'][0]:+.2f} [{a_['mean_back_minus_front_lineage_ci'][1]:+.2f},{a_['mean_back_minus_front_lineage_ci'][2]:+.2f}]"
                     f"  fail lineages(back<0): {a_['back_lt0']['which_lineages_fail']}")
s_ = A["serial_vs_control"]
print(f"  serial back < control back: {s_['serial_back_lt_control_back']['models']}/17 models, {s_['serial_back_lt_control_back']['lineages']}/11 lineages; lineage mean diff {s_['diff_lineage_ci']}")
print("  models with back<0 in >=3 of 4 tasks:", A["models_back_lt0_in_ge3_tasks"], " lineages:", A["lineages_back_lt0_in_ge3_tasks"])

# ------------------------------------------------------------------ B. 40-50% concentration
B = {}
for j, t in enumerate(T + ["mean4"]):
    v = RG["d40_50"][:, j] if j < 4 else RGa["d40_50"]
    B[t] = {"d40_50_gt0": summary(v, lambda x: x > 0), "lineage_mean_ci": boot_ci(lin_mean(v)), "model_mean": float(v.mean())}
B["fixed_digit"] = {"d40_50_gt0": summary(RGfd["d40_50"], lambda x: x > 0), "lineage_mean_ci": boot_ci(lin_mean(RGfd["d40_50"])),
                    "d30_40_gt0": summary(RGfd["d30_40"], lambda x: x > 0), "d30_40_lineage_mean_ci": boot_ci(lin_mean(RGfd["d30_40"])),
                    "max_d30_50_gt0": summary(np.maximum(RGfd["d30_40"], RGfd["d40_50"]), lambda x: x > 0)}
# serial tasks 30-40 too, and 'peak of the front half lies at 30-50%'
B["mean4_d30_40_gt0"] = summary(RGa["d30_40"], lambda x: x > 0)
B["write_k_tasks_mean_d40_50_gt0"] = summary(RG["d40_50"][:, :2].mean(1), lambda x: x > 0)
B["chain_tasks_mean_d40_50_gt0"] = summary(RG["d40_50"][:, 2:].mean(1), lambda x: x > 0)
B["write_k_minus_chain_lineage_ci"] = boot_ci(lin_mean(RG["d40_50"][:, :2].mean(1) - RG["d40_50"][:, 2:].mean(1)))
out["B_concentration_40_50"] = B
print("\nB. 40-50% decile mean z > 0: models/17, lineages/11 | lineage mean [95% CI]")
for t in T + ["mean4", "fixed_digit"]:
    b_ = B[t]; print(f"  {t:12s} {b_['d40_50_gt0']['models']:2d}/17 {b_['d40_50_gt0']['lineages']:2d}/11 bases {b_['d40_50_gt0']['bases']:2d} | {b_['lineage_mean_ci'][0]:+.2f} [{b_['lineage_mean_ci'][1]:+.2f},{b_['lineage_mean_ci'][2]:+.2f}]")
print(f"  fixed_digit 30-40%: {B['fixed_digit']['d30_40_gt0']['models']}/17, {B['fixed_digit']['d30_40_gt0']['lineages']}/11, lineage mean {B['fixed_digit']['d30_40_lineage_mean_ci']}")
print(f"  write-k tasks (pointer+arith) 40-50>0: {B['write_k_tasks_mean_d40_50_gt0']['models']}/17 {B['write_k_tasks_mean_d40_50_gt0']['lineages']}/11; chain tasks: {B['chain_tasks_mean_d40_50_gt0']['models']}/17 {B['chain_tasks_mean_d40_50_gt0']['lineages']}/11; diff {B['write_k_minus_chain_lineage_ci']}")

# ------------------------------------------------------------------ C. scale
logN = np.log10(np.array([PARAMS[m] for m in ORDER])); Hh = np.array([MD2["H"][m] for m in ORDER], float)
metrics = {"back": RGa["back"], "d50_70": RGa["d50_70"], "back_minus_front": RGa["back"] - RGa["front"], "pc1": PC1}
def sp(x, y):
    r = spearmanr(x, y); return [float(r.correlation), float(r.pvalue), len(x)]
bi = np.array([ORDER.index(m) for m in BASES])
FAMS_MULTI = {"Llama": ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B"], "Qwen": ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-14B"],
              "Mistral": ["Mistral-7B-v0.1", "Mistral-Small-24B-Base-2501"]}
def within_pairs(vals, fams):
    """concordance of size with metric over within-family pairs; returns (#pairs where the larger model has the LOWER metric,
    i.e. spreads more, #pairs), exact within-family permutation p (one-sided, larger spreads more)"""
    def S(assign):
        s = 0; n = 0
        for f, ms in fams.items():
            for x, y in itertools.combinations(ms, 2):
                big, small = (x, y) if PARAMS[x] > PARAMS[y] else (y, x); s += assign[big] < assign[small]; n += 1
        return s, n
    obs, n = S(vals)
    perms = [list(itertools.permutations(ms)) for ms in fams.values()]; null = []
    for combo in itertools.product(*perms):
        asg = {}
        for ms, pm in zip(fams.values(), combo):
            for m_, pm_ in zip(ms, pm): asg[m_] = vals[pm_]
        null.append(S(asg)[0])
    null = np.array(null); return int(obs), int(n), float((null >= obs).mean())
Cs = {}
for name, v in metrics.items():
    vd = dict(zip(ORDER, v))
    d = {"bases_11": sp(logN[bi], v[bi]),
         "bases_excl_qwen": sp([logN[ORDER.index(m)] for m in BASES if FAMOF[m] != "Qwen"], [vd[m] for m in BASES if FAMOF[m] != "Qwen"]),
         "bases_excl_qwen7b": sp([logN[ORDER.index(m)] for m in BASES if m != "Qwen2.5-7B"], [vd[m] for m in BASES if m != "Qwen2.5-7B"]),
         "all_17": sp(logN, v),
         "qwen_bases": sp([logN[ORDER.index(m)] for m in FAMS_MULTI["Qwen"]], [vd[m] for m in FAMS_MULTI["Qwen"]]),
         "qwen_bases_excl_7b": sp([logN[ORDER.index(m)] for m in FAMS_MULTI["Qwen"] if m != "Qwen2.5-7B"], [vd[m] for m in FAMS_MULTI["Qwen"] if m != "Qwen2.5-7B"]),
         "llama_bases_values": {m: float(vd[m]) for m in FAMS_MULTI["Llama"]}, "mistral_values": {m: float(vd[m]) for m in FAMS_MULTI["Mistral"]},
         "qwen_values": {m: float(vd[m]) for m in FAMS_MULTI["Qwen"]}}
    d["within_family_pairs_larger_spreads_more"] = within_pairs(vd, FAMS_MULTI)
    d["within_family_pairs_excl_qwen"] = within_pairs(vd, {k: v_ for k, v_ in FAMS_MULTI.items() if k != "Qwen"})
    d["within_family_pairs_llama"] = within_pairs(vd, {"Llama": FAMS_MULTI["Llama"]})
    if name == "pc1":  # PC1 + = more spreading at 50-70%: flip so 'lower = spreads more' semantics hold
        vneg = {m: -x for m, x in vd.items()}
        d["within_family_pairs_larger_spreads_more"] = within_pairs(vneg, FAMS_MULTI); d["within_family_pairs_excl_qwen"] = within_pairs(vneg, {k: v_ for k, v_ in FAMS_MULTI.items() if k != "Qwen"})
        d["within_family_pairs_llama"] = within_pairs(vneg, {"Llama": FAMS_MULTI["Llama"]})
    Cs[name] = d
# partial Spearman back ~ logN | H across bases (rank residualisation)
def partial_sp(x, y, zc):
    rx, ry, rz = [np.argsort(np.argsort(a)).astype(float) for a in (x, y, zc)]
    X = np.c_[np.ones_like(rz), rz]; ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]; ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1]); n = len(x); tt = r * np.sqrt((n - 3) / max(1e-12, 1 - r * r))
    return [r, float(2 * tdist.sf(abs(tt), n - 3)), n]
Cs["logN_vs_H_bases"] = sp(logN[bi], Hh[bi])
Cs["back_vs_logN_given_H_bases"] = partial_sp(logN[bi], RGa["back"][bi], Hh[bi])
Cs["back_vs_H_given_logN_bases"] = partial_sp(Hh[bi], RGa["back"][bi], logN[bi])
Cs["acc_vs_logN_bases"] = sp(logN[bi], np.array([MD2["acc"][m] for m in BASES]))
out["C_scale"] = Cs
print("\nC. scale (Spearman with log10 params; negative = larger spreads more for back/d50_70/back-front; positive for PC1)")
for name in metrics:
    d = Cs[name]
    print(f"  {name:16s} 11 bases {d['bases_11'][0]:+.2f} (p={d['bases_11'][1]:.3f}) | excl Qwen {d['bases_excl_qwen'][0]:+.2f} (p={d['bases_excl_qwen'][1]:.2f}, n={d['bases_excl_qwen'][2]})"
          f" | excl Q7B {d['bases_excl_qwen7b'][0]:+.2f} (p={d['bases_excl_qwen7b'][1]:.3f}) | Qwen {d['qwen_bases'][0]:+.2f} (p={d['qwen_bases'][1]:.3f}); Qwen excl 7B {d['qwen_bases_excl_7b'][0]:+.2f}"
          f" | within-family pairs larger spreads more {d['within_family_pairs_larger_spreads_more']}, excl Qwen {d['within_family_pairs_excl_qwen']}")
print("  Llama bases back:", {k: round(v, 2) for k, v in Cs['back']['llama_bases_values'].items()}, " Mistral:", {k: round(v, 2) for k, v in Cs['back']['mistral_values'].items()},
      " Qwen:", {k: round(v, 2) for k, v in Cs['back']['qwen_values'].items()})
print("  logN vs H (bases)", np.round(Cs["logN_vs_H_bases"], 3), " partial back~logN|H", np.round(Cs["back_vs_logN_given_H_bases"], 3), " partial back~H|logN", np.round(Cs["back_vs_H_given_logN_bases"], 3),
      " acc~logN", np.round(Cs["acc_vs_logN_bases"], 3))

# ------------------------------------------------------------------ D. accuracy
acc = np.array([MD2["acc"][m] for m in ORDER]); acc1 = np.array([MD2["acc_k1"][m] for m in ORDER])
Dd = {}
for an, av in [("acc", acc), ("acc_k1", acc1)]:
    for mn in ["back", "d50_70"]:
        v = RGa[mn]
        Dd[f"{an}~{mn}"] = {"all_17": sp(av, v), "bases_11": sp(av[bi], v[bi]),
                            "all_17_partial_H": partial_sp(av, v, Hh), "bases_11_partial_H": partial_sp(av[bi], v[bi], Hh[bi]),
                            "H32_9models": sp(av[Hh == 32], v[Hh == 32]), "H32_5bases": sp(av[bi][Hh[bi] == 32], v[bi][Hh[bi] == 32]),
                            "non_qwen_11": sp(av[[FAMOF[m] != "Qwen" for m in ORDER]], v[[FAMOF[m] != "Qwen" for m in ORDER]]),
                            "excl_qwen7b_16": sp(av[[m != "Qwen2.5-7B" for m in ORDER]], v[[m != "Qwen2.5-7B" for m in ORDER]])}
Dd["H~back_all17"] = sp(Hh, RGa["back"]); Dd["H~acc_all17"] = sp(Hh, acc); Dd["H~acc_bases"] = sp(Hh[bi], acc[bi])
# per task: task accuracy vs task back z across models
pt = {}
for j, t in enumerate(T):
    at = np.array([ACC[m][t] for m in ORDER]); v = RG["back"][:, j]; vr = Rel[:, j, 5:].mean(1)
    pt[t] = {"all_17": sp(at, v), "bases_11": sp(at[bi], v[bi]), "all_17_relslope": sp(at, vr), "partial_H_17": partial_sp(at, v, Hh)}
# pooled, task-demeaned, lineage-level: centre acc and back z within task, average within lineage, Spearman over 44 lineage x task points
Ac = np.array([[ACC[m][t] for t in T] for m in ORDER]); Ac = Ac - Ac.mean(0); Bc = RG["back"] - RG["back"].mean(0)
AL, BL = lin_mean(Ac), lin_mean(Bc)
pt["pooled_task_centred_lineages"] = sp(AL.ravel(), BL.ravel())
# lineage-cluster permutation for the pooled statistic: permute lineage labels within each task
obs = spearmanr(AL.ravel(), BL.ravel()).correlation; nl = []
for _ in range(5000):
    BLp = np.stack([BL[rng.permutation(11), j] for j in range(4)], 1); nl.append(spearmanr(AL.ravel(), BLp.ravel()).correlation)
pt["pooled_task_centred_lineages_perm_p"] = float((np.sum(np.abs(nl) >= abs(obs)) + 1) / 5001)
# within-model across tasks: does the task a model solves best show the most spreading, after removing task means? (model-level Spearman over 4 tasks, averaged)
wm = [spearmanr(Ac[i], Bc[i]).correlation for i in range(17)]
pt["within_model_task_centred_mean_rho"] = float(np.nanmean(wm)); pt["within_model_task_centred_n_negative"] = cnt(np.array(wm) < 0)
Dd["per_task"] = pt
out["D_accuracy"] = Dd
print("\nD. accuracy vs spreading (negative rho = more accurate models spread more)")
for k_, d in Dd.items():
    if k_ == "per_task" or k_.startswith("H~"): continue
    print(f"  {k_:14s} 17: {d['all_17'][0]:+.2f} (p={d['all_17'][1]:.3f}) | 11 bases: {d['bases_11'][0]:+.2f} (p={d['bases_11'][1]:.3f}) | 17 partial H: {d['all_17_partial_H'][0]:+.2f} (p={d['all_17_partial_H'][1]:.3f})"
          f" | bases partial H: {d['bases_11_partial_H'][0]:+.2f} (p={d['bases_11_partial_H'][1]:.3f}) | H=32 (9): {d['H32_9models'][0]:+.2f} (p={d['H32_9models'][1]:.3f}) | H=32 bases (5): {d['H32_5bases'][0]:+.2f}"
          f" | non-Qwen (11): {d['non_qwen_11'][0]:+.2f} (p={d['non_qwen_11'][1]:.3f})")
print("  H~back 17", np.round(Dd["H~back_all17"], 3), " H~acc 17", np.round(Dd["H~acc_all17"], 3), " H~acc bases", np.round(Dd["H~acc_bases"], 3))
for t in T: print(f"  per task {t:10s}: acc_t vs back_t 17 {pt[t]['all_17'][0]:+.2f} (p={pt[t]['all_17'][1]:.3f}), bases {pt[t]['bases_11'][0]:+.2f} (p={pt[t]['bases_11'][1]:.3f}), rel-slope {pt[t]['all_17_relslope'][0]:+.2f}, partial H {pt[t]['partial_H_17'][0]:+.2f} (p={pt[t]['partial_H_17'][1]:.3f})")
print(f"  pooled task-centred (44 lineage x task): rho {pt['pooled_task_centred_lineages'][0]:+.2f}, perm p {pt['pooled_task_centred_lineages_perm_p']:.4f}; within-model task-centred mean rho {pt['within_model_task_centred_mean_rho']:+.2f}, negative in {pt['within_model_task_centred_n_negative']}/17")

# ------------------------------------------------------------------ E. final layer, first decile / first layer
fz = np.array([[zl[(m, t)][-1] for t in T] for m in ORDER]); fp = np.array([[pl[(m, t)][-1] for t in T] for m in ORDER])
fzfd = np.array([CC[(m, "fixed_digit")]["z"][-1] for m in ORDER])
z0 = np.array([[zl[(m, t)][0] for t in T] for m in ORDER]); p0 = np.array([[pl[(m, t)][0] for t in T] for m in ORDER])
E = {"final_layer_z": {m: fz[i].round(2).tolist() + [round(float(fzfd[i]), 2)] for i, m in enumerate(ORDER)},
     "final_conc_all4": cnt((fz > 0).all(1)), "final_conc_ge3": cnt((fz > 0).sum(1) >= 3), "final_spread_all4": cnt((fz < 0).all(1)), "final_spread_ge3": cnt((fz < 0).sum(1) >= 3),
     "final_mean_gt0": summary(fz.mean(1), lambda v: v > 0), "final_mean_lt0": summary(fz.mean(1), lambda v: v < 0),
     "final_sig_conc_any": cnt(((fz > 0) & (fp < .05)).any(1)), "final_sig_spread_any": cnt(((fz < 0) & (fp < .05)).any(1)),
     "final_sig_conc_ge2": cnt(((fz > 0) & (fp < .05)).sum(1) >= 2), "final_sig_spread_ge2": cnt(((fz < 0) & (fp < .05)).sum(1) >= 2),
     "final_cells_sig_conc": cnt((fz > 0) & (fp < .05)), "final_cells_sig_spread": cnt((fz < 0) & (fp < .05)), "final_cells": 68,
     "final_fd_gt0": summary(fzfd, lambda v: v > 0),
     "final_minus_d90_lineage_ci": boot_ci(lin_mean(fz.mean(1) - RGa["d90_100"])),
     "final_gt_rest_of_last_decile": summary(fz.mean(1) - RGa["d90_100"], lambda v: v > 0),
     "d0_gt0": summary(RGa["d0"], lambda v: v > 0), "d0_lineage_ci": boot_ci(lin_mean(RGa["d0"])),
     "d0_gt0_per_task": {t: cnt(RG["d0"][:, j] > 0) for j, t in enumerate(T)},
     "layer0_mean_gt0": summary(z0.mean(1), lambda v: v > 0), "layer0_cells_sig_conc": cnt((z0 > 0) & (p0 < .05)), "layer0_cells_sig_spread": cnt((z0 < 0) & (p0 < .05)),
     "d90_100_lt0": summary(RGa["d90_100"], lambda v: v < 0), "d90_100_lineage_ci": boot_ci(lin_mean(RGa["d90_100"]))}
# final layer per task counts
E["final_gt0_per_task"] = {t: cnt(fz[:, j] > 0) for j, t in enumerate(T)}
out["E_final_first"] = E
print("\nE. final layer (per-layer z of the last layer)")
print(f"  concentrates in all 4 tasks: {E['final_conc_all4']}/17, >=3: {E['final_conc_ge3']}; spreads in all 4: {E['final_spread_all4']}, >=3: {E['final_spread_ge3']}")
print(f"  task-mean >0: {E['final_mean_gt0']['models']}/17 models, {E['final_mean_gt0']['lineages']}/11 lineages (fail: {E['final_mean_gt0']['which_models_fail']})")
print(f"  significant cells: conc {E['final_cells_sig_conc']}/68, spread {E['final_cells_sig_spread']}/68; models with >=1 sig conc {E['final_sig_conc_any']}, >=1 sig spread {E['final_sig_spread_any']}; >=2 sig conc {E['final_sig_conc_ge2']}")
print(f"  per task final z>0: {E['final_gt0_per_task']}; numeral control final>0: {E['final_fd_gt0']['models']}/17 {E['final_fd_gt0']['lineages']}/11")
print(f"  final layer above the rest of the last decile: {E['final_gt_rest_of_last_decile']['models']}/17 {E['final_gt_rest_of_last_decile']['lineages']}/11; lineage diff {E['final_minus_d90_lineage_ci']}")
print(f"  first decile >0: {E['d0_gt0']['models']}/17 {E['d0_gt0']['lineages']}/11, lineage mean {E['d0_lineage_ci']}; per task {E['d0_gt0_per_task']}; layer0 mean>0 {E['layer0_mean_gt0']['models']}/17; layer0 sig cells conc {E['layer0_cells_sig_conc']} spread {E['layer0_cells_sig_spread']}")
print(f"  last decile (90-100%) <0: {E['d90_100_lt0']['models']}/17 {E['d90_100_lt0']['lineages']}/11; lineage mean {E['d90_100_lineage_ci']}")
for i, m in enumerate(ORDER): print(f"    {m:28s} final z P A B R | FD: " + " ".join(f"{x:+5.2f}" for x in fz[i]) + f" | {fzfd[i]:+5.2f}")

# ------------------------------------------------------------------ F. task ranking
def task_rank(Bm):  # Bm 17 x 4 back values (lower = more spreading)
    strongest = [T[j] for j in Bm.argmin(1)]; weakest = [T[j] for j in Bm.argmax(1)]
    return {"strongest": {t: strongest.count(t) for t in T}, "weakest": {t: weakest.count(t) for t in T},
            "mean_rank": {t: float(v) for t, v in zip(T, np.argsort(np.argsort(Bm, 1), 1).mean(0) + 1)}}
Bz = RG["back"]; Brel = Rel[:, :, 5:].mean(2)
F = {"z": task_rank(Bz), "rel_slope": task_rank(Brel), "z_lineages": task_rank(lin_mean(Bz)), "rel_lineages": task_rank(lin_mean(Brel)),
     "z_bases": task_rank(base_sel(Bz))}
# binding vs each other task, paired across models and lineages
F["binding_below_other"] = {t: {"models": cnt(Bz[:, 2] < Bz[:, j]), "lineages": cnt(lin_mean(Bz)[:, 2] < lin_mean(Bz)[:, j]),
                                "models_rel": cnt(Brel[:, 2] < Brel[:, j])} for j, t in enumerate(T) if t != "binding"}
F["arith_above_other"] = {t: {"models": cnt(Bz[:, 1] > Bz[:, j]), "lineages": cnt(lin_mean(Bz)[:, 1] > lin_mean(Bz)[:, j]),
                              "models_rel": cnt(Brel[:, 1] > Brel[:, j])} for j, t in enumerate(T) if t != "arith"}
F["mean_back_rel_by_task_lineage_ci"] = {t: boot_ci(lin_mean(Brel[:, j])) for j, t in enumerate(T)}
F["mean_back_z_by_task_lineage_ci"] = {t: boot_ci(lin_mean(Bz[:, j])) for j, t in enumerate(T)}
F["n_items"] = {t: [C[m]["families"][t]["n"] for m in ORDER] for t in T}
out["F_task_rank"] = F
print("\nF. which task spreads most in the back half (per model argmin of back mean)")
for k_ in ["z", "rel_slope", "z_lineages", "rel_lineages", "z_bases"]:
    print(f"  {k_:12s} strongest {F[k_]['strongest']} | weakest {F[k_]['weakest']} | mean rank {({t: round(v, 2) for t, v in F[k_]['mean_rank'].items()})}")
print("  binding more negative than:", F["binding_below_other"]); print("  arith less negative than:", F["arith_above_other"])
print("  lineage mean back rel slope (% of layer mean per step):", {t: [round(x, 3) for x in v] for t, v in F["mean_back_rel_by_task_lineage_ci"].items()})

# ------------------------------------------------------------------ G. scan for other near-universal features
prof = Z.mean(1)                                                     # task-averaged profile 17 x 10
lo_mu = []
for i, m in enumerate(ORDER):
    others = [j for j, x in enumerate(ORDER) if LIN[x] != LIN[m]]
    Lo = sorted(set(LIN[ORDER[j]] for j in others)); mu_lo = np.mean([prof[[j for j in others if LIN[ORDER[j]] == L_]].mean(0) for L_ in Lo], 0)
    lo_mu.append(np.corrcoef(prof[i], mu_lo)[0, 1])
lo_mu = np.array(lo_mu)
profL0 = lin_mean(prof); lo_mu_L = np.array([np.corrcoef(profL0[i], np.delete(profL0, i, 0).mean(0))[0, 1] for i in range(11)])
# per task: correlation of each model's task profile with the leave-lineage-out mean profile of that task
lo_task = np.zeros((17, 4))
for j in range(4):
    for i, m in enumerate(ORDER):
        others = [q for q, x in enumerate(ORDER) if LIN[x] != LIN[m]]; Lo = sorted(set(LIN[ORDER[q]] for q in others))
        mu_lo = np.mean([Z[[q for q in others if LIN[ORDER[q]] == L_], j].mean(0) for L_ in Lo], 0); lo_task[i, j] = np.corrcoef(Z[i, j], mu_lo)[0, 1]
argmin_dec = prof.argmin(1); argmax_front = prof[:, :5].argmax(1)
feat = {
    "back<0": RGa["back"] < 0, "back<front": RGa["back"] < RGa["front"], "d40_50>0": RGa["d40_50"] > 0,
    "d40_50>d50_60 (switch at mid-depth)": prof[:, 4] > prof[:, 5], "d40_50>back": RGa["d40_50"] > RGa["back"],
    "min decile at 50-80%": np.isin(argmin_dec, [5, 6, 7]), "min decile in back half": argmin_dec >= 5,
    "front-half max at 30-50%": np.isin(argmax_front, [3, 4]),
    "d50_60<0": prof[:, 5] < 0, "d60_70<0": prof[:, 6] < 0, "d70_80<0": prof[:, 7] < 0, "d80_90<0": prof[:, 8] < 0, "d90_100<0": prof[:, 9] < 0,
    "d50_70<0": RGa["d50_70"] < 0, "d0>0": prof[:, 0] > 0, "d10_20>0": prof[:, 1] > 0, "front>0": RGa["front"] > 0,
    "|back|>|front|": np.abs(RGa["back"]) > np.abs(RGa["front"]),
    "final layer z>0 (task mean)": fz.mean(1) > 0,
    "profile r>0 with leave-lineage-out mean": lo_mu > 0, "profile r>0.5 with leave-lineage-out mean": lo_mu > 0.5,
    "task profiles r>0 with LLO task mean in >=3 tasks": (lo_task > 0).sum(1) >= 3,
    "serial back < control back": RGa["back"] < RGfd["back"], "control back<0": RGfd["back"] < 0, "control d30_40>0": RGfd["d30_40"] > 0,
    "control d40_50>0": RGfd["d40_50"] > 0, "control max(d30_40,d40_50)>0": np.maximum(RGfd["d30_40"], RGfd["d40_50"]) > 0,
    "binding back<0": RG["back"][:, 2] < 0, "pointer back<0": RG["back"][:, 0] < 0, "relational back<0": RG["back"][:, 3] < 0, "arith back<0": RG["back"][:, 1] < 0,
    "binding strongest back task": Bz.argmin(1) == 2, "arith weakest back task": Bz.argmax(1) == 1,
    "binding back < arith back": Bz[:, 2] < Bz[:, 1], "chain-task back < write-k back": Bz[:, 2:].mean(1) < Bz[:, :2].mean(1),
    "arith d40_50>0": RG["d40_50"][:, 1] > 0, "pointer d40_50>0": RG["d40_50"][:, 0] > 0, "write-k d40_50 > chain d40_50": RG["d40_50"][:, :2].mean(1) > RG["d40_50"][:, 2:].mean(1),
    "entropy slope>0": np.array([MD2["ent_slope"][m] for m in ORDER]) > 0,
}
def lin_feat(vals_fn_mask, name):
    return None
G = {}
# lineage versions computed from lineage-mean underlying quantities where possible; otherwise majority of members
contin = {"back<0": (RGa["back"], lambda v: v < 0), "back<front": (RGa["back"] - RGa["front"], lambda v: v < 0), "d40_50>0": (RGa["d40_50"], lambda v: v > 0),
          "d40_50>d50_60 (switch at mid-depth)": (prof[:, 4] - prof[:, 5], lambda v: v > 0), "d40_50>back": (RGa["d40_50"] - RGa["back"], lambda v: v > 0),
          "d50_60<0": (prof[:, 5], lambda v: v < 0), "d60_70<0": (prof[:, 6], lambda v: v < 0), "d70_80<0": (prof[:, 7], lambda v: v < 0), "d80_90<0": (prof[:, 8], lambda v: v < 0),
          "d90_100<0": (prof[:, 9], lambda v: v < 0), "d50_70<0": (RGa["d50_70"], lambda v: v < 0), "d0>0": (prof[:, 0], lambda v: v > 0), "d10_20>0": (prof[:, 1], lambda v: v > 0),
          "front>0": (RGa["front"], lambda v: v > 0), "final layer z>0 (task mean)": (fz.mean(1), lambda v: v > 0),
          "serial back < control back": (RGa["back"] - RGfd["back"], lambda v: v < 0), "control back<0": (RGfd["back"], lambda v: v < 0),
          "control d30_40>0": (RGfd["d30_40"], lambda v: v > 0), "control d40_50>0": (RGfd["d40_50"], lambda v: v > 0),
          "binding back<0": (RG["back"][:, 2], lambda v: v < 0), "pointer back<0": (RG["back"][:, 0], lambda v: v < 0), "relational back<0": (RG["back"][:, 3], lambda v: v < 0),
          "arith back<0": (RG["back"][:, 1], lambda v: v < 0), "binding back < arith back": (Bz[:, 2] - Bz[:, 1], lambda v: v < 0),
          "chain-task back < write-k back": (Bz[:, 2:].mean(1) - Bz[:, :2].mean(1), lambda v: v < 0),
          "arith d40_50>0": (RG["d40_50"][:, 1], lambda v: v > 0), "pointer d40_50>0": (RG["d40_50"][:, 0], lambda v: v > 0),
          "write-k d40_50 > chain d40_50": (RG["d40_50"][:, :2].mean(1) - RG["d40_50"][:, 2:].mean(1), lambda v: v > 0),
          "|back|>|front|": (np.abs(RGa["back"]) - np.abs(RGa["front"]), lambda v: v > 0)}
profL = lin_mean(prof)
for name, mask in feat.items():
    nm = cnt(mask)
    if name in contin: nl_ = cnt(contin[name][1](lin_mean(contin[name][0]))); how = "lineage-mean"
    elif name == "min decile at 50-80%": nl_ = cnt(np.isin(profL.argmin(1), [5, 6, 7])); how = "lineage-mean profile"
    elif name == "min decile in back half": nl_ = cnt(profL.argmin(1) >= 5); how = "lineage-mean profile"
    elif name == "front-half max at 30-50%": nl_ = cnt(np.isin(profL[:, :5].argmax(1), [3, 4])); how = "lineage-mean profile"
    elif name == "profile r>0 with leave-lineage-out mean": nl_ = cnt(lo_mu_L > 0); how = "lineage-mean profile vs other lineages"
    elif name == "profile r>0.5 with leave-lineage-out mean": nl_ = cnt(lo_mu_L > 0.5); how = "lineage-mean profile vs other lineages"
    elif name == "binding strongest back task": nl_ = cnt(lin_mean(Bz).argmin(1) == 2); how = "lineage-mean"
    elif name == "arith weakest back task": nl_ = cnt(lin_mean(Bz).argmax(1) == 1); how = "lineage-mean"
    else: nl_ = cnt(lin_mean(mask.astype(float)) > .5); how = "majority of members"
    G[name] = {"models": nm, "lineages": nl_, "lineage_rule": how, "bases": cnt(base_sel(mask)), "meets_13_8": bool(nm >= 13 and nl_ >= 8),
               "fail_models": [m for m, ok in zip(ORDER, mask) if not ok]}
G["_leave_lineage_out_r"] = dict(zip(ORDER, np.round(lo_mu, 3).tolist())); G["_leave_lineage_out_r_median"] = float(np.median(lo_mu))
G["_leave_lineage_out_r_lineages"] = dict(zip(LINS, np.round(lo_mu_L, 3).tolist()))
G["_argmin_decile"] = dict(zip(ORDER, argmin_dec.tolist()))
out["G_scan"] = G
print("\nG. scan (models/17, lineages/11, bases/11; * = meets >=13/17 and >=8/11)")
for name, d in G.items():
    if name.startswith("_"): continue
    print(f"  {'*' if d['meets_13_8'] else ' '} {name:52s} {d['models']:2d}/17 {d['lineages']:2d}/11 bases {d['bases']:2d}  fail: {', '.join(x.replace('Llama-','L').replace('Qwen2.5-','Q').replace('OLMo-2-1124-','O').replace('Mistral-','M') for x in d['fail_models'])}")
print("  leave-lineage-out profile r: median", round(G["_leave_lineage_out_r_median"], 2), {k.replace("Qwen2.5-", "Q"): v for k, v in G["_leave_lineage_out_r"].items()})

# ------------------------------------------------------------------ R. robustness of key counts
def key_counts(Zv, models, Zfdv=None):
    """Zv: n x 4 x 10 over `models`; counts over models and lineages present"""
    pr = Zv.mean(1); back = pr[:, 5:].mean(1); front = pr[:, :5].mean(1); d4 = pr[:, 4]; sw = pr[:, 4] - pr[:, 5]
    bz = Zv[:, :, 5:].mean(2)
    lins = sorted(set(LIN[m] for m in models), key=lambda x: ORDER.index(x))
    def lm(v): return np.array([np.mean([v[i] for i, m in enumerate(models) if LIN[m] == L_], 0) for L_ in lins])
    r = {"n_models": len(models), "n_lineages": len(lins),
         "back<0": [cnt(back < 0), cnt(lm(back) < 0)], "back<front": [cnt(back < front), cnt(lm(back - front) < 0)],
         "d40_50>0": [cnt(d4 > 0), cnt(lm(d4) > 0)], "switch d40_50>d50_60": [cnt(sw > 0), cnt(lm(sw) > 0)],
         "min decile at 50-80%": [cnt(np.isin(pr.argmin(1), [5, 6, 7])), cnt(np.isin(lm(pr).argmin(1), [5, 6, 7]))],
         "binding strongest": [cnt(bz.argmin(1) == 2), cnt(lm(bz).argmin(1) == 2)], "arith weakest": [cnt(bz.argmax(1) == 1), cnt(lm(bz).argmax(1) == 1)],
         "binding back<0": [cnt(bz[:, 2] < 0), cnt(lm(bz[:, 2]) < 0)], "arith back<0": [cnt(bz[:, 1] < 0), cnt(lm(bz[:, 1]) < 0)],
         "d0>0": [cnt(pr[:, 0] > 0), cnt(lm(pr[:, 0]) > 0)],
         "pointer back<0": [cnt(bz[:, 0] < 0), cnt(lm(bz[:, 0]) < 0)], "relational back<0": [cnt(bz[:, 3] < 0), cnt(lm(bz[:, 3]) < 0)],
         "arith d40_50>0": [cnt(Zv[:, 1, 4] > 0), cnt(lm(Zv[:, 1, 4]) > 0)], "back<0 in >=3 tasks": [cnt((bz < 0).sum(1) >= 3), cnt((lm(bz) < 0).sum(1) >= 3)]}
    return r
R = {"primary_q10": key_counts(Z, ORDER)}
for st in ["q25", "q50", "entropy", "top1"]: R[st] = key_counts(from_consist(st), ORDER)
m16 = [m for m in ORDER if m != "Qwen2.5-7B-Instruct"]
Zs, miss = from_cells("eager", "sink_head", m16); R["sink_heads_removed_16_no_Q7I"] = key_counts(Zs, m16) if Zs is not None else {"missing": miss}
Ze, _ = from_cells("eager", "all", m16); R["eager_per_wording_z_16_no_Q7I"] = key_counts(Ze, m16)
msd = [m for m in ORDER if from_cells("sdpa", "all", [m])[0] is not None]
Zsd, _ = from_cells("sdpa", "all", msd); R["sdpa_%d" % len(msd)] = key_counts(Zsd, msd); R["sdpa_models"] = msd
Zeq, _ = from_cells("eager", "all", [m for m in msd if m != "Qwen2.5-7B-Instruct"])
R["eager_same_as_sdpa_minus_Q7I"] = key_counts(Zeq, [m for m in msd if m != "Qwen2.5-7B-Instruct"])
R["bases_only_11"] = key_counts(Z[bi], BASES)
# per wording (eager/all; Q7I sdpa)
Cl = json.load(open(f"{SP}/vet3/cells.json")); Dw = {(r["model"], r["fam"], r["w"]): np.array(r["z"]) for r in Cl if r["variant"] == "all" and r["fam"] in T and r["attn"] == ATTN[r["model"]]}
for w in ("w0", "w1"):
    Zw = np.array([[dec_prof(Dw[(m, t, w)]) for t in T] for m in ORDER]); R[f"wording_{w}"] = key_counts(Zw, ORDER)
out["R_robustness"] = R
print("\nR. robustness of key counts [models, lineages]")
keys = ["back<0", "back<front", "d40_50>0", "switch d40_50>d50_60", "min decile at 50-80%", "binding strongest", "arith weakest", "binding back<0", "arith back<0", "d0>0",
        "pointer back<0", "relational back<0", "arith d40_50>0", "back<0 in >=3 tasks"]
print(f"  {'variant':32s} n  " + " | ".join(k[:14] for k in keys))
for name, r in R.items():
    if name == "sdpa_models" or "missing" in r: continue
    print(f"  {name:32s} {r['n_models']:2d}/{r['n_lineages']:2d} " + " | ".join(f"{r[k][0]:2d},{r[k][1]:2d}".ljust(14) for k in keys))

# ------------------------------------------------------------------ H. follow-ups
Hx = {}
# (1) mid-depth switch per task and in the control; robustness
sw_t = RG["d40_50"] - Z[:, :, 5]
Hx["switch_per_task"] = {t: summary(sw_t[:, j], lambda v: v > 0) for j, t in enumerate(T)}
Hx["switch_control"] = summary(RGfd["d40_50"] - Zfd[:, 5], lambda v: v > 0)
Hx["switch_size_lineage_ci"] = boot_ci(lin_mean(prof[:, 4] - prof[:, 5]))
Hx["switch_control_lineage_ci"] = boot_ci(lin_mean(RGfd["d40_50"] - Zfd[:, 5]))
Hx["serial_d30_40_gt0"] = summary(RGa["d30_40"], lambda v: v > 0)
# (2) response magnitude: mean |z| over layers, back vs front; final layer vs others
absz = {m: np.mean([np.abs(zl[(m, t)]) for t in T], 0) for m in ORDER}
absfd = {m: np.abs(CC[(m, "fixed_digit")]["z"]) for m in ORDER}
def halves(v):
    L_ = len(v); x = np.arange(L_) / (L_ - 1); return v[x < .5].mean(), v[x >= .5].mean()
bf = np.array([halves(absz[m]) for m in ORDER]); bffd = np.array([halves(absfd[m]) for m in ORDER])
Hx["mean_abs_z_back_gt_front"] = summary(bf[:, 1] - bf[:, 0], lambda v: v > 0)
Hx["mean_abs_z_back_minus_front_lineage_ci"] = boot_ci(lin_mean(bf[:, 1] - bf[:, 0]))
Hx["mean_abs_z_back_gt_front_control"] = summary(bffd[:, 1] - bffd[:, 0], lambda v: v > 0)
Hx["mean_abs_z_front_back"] = {m: bf[i].round(2).tolist() for i, m in enumerate(ORDER)}
fin_rank = np.array([(absz[m] >= absz[m][-1]).sum() for m in ORDER])       # 1 = final layer has the largest mean |z|
fin_pct = np.array([(absz[m] < absz[m][-1]).mean() for m in ORDER])         # fraction of layers below the final layer
fin_rank_fd = np.array([(absfd[m] >= absfd[m][-1]).sum() for m in ORDER])
Hx["final_layer_absz_rank"] = dict(zip(ORDER, fin_rank.tolist()))
Hx["final_layer_largest_absz"] = cnt(fin_rank == 1); Hx["final_layer_top3_absz"] = cnt(fin_rank <= 3)
Hx["final_layer_above_90pct_of_layers"] = summary(fin_pct, lambda v: v >= .9)
Hx["final_layer_largest_absz_control"] = cnt(fin_rank_fd == 1); Hx["final_layer_top3_absz_control"] = cnt(fin_rank_fd <= 3)
# significance rate by layer position: final layer vs all other layers (pooled over 68 cells)
sig_fin = np.mean([pl[(m, t)][-1] < .05 for m in ORDER for t in T]); sig_oth = np.mean(np.concatenate([pl[(m, t)][:-1] < .05 for m in ORDER for t in T]))
sig_pen = np.mean([pl[(m, t)][-2] < .05 for m in ORDER for t in T])
Hx["sig_rate_final_vs_other_vs_penultimate"] = [float(sig_fin), float(sig_oth), float(sig_pen)]
Hx["final_layer_direction_by_family"] = {fam: [round(float(fz[ORDER.index(m)].mean()), 2) for m in ORDER if FAMOF[m] == fam] for fam in ["Llama", "Qwen", "OLMo", "Mistral"]}
# (3) scale: where the spreading sits (50-70% vs 70-100%), partials for 50-70%, other statistics
e_l = RGa["d50_70"] - RGa["d70_100"]            # negative = spreading sits earlier (50-70) rather than late
vd = dict(zip(ORDER, e_l))
Hx["scale_early_minus_late"] = {"bases_11": sp(logN[bi], e_l[bi]), "within_family_pairs_larger_earlier": within_pairs(vd, FAMS_MULTI),
                                "within_family_pairs_excl_qwen": within_pairs(vd, {k: v_ for k, v_ in FAMS_MULTI.items() if k != "Qwen"})}
Hx["scale_d70_100"] = {"bases_11": sp(logN[bi], RGa["d70_100"][bi]), "within_family_pairs_larger_spreads_more": within_pairs(dict(zip(ORDER, RGa["d70_100"])), FAMS_MULTI)}
Hx["scale_d50_70_partial"] = {"logN|H": partial_sp(logN[bi], RGa["d50_70"][bi], Hh[bi]), "H|logN": partial_sp(Hh[bi], RGa["d50_70"][bi], logN[bi]),
                              "logN|acc": partial_sp(logN[bi], RGa["d50_70"][bi], acc[bi]), "acc|logN": partial_sp(acc[bi], RGa["d50_70"][bi], logN[bi])}
Hx["scale_d50_70_values"] = {m: round(float(RGa["d50_70"][ORDER.index(m)]), 2) for m in BASES}
Hx["scale_d70_100_values"] = {m: round(float(RGa["d70_100"][ORDER.index(m)]), 2) for m in BASES}
alt = {}
for st in ["q25", "q50", "entropy", "top1"]:
    Zv = from_consist(st); v5 = Zv.mean(1)[:, 5:7].mean(1); vb = Zv.mean(1)[:, 5:].mean(1)
    alt[st] = {"d50_70_bases": sp(logN[bi], v5[bi]), "d50_70_pairs": within_pairs(dict(zip(ORDER, v5)), FAMS_MULTI),
               "d50_70_pairs_excl_qwen": within_pairs(dict(zip(ORDER, v5)), {k: v_ for k, v_ in FAMS_MULTI.items() if k != "Qwen"}),
               "back_bases": sp(logN[bi], vb[bi]), "back_pairs": within_pairs(dict(zip(ORDER, vb)), FAMS_MULTI),
               "acc_d50_70_H32": sp(acc[Hh == 32], v5[Hh == 32]), "acc_d50_70_nonqwen": sp(acc[[FAMOF[m] != "Qwen" for m in ORDER]], v5[[FAMOF[m] != "Qwen" for m in ORDER]])}
Hx["scale_and_acc_other_stats"] = alt
# (4) accuracy, within-model task-centred association: permutation p (shuffle task labels of back z within model)
obs_wm = np.nanmean([spearmanr(Ac[i], Bc[i]).correlation for i in range(17)]); nullwm = []
for _ in range(5000): nullwm.append(np.nanmean([spearmanr(Ac[i], Bc[i][rng.permutation(4)]).correlation for i in range(17)]))
Hx["acc_within_model_task_centred"] = {"mean_rho": float(obs_wm), "p_perm": float((np.sum(np.array(nullwm) <= obs_wm) + 1) / 5001),
                                       "lineage_level_mean_rho": float(np.nanmean([spearmanr(AL[i], BL[i]).correlation for i in range(11)])),
                                       "lineage_level_n_negative": cnt(np.array([spearmanr(AL[i], BL[i]).correlation for i in range(11)]) < 0)}
# (5) task pairs under robustness variants
def pair_counts(Zv, models):
    bz = Zv[:, :, 5:].mean(2); lins = sorted(set(LIN[m] for m in models), key=lambda x: ORDER.index(x))
    lm = lambda v: np.array([np.mean([v[i] for i, m in enumerate(models) if LIN[m] == L_], 0) for L_ in lins])
    return {"binding<relational": [cnt(bz[:, 2] < bz[:, 3]), cnt(lm(bz)[:, 2] < lm(bz)[:, 3])], "binding<arith": [cnt(bz[:, 2] < bz[:, 1]), cnt(lm(bz)[:, 2] < lm(bz)[:, 1])],
            "binding<pointer": [cnt(bz[:, 2] < bz[:, 0]), cnt(lm(bz)[:, 2] < lm(bz)[:, 0])], "arith>pointer": [cnt(bz[:, 1] > bz[:, 0]), cnt(lm(bz)[:, 1] > lm(bz)[:, 0])],
            "arith>relational": [cnt(bz[:, 1] > bz[:, 3]), cnt(lm(bz)[:, 1] > lm(bz)[:, 3])], "n": [len(models), len(lins)]}
PCn = {"primary_q10": pair_counts(Z, ORDER)}
for st in ["q25", "q50", "entropy", "top1"]: PCn[st] = pair_counts(from_consist(st), ORDER)
PCn["sink_heads_removed_16"] = pair_counts(Zs, m16); PCn["sdpa_14"] = pair_counts(Zsd, msd)
for w in ("w0", "w1"): PCn[f"wording_{w}"] = pair_counts(np.array([[dec_prof(Dw[(m, t, w)]) for t in T] for m in ORDER]), ORDER)
Hx["task_pairs_robustness"] = PCn
# (6) switch and LLO-correlation under robustness variants
def more_counts(Zv, models):
    pr = Zv.mean(1); lins = sorted(set(LIN[m] for m in models), key=lambda x: ORDER.index(x))
    lm = lambda v: np.array([np.mean([v[i] for i, m in enumerate(models) if LIN[m] == L_], 0) for L_ in lins])
    r = []
    for i, m in enumerate(models):
        oth = [q for q, x in enumerate(models) if LIN[x] != LIN[m]]; Lo = sorted(set(LIN[models[q]] for q in oth))
        mu_lo = np.mean([pr[[q for q in oth if LIN[models[q]] == L_]].mean(0) for L_ in Lo], 0); r.append(np.corrcoef(pr[i], mu_lo)[0, 1])
    r = np.array(r); PL = lm(pr); rL = np.array([np.corrcoef(PL[i], np.delete(PL, i, 0).mean(0))[0, 1] for i in range(len(PL))])
    return {"LLO_r>0": [cnt(r > 0), cnt(rL > 0)], "LLO_r_median": float(np.median(r)), "n": [len(models), len(lins)]}
MC = {"primary_q10": more_counts(Z, ORDER)}
for st in ["q25", "q50", "entropy", "top1"]: MC[st] = more_counts(from_consist(st), ORDER)
MC["sink_heads_removed_16"] = more_counts(Zs, m16); MC["sdpa_14"] = more_counts(Zsd, msd)
Hx["LLO_robustness"] = MC
# (7) final-layer direction split by family (Qwen vs others): Fisher exact on models and on lineages
from scipy.stats import fisher_exact
isq = np.array([FAMOF[m] == "Qwen" for m in ORDER]); fneg = fz.mean(1) < 0
fL = lin_mean(fz.mean(1)); isqL = np.array([L_.startswith("Qwen") for L_ in LINS])
Hx["final_layer_family_split"] = {"qwen_neg_models": [cnt(fneg & isq), cnt(isq)], "other_neg_models": [cnt(fneg & ~isq), cnt(~isq)],
                                  "fisher_p_models": float(fisher_exact([[cnt(fneg & isq), cnt(~fneg & isq)], [cnt(fneg & ~isq), cnt(~fneg & ~isq)]])[1]),
                                  "qwen_neg_lineages": [cnt((fL < 0) & isqL), cnt(isqL)], "other_neg_lineages": [cnt((fL < 0) & ~isqL), cnt(~isqL)],
                                  "fisher_p_lineages": float(fisher_exact([[cnt((fL < 0) & isqL), cnt((fL >= 0) & isqL)], [cnt((fL < 0) & ~isqL), cnt((fL >= 0) & ~isqL)]])[1]),
                                  "control_qwen_neg": [cnt((fzfd < 0) & isq), cnt(isq)], "control_other_neg": [cnt((fzfd < 0) & ~isq), cnt(~isq)],
                                  "tasks_conc_all4": [m for i, m in enumerate(ORDER) if (fz[i] > 0).all()], "tasks_spread_all4": [m for i, m in enumerate(ORDER) if (fz[i] < 0).all()]}
# (8) 40-50% without arithmetic
na = RG["d40_50"][:, [0, 2, 3]].mean(1)
Hx["d40_50_gt0_without_arith"] = summary(na, lambda v: v > 0); Hx["d40_50_without_arith_lineage_ci"] = boot_ci(lin_mean(na))
# (9) PC1 scale under other statistics (PCA of each statistic's signatures, oriented so + = more spreading at 50-70%)
pcs = {}
for st in ["q10", "q25", "q50", "entropy", "top1"]:
    Zv = from_consist(st); av = decompose(Zv)[1]; U, s_, Vt = np.linalg.svd(av - av.mean(0), full_matrices=False); sc = U[:, 0] * s_[0]
    if np.corrcoef(sc, av[:, 5:7].mean(1))[0, 1] > 0: sc = -sc
    pcs[st] = {"bases_11": sp(logN[bi], sc[bi]), "pairs": within_pairs(dict(zip(ORDER, -sc)), FAMS_MULTI),
               "pairs_excl_qwen": within_pairs(dict(zip(ORDER, -sc)), {k: v_ for k, v_ in FAMS_MULTI.items() if k != "Qwen"}),
               "r_with_d50_70_signature": float(np.corrcoef(sc, av[:, 5:7].mean(1))[0, 1]), "explained": float(s_[0] ** 2 / (s_ ** 2).sum())}
Hx["pc1_scale_by_stat"] = pcs
Hx["pc1_values_bases"] = {m: round(float(PC1[ORDER.index(m)]), 2) for m in BASES}
Hx["model_mean_back_per_task"] = {t: float(RG["back"][:, j].mean()) for j, t in enumerate(T)}
Hx["serial_minus_control_back_per_model"] = {m: round(float(dser[i]), 2) for i, m in enumerate(ORDER)}
Hx["control_back_per_model"] = {m: round(float(RGfd["back"][i]), 2) for i, m in enumerate(ORDER)}
# (10) null for the leave-lineage-out profile correlation: circularly shift each model's profile by s = 1..9 deciles
#      (keeps its shape, destroys its depth alignment); expected count of r > 0 over the 9 shifts
shift_cnt = []
for s_ in range(1, 10):
    c_ = 0
    for i, m in enumerate(ORDER):
        others = [j for j, x in enumerate(ORDER) if LIN[x] != LIN[m]]; Lo = sorted(set(LIN[ORDER[j]] for j in others))
        mu_lo = np.mean([prof[[j for j in others if LIN[ORDER[j]] == L_]].mean(0) for L_ in Lo], 0)
        c_ += np.corrcoef(np.roll(prof[i], s_), mu_lo)[0, 1] > 0
    shift_cnt.append(int(c_))
Hx["LLO_shift_null_counts"] = shift_cnt; Hx["LLO_shift_null_mean"] = float(np.mean(shift_cnt)); Hx["LLO_shift_null_max"] = int(max(shift_cnt))
# (11) lineage-level permutation p for the within-lineage task-centred accuracy association
obsL = np.nanmean([spearmanr(AL[i], BL[i]).correlation for i in range(11)]); nL = []
for _ in range(5000): nL.append(np.nanmean([spearmanr(AL[i], BL[i][rng.permutation(4)]).correlation for i in range(11)]))
Hx["acc_within_model_task_centred"]["lineage_level_p_perm"] = float((np.sum(np.array(nL) <= obsL) + 1) / 5001)
# (12) binding vs relational: is binding also the more accurate task, and does the accuracy gap track the spreading gap?
accB = np.array([ACC[m]["binding"] for m in ORDER]); accR = np.array([ACC[m]["relational"] for m in ORDER])
Hx["binding_vs_relational"] = {"acc_binding_gt_relational_models": cnt(accB > accR), "spread_gap_vs_acc_gap_17": sp(accB - accR, Bz[:, 2] - Bz[:, 3]),
                               "spread_gap_vs_acc_gap_bases": sp((accB - accR)[bi], (Bz[:, 2] - Bz[:, 3])[bi]),
                               "mean_acc_binding": float(accB.mean()), "mean_acc_relational": float(accR.mean())}
Hx["task_accuracy"] = {m: {f: round(ACC[m][f], 3) for f in T5} for m in ORDER}
out["H_followups"] = Hx
print("  binding vs relational:", Hx["binding_vs_relational"])
print("  LLO shift null counts:", shift_cnt, " lineage-level within task-centred acc p:", Hx["acc_within_model_task_centred"]["lineage_level_p_perm"])
print("  final-layer family split:", Hx["final_layer_family_split"])
print("  40-50 >0 without arith:", Hx["d40_50_gt0_without_arith"]["models"], Hx["d40_50_gt0_without_arith"]["lineages"], np.round(Hx["d40_50_without_arith_lineage_ci"], 2))
for st, d in pcs.items(): print("  PC1 scale", st, d)
print("  PC1 bases:", Hx["pc1_values_bases"])
print("  model-mean back z per task:", {k: round(v, 2) for k, v in Hx["model_mean_back_per_task"].items()})
print("  control back per model:", Hx["control_back_per_model"])
print("  serial - control back:", Hx["serial_minus_control_back_per_model"])
print("\nH. follow-ups")
print("  switch (40-50 > 50-60) per task:", {t: (d["models"], d["lineages"]) for t, d in Hx["switch_per_task"].items()}, " control:", (Hx["switch_control"]["models"], Hx["switch_control"]["lineages"]),
      " lineage mean jump", np.round(Hx["switch_size_lineage_ci"], 2), " control", np.round(Hx["switch_control_lineage_ci"], 2))
print("  serial 30-40% > 0:", Hx["serial_d30_40_gt0"]["models"], Hx["serial_d30_40_gt0"]["lineages"])
print("  mean |z| back > front:", Hx["mean_abs_z_back_gt_front"]["models"], Hx["mean_abs_z_back_gt_front"]["lineages"], np.round(Hx["mean_abs_z_back_minus_front_lineage_ci"], 2),
      "| control:", Hx["mean_abs_z_back_gt_front_control"]["models"], Hx["mean_abs_z_back_gt_front_control"]["lineages"])
print("  final layer mean|z| rank:", Hx["final_layer_absz_rank"], " largest:", Hx["final_layer_largest_absz"], " top3:", Hx["final_layer_top3_absz"],
      " >=90th pct:", Hx["final_layer_above_90pct_of_layers"]["models"], Hx["final_layer_above_90pct_of_layers"]["lineages"], " control largest/top3:", Hx["final_layer_largest_absz_control"], Hx["final_layer_top3_absz_control"])
print("  sig rate final / other / penultimate:", np.round(Hx["sig_rate_final_vs_other_vs_penultimate"], 3), " final-layer direction by family:", Hx["final_layer_direction_by_family"])
print("  scale early-minus-late:", Hx["scale_early_minus_late"], "\n  scale d70_100:", Hx["scale_d70_100"])
print("  d50_70 partials:", {k: np.round(v, 3).tolist() for k, v in Hx["scale_d50_70_partial"].items()})
print("  d50_70 values:", Hx["scale_d50_70_values"], "\n  d70_100 values:", Hx["scale_d70_100_values"])
for st, d in alt.items(): print("  ", st, {k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in d.items()})
print("  acc within-model task-centred:", Hx["acc_within_model_task_centred"])
for k_, v in PCn.items(): print("  task pairs", k_, v)
for k_, v in MC.items(): print("  LLO", k_, v)

json.dump(out, open(f"{OUT}/other_trends.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("\nwrote", f"{OUT}/other_trends.json")
