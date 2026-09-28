"""Skeptical-review spot checks of the 'other trends' claims (T1-T15).
Independent recomputation from consist.json (primary q10), vet3 cache (numeral control, effect sizes, item counts),
altexp_layers.csv (entropy/correctness-adjusted k-slope t), model_diff3.json, pairs_q10.json.
Writes r1.json next to this file."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, sys, json, itertools, numpy as np, pandas as pd
from scipy.stats import spearmanr, binomtest
SP = str(paths.ANALYSIS)
OUT = os.path.join(SP, "trends", "review_other")
os.environ["VPRIM"] = "y10"
sys.path.insert(0, SP); sys.path.insert(0, f"{SP}/vet3")
import model_diff as MDm; MDm.SP = SP
from model_diff import ORDER, T, LIN, dec_prof
from v2lib import load as vload, zprof
rng = np.random.default_rng(1)
LINS = sorted(set(LIN.values()), key=ORDER.index)
FAM = {m: ("Llama" if m.startswith("Llama") else "Qwen" if m.startswith("Qwen") else "OLMo" if m.startswith("OLMo") else "Mistral") for m in ORDER}
FAMS = ["Llama", "Qwen", "OLMo", "Mistral"]
BASES = [m for m in ORDER if LIN[m] == m]
out = {}
def lm(v):  # lineage mean over axis 0
    v = np.asarray(v, float); return np.array([v[[i for i, m in enumerate(ORDER) if LIN[m] == L]].mean(0) for L in LINS])
def fm(v):  # family mean of lineage means
    vl = lm(v); return np.array([vl[[i for i, L in enumerate(LINS) if FAM[L] == f]].mean(0) for f in FAMS])
def binp(k, n): return float(binomtest(int(k), int(n), .5).pvalue)
def tci(v):
    from scipy.stats import t as td
    v = np.asarray(v, float); n = len(v); se = v.std(ddof=1) / np.sqrt(n); q = td.ppf(.975, n - 1)
    return [float(v.mean()), float(v.mean() - q * se), float(v.mean() + q * se)]
def boot(v, NB=10000):
    v = np.asarray(v, float); b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)])
    return [float(v.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))]
def cluster_boot(v, NB=10000):
    """two-stage: resample families, then lineages within family; statistic = mean over sampled lineages"""
    vl = lm(v); groups = [[i for i, L in enumerate(LINS) if FAM[L] == f] for f in FAMS]; res = []
    for _ in range(NB):
        fs = rng.integers(0, 4, 4); vals = []
        for f in fs:
            g = groups[f]; vals += list(vl[np.array(g)[rng.integers(0, len(g), len(g))]])
        res.append(np.mean(vals))
    return [float(vl.mean()), float(np.quantile(res, .025)), float(np.quantile(res, .975))]

# ---------------- primary decile profiles
C = {r["model"]: r for r in json.load(open(f"{SP}/consist.json")) if r["stat"] == "q10"}
Z = np.array([[dec_prof(np.array(C[m]["families"][t]["z"])) for t in T] for m in ORDER])
NIT = {m: {t: C[m]["families"][t]["n"] for t in T} for m in ORDER}
prof = Z.mean(1); back = prof[:, 5:].mean(1); front = prof[:, :5].mean(1)
# T1 recompute
t1 = {"back<front_models": int((back < front).sum()), "back<front_lineages": int((lm(back - front) < 0).sum()),
      "back<front_families": int((fm(back - front) < 0).sum()),
      "back<0_models": int((back < 0).sum()), "back<0_lineages": int((lm(back) < 0).sum()), "back<0_families": int((fm(back) < 0).sum()),
      "family_means_back": dict(zip(FAMS, np.round(fm(back), 2).tolist())), "family_means_back_minus_front": dict(zip(FAMS, np.round(fm(back - front), 2).tolist())),
      "lineage_back_minus_front_boot": boot(lm(back - front)), "lineage_back_minus_front_t": tci(lm(back - front)),
      "back_minus_front_family_cluster_boot": cluster_boot(back - front), "back_family_cluster_boot": cluster_boot(back),
      "lineage_back_t": tci(lm(back)),
      "bases_back<0": int((back[[ORDER.index(m) for m in BASES]] < 0).sum())}
for j, t in enumerate(T):
    b = Z[:, j, 5:].mean(1); t1[f"{t}_back<0"] = [int((b < 0).sum()), int((lm(b) < 0).sum()), int((fm(b) < 0).sum())]
    t1[f"{t}_lineage_back_boot"] = boot(lm(b)); t1[f"{t}_family_cluster_boot"] = cluster_boot(b)
out["T1"] = t1

# ---------------- entropy / correctness adjustment of the k-slope (altexp_layers.csv; per-wording t, averaged)
A = pd.read_csv(f"{SP}/vet3/altexp_layers.csv")
def adj_profiles(fit):
    P = np.full((17, 4, 10), np.nan)
    for i, m in enumerate(ORDER):
        for j, t in enumerate(T):
            s = A[(A.model == m) & (A.task == t) & (A.fit == fit)]
            if s.empty: continue
            ws = [s[s.w == w].sort_values("x")["t"].values for w in ("w0", "w1")]
            if any(len(x) == 0 for x in ws): continue
            P[i, j] = dec_prof(0.5 * (ws[0] + ws[1]))
    return P
adj = {}
for fit in ["k", "k+ent", "k+corr", "k+ent+corr", "k+ent+corr+logpgold"]:
    P = adj_profiles(fit); pr = np.nanmean(P, 1); bk = pr[:, 5:].mean(1); fr = pr[:, :5].mean(1); sw = pr[:, 4] - pr[:, 5]
    ok = ~np.isnan(bk)
    adj[fit] = {"n_models": int(ok.sum()), "back<0": [int((bk[ok] < 0).sum()), int((lm(np.where(ok, bk, 0))[:] < 0).sum())],
                "back<front": [int(((bk - fr)[ok] < 0).sum()), int((lm(bk - fr) < 0).sum())],
                "switch": [int((sw[ok] > 0).sum()), int((lm(sw) > 0).sum())], "d40_50>0": [int((pr[ok, 4] > 0).sum()), int((lm(pr[:, 4]) > 0).sum())],
                "lineage_mean_back_t": boot(lm(bk)), "lineage_mean_back_minus_front_t": boot(lm(bk - fr)),
                "per_model_back": dict(zip(ORDER, np.round(bk, 2).tolist())),
                "per_task_back<0_models": [int((np.nanmean(P[:, j, 5:], 1) < 0).sum()) for j in range(4)],
                "d50_70": dict(zip(ORDER, np.round(pr[:, 5:7].mean(1), 2).tolist()))}
out["adjusted"] = adj

# ---------------- numeral control: effect size, item counts, permutation scheme, entropy/accuracy vs k
def cell(m, f, attn):
    a, b = vload(m, f, "w0", attn, True), vload(m, f, "w1", attn, True)
    if a is None or b is None: return None
    ka = {(int(k), int(bd)): i for i, (k, bd) in enumerate(zip(a["k"], a["body"]))}
    kb = {(int(k), int(bd)): i for i, (k, bd) in enumerate(zip(b["k"], b["body"]))}
    keys = sorted(set(ka) & set(kb)); ia = np.array([ka[x] for x in keys]); ib = np.array([kb[x] for x in keys])
    Y = 0.5 * (a["y10"][ia] + b["y10"][ib]); k = np.array([x[0] for x in keys], float); body = np.array([x[1] for x in keys])
    within = f in ("binding", "relational")
    sl, sd, z = zprof(Y, k, body, within)
    sl2, sd2, z2 = zprof(Y, k, body, not within)       # the other permutation scheme
    ent = 0.5 * (a["ent"][ia] + b["ent"][ib]); corr = 0.5 * (a["correct"][ia].astype(float) + b["correct"][ib].astype(float))
    kc = k - k.mean()
    return {"z": z, "z_other_scheme": z2, "rel": 100 * sl / np.abs(Y.mean(0)), "n": len(keys),
            "ent_slope": float(kc @ (ent - ent.mean()) / (kc @ kc)), "acc_slope": float(kc @ (corr - corr.mean()) / (kc @ kc)),
            "acc": float(corr.mean()), "acc_k1": float(corr[k == k.min()].mean()), "acc_kmax": float(corr[k == k.max()].mean())}
ATTN = {m: ("sdpa" if m == "Qwen2.5-7B-Instruct" else "eager") for m in ORDER}
T5 = T + ["fixed_digit"]
CC = {(m, f): cell(m, f, ATTN[m]) for m in ORDER for f in T5}
json.dump({f"{m}|{f}": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()} for (m, f), d in CC.items()},
          open(f"{OUT}/r1_cells.json", "w"))
Zfd = np.array([dec_prof(CC[(m, "fixed_digit")]["z"]) for m in ORDER])
Rel = np.array([[dec_prof(CC[(m, t)]["rel"]) for t in T5] for m in ORDER])       # 17 x 5 x 10
Zc = np.array([[dec_prof(CC[(m, t)]["z"]) for t in T5] for m in ORDER])
Zalt = np.array([[dec_prof(CC[(m, t)]["z_other_scheme"]) for t in T5] for m in ORDER])
ctl = {}
ctl["n_items"] = {t: [CC[(m, t)]["n"] for m in ORDER] for t in T5}
ctl["ent_slope_gt0"] = {t: int(sum(CC[(m, t)]["ent_slope"] > 0 for m in ORDER)) for t in T5}
ctl["ent_slope_mean"] = {t: float(np.mean([CC[(m, t)]["ent_slope"] for m in ORDER])) for t in T5}
ctl["acc_slope_lt0"] = {t: int(sum(CC[(m, t)]["acc_slope"] < 0 for m in ORDER)) for t in T5}
ctl["acc_k1_vs_kmax_mean"] = {t: [float(np.mean([CC[(m, t)]["acc_k1"] for m in ORDER])), float(np.mean([CC[(m, t)]["acc_kmax"] for m in ORDER]))] for t in T5}
ctl["fixed_digit_ent_slope_per_model"] = {m: round(CC[(m, "fixed_digit")]["ent_slope"], 3) for m in ORDER}
ctl["fixed_digit_acc_slope_per_model"] = {m: round(CC[(m, "fixed_digit")]["acc_slope"], 4) for m in ORDER}
# serial vs control: z, per-task z, effect size
bz = Zc[:, :, 5:].mean(2); brel = Rel[:, :, 5:].mean(2)
ser_z = bz[:, :4].mean(1); ser_rel = brel[:, :4].mean(1)
ctl["serial<control_z"] = [int((ser_z < bz[:, 4]).sum()), int((lm(ser_z - bz[:, 4]) < 0).sum())]
ctl["serial<control_rel"] = [int((ser_rel < brel[:, 4]).sum()), int((lm(ser_rel - brel[:, 4]) < 0).sum())]
ctl["serial_minus_control_rel_lineage_boot"] = boot(lm(ser_rel - brel[:, 4]))
ctl["serial_minus_control_z_lineage_boot"] = boot(lm(ser_z - bz[:, 4]))
ctl["serial_minus_control_z_family_cluster"] = cluster_boot(ser_z - bz[:, 4])
ctl["each_task<control_z"] = {t: [int((bz[:, j] < bz[:, 4]).sum()), int((lm(bz[:, j] - bz[:, 4]) < 0).sum())] for j, t in enumerate(T)}
ctl["each_task<control_rel"] = {t: [int((brel[:, j] < brel[:, 4]).sum()), int((lm(brel[:, j] - brel[:, 4]) < 0).sum())] for j, t in enumerate(T)}
# same permutation scheme: pointer+arith (not within) vs control (not within)
pa_z = bz[:, :2].mean(1)
ctl["pointer_arith<control_z_same_scheme"] = [int((pa_z < bz[:, 4]).sum()), int((lm(pa_z - bz[:, 4]) < 0).sum())]
# chain tasks under the non-within scheme
bza = Zalt[:, :, 5:].mean(2)
ser_z_same = np.c_[bz[:, :2], bza[:, 2:4]].mean(1)
ctl["serial(all not-within)<control_z"] = [int((ser_z_same < bz[:, 4]).sum()), int((lm(ser_z_same - bz[:, 4]) < 0).sum())]
ctl["ratio_absz_within_vs_not_chain"] = float(np.mean(np.abs(Zc[:, 2:4]) / np.maximum(np.abs(Zalt[:, 2:4]), 1e-9)))
ctl["control_back_z"] = dict(zip(ORDER, np.round(bz[:, 4], 2).tolist()))
ctl["control_back_rel"] = dict(zip(ORDER, np.round(brel[:, 4], 3).tolist()))
ctl["serial_back_rel"] = dict(zip(ORDER, np.round(ser_rel, 3).tolist()))
# control: switch at other boundaries; control profile vs serial shared profile
ctl["control_boundary_counts(d_i>d_i+1)"] = [int((Zfd[:, i] > Zfd[:, i + 1]).sum()) for i in range(9)]
ctl["serial_boundary_counts(d_i>d_i+1)"] = [int((prof[:, i] > prof[:, i + 1]).sum()) for i in range(9)]
ctl["serial_boundary_counts_lineage"] = [int((lm(prof[:, i] - prof[:, i + 1]) > 0).sum()) for i in range(9)]
ctl["control_boundary_counts_lineage"] = [int((lm(Zfd[:, i] - Zfd[:, i + 1]) > 0).sum()) for i in range(9)]
# serial per-task boundary counts
ctl["serial_per_task_boundary_counts"] = {t: [int((Z[:, j, i] > Z[:, j, i + 1]).sum()) for i in range(9)] for j, t in enumerate(T)}
# control profile vs leave-lineage-out serial mean
def llo_r(P, Ref):
    r = []
    for i, m in enumerate(ORDER):
        oth = [q for q, x in enumerate(ORDER) if LIN[x] != LIN[m]]; Lo = sorted(set(LIN[ORDER[q]] for q in oth))
        mu = np.mean([Ref[[q for q in oth if LIN[ORDER[q]] == L]].mean(0) for L in Lo], 0); r.append(np.corrcoef(P[i], mu)[0, 1])
    return np.array(r)
r_ctl_vs_serial = llo_r(Zfd, prof); r_ctl_vs_ctl = llo_r(Zfd, Zfd); r_ser = llo_r(prof, prof)
ctl["control_profile_vs_LLO_serial_mean"] = {"r>0": int((r_ctl_vs_serial > 0).sum()), "median": float(np.median(r_ctl_vs_serial)), "per_model": dict(zip(ORDER, np.round(r_ctl_vs_serial, 2).tolist()))}
ctl["control_profile_vs_LLO_control_mean"] = {"r>0": int((r_ctl_vs_ctl > 0).sum()), "median": float(np.median(r_ctl_vs_ctl))}
ctl["serial_profile_vs_LLO_serial_mean"] = {"r>0": int((r_ser > 0).sum()), "median": float(np.median(r_ser))}
# single serial task vs LLO mean of serial (fair comparison to one control task): per task median r
ctl["single_task_vs_LLO_serial_mean_median_r"] = {t: float(np.median(llo_r(Z[:, j], prof))) for j, t in enumerate(T)}
ctl["single_task_vs_LLO_serial_mean_rpos"] = {t: int((llo_r(Z[:, j], prof) > 0).sum()) for j, t in enumerate(T)}
out["control"] = ctl

# ---------------- mean |z| back vs front (T15) per task and control, and in effect size
def halves(v):
    L_ = len(v); x = np.arange(L_) / (L_ - 1); return v[x < .5].mean(), v[x >= .5].mean()
t15 = {}
for f in T5:
    d = np.array([np.subtract(*halves(np.abs(CC[(m, f)]["z"]))[::-1]) for m in ORDER])
    t15[f] = [int((d > 0).sum()), int((lm(d) > 0).sum())]
d_ser = np.array([np.subtract(*halves(np.mean([np.abs(np.array(C[m]["families"][t]["z"])) for t in T], 0))[::-1]) for m in ORDER])
t15["serial_mean_abs"] = [int((d_ser > 0).sum()), int((lm(d_ser) > 0).sum()), boot(lm(d_ser))]
# layer counts in each half differ? effective number of layers; also |rel| halves
for f in T5:
    d = np.array([np.subtract(*halves(np.abs(CC[(m, f)]["rel"]))[::-1]) for m in ORDER]); t15[f + "_absrel"] = [int((d > 0).sum()), int((lm(d) > 0).sum())]
out["T15"] = t15

# ---------------- T3: 40-50 > back-half mean given back<front ; switch null by circular shift
t3 = {"switch": [int((prof[:, 4] > prof[:, 5]).sum()), int((lm(prof[:, 4] - prof[:, 5]) > 0).sum()), int((fm(prof[:, 4] - prof[:, 5]) > 0).sum())],
      "switch_lineage_boot": boot(lm(prof[:, 4] - prof[:, 5])), "switch_family_cluster": cluster_boot(prof[:, 4] - prof[:, 5]),
      "max_front_gt_back_mean": int((prof[:, :5].max(1) > back).sum()),
      "d40_50_gt_back": int((prof[:, 4] > back).sum()),
      "argmax_front_decile": dict(zip(ORDER, prof[:, :5].argmax(1).tolist())),
      "largest_adjacent_drop_boundary": dict(zip(ORDER, np.argmax(prof[:, :-1] - prof[:, 1:], 1).tolist()))}
t3["largest_drop_at_40_50|50_60"] = int((np.argmax(prof[:, :-1] - prof[:, 1:], 1) == 4).sum())
out["T3"] = t3

# ---------------- T11/T8: exact Spearman p for small n, non-Qwen bases
def exact_sp(x, y):
    x = np.asarray(x); y = np.asarray(y); n = len(x); obs = spearmanr(x, y).correlation
    null = [spearmanr(x, np.array(p)).correlation for p in itertools.permutations(y)]
    return [float(obs), float(np.mean(np.abs(null) >= abs(obs) - 1e-12)), n]
MD2 = json.load(open(f"{SP}/model_diff2.json"))["features"]; acc = np.array([MD2["acc"][m] for m in ORDER]); H = np.array([MD2["H"][m] for m in ORDER]) if "H" in MD2 else None
d5070 = prof[:, 5:7].mean(1)
bi = [ORDER.index(m) for m in BASES]
h32b = [i for i in bi if H[i] == 32]; nqb = [i for i in bi if FAM[ORDER[i]] != "Qwen"]
t11 = {"H": dict(zip(ORDER, H.tolist())), "H32_bases": [ORDER[i] for i in h32b],
       "acc~d50_70_H32_bases_exact": exact_sp(acc[h32b], d5070[h32b]),
       "acc~d50_70_nonQwen_bases_exact": exact_sp(acc[nqb], d5070[nqb]),
       "acc~d50_70_H32_lineages": None}
# lineage-level H=32 (5 lineages: L1B, L8B, OLMo, M7, M24)
Lh32 = [L for L in LINS if H[ORDER.index(L)] == 32]
al = lm(acc); dl = lm(d5070); idx = [LINS.index(L) for L in Lh32]
t11["acc~d50_70_H32_lineages"] = exact_sp(al[idx], dl[idx]); t11["H32_lineages"] = Lh32
nq = [i for i, L in enumerate(LINS) if FAM[L] != "Qwen"]
t11["acc~d50_70_nonQwen_lineages_exact"] = exact_sp(al[nq], dl[nq])
t11["acc_values_H32_lineages"] = dict(zip(Lh32, np.round(al[idx], 3).tolist())); t11["d50_70_H32_lineages"] = dict(zip(Lh32, np.round(dl[idx], 2).tolist()))
PARAMS = {"Llama-3.2-1B": 1.2, "Llama-3.2-3B": 3.2, "Llama-3.1-8B": 8.0, "Qwen2.5-0.5B": 0.5, "Qwen2.5-1.5B": 1.5, "Qwen2.5-3B": 3.1,
          "Qwen2.5-7B": 7.6, "Qwen2.5-14B": 14.7, "OLMo-2-1124-7B": 7.3, "Mistral-7B-v0.1": 7.2, "Mistral-Small-24B-Base-2501": 24.0}
t11["logN~d50_70_nonQwen_bases_exact"] = exact_sp(np.log([PARAMS[ORDER[i]] for i in nqb]), d5070[nqb])
t11["acc~logN_nonQwen_bases"] = exact_sp(np.log([PARAMS[ORDER[i]] for i in nqb]), acc[nqb])
# pseudo-replication: 17-model acc~d50_70 vs lineage-level (11)
t11["acc~d50_70_lineages11"] = [float(spearmanr(al, dl).correlation), float(spearmanr(al, dl).pvalue)]
t11["acc~back_lineages11"] = [float(spearmanr(al, lm(back)).correlation), float(spearmanr(al, lm(back)).pvalue)]
out["T11"] = t11

# ---------------- T12: per-task accuracy vs back, lineage level; within-lineage task-centred
Ac = np.array([[CC[(m, t)]["acc"] for t in T] for m in ORDER]); Bz4 = Z[:, :, 5:].mean(2)
t12 = {}
for j, t in enumerate(T):
    r = spearmanr(lm(Ac[:, j]), lm(Bz4[:, j])); t12[f"{t}_lineages11"] = [float(r.correlation), float(r.pvalue)]
out["T12"] = t12

# ---------------- T6: binding vs relational within lineage, family counts; effect size
bb = Bz4[:, 2] - Bz4[:, 3]; bbr = Rel[:, 2, 5:].mean(1) - Rel[:, 3, 5:].mean(1)
out["T6"] = {"z": [int((bb < 0).sum()), int((lm(bb) < 0).sum()), int((fm(bb) < 0).sum())], "rel": [int((bbr < 0).sum()), int((lm(bbr) < 0).sum())],
             "lineage_boot_z": boot(lm(bb)), "family_cluster_z": cluster_boot(bb),
             "acc_bind_gt_rel": int((Ac[:, 2] > Ac[:, 3]).sum()),
             "gap_vs_accgap_lineages": [float(spearmanr(lm(Ac[:, 2] - Ac[:, 3]), lm(bb)).correlation), float(spearmanr(lm(Ac[:, 2] - Ac[:, 3]), lm(bb)).pvalue)],
             "n_items_binding_relational": [NIT[m]["binding"] for m in ORDER] + [NIT[m]["relational"] for m in ORDER]}
# is binding<relational explained by accuracy? binding<relational among models where binding is NOT more accurate
nb = Ac[:, 2] <= Ac[:, 3]
out["T6"]["models_binding_not_more_accurate"] = [m for m, x in zip(ORDER, nb) if x]
out["T6"]["binding<relational_among_those"] = int((bb[nb] < 0).sum())

# ---------------- T13 final layer, lineage-level Fisher already; within-family consistency: check Mistral-7B ~0
fz = np.array([[C[m]["families"][t]["z"][-1] for t in T] for m in ORDER]).mean(1)
out["T13"] = {"final_task_mean": dict(zip(ORDER, np.round(fz, 2).tolist())), "gt0.5": int((fz > .5).sum()), "lt-0.5": int((fz < -.5).sum())}

# ---------------- post-training (for 'missing'): pairs_q10.json summary per cell, and control spreading of post-trained models
P = json.load(open(f"{SP}/pairs_q10.json"))
pt = {}
for r in P:
    key = f"{r['base']}->{r['post']}"
    pt.setdefault(key, {})[r["cell"]] = {k: r.get(k) for k in ("mean_all", "mean_both_wrong", "nosink_all", "median_all")}
out["pairs_q10"] = pt
# serial-task post-minus-base back-half z (primary) per pair
pairs = [("Llama-3.2-3B", "Llama-3.2-3B-Instruct"), ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"), ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
         ("OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-DPO"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct")]
ptz = {}
for a, b in pairs:
    ia, ib = ORDER.index(a), ORDER.index(b)
    ptz[f"{a}->{b}"] = {"serial_back_z": [round(float(back[ia]), 2), round(float(back[ib]), 2)],
                        "control_back_z": [round(float(bz[ia, 4]), 2), round(float(bz[ib, 4]), 2)],
                        "serial_back_rel": [round(float(ser_rel[ia]), 3), round(float(ser_rel[ib]), 3)],
                        "control_back_rel": [round(float(brel[ia, 4]), 3), round(float(brel[ib, 4]), 3)],
                        "per_task_back_z_post_minus_base": np.round(Bz4[ib] - Bz4[ia], 2).tolist(),
                        "acc": [round(float(acc[ia]), 3), round(float(acc[ib]), 3)]}
    for fit in ["k", "k+ent", "k+ent+corr"]:
        ptz[f"{a}->{b}"][f"back_t_{fit}"] = [adj[fit]["per_model_back"][a], adj[fit]["per_model_back"][b]]
out["posttrain_back"] = ptz
json.dump(out, open(f"{OUT}/r1.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k not in ("pairs_q10",)}, indent=1, default=float)[:60000])
