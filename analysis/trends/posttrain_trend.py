"""Secondary trend: do post-trained (instruction-tuned / SFT / DPO) models SPREAD head activity more as serial
demand k grows than their base models?

Six base -> post pairs, four lineages (Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-7B, OLMo-2-7B; the three OLMo
post-trained checkpoints share one base).  Lineage-weighted = average the OLMo pairs first, then the 4 lineages.
Statistic: top-10% head share per layer; per-layer OLS slope on k standardised by the permutation null -> z
(negative = spreads as k grows).  Items wording-averaged.  Qwen2.5-7B-Instruct under SDPA in the primary
pipeline (Qwen2.5-7B base under eager: KERNEL MISMATCH inside that pair, checked in section 4).

Sections (keys of posttrain_trend.json):
  s1_model_level   consist.json q10: back-half / front-half / whole mean z, fraction of significant layers,
                   relative slope, post-minus-base per pair x task, counts, PC1 shift, item-level paired bootstrap
  s2_depth         decile-wise post-minus-base of Z (pair mean and lineage-weighted), sign counts, hierarchical
                   (lineage x task) bootstrap and item-bootstrap CIs, region summary
  s3_same_item     pairs_q10.json counts (serial tasks and numeral control separately) + recomputation from the
                   vet3 cache with Qwen2.5-7B-Instruct under SDPA and a pointer/w0 all-SDPA check
  s4_robustness    other statistics (q25, q50, entropy, top1), cells.json variants (sink_head, sink_item,
                   eager-only, SDPA, noclean), kernel check on pointer/w0
  s5_confounds     accuracy (k=1, overall), answer-entropy slope, between-base-model accuracy regression,
                   item-level entropy/correctness-adjusted k-slopes (recomputed with Qwen-I SDPA and eager),
                   numeral control fixed_digit at the model level
  s6_concentrating front-half / pre-mid-depth concentration, significant concentrating layers, peak decile
Run: python3 posttrain_trend.py   (reads only; writes posttrain_trend.json next to this script)
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, sys, json, itertools, warnings
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore")
os.environ["VPRIM"] = "y10"
SP = str(paths.ANALYSIS)
OUTD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP); sys.path.insert(0, f"{SP}/vet3")
import model_diff as md
md.SP = SP
from model_diff import ORDER, T, LIN, decompose, from_consist, from_cells, dec_prof
from v2lib import load as vload, zprof

PAIRS = [("Llama-3.2-3B", "Llama-3.2-3B-Instruct"), ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"),
         ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT"),
         ("OLMo-2-1124-7B", "OLMo-2-1124-7B-DPO"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct")]
PN = [p for _, p in PAIRS]
LINEAGES = ["Llama-3.2-3B", "Llama-3.1-8B", "Qwen2.5-7B", "OLMo-2-1124-7B"]
LIDX = [[i for i, (b, _) in enumerate(PAIRS) if b == L] for L in LINEAGES]
ATTN_PRIMARY = {"Qwen2.5-7B-Instruct": "sdpa"}
WITHIN = {"binding", "relational"}
FAMS5 = T + ["fixed_digit"]
rng = np.random.default_rng(12345)


def lw(v):
    """lineage-weighted mean of a per-pair vector (axis 0 = pairs)"""
    v = np.asarray(v, float); return np.mean([v[ix].mean(0) for ix in LIDX], 0)


def per_lineage(v):
    v = np.asarray(v, float); return np.array([v[ix].mean(0) for ix in LIDX])


def sign_p(nneg, n):
    """one-sided binomial p for >= nneg negatives out of n under p=.5"""
    return float(stats.binom.sf(nneg - 1, n, 0.5))


def r3(x):
    if isinstance(x, dict): return {k: r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [r3(v) for v in x]
    if isinstance(x, (np.floating, float)): return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.ndarray): return r3(x.tolist())
    return x


def xgrid(L): return np.arange(L) / (L - 1)


def regions(z):
    z = np.asarray(z, float); x = xgrid(len(z))
    return {"front": np.nanmean(z[x < .5]), "back": np.nanmean(z[x >= .5]), "whole": np.nanmean(z),
            "40-50": np.nanmean(z[(x >= .4) & (x < .5)]), "50-70": np.nanmean(z[(x >= .5) & (x < .7)]),
            "70-100": np.nanmean(z[x >= .7])}


def pair_summary(D):
    """D: dict pair-name -> scalar difference (post - base).  Counts and lineage-weighted mean."""
    v = np.array([D[p] for p in PN]); lv = per_lineage(v)
    return {"per_pair": dict(zip(PN, v)), "n_pairs_neg": int((v < 0).sum()), "pair_mean": v.mean(),
            "per_lineage": dict(zip(LINEAGES, lv)), "n_lineages_neg": int((lv < 0).sum()), "lineage_weighted": lv.mean(),
            "sign_p_lineages_one_sided": sign_p(int((lv < 0).sum()), 4)}


# ---------------------------------------------------------------------------------------------------------
# data access from the vet3 cache (reproduces consist.json q10 z with exact permutation SD; r >= .9998)
_cache = {}


def wavg(m, f, attn, key="y10", clean=True):
    """wording-averaged item x layer matrix, matched on (k, body); returns dict (k, body) -> row plus aux"""
    ck = (m, f, attn, key, clean)
    if ck in _cache: return _cache[ck]
    a, b = vload(m, f, "w0", attn, clean), vload(m, f, "w1", attn, clean)
    if a is None or b is None: _cache[ck] = None; return None
    ka = {(int(k), int(bb)): j for j, (k, bb) in enumerate(zip(a["k"], a["body"]))}
    kb = {(int(k), int(bb)): j for j, (k, bb) in enumerate(zip(b["k"], b["body"]))}
    keys = sorted(set(ka) & set(kb)); ia = np.array([ka[x] for x in keys]); ib = np.array([kb[x] for x in keys])
    out = {"keys": keys, "Y": 0.5 * (a[key][ia] + b[key][ib]), "k": np.array([x[0] for x in keys], float),
           "body": np.array([x[1] for x in keys]), "correct": 0.5 * (a["correct"][ia].astype(float) + b["correct"][ib].astype(float)),
           "ent": 0.5 * (a["ent"][ia] + b["ent"][ib])}
    _cache[ck] = out; return out


def model_z(m, f, attn=None, key="y10", clean=True):
    attn = attn or ATTN_PRIMARY.get(m, "eager"); d = wavg(m, f, attn, key, clean)
    if d is None: return None
    return zprof(d["Y"], d["k"], d["body"], f in WITHIN)[2]


def matched(base, post, f, ab, ap, key="y10"):
    A, B = wavg(base, f, ab, key), wavg(post, f, ap, key)
    ia = {x: j for j, x in enumerate(A["keys"])}; ib = {x: j for j, x in enumerate(B["keys"])}
    keys = [x for x in A["keys"] if x in ib]
    ja = np.array([ia[x] for x in keys]); jb = np.array([ib[x] for x in keys])
    return A["Y"][ja], B["Y"][jb], np.array([x[0] for x in keys], float), np.array([x[1] for x in keys])


def boot_idx(k, body, within, rng):
    if within:
        ub = np.unique(body); draw = rng.choice(ub, len(ub), replace=True)
        idx, nb = [], []
        for j, g in enumerate(draw):
            ii = np.nonzero(body == g)[0]; idx.append(ii); nb.append(np.full(len(ii), j))
        return np.concatenate(idx), np.concatenate(nb)
    idx = np.concatenate([rng.choice(np.nonzero(k == kk)[0], (k == kk).sum(), replace=True) for kk in np.unique(k)])
    return idx, body[idx]


out = {"meta": {"pairs": PAIRS, "lineages": LINEAGES, "statistic": "top-10% head share (q10)",
                "sign": "z < 0 = spreads as k grows; differences are post minus base (negative = post-trained spreads more)",
                "kernel": "Qwen2.5-7B-Instruct SDPA in primary; Qwen2.5-7B base eager (SDPA only pointer/w0)",
                "count_semantics": ("n_*_neg counts post-minus-base < 0. For z and rel metrics that means post-trained spreads more; "
                                    "for frac_sig_spread* a POSITIVE difference means more significantly spreading layers; "
                                    "for PC1 a POSITIVE shift means more back-half spreading (see n_lineages_pos)."),
                "sign_test_note": "with 4 lineages the smallest attainable one-sided sign-test p is 1/16 = 0.0625"}}

# =========================================================================================================
# 1. MODEL LEVEL (consist.json, q10)
C = {(r["model"], r["stat"]): r for r in json.load(open(f"{SP}/consist.json"))}


def mstats(m, stat="q10"):
    r = C[(m, stat)]; o = {}
    for t in T:
        F = r["families"][t]; z = np.array(F["z"]); p = np.array(F["p"]); rel = np.array(F["rel"]); x = xgrid(len(z))
        sp, cc = (p < .05) & (z < 0), (p < .05) & (z > 0)
        o[t] = {**{f"{k}_z": v for k, v in regions(z).items()},
                "frac_sig_spread": sp.mean(), "frac_sig_conc": cc.mean(),
                "frac_sig_spread_back": sp[x >= .5].mean(), "frac_sig_conc_front": cc[x < .5].mean(),
                "frac_sig_conc_back": cc[x >= .5].mean(), "frac_sig_spread_front": sp[x < .5].mean(),
                "back_rel": rel[x >= .5].mean(), "front_rel": rel[x < .5].mean()}
    o["mean"] = {k: float(np.mean([o[t][k] for t in T])) for k in o[T[0]]}
    return o


MS = {m: mstats(m) for m in ORDER}
METRICS = ["back_z", "front_z", "whole_z", "40-50_z", "50-70_z", "70-100_z", "frac_sig_spread", "frac_sig_conc",
           "frac_sig_spread_back", "frac_sig_conc_front", "frac_sig_conc_back", "frac_sig_spread_front", "back_rel", "front_rel"]
s1 = {"per_model": {m: {t: {k: MS[m][t][k] for k in METRICS} for t in T + ["mean"]} for b, p in PAIRS for m in (b, p)}}
diffs = {}
for mt in METRICS:
    D = {p: {t: MS[p][t][mt] - MS[b][t][mt] for t in T + ["mean"]} for b, p in PAIRS}
    ps = pair_summary({p: D[p]["mean"] for p in PN})
    combos = np.array([D[p][t] for p in PN for t in T])
    ps["per_pair_task"] = D
    ps["n_pair_task_neg"] = int((combos < 0).sum()); ps["n_pair_task"] = len(combos)
    lt = np.array([[np.mean([D[PN[i]][t] for i in ix]) for t in T] for ix in LIDX])
    ps["n_lineage_task_neg"] = int((lt < 0).sum()); ps["n_lineage_task"] = lt.size
    ps["per_task_lineage_weighted"] = dict(zip(T, lt.mean(0)))
    ps["n_pairs_neg_per_task"] = {t: int(sum(D[p][t] < 0 for p in PN)) for t in T}
    diffs[mt] = ps
s1["diff"] = diffs
MD = json.load(open(f"{SP}/model_diff.json")); sc = MD["pca"]["scores"]
s1["pc1"] = {"loadings_by_decile": MD["pca"]["loadings"][0], "explained": MD["pca"]["explained"][0],
             "note": "PC1 loads negatively on deciles 5-8 (back half) and positively on decile 3: + score = more back-half spreading",
             "scores": {m: sc[m][0] for b, p in PAIRS for m in (b, p)},
             **pair_summary({p: sc[p][0] - sc[b][0] for b, p in PAIRS})}
s1["pc1"]["n_lineages_pos"] = int(sum(v > 0 for v in s1["pc1"]["per_lineage"].values()))
s1["pc1"]["sign_p_lineages_pos_one_sided"] = sign_p(s1["pc1"]["n_lineages_pos"], 4)
s1["decile_back_model_diff_json"] = pair_summary({p: MD["per_model"][p]["back"] - MD["per_model"][b]["back"] for b, p in PAIRS})
# reference scale: spread of back-half z among the 11 base models, and among unrelated same-size pairs
bases = [m for m in ORDER if LIN[m] == m]
bb = np.array([MS[m]["mean"]["back_z"] for m in bases])
s1["reference_scale"] = {"base_models_back_z_sd": bb.std(ddof=1), "base_models_back_z_range": [bb.min(), bb.max()],
                         "mean_abs_diff_between_base_models": np.mean([abs(x - y) for x, y in itertools.combinations(bb, 2)])}

# item-level bootstrap of the back-half / front-half mean z difference and of the decile-profile difference.
#  unpaired: each model on its own (cleaned, wording-matched) items, resampled independently -> reproduces consist.json
#  paired:   items common to base and post, resampled jointly (more power; differs for OLMo-2 because the
#            massive-activation exclusion removes different items in base and post, e.g. OLMo-2 base arith w0 146/252)
# strata: k for pointer/arith (items resampled within k); prompt-body clusters for binding/relational
NBOOT = 400


def zb_of(Y, k, body, wi, idx=None, nb=None):
    if idx is None: return zprof(Y, k, body, wi)[2]
    return zprof(Y[idx], k[idx], nb, wi)[2]


def summarize_boot(kind, get):
    boot = {}; dec = np.zeros((len(PAIRS), len(T), NBOOT, 10))
    for i, (b, p) in enumerate(PAIRS):
        boot[p] = {}; BKs, FRs = [], []
        for j, t in enumerate(T):
            (Yb, kb_, bb_), (Yp, kp_, bp_) = get(b, p, t)
            wi = t in WITHIN; x = xgrid(Yb.shape[1])
            zb, zp = zb_of(Yb, kb_, bb_, wi), zb_of(Yp, kp_, bp_, wi)
            ob = zp[x >= .5].mean() - zb[x >= .5].mean(); of = zp[x < .5].mean() - zb[x < .5].mean()
            bk, fr = np.empty(NBOOT), np.empty(NBOOT)
            for r_ in range(NBOOT):
                ib, nbb = boot_idx(kb_, bb_, wi, rng)
                if kind == "paired": ip, nbp = ib, nbb
                else: ip, nbp = boot_idx(kp_, bp_, wi, rng)
                z1 = zb_of(Yb, kb_, bb_, wi, ib, nbb); z2 = zb_of(Yp, kp_, bp_, wi, ip, nbp)
                bk[r_] = z2[x >= .5].mean() - z1[x >= .5].mean(); fr[r_] = z2[x < .5].mean() - z1[x < .5].mean()
                dec[i, j, r_] = dec_prof(z2) - dec_prof(z1)
            boot[p][t] = {"back_diff": ob, "back_ci": np.percentile(bk, [2.5, 97.5]), "back_p_ge0": float((bk >= 0).mean()),
                          "front_diff": of, "front_ci": np.percentile(fr, [2.5, 97.5]), "n_items_base": len(kb_), "n_items_post": len(kp_)}
            BKs.append(bk); FRs.append(fr)
        bkm, frm = np.mean(BKs, 0), np.mean(FRs, 0)
        boot[p]["task_mean"] = {"back_diff": np.mean([boot[p][t]["back_diff"] for t in T]), "back_ci": np.percentile(bkm, [2.5, 97.5]),
                                "back_p_ge0": float((bkm >= 0).mean()), "front_diff": np.mean([boot[p][t]["front_diff"] for t in T]),
                                "front_ci": np.percentile(frm, [2.5, 97.5]), "front_p_le0": float((frm <= 0).mean())}
        boot[p]["_bkm"] = bkm; boot[p]["_frm"] = frm
    BK = np.array([boot[p].pop("_bkm") for p in PN]); FR = np.array([boot[p].pop("_frm") for p in PN])
    res = {"per_pair": boot,
           "lineage_weighted_back": {"diff": lw([boot[p]["task_mean"]["back_diff"] for p in PN]),
                                     "ci": np.percentile(lw(BK), [2.5, 97.5]), "p_ge0": float((lw(BK) >= 0).mean())},
           "lineage_weighted_back_without_qwen": {"diff": np.mean(per_lineage([boot[p]["task_mean"]["back_diff"] for p in PN])[[0, 1, 3]]),
                                                  "ci": np.percentile(per_lineage(BK)[[0, 1, 3]].mean(0), [2.5, 97.5])},
           "per_lineage_back": {L: {"diff": per_lineage([boot[p]["task_mean"]["back_diff"] for p in PN])[li],
                                    "ci": np.percentile(per_lineage(BK)[li], [2.5, 97.5])} for li, L in enumerate(LINEAGES)},
           "lineage_weighted_front": {"diff": lw([boot[p]["task_mean"]["front_diff"] for p in PN]),
                                      "ci": np.percentile(lw(FR), [2.5, 97.5])}}
    return res, dec


def get_unpaired(b, p, t):
    A = wavg(b, t, ATTN_PRIMARY.get(b, "eager")); B = wavg(p, t, ATTN_PRIMARY.get(p, "eager"))
    return (A["Y"], A["k"], A["body"]), (B["Y"], B["k"], B["body"])


def get_paired(b, p, t):
    Yb, Yp, k, body = matched(b, p, t, ATTN_PRIMARY.get(b, "eager"), ATTN_PRIMARY.get(p, "eager"))
    return (Yb, k, body), (Yp, k, body)


ub, bootdec = summarize_boot("unpaired", get_unpaired)
pb_, bootdec_p = summarize_boot("paired", get_paired)
s1["item_bootstrap_unpaired"] = {"note": f"{NBOOT} independent item resamples per model (own items; point estimates = consist.json); p_ge0 = share of resamples with diff >= 0", **ub}
s1["item_bootstrap_paired_common_items"] = {"note": f"{NBOOT} joint resamples of the items common to base and post", **pb_}
out["s1_model_level"] = s1

# =========================================================================================================
# 2. WHERE IN DEPTH (decile profiles)
Z = from_consist("q10"); I = {m: i for i, m in enumerate(ORDER)}
DD = np.array([[Z[I[p], j] - Z[I[b], j] for j in range(4)] for b, p in PAIRS])        # [6, 4, 10]
pairdec = DD.mean(1); lindec = per_lineage(pairdec); lwdec = lindec.mean(0)
LT = np.array([DD[ix].mean(0) for ix in LIDX])                                           # [4 lineages, 4 tasks, 10]
hb = np.empty((5000, 10))
for r_ in range(5000):
    li = rng.integers(0, 4, 4); ti = rng.integers(0, 4, 4); hb[r_] = LT[li][:, ti].mean((0, 1))
itb = np.array([bootdec[ix].mean((0, 1)) for ix in LIDX]).mean(0)                     # [NBOOT, 10] lineage-weighted (unpaired)
itbp = np.array([bootdec_p[ix].mean((0, 1)) for ix in LIDX]).mean(0)                  # paired, common items
s2 = {"deciles": [f"{10*d}-{10*d+10}%" for d in range(10)],
      "shared_profile_mu": MD["primary"]["mu"],
      "base_mean_profile": Z[[I[b] for b, _ in PAIRS]].mean((0, 1)), "post_mean_profile": Z[[I[p] for _, p in PAIRS]].mean((0, 1)),
      "base_lineage_profile": np.array([Z[I[L]].mean(0) for L in LINEAGES]).mean(0),
      "post_lineage_weighted_profile": np.array([np.mean([Z[I[PN[i]]].mean(0) for i in ix], 0) for ix in LIDX]).mean(0),
      "pair_mean_diff": pairdec.mean(0), "lineage_weighted_diff": lwdec,
      "per_pair_diff": dict(zip(PN, pairdec)), "per_lineage_diff": dict(zip(LINEAGES, lindec)),
      "n_pairs_neg": (pairdec < 0).sum(0), "n_lineages_neg": (lindec < 0).sum(0), "n_pair_task_neg_of24": (DD < 0).sum((0, 1)),
      "hier_boot_ci_lineage_task": np.percentile(hb, [2.5, 97.5], 0), "hier_boot_p_ge0": (hb >= 0).mean(0),
      "item_boot_ci_lineage_weighted": np.percentile(itb, [2.5, 97.5], 0), "item_boot_p_ge0": (itb >= 0).mean(0),
      "item_boot_paired_ci_lineage_weighted": np.percentile(itbp, [2.5, 97.5], 0), "item_boot_paired_p_ge0": (itbp >= 0).mean(0)}
reg = {"0-40": slice(0, 4), "40-50": slice(4, 5), "50-70": slice(5, 7), "70-100": slice(7, 10), "front(0-50)": slice(0, 5), "back(50-100)": slice(5, 10)}
s2["regions"] = {k: {"lineage_weighted": lwdec[s_].mean(), "pair_mean": pairdec[:, s_].mean(),
                     "n_pairs_neg": int((pairdec[:, s_].mean(1) < 0).sum()), "n_lineages_neg": int((lindec[:, s_].mean(1) < 0).sum()),
                     "n_pair_task_neg_of24": int((DD[:, :, s_].mean(2) < 0).sum()),
                     "item_boot_ci": np.percentile(itb[:, s_].mean(1), [2.5, 97.5]),
                     "item_boot_paired_ci": np.percentile(itbp[:, s_].mean(1), [2.5, 97.5]),
                     "hier_boot_ci": np.percentile(hb[:, s_].mean(1), [2.5, 97.5])} for k, s_ in reg.items()}
out["s2_depth"] = s2

# =========================================================================================================
# 3. SAME-ITEM PAIRED COMPARISON
PQ = json.load(open(f"{SP}/pairs_q10.json"))
KEYS = ["mean_all", "mean_k2plus", "mean_both_wrong", "median_all", "median_k2plus", "median_both_wrong",
        "nosink_all", "nosink_k2plus", "nosink_both_wrong"]


def count_cells(R, keys=KEYS):
    o = {}
    for key in keys:
        v = np.array([r[key][0] for r in R if key in r]); p = np.array([r[key][1] for r in R if key in r])
        if not len(v): continue
        lin_ = {L: np.mean([r[key][0] for r in R if key in r and r["base"] == L]) for L in LINEAGES}
        o[key] = {"n": len(v), "n_neg": int((v < 0).sum()), "n_sig_neg": int(((v < 0) & (p < .05)).sum()),
                  "n_sig_pos": int(((v > 0) & (p < .05)).sum()), "mean": v.mean(),
                  "lineage_weighted_mean": float(np.mean(list(lin_.values()))),
                  "per_pair": {pp: {"n_neg": int(sum(r[key][0] < 0 for r in R if r["post"] == pp and key in r)),
                                    "n_sig_neg": int(sum((r[key][0] < 0) and (r[key][1] < .05) for r in R if r["post"] == pp and key in r)),
                                    "n_sig_pos": int(sum((r[key][0] > 0) and (r[key][1] < .05) for r in R if r["post"] == pp and key in r)),
                                    "mean": float(np.mean([r[key][0] for r in R if r["post"] == pp and key in r]))} for pp in PN}}
    return o


ser = [r for r in PQ if r["cell"].split("/")[0] in T]; ctl = [r for r in PQ if r["cell"].startswith("fixed_digit")]
s3 = {"note": ("pairs_q10.json (pairs.py): per item, 40-80% depth-window mean top-10% share relative to the model's mean; "
               "post-minus-base slope on k (x100, % of mean per step), permutation p. Qwen2.5-7B-Instruct EAGER in that file. "
               "OLMo-2 has no sink heads (pos0_attn>0.9 never), so nosink == mean for OLMo pairs."),
      "serial": count_cells(ser), "numeral_control": count_cells(ctl),
      "serial_per_task": {t: count_cells([r for r in ser if r["cell"].startswith(t)], ["mean_all", "mean_k2plus", "mean_both_wrong"]) for t in T}}


# recomputation from the vet3 cache (verifies pairs_q10 and swaps the Qwen2.5-7B-Instruct kernel)
def win(L): return slice(int(.4 * (L - 1)), int(.8 * (L - 1)) + 1)


def perm_p(y, k, n=1000, seed=0):
    r_ = np.random.default_rng(seed); kc = k - k.mean(); ss = kc @ kc; yc = y - y.mean(); o = kc @ yc / ss
    nl = np.array([r_.permutation(kc) @ yc / ss for _ in range(n)])
    return float(o), float((np.sum(np.abs(nl) >= abs(o) - 1e-15) + 1) / (n + 1))


def same_item(base, post, fam, w, ab, ap, clean=True, key="y10"):
    A, B = vload(base, fam, w, ab, clean), vload(post, fam, w, ap, clean)
    if A is None or B is None: return None
    ka = {(int(k), int(bb)): j for j, (k, bb) in enumerate(zip(A["k"], A["body"]))}
    kb = {(int(k), int(bb)): j for j, (k, bb) in enumerate(zip(B["k"], B["body"]))}
    keys = [x for x in ka if x in kb]; ia = np.array([ka[x] for x in keys]); ib = np.array([kb[x] for x in keys])
    L = A[key].shape[1]; wd = win(L)
    ya, yb = np.nanmean(A[key][ia][:, wd], 1), np.nanmean(B[key][ib][:, wd], 1)
    k = A["k"][ia]; diff = yb / np.nanmean(yb) - ya / np.nanmean(ya)
    o = {}
    for sub, sel in (("all", np.ones(len(k), bool)), ("k2plus", k >= sorted(set(k))[1]),
                     ("both_wrong", ~A["correct"][ia].astype(bool) & ~B["correct"][ib].astype(bool))):
        if len(set(k[sel])) < 2: continue
        s, p = perm_p(diff[sel], k[sel]); o[sub] = (100 * s, p)
    return o


# reproduction check against pairs_q10 (all eager)
chk = []
for r in PQ:
    f, w = r["cell"].split("/")
    for clean in (True, False):
        o = same_item(r["base"], r["post"], f, w, "eager", "eager", clean)
        if o is None: continue
        chk.append((clean, abs(o["all"][0] - r["mean_all"][0])))
s3["reproduction_max_abs_err"] = {"clean": max(e for c, e in chk if c), "noclean": max(e for c, e in chk if not c)}
recomp = []
for b, p in PAIRS:
    for f in FAMS5:
        for w in ("w0", "w1"):
            o = same_item(b, p, f, w, ATTN_PRIMARY.get(b, "eager"), ATTN_PRIMARY.get(p, "eager"))
            recomp.append({"base": b, "post": p, "cell": f"{f}/{w}", **{f"mean_{k}": v for k, v in o.items()}})
s3["recomputed_primary_kernel"] = {
    "note": "same test from the vet3 cache with Qwen2.5-7B-Instruct under SDPA (base Qwen under eager: mixed kernel)",
    "serial": count_cells([r for r in recomp if r["cell"].split("/")[0] in T], ["mean_all", "mean_k2plus", "mean_both_wrong"]),
    "numeral_control": count_cells([r for r in recomp if r["cell"].startswith("fixed_digit")], ["mean_all", "mean_k2plus", "mean_both_wrong"]),
    "qwen_cells": {r["cell"]: {k: v for k, v in r.items() if k.startswith("mean")} for r in recomp if r["post"] == "Qwen2.5-7B-Instruct"}}
# all-SDPA pointer/w0 (the one cell where every base has an SDPA capture) vs all-eager
s3["pointer_w0_kernel"] = {p: {"eager_eager": same_item(b, p, "pointer", "w0", "eager", "eager"),
                               "sdpa_sdpa": same_item(b, p, "pointer", "w0", "sdpa", "sdpa"),
                               "base_eager_post_sdpa": same_item(b, p, "pointer", "w0", "eager", "sdpa")} for b, p in PAIRS}
# Llama-3.2-3B pair fully under SDPA
l3 = [{"cell": f"{f}/{w}", **{f"mean_{k}": v for k, v in same_item("Llama-3.2-3B", "Llama-3.2-3B-Instruct", f, w, "sdpa", "sdpa").items()}}
      for f in FAMS5 for w in ("w0", "w1")]
s3["llama32_3b_all_sdpa"] = {"serial_n_neg": sum(r["mean_all"][0] < 0 for r in l3 if not r["cell"].startswith("fixed")),
                             "serial_n_sig_neg": sum(r["mean_all"][0] < 0 and r["mean_all"][1] < .05 for r in l3 if not r["cell"].startswith("fixed")),
                             "serial_n_sig_pos": sum(r["mean_all"][0] > 0 and r["mean_all"][1] < .05 for r in l3 if not r["cell"].startswith("fixed")),
                             "control_n_neg": sum(r["mean_all"][0] < 0 for r in l3 if r["cell"].startswith("fixed")),
                             "cells": {r["cell"]: r["mean_all"] for r in l3}}
out["s3_same_item"] = s3

# =========================================================================================================
# 4. ROBUSTNESS of the model-level trend (back-half and front-half mean z, post minus base)
s4 = {}
for stat in ("q10", "q25", "q50", "entropy", "top1"):
    MSx = {m: mstats(m, stat) for b, p in PAIRS for m in (b, p)}
    s4[f"consist_{stat}"] = {"back": pair_summary({p: MSx[p]["mean"]["back_z"] - MSx[b]["mean"]["back_z"] for b, p in PAIRS}),
                             "front": pair_summary({p: MSx[p]["mean"]["front_z"] - MSx[b]["mean"]["front_z"] for b, p in PAIRS}),
                             "50-70": pair_summary({p: MSx[p]["mean"]["50-70_z"] - MSx[b]["mean"]["50-70_z"] for b, p in PAIRS}),
                             "70-100": pair_summary({p: MSx[p]["mean"]["70-100_z"] - MSx[b]["mean"]["70-100_z"] for b, p in PAIRS}),
                             "n_pair_task_back_neg": int(sum(MSx[p][t]["back_z"] < MSx[b][t]["back_z"] for b, p in PAIRS for t in T))}
CE = {}
for r in json.load(open(f"{SP}/vet3/cells.json")):
    CE[(r["model"], r["fam"], r["w"], r["attn"], r["variant"])] = np.array(r["z"], float)


def cell_reg(m, t, attn, var, ws=("w0", "w1")):
    zs = [CE.get((m, t, w, attn, var)) for w in ws]
    if any(z is None for z in zs): return None
    return regions(np.mean(zs, 0))


def variant_pairs(spec, tasks=T, ws=("w0", "w1")):
    """spec: function(model) -> (attn, variant).  Returns pair summary of task-mean back/front difference where available."""
    res = {"back": {}, "front": {}, "50-70": {}, "n_pair_task_back_neg": 0, "n_pair_task": 0}
    for b, p in PAIRS:
        rb = [cell_reg(b, t, *spec(b), ws) for t in tasks]; rp = [cell_reg(p, t, *spec(p), ws) for t in tasks]
        if any(x is None for x in rb + rp): continue
        res["back"][p] = np.mean([y["back"] - x["back"] for x, y in zip(rb, rp)])
        res["front"][p] = np.mean([y["front"] - x["front"] for x, y in zip(rb, rp)])
        res["50-70"][p] = np.mean([y["50-70"] - x["50-70"] for x, y in zip(rb, rp)])
        res["n_pair_task_back_neg"] += int(sum(y["back"] < x["back"] for x, y in zip(rb, rp))); res["n_pair_task"] += len(tasks)
    if len(res["back"]) == 6:
        res["back_summary"] = pair_summary(res["back"]); res["front_summary"] = pair_summary(res["front"]); res["50-70_summary"] = pair_summary(res["50-70"])
    return res


mixed = lambda v: (lambda m: ("sdpa", "all") if m == "Qwen2.5-7B-Instruct" and v == "all" else ("eager", v))
s4["cells_primary_kernel_all"] = variant_pairs(mixed("all"))
s4["cells_all_eager"] = variant_pairs(lambda m: ("eager", "all"))
s4["cells_sink_head_removed_eager"] = variant_pairs(lambda m: ("eager", "sink_head"))
s4["cells_sink_item_removed_eager"] = variant_pairs(lambda m: ("eager", "sink_item"))
s4["cells_olmo_noclean"] = variant_pairs(lambda m: ("eager", "noclean") if m.startswith("OLMo") else (ATTN_PRIMARY.get(m, "eager"), "all"))
s4["cells_sdpa_all_tasks_llama32_3b"] = {"back": np.mean([cell_reg("Llama-3.2-3B-Instruct", t, "sdpa", "all")["back"] - cell_reg("Llama-3.2-3B", t, "sdpa", "all")["back"] for t in T]),
                                          "front": np.mean([cell_reg("Llama-3.2-3B-Instruct", t, "sdpa", "all")["front"] - cell_reg("Llama-3.2-3B", t, "sdpa", "all")["front"] for t in T]),
                                          "eager_back": np.mean([cell_reg("Llama-3.2-3B-Instruct", t, "eager", "all")["back"] - cell_reg("Llama-3.2-3B", t, "eager", "all")["back"] for t in T])}
kp = {}
for b, p in PAIRS:
    g = lambda m, a: cell_reg(m, "pointer", a, "all", ("w0",))
    kp[p] = {"base_eager": g(b, "eager")["back"], "base_sdpa": g(b, "sdpa")["back"], "post_eager": g(p, "eager")["back"], "post_sdpa": g(p, "sdpa")["back"]}
    kp[p].update({"diff_eager_eager": kp[p]["post_eager"] - kp[p]["base_eager"], "diff_sdpa_sdpa": kp[p]["post_sdpa"] - kp[p]["base_sdpa"],
                  "diff_mixed(base eager, post sdpa)": kp[p]["post_sdpa"] - kp[p]["base_eager"]})
s4["kernel_pointer_w0_back_z"] = {"per_pair": kp,
                                  "sdpa_sdpa": pair_summary({p: kp[p]["diff_sdpa_sdpa"] for p in PN}),
                                  "eager_eager": pair_summary({p: kp[p]["diff_eager_eager"] for p in PN})}
# Qwen pair decomposed: wording-averaged, cache-based, eager-eager vs mixed
qz = {}
for t in T:
    zb = model_z("Qwen2.5-7B", t, "eager"); zi_s = model_z("Qwen2.5-7B-Instruct", t, "sdpa"); zi_e = model_z("Qwen2.5-7B-Instruct", t, "eager")
    x = xgrid(len(zb)); qz[t] = {"base_eager": zb[x >= .5].mean(), "instr_sdpa": zi_s[x >= .5].mean(), "instr_eager": zi_e[x >= .5].mean()}
qz["mean"] = {k: np.mean([qz[t][k] for t in T]) for k in qz[T[0]]}
qz["diff_mixed"] = qz["mean"]["instr_sdpa"] - qz["mean"]["base_eager"]; qz["diff_both_eager"] = qz["mean"]["instr_eager"] - qz["mean"]["base_eager"]
qz["kernel_offset_base_pointer_w0"] = kp["Qwen2.5-7B-Instruct"]["base_sdpa"] - kp["Qwen2.5-7B-Instruct"]["base_eager"]
s4["qwen_pair_back_z"] = qz
# lineage-weighted back-half difference with the Qwen pair under both-eager instead of mixed
alt = {p: s1["diff"]["back_z"]["per_pair"][p] for p in PN}; alt["Qwen2.5-7B-Instruct"] = qz["diff_both_eager"]
s4["primary_with_qwen_both_eager"] = pair_summary(alt)
out["s4_robustness"] = s4

# =========================================================================================================
# 5. CONFOUNDS
M2 = json.load(open(f"{SP}/model_diff2.json"))["features"]
s5 = {"accuracy": {f: {p: {"base": M2[f][b], "post": M2[f][p], "diff": M2[f][p] - M2[f][b]} for b, p in PAIRS}
                   for f in ("acc", "acc_k1", "acc_k2", "ent_slope", "base_share_back")},
      "n_post_higher_acc_k1": int(sum(M2["acc_k1"][p] > M2["acc_k1"][b] for b, p in PAIRS)),
      "n_post_higher_ent_slope": int(sum(M2["ent_slope"][p] > M2["ent_slope"][b] for b, p in PAIRS))}
# between-base-model association: back-half z ~ acc_k1 across the 11 base models; predicted shift for each pair
bx = np.array([M2["acc_k1"][m] for m in bases]); by = np.array([MS[m]["mean"]["back_z"] for m in bases])
sl, ic, rr, pv, se = stats.linregress(bx, by)
pred = {p: sl * (M2["acc_k1"][p] - M2["acc_k1"][b]) for b, p in PAIRS}
obs = {p: s1["diff"]["back_z"]["per_pair"][p] for p in PN}
s5["between_base_regression"] = {"slope_backz_per_acc_k1": sl, "r": rr, "p": pv, "n_bases": len(bases),
                                 "rho_spearman": stats.spearmanr(bx, by)[0],
                                 "predicted_shift": pred, "observed_shift": obs,
                                 "residual_shift": {p: obs[p] - pred[p] for p in PN},
                                 "lineage_weighted_pred": lw([pred[p] for p in PN]), "lineage_weighted_obs": lw([obs[p] for p in PN]),
                                 "lineage_weighted_residual": lw([obs[p] - pred[p] for p in PN])}
ex = np.array([M2["ent_slope"][m] for m in bases]); sl2, ic2, rr2, pv2, _ = stats.linregress(ex, by)
s5["between_base_regression_ent_slope"] = {"slope": sl2, "r": rr2, "p": pv2,
                                           "predicted_shift": {p: sl2 * (M2["ent_slope"][p] - M2["ent_slope"][b]) for b, p in PAIRS},
                                           "lineage_weighted_pred": lw([sl2 * (M2["ent_slope"][p] - M2["ent_slope"][b]) for b, p in PAIRS])}


# item-level adjusted k-slopes (replicates vet3/altexp.py per wording, then averages wordings and tasks)
def demean(v, body, within):
    if not within: return v - v.mean(0)
    o = v.astype(float).copy()
    for g in np.unique(body):
        i = body == g; o[i] = v[i] - v[i].mean(0)
    return o


def ols_t(Y, X):
    n, p = X.shape; XtX = np.linalg.pinv(X.T @ X); B = XtX @ X.T @ Y; E = Y - X @ B
    s2 = (E ** 2).sum(0) / (n - p - 1); se = np.sqrt(np.outer(np.diag(XtX), s2)); return B / se


def adj_back_t(m, f, attn):
    vals = {}
    for w in ("w0", "w1"):
        d = vload(m, f, w, attn, True); body = d["body"]; wi = f in WITHIN; x = d["x"]
        Y = demean(d["y10"], body, wi); k = demean(d["k"][:, None], body, wi)
        e = demean(d["ent"][:, None], body, wi); c = demean(d["correct"].astype(float)[:, None], body, wi)
        for name, X in {"k": k, "k+ent": np.hstack([k, e]), "k+ent+corr": np.hstack([k, e, c])}.items():
            if np.linalg.matrix_rank(X) < X.shape[1]: continue
            t_ = ols_t(Y, X)[0]; vals.setdefault(name, []).append((t_[x >= .5].mean(), t_[x < .5].mean()))
    return {n: np.mean(v, 0) for n, v in vals.items()}


adj = {}
for fam_set, label in ((T, "serial"), (["fixed_digit"], "numeral_control")):
    adj[label] = {}
    for kern in ("primary", "all_eager"):
        R = {}
        for b, p in PAIRS:
            for m in (b, p):
                if m in R: continue
                a_ = "eager" if kern == "all_eager" else ATTN_PRIMARY.get(m, "eager")
                per = [adj_back_t(m, f, a_) for f in fam_set]
                R[m] = {n: np.mean([q[n] for q in per if n in q], 0) for n in ("k", "k+ent", "k+ent+corr")}
        adj[label][kern] = {n: {"back": pair_summary({p: R[p][n][0] - R[b][n][0] for b, p in PAIRS}),
                                "front": pair_summary({p: R[p][n][1] - R[b][n][1] for b, p in PAIRS}),
                                "per_model_back_t": {m: R[m][n][0] for m in R}} for n in ("k", "k+ent", "k+ent+corr")}
# check against vet3/altexp_layers.csv / model_diff3.json (Qwen2.5-7B-Instruct eager there)
M3 = json.load(open(f"{SP}/model_diff3.json"))
adj["model_diff3_decile_back_mean_t"] = {n: pair_summary({p: M3[n]["back_mean_t"][p] - M3[n]["back_mean_t"][b] for b, p in PAIRS})
                                         for n in ("k", "k+ent", "k+ent+corr")}
s5["adjusted_slopes"] = adj
# numeral control at the model level (wording-averaged exact z, Qwen2.5-7B-Instruct SDPA)
nc = {}
for b, p in PAIRS:
    for m in (b, p):
        if m in nc: continue
        nc[m] = regions(model_z(m, "fixed_digit"))
nc_all = {m: regions(model_z(m, "fixed_digit")) for m in ORDER}
s5["numeral_control_model_level"] = {
    "per_model": nc, "all_models_back_z": {m: v["back"] for m, v in nc_all.items()},
    "back": pair_summary({p: nc[p]["back"] - nc[b]["back"] for b, p in PAIRS}),
    "front": pair_summary({p: nc[p]["front"] - nc[b]["front"] for b, p in PAIRS}),
    "serial_minus_control_back": pair_summary({p: (s1["diff"]["back_z"]["per_pair"][p]) - (nc[p]["back"] - nc[b]["back"]) for b, p in PAIRS}),
    "qwen_both_eager_control_back_diff": regions(model_z("Qwen2.5-7B-Instruct", "fixed_digit", "eager"))["back"] - nc["Qwen2.5-7B"]["back"],
    "llama32_3b_pair_control_back_diff_all_sdpa": regions(model_z("Llama-3.2-3B-Instruct", "fixed_digit", "sdpa"))["back"] - regions(model_z("Llama-3.2-3B", "fixed_digit", "sdpa"))["back"],
    "base_models_mean_back_z": float(np.mean([nc_all[m]["back"] for m in bases])),
    "base_models_n_back_lt0": int(sum(nc_all[m]["back"] < 0 for m in bases)),
    "post_models_mean_back_z": float(np.mean([nc_all[p]["back"] for p in PN])),
    "rank_from_most_spreading_of17": {m: int(sorted(ORDER, key=lambda q: nc_all[q]["back"]).index(m)) + 1 for m in PN},
    "serial_base_models_mean_back_z": float(np.mean([MS[m]["mean"]["back_z"] for m in bases])),
    "serial_rank_from_most_spreading_of17": {m: int(sorted(ORDER, key=lambda q: MS[q]["mean"]["back_z"]).index(m)) + 1 for m in PN}}
out["s5_confounds"] = s5

# =========================================================================================================
# 6. CONCENTRATING DIRECTION
s6 = {}
for mt in ("front_z", "40-50_z", "frac_sig_conc", "frac_sig_conc_front", "frac_sig_conc_back"):
    s6[mt] = {k: v for k, v in s1["diff"][mt].items() if k != "per_pair_task"}
    s6[mt]["per_pair_task"] = s1["diff"][mt]["per_pair_task"]
peak = {}
for b, p in PAIRS:
    pb, pp_ = Z[I[b]].mean(0), Z[I[p]].mean(0)
    peak[p] = {"base_peak_decile": int(pb.argmax()), "base_peak_z": pb.max(), "post_peak_decile": int(pp_.argmax()), "post_peak_z": pp_.max(),
               "base_max_front_decile_z": pb[:5].max(), "post_max_front_decile_z": pp_[:5].max()}
s6["peak_concentration"] = peak
s6["peak_diff"] = pair_summary({p: peak[p]["post_peak_z"] - peak[p]["base_peak_z"] for p in PN})
s6["front_max_diff"] = pair_summary({p: peak[p]["post_max_front_decile_z"] - peak[p]["base_max_front_decile_z"] for p in PN})
out["s6_concentrating"] = s6

json.dump(r3(out), open(f"{OUTD}/posttrain_trend.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------ summary
def f2(v): return f"{v:+.2f}"
print("== 1. model level (consist q10), post minus base, task-mean")
for mt in ("back_z", "front_z", "whole_z", "50-70_z", "70-100_z", "frac_sig_spread", "frac_sig_spread_back", "frac_sig_conc", "back_rel"):
    d = s1["diff"][mt]
    print(f"  {mt:18s} " + " ".join(f"{p.split('-')[0][:5]}{p.split('-')[-1][:5]}:{f2(v)}" for p, v in d["per_pair"].items())
          + f" | pairs<0 {d['n_pairs_neg']}/6 lin<0 {d['n_lineages_neg']}/4 pxt<0 {d['n_pair_task_neg']}/24 LW {f2(d['lineage_weighted'])}")
print("  per-task LW back diff:", {t: round(v, 2) for t, v in s1['diff']['back_z']['per_task_lineage_weighted'].items()}, "pairs<0 per task", s1['diff']['back_z']['n_pairs_neg_per_task'])
print("  frac sig spread per model:", {m: round(MS[m]['mean']['frac_sig_spread'], 3) for b, p in PAIRS for m in (b, p)})
print("  back per model:", {m: round(MS[m]['mean']['back_z'], 2) for b, p in PAIRS for m in (b, p)})
print("  PC1 shift:", {p: round(v, 2) for p, v in s1["pc1"]["per_pair"].items()})
for kb in ("item_bootstrap_unpaired", "item_bootstrap_paired_common_items"):
    bt = s1[kb]["per_pair"]
    print(f"  {kb} back diff:", {p: (round(bt[p]['task_mean']['back_diff'], 2), np.round(bt[p]['task_mean']['back_ci'], 2).tolist(), bt[p]['task_mean']['back_p_ge0']) for p in PN})
    print("     LW:", r3(s1[kb]["lineage_weighted_back"]), "without Qwen", r3(s1[kb]["lineage_weighted_back_without_qwen"]), "front LW", r3(s1[kb]["lineage_weighted_front"]))
    print("     per lineage:", r3(s1[kb]["per_lineage_back"]))
print("  per pair x task back diff:", {p: [round(s1['diff']['back_z']['per_pair_task'][p][t], 2) for t in T] for p in PN})
print("  ref scale:", r3(s1["reference_scale"]))
print("== 2. deciles: lineage-weighted diff", np.round(lwdec, 2), "\n   n_lin<0", s2["n_lineages_neg"], " n_pairs<0", s2["n_pairs_neg"], " n_pxt<0", s2["n_pair_task_neg_of24"])
print("   hier CI lo", np.round(s2["hier_boot_ci_lineage_task"][0], 2), "\n   hier CI hi", np.round(s2["hier_boot_ci_lineage_task"][1], 2))
print("   item CI lo", np.round(s2["item_boot_ci_lineage_weighted"][0], 2), "\n   item CI hi", np.round(s2["item_boot_ci_lineage_weighted"][1], 2))
print("   paired item CI lo", np.round(s2["item_boot_paired_ci_lineage_weighted"][0], 2), "\n   paired item CI hi", np.round(s2["item_boot_paired_ci_lineage_weighted"][1], 2))
for k, v in s2["regions"].items(): print("   ", k, r3({kk: vv for kk, vv in v.items()}))
print("== 3. same item")
for lab in ("serial", "numeral_control"):
    for key in ("mean_all", "mean_k2plus", "mean_both_wrong", "median_all", "nosink_all"):
        v = s3[lab][key]; print(f"  {lab:16s} {key:16s} n={v['n']} neg {v['n_neg']} signeg {v['n_sig_neg']} sigpos {v['n_sig_pos']} mean {v['mean']:+.3f} LW {v['lineage_weighted_mean']:+.3f}")
for lab in ("serial", "numeral_control"):
    for key in ("mean_all", "mean_k2plus", "mean_both_wrong"):
        v = s3["recomputed_primary_kernel"][lab][key]; print(f"  RECOMP {lab:16s} {key:16s} n={v['n']} neg {v['n_neg']} signeg {v['n_sig_neg']} sigpos {v['n_sig_pos']} mean {v['mean']:+.3f} LW {v['lineage_weighted_mean']:+.3f}")
print("  reproduction err", s3["reproduction_max_abs_err"])
for key in ("mean_all",):
    print("  per pair serial:", {p: (v['n_neg'], v['n_sig_neg'], v['n_sig_pos'], round(v['mean'], 2)) for p, v in s3['serial'][key]['per_pair'].items()})
    print("  per pair control:", {p: (v['n_neg'], v['n_sig_neg'], v['n_sig_pos'], round(v['mean'], 2)) for p, v in s3['numeral_control'][key]['per_pair'].items()})
print("  per task serial mean_all:", {t: (v['mean_all']['n_neg'], v['mean_all']['n_sig_neg'], v['mean_all']['n_sig_pos']) for t, v in s3['serial_per_task'].items()})
print("  pointer/w0 kernel:", {p: {k: (None if v is None else round(v['all'][0], 2)) for k, v in d.items()} for p, d in s3['pointer_w0_kernel'].items()})
print("  L3B all-sdpa:", {k: v for k, v in s3['llama32_3b_all_sdpa'].items() if k != 'cells'})
print("== 4. robustness (back diff per pair | LW | lineages<0)")
for k, v in s4.items():
    if k.startswith("consist_"):
        print(f"  {k:28s}", {p: round(x, 2) for p, x in v['back']['per_pair'].items()}, f2(v['back']['lineage_weighted']), v['back']['n_lineages_neg'], "pxt", v['n_pair_task_back_neg'])
    elif k.startswith("cells_") and "back_summary" in v:
        print(f"  {k:28s}", {p: round(x, 2) for p, x in v['back'].items()}, f2(v['back_summary']['lineage_weighted']), v['back_summary']['n_lineages_neg'], "pxt", v['n_pair_task_back_neg'])
print("  L3B sdpa:", r3(s4["cells_sdpa_all_tasks_llama32_3b"]))
print("  kernel pointer/w0:", r3(s4["kernel_pointer_w0_back_z"]["per_pair"]))
print("  qwen pair:", r3(qz))
print("  primary with qwen both eager:", r3({k: v for k, v in s4['primary_with_qwen_both_eager'].items()}))
print("== 5. confounds")
print("  acc_k1:", {p: round(v['diff'], 3) for p, v in s5['accuracy']['acc_k1'].items()}, " ent_slope:", {p: round(v['diff'], 3) for p, v in s5['accuracy']['ent_slope'].items()})
print("  between-base reg:", r3({k: v for k, v in s5['between_base_regression'].items()}))
print("  ent_slope reg:", r3(s5['between_base_regression_ent_slope']))
for lab in ("serial", "numeral_control"):
    for kern in ("primary", "all_eager"):
        for n in ("k", "k+ent", "k+ent+corr"):
            v = adj[lab][kern][n]["back"]; print(f"  adj {lab:15s} {kern:9s} {n:11s}", {p: round(x, 2) for p, x in v['per_pair'].items()}, "LW", f2(v['lineage_weighted']), "lin<0", v['n_lineages_neg'])
for n in ("k", "k+ent", "k+ent+corr"):
    v = adj["model_diff3_decile_back_mean_t"][n]; print(f"  model_diff3 {n:11s}", {p: round(x, 2) for p, x in v['per_pair'].items()}, "LW", f2(v['lineage_weighted']))
v = s5["numeral_control_model_level"]
print("  numeral control back diff:", {p: round(x, 2) for p, x in v['back']['per_pair'].items()}, "LW", f2(v['back']['lineage_weighted']), "lin<0", v['back']['n_lineages_neg'])
print("  numeral control per model back:", {m: round(x['back'], 2) for m, x in v['per_model'].items()})
print("  serial minus control:", {p: round(x, 2) for p, x in v['serial_minus_control_back']['per_pair'].items()}, "LW", f2(v['serial_minus_control_back']['lineage_weighted']))
print("  qwen both-eager control diff", round(v['qwen_both_eager_control_back_diff'], 2), "L3B pair control all-SDPA", round(v['llama32_3b_pair_control_back_diff_all_sdpa'], 2))
print("  control: base mean back", round(v['base_models_mean_back_z'], 2), "bases<0", v['base_models_n_back_lt0'], "post mean", round(v['post_models_mean_back_z'], 2), "ranks", v['rank_from_most_spreading_of17'])
print("  serial: base mean back", round(v['serial_base_models_mean_back_z'], 2), "ranks", v['serial_rank_from_most_spreading_of17'])
for k in ("consist_q25", "consist_q50", "consist_entropy", "consist_top1"):
    print("  50-70", k, {p: round(x, 2) for p, x in s4[k]['50-70']['per_pair'].items()}, "LW", round(s4[k]['50-70']['lineage_weighted'], 2), "lin<0", s4[k]['50-70']['n_lineages_neg'])
for k in ("cells_all_eager", "cells_sink_head_removed_eager", "cells_sink_item_removed_eager", "cells_olmo_noclean"):
    print("  50-70", k, {p: round(x, 2) for p, x in s4[k]['50-70'].items()}, "LW", round(s4[k]['50-70_summary']['lineage_weighted'], 2), "lin<0", s4[k]['50-70_summary']['n_lineages_neg'])
print("== 6. concentrating")
for mt in ("front_z", "40-50_z", "frac_sig_conc", "frac_sig_conc_front"):
    d = s6[mt]; print(f"  {mt:20s}", {p: round(x, 3) for p, x in d['per_pair'].items()}, "LW", f2(d['lineage_weighted']), "pairs<0", d['n_pairs_neg'], "pxt<0", d.get('n_pair_task_neg'))
print("  peak:", r3(peak))
print("  peak diff LW", round(s6['peak_diff']['lineage_weighted'], 2), "front max diff LW", round(s6['front_max_diff']['lineage_weighted'], 2))
