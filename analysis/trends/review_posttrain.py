"""Reviewer spot-checks of the post-training trend claims (C1-C7).  Read-only on sources; writes review_posttrain.json.
Independent re-implementation from consist.json / pairs_q10.json / vet3 cache (not importing posttrain_trend.py)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, sys, json, itertools, warnings
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore")
os.environ["VPRIM"] = "y10"
SP = str(paths.ANALYSIS)
sys.path.insert(0, f"{SP}/vet3")
from v2lib import load as vload, zprof
OUT = {}
T = ["pointer", "arith", "binding", "relational"]
PAIRS = [("Llama-3.2-3B", "Llama-3.2-3B-Instruct"), ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"),
         ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT"),
         ("OLMo-2-1124-7B", "OLMo-2-1124-7B-DPO"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct")]
LIN = [[0], [1], [2], [3, 4, 5]]
ORDER = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.2-3B-Instruct", "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
         "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
         "OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct",
         "Mistral-7B-v0.1", "Mistral-Small-24B-Base-2501"]
BASES = [m for m in ORDER if not any(s in m for s in ("Instruct", "SFT", "DPO"))]


def lineage(v): v = np.asarray(v, float); return np.array([v[i].mean() for i in LIN])


def signflip(lv):
    """exact one-sided sign-flip p (mean <= observed) over the 4 lineage values, and a t-based 95% CI (df=3)"""
    lv = np.asarray(lv, float); obs = lv.mean()
    null = [np.mean(lv * np.array(s)) for s in itertools.product([1, -1], repeat=len(lv))]
    p1 = float(np.mean([x <= obs + 1e-12 for x in null]))
    se = lv.std(ddof=1) / np.sqrt(len(lv)); tq = stats.t.ppf(.975, len(lv) - 1)
    return {"lineages": lv.round(3).tolist(), "LW": round(obs, 3), "signflip_p_one_sided": p1,
            "t": round(obs / se, 2), "t_ci95": [round(obs - tq * se, 2), round(obs + tq * se, 2)],
            "p_two_sided_t": round(float(2 * stats.t.sf(abs(obs / se), len(lv) - 1)), 3)}


# ---- 1. consist.json: region means per model x task
C = {(r["model"], r["stat"]): r for r in json.load(open(f"{SP}/consist.json"))}


def reg(z, lo, hi):
    z = np.asarray(z, float); x = np.arange(len(z)) / (len(z) - 1); return np.nanmean(z[(x >= lo) & (x < hi)])


def mreg(m, lo, hi, stat="q10", tasks=T):
    return np.mean([reg(C[(m, stat)]["families"][t]["z"], lo, hi) for t in tasks])


R = {}
for name, (lo, hi) in {"back": (.5, 1.01), "front": (0, .5), "whole": (0, 1.01), "50-70": (.5, .7), "70-100": (.7, 1.01), "40-50": (.4, .5), "20-30": (.2, .3)}.items():
    d = np.array([mreg(p, lo, hi) - mreg(b, lo, hi) for b, p in PAIRS])
    R[name] = {"per_pair": d.round(3).tolist(), **signflip(lineage(d))}
    # without Qwen (3 lineages)
    lv = lineage(d)[[0, 1, 3]]; R[name]["LW_without_qwen"] = round(lv.mean(), 3)
OUT["region_diffs_consist_q10"] = R
# per-task back diffs and per-lineage-task
bt = np.array([[reg(C[(p, "q10")]["families"][t]["z"], .5, 1.01) - reg(C[(b, "q10")]["families"][t]["z"], .5, 1.01) for t in T] for b, p in PAIRS])
OUT["back_per_pair_task"] = bt.round(2).tolist()
lt = np.array([bt[i].mean(0) for i in LIN])
OUT["back_lineage_x_task_neg"] = int((lt < 0).sum())
# base-model back z SD (reference scale)
OUT["base_back_sd"] = round(float(np.std([mreg(m, .5, 1.01) for m in BASES], ddof=1)), 3)

# ---- 2. deciles: is 50-70 a post-hoc window?  enumerate all contiguous decile windows
def dec(z):
    z = np.asarray(z, float); L = len(z); x = np.arange(L) / (L - 1); d = np.minimum((x * 10).astype(int), 9)
    return np.array([np.nanmean(z[d == i]) for i in range(10)])


D = np.array([[dec(C[(p, "q10")]["families"][t]["z"]) - dec(C[(b, "q10")]["families"][t]["z"]) for t in T] for b, p in PAIRS])  # 6x4x10
lwdec = np.array([D[i].mean((0, 1)) for i in LIN]).mean(0)
OUT["decile_LW"] = lwdec.round(2).tolist()
wins = []
for a in range(10):
    for b_ in range(a + 1, 11):
        lv = np.array([D[i][:, :, a:b_].mean() for i in LIN]); pr = D[:, :, a:b_].mean((1, 2))
        wins.append({"win": f"{10*a}-{10*b_}", "LW": round(lv.mean(), 3), "n_pairs_neg": int((pr < 0).sum()), "n_lin_neg": int((lv < 0).sum())})
OUT["n_windows"] = len(wins)
OUT["windows_all_6_pairs_neg"] = [w for w in wins if w["n_pairs_neg"] == 6]
OUT["most_negative_2dec_windows"] = sorted([w for w in wins if int(w["win"].split("-")[1]) - int(w["win"].split("-")[0]) == 20], key=lambda w: w["LW"])[:4]
OUT["50-70_decile_vs_layer"] = {"decile_LW": round(lwdec[5:7].mean(), 3), "layer_LW": R["50-70"]["LW"]}
# selection-aware: sign-flip over lineages of the MIN over all 2-decile windows (lineage-level; 16 flips)
L4 = np.array([D[i].mean((0, 1)) for i in LIN])  # 4 x 10
obs_min = min(L4[:, a:a + 2].mean(1).mean() for a in range(9))
nulls = [min((L4 * np.array(s)[:, None])[:, a:a + 2].mean(1).mean() for a in range(9)) for s in itertools.product([1, -1], repeat=4)]
OUT["min_2dec_window_signflip_p"] = float(np.mean([x <= obs_min + 1e-12 for x in nulls]))

# ---- 3. same-item (pairs_q10): lineage-level summary, within-body permutation check, both-wrong n
PQ = json.load(open(f"{SP}/pairs_q10.json"))
ser = [r for r in PQ if r["cell"].split("/")[0] in T]; ctl = [r for r in PQ if r["cell"].startswith("fixed_digit")]
def cnt(Rs, key):
    v = np.array([r[key][0] for r in Rs]); p = np.array([r[key][1] for r in Rs])
    per_pair = np.array([np.mean([r[key][0] for r in Rs if r["post"] == pp]) for _, pp in PAIRS])
    return {"n": len(v), "neg": int((v < 0).sum()), "sig_neg": int(((v < 0) & (p < .05)).sum()), "sig_pos": int(((v > 0) & (p < .05)).sum()),
            "per_pair_mean": per_pair.round(3).tolist(), **signflip(lineage(per_pair))}
OUT["same_item"] = {f"serial_{k}": cnt(ser, k) for k in ("mean_all", "mean_k2plus", "mean_both_wrong")}
OUT["same_item"].update({f"control_{k}": cnt(ctl, k) for k in ("mean_all", "mean_k2plus", "mean_both_wrong")})
# expected false positives per direction at alpha .05 two-sided over 48 tests
OUT["same_item"]["expected_sig_per_direction_under_null_48"] = 48 * .025


def win(L): return slice(int(.4 * (L - 1)), int(.8 * (L - 1)) + 1)


def perm_p(y, k, body=None, n=2000, seed=0):
    r_ = np.random.default_rng(seed); kc = k - k.mean(); ss = kc @ kc; yc = y - y.mean(); o = kc @ yc / ss
    if body is None:
        nl = np.array([r_.permutation(kc) @ yc / ss for _ in range(n)])
    else:
        grp = [np.nonzero(body == g)[0] for g in np.unique(body)]; nl = np.empty(n)
        for i in range(n):
            kp = k.copy()
            for g in grp: kp[g] = k[r_.permutation(g)]
            kpc = kp - kp.mean(); nl[i] = kpc @ yc / ss
    return float(o), float((np.sum(np.abs(nl) >= abs(o) - 1e-15) + 1) / (n + 1))


wb = []; nbw = []
for b, p in PAIRS:
    for f in T:
        for w in ("w0", "w1"):
            A, B = vload(b, f, w, "eager", True), vload(p, f, w, "eager", True)
            ka = {(int(k), int(bb)): j for j, (k, bb) in enumerate(zip(A["k"], A["body"]))}
            kb = {(int(k), int(bb)): j for j, (k, bb) in enumerate(zip(B["k"], B["body"]))}
            keys = [x for x in ka if x in kb]; ia = np.array([ka[x] for x in keys]); ib = np.array([kb[x] for x in keys])
            L = A["y10"].shape[1]; wd = win(L)
            ya, yb = np.nanmean(A["y10"][ia][:, wd], 1), np.nanmean(B["y10"][ib][:, wd], 1)
            k = A["k"][ia]; body = A["body"][ia]; diff = yb / np.nanmean(yb) - ya / np.nanmean(ya)
            bw = ~A["correct"][ia].astype(bool) & ~B["correct"][ib].astype(bool)
            nbw.append((p, f, w, int(bw.sum()), len(k)))
            if f in ("binding", "relational"):
                s0, p0 = perm_p(diff, k); s1, p1 = perm_p(diff, k, body)
                wb.append({"pair": p, "cell": f"{f}/{w}", "slope": round(100 * s0, 3), "p_across": round(p0, 4), "p_within_body": round(p1, 4),
                           "n_bodies": int(len(np.unique(body))), "n": len(k)})
OUT["within_body_check"] = {"cells": wb,
                            "sig_neg_across": sum(r["slope"] < 0 and r["p_across"] < .05 for r in wb),
                            "sig_neg_within": sum(r["slope"] < 0 and r["p_within_body"] < .05 for r in wb),
                            "sig_pos_across": sum(r["slope"] > 0 and r["p_across"] < .05 for r in wb),
                            "sig_pos_within": sum(r["slope"] > 0 and r["p_within_body"] < .05 for r in wb)}
OUT["both_wrong_n"] = {"median_frac": float(np.median([a / b_ for _, _, _, a, b_ in nbw])), "min": min(a for *_, a, _ in nbw),
                       "per_cell": [(p, f, w, a, n) for p, f, w, a, n in nbw]}

# ---- 4. numeral control at the model level (wording-averaged exact z from cache; Qwen-I SDPA)
def wz(m, f, attn):
    a, b = vload(m, f, "w0", attn, True), vload(m, f, "w1", attn, True)
    if a is None or b is None: return None
    ka = {(int(k), int(bb)): j for j, (k, bb) in enumerate(zip(a["k"], a["body"]))}
    kb = {(int(k), int(bb)): j for j, (k, bb) in enumerate(zip(b["k"], b["body"]))}
    keys = sorted(set(ka) & set(kb)); ia = np.array([ka[x] for x in keys]); ib = np.array([kb[x] for x in keys])
    Y = .5 * (a["y10"][ia] + b["y10"][ib]); k = np.array([x[0] for x in keys], float); body = np.array([x[1] for x in keys])
    return Y, k, body


def backz(Y, k, body, within=False):
    z = zprof(Y, k, body, within)[2]; x = np.arange(len(z)) / (len(z) - 1); return z[x >= .5].mean()


ATT = lambda m: "sdpa" if m == "Qwen2.5-7B-Instruct" else "eager"
ctl_back = {m: backz(*wz(m, "fixed_digit", ATT(m))) for m in ORDER}
cd = np.array([ctl_back[p] - ctl_back[b] for b, p in PAIRS])
OUT["control_back"] = {"per_model": {m: round(v, 2) for m, v in ctl_back.items()}, "per_pair": cd.round(2).tolist(), **signflip(lineage(cd)),
                       "rank_most_spreading": {m: sorted(ORDER, key=lambda q: ctl_back[q]).index(m) + 1 for m in ORDER if m in [p for _, p in PAIRS]},
                       "bases_mean": round(np.mean([ctl_back[m] for m in BASES]), 2), "bases_ge0": int(sum(ctl_back[m] >= 0 for m in BASES))}
qe = backz(*wz("Qwen2.5-7B-Instruct", "fixed_digit", "eager"))
OUT["control_back"]["qwen_instr_eager"] = round(qe, 2)
OUT["control_back"]["qwen_instr_rank_if_eager"] = sorted([ctl_back[m] for m in ORDER if m != "Qwen2.5-7B-Instruct"] + [qe]).index(qe) + 1
cd_e = cd.copy(); cd_e[2] = qe - ctl_back["Qwen2.5-7B"]
OUT["control_back"]["qwen_both_eager"] = signflip(lineage(cd_e))
# item bootstrap per pair in the control (unpaired, within-k resampling)
rng = np.random.default_rng(7)
def bidx(k):
    return np.concatenate([rng.choice(np.nonzero(k == kk)[0], (k == kk).sum(), replace=True) for kk in np.unique(k)])
cb = {}
for b, p in PAIRS:
    Yb, kb_, bb_ = wz(b, "fixed_digit", ATT(b)); Yp, kp_, bp_ = wz(p, "fixed_digit", ATT(p)); v = []
    for _ in range(300):
        i1, i2 = bidx(kb_), bidx(kp_); v.append(backz(Yp[i2], kp_[i2], None) - backz(Yb[i1], kb_[i1], None))
    cb[p] = [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]
OUT["control_item_boot_ci"] = cb
# serial vs control shift across pairs
sd = np.array(R["back"]["per_pair"])
OUT["serial_vs_control_shift_corr_pairs"] = round(float(np.corrcoef(sd, cd)[0, 1]), 3)
# serial DiD at lineage level
OUT["serial_minus_control_lineage"] = signflip(lineage(sd - cd))

# ---- 5. confounds
M2 = json.load(open(f"{SP}/model_diff2.json"))["features"]
OUT["acc_k1_diff"] = [round(M2["acc_k1"][p] - M2["acc_k1"][b], 3) for b, p in PAIRS]
OUT["acc_diff"] = [round(M2["acc"][p] - M2["acc"][b], 3) for b, p in PAIRS]
OUT["ent_slope_diff"] = [round(M2["ent_slope"][p] - M2["ent_slope"][b], 3) for b, p in PAIRS]
bx = np.array([M2["acc_k1"][m] for m in BASES]); by = np.array([mreg(m, .5, 1.01) for m in BASES])
lr = stats.linregress(bx, by); tq = stats.t.ppf(.975, len(bx) - 2)
OUT["between_base_reg"] = {"slope": round(lr.slope, 3), "slope_ci95": [round(lr.slope - tq * lr.stderr, 2), round(lr.slope + tq * lr.stderr, 2)],
                           "r": round(lr.rvalue, 3), "p": round(lr.pvalue, 3)}
# adjusted slopes from altexp_layers.csv (Qwen-I eager in that file): back-half mean t per model, wording/task averaged
import pandas as pd
A = pd.read_csv(f"{SP}/vet3/altexp_layers.csv")
A = A[A.task.isin(T) & (A.x >= .5)]
bm = A.groupby(["model", "fit", "task", "w"]).t.mean().groupby(["model", "fit"]).mean()
adj = {}
for fit in ("k", "k+ent", "k+ent+corr", "k+ent+corr+logpgold"):
    d = np.array([bm[(p, fit)] - bm[(b, fit)] for b, p in PAIRS]); adj[fit] = {"per_pair": d.round(2).tolist(), **signflip(lineage(d))}
OUT["altexp_adjusted_back_t_all_eager"] = adj

json.dump(OUT, open(f"{SP}/trends/review_posttrain.json", "w"), indent=1, default=float)
for k, v in OUT.items():
    if k in ("both_wrong_n",): print(k, {kk: vv for kk, vv in v.items() if kk != "per_cell"}); continue
    if k == "within_body_check": print(k, {kk: vv for kk, vv in v.items() if kk != "cells"}); continue
    print(k, json.dumps(v, default=float)[:1500])
