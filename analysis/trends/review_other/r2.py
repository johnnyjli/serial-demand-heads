"""Follow-up checks: (1) permutation-scheme comparability (binding/relational z use within-body permutation, pointer/arith/control do not);
T1/T3/T4/T15 counts with every task on the same scheme and in effect size; (2) post-training summary from pairs_q10 (sign/significance tallies,
serial vs control); (3) Bonferroni-style correction for the mid-depth boundary scan.  Reads r1_cells.json (written by r1.py)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, sys, json, numpy as np
from scipy.stats import binomtest
SP = str(paths.ANALYSIS)
OUT = os.path.join(SP, "trends", "review_other")
sys.path.insert(0, SP)
import model_diff as MDm; MDm.SP = SP
from model_diff import ORDER, T, LIN, dec_prof
LINS = sorted(set(LIN.values()), key=ORDER.index)
def lm(v):
    v = np.asarray(v, float); return np.array([v[[i for i, m in enumerate(ORDER) if LIN[m] == L]].mean(0) for L in LINS])
CC = json.load(open(f"{OUT}/r1_cells.json"))
T5 = T + ["fixed_digit"]
Zw = np.array([[dec_prof(np.array(CC[f"{m}|{t}"]["z"])) for t in T5] for m in ORDER])
Zo = np.array([[dec_prof(np.array(CC[f"{m}|{t}"]["z_other_scheme"])) for t in T5] for m in ORDER])
Rl = np.array([[dec_prof(np.array(CC[f"{m}|{t}"]["rel"])) for t in T5] for m in ORDER])
out = {}
# |z| ratio within vs not-within for chain tasks (layer level)
rat = []
for m in ORDER:
    for t in ("binding", "relational"):
        a = np.abs(np.array(CC[f"{m}|{t}"]["z"])); b = np.abs(np.array(CC[f"{m}|{t}"]["z_other_scheme"])); rat.append(a.mean() / b.mean())
out["chain_mean_absz_within_over_notwithin"] = [float(np.median(rat)), float(np.min(rat)), float(np.max(rat))]
# same-scheme serial profiles (all tasks not-within)
Zsame = np.concatenate([Zw[:, :2], Zo[:, 2:4]], 1)
def counts(P4, name):
    pr = P4.mean(1); bk = pr[:, 5:].mean(1); fr = pr[:, :5].mean(1); sw = pr[:, 4] - pr[:, 5]
    return {"back<0": [int((bk < 0).sum()), int((lm(bk) < 0).sum())], "back<front": [int((bk < fr).sum()), int((lm(bk - fr) < 0).sum())],
            "switch": [int((sw > 0).sum()), int((lm(sw) > 0).sum())], "d40_50>0": [int((pr[:, 4] > 0).sum()), int((lm(pr[:, 4]) > 0).sum())],
            "binding_strongest": int((P4[:, :, 5:].mean(2).argmin(1) == 2).sum()), "arith_weakest": int((P4[:, :, 5:].mean(2).argmax(1) == 1).sum())}
out["serial_counts_within_scheme(primary_rebuild)"] = counts(Zw[:, :4], "w")
out["serial_counts_all_not_within"] = counts(Zsame, "same")
out["serial_counts_rel_effect_size"] = counts(Rl[:, :4], "rel")
# T15 under the same scheme
def halves_abs(z):
    z = np.abs(np.asarray(z)); L_ = len(z); x = np.arange(L_) / (L_ - 1); return z[x >= .5].mean() - z[x < .5].mean()
d_same = np.array([np.mean([halves_abs(CC[f"{m}|{t}"]["z"] if t in ("pointer", "arith") else CC[f"{m}|{t}"]["z_other_scheme"]) for t in T]) for m in ORDER])
d_ser_same = np.array([halves_abs(np.mean([np.abs(np.array(CC[f"{m}|{t}"]["z"] if t in ("pointer", "arith") else CC[f"{m}|{t}"]["z_other_scheme"])) for t in T], 0)) for m in ORDER])
out["T15_serial_meanabs_back>front_same_scheme"] = [int((d_ser_same > 0).sum()), int((lm(d_ser_same) > 0).sum())]
out["T15_per_task_not_within"] = {t: int(sum(halves_abs(CC[f"{m}|{t}"]["z_other_scheme" if t in ("binding", "relational") else "z"]) > 0 for m in ORDER)) for t in T}
out["T15_per_task_within"] = {t: int(sum(halves_abs(CC[f"{m}|{t}"]["z_other_scheme" if t in ("pointer", "arith", "fixed_digit") else "z"]) > 0 for m in ORDER)) for t in T5}
# (3) boundary scan correction
sc = [6, 7, 7, 5, 16, 13, 6, 9, 7]
out["switch_boundary_scan_bonferroni9_models"] = float(min(1, 9 * binomtest(16, 17, .5).pvalue))
out["switch_boundary_scan_bonferroni9_lineages"] = float(min(1, 9 * binomtest(10, 11, .5).pvalue))
# (2) post-training tallies from pairs_q10 (post minus base relative k-slope of top-share, 40-80% depth; negative = post spreads more)
P = json.load(open(f"{SP}/pairs_q10.json")); tal = {}
for r in P:
    key = r["post"]; fam = r["cell"].split("/")[0]; grp = "control" if fam == "fixed_digit" else "serial"
    for stat in ("mean_all", "mean_both_wrong", "nosink_all", "median_all"):
        v = r.get(stat)
        if v is None: continue
        d = tal.setdefault(key, {}).setdefault(stat, {"serial": [0, 0, 0, 0], "control": [0, 0, 0, 0]})   # [neg, neg&p<.05, pos&p<.05, n]
        d[grp][0] += v[0] < 0; d[grp][1] += (v[0] < 0) and v[1] < .05; d[grp][2] += (v[0] > 0) and v[1] < .05; d[grp][3] += 1
out["pairs_q10_tally_[neg, sig_neg, sig_pos, n]"] = tal
# lineage-level summary: mean over the 8 serial cells of the post-minus-base difference, per pair
ms = {}
for r in P:
    fam = r["cell"].split("/")[0]; grp = "control" if fam == "fixed_digit" else "serial"
    ms.setdefault(r["post"], {}).setdefault(grp, []).append(r["mean_all"][0])
out["pairs_q10_mean_diff"] = {k: {g: round(float(np.mean(v)), 3) for g, v in d.items()} for k, d in ms.items()}
json.dump(out, open(f"{OUT}/r2.json", "w"), indent=1, default=float)
for k, v in out.items(): print(k, v)
