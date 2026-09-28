import json, numpy as np
from ilib import *
R = json.load(open(f"{SP}/pairs_q10.json"))
pidx = {p: i for i, (b, p) in enumerate(PAIRS)}
def count(rows, key):
    v = np.array([r[key][0] for r in rows if key in r]); p = np.array([r[key][1] for r in rows if key in r])
    return len(v), int((v < 0).sum()), int(((v < 0) & (p < .05)).sum()), int(((v > 0) & (p < .05)).sum()), float(v.mean())
def lwmean(rows, key):
    per = np.array([np.mean([r[key][0] for r in rows if r["post"] == p and key in r]) for b, p in PAIRS]); return lw(per), per
serial = [r for r in R if r["cell"].split("/")[0] in TASKS]; ctrl = [r for r in R if r["cell"].startswith("fixed_digit")]
for key in ["mean_all", "mean_k2plus", "mean_both_wrong", "median_all", "median_k2plus", "median_both_wrong", "nosink_all", "nosink_k2plus", "nosink_both_wrong"]:
    n, neg, sn, sp, mu = count(serial, key); L, per = lwmean(serial, key)
    n2, neg2, sn2, sp2, mu2 = count(ctrl, key); L2, per2 = lwmean(ctrl, key)
    print(f"{key:18s} serial n={n} neg {neg} signeg {sn} sigpos {sp} mean {mu:+.3f} LW {L:+.3f} | control n={n2} neg {neg2} signeg {sn2} sigpos {sp2} LW {L2:+.3f}")
print("per pair mean_all (neg/signeg/sigpos of 8):")
for b, p in PAIRS:
    rr = [r for r in serial if r["post"] == p]; print("  ", p, count(rr, "mean_all")[1:4], " nosink:", count(rr, "nosink_all")[1:4], " both_wrong:", count(rr, "mean_both_wrong")[1:4], " k2plus:", count(rr, "mean_k2plus")[1:4])
print("per task mean_all (of 12):")
for f in TASKS:
    rr = [r for r in serial if r["cell"].startswith(f)]; print("  ", f, count(rr, "mean_all")[1:4], " both_wrong:", count(rr, "mean_both_wrong")[1:4], "k2plus", count(rr, "mean_k2plus")[1:4])
# Llama-only
rr = [r for r in serial if r["post"].startswith("Llama")]; print("Llama pairs mean_all", count(rr, "mean_all"), "nosink", count(rr, "nosink_all"))
