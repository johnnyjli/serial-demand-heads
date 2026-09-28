"""T6, T7: within-model task comparisons of back-half spreading (z and relative slope), accuracy link."""
import json, numpy as np
from vo_lib import *
from s1_profiles import cells_Z
S0 = json.load(open(f"{OUT}/s0_build.json"))
def back_t(Z): return Z[:, :, 5:].mean(2)     # models x tasks
Bz = back_t(Zdec("q10")); Br = back_t(Zdec("q10", "rel"))
P, A, Bi, R = 0, 1, 2, 3
def pair_counts(B, ms, i, j, name):
    d = B[:, i] - B[:, j]; ld = lin_mean(dict(zip(ms, d))) if len(ms) == 17 else {L: np.mean([d[k] for k, m in enumerate(ms) if LINEAGE[m] == L]) for L in sorted(set(LINEAGE[m] for m in ms), key=M.index)}
    return int((d < 0).sum()), len(ms), int(sum(v < 0 for v in ld.values())), len(ld)
out = {}
for lab, (i, j) in {"binding<relational": (Bi, R), "binding<arith": (Bi, A), "binding<pointer": (Bi, P), "relational<arith": (R, A), "pointer<arith": (P, A), "pointer<relational": (P, R)}.items():
    cz = pair_counts(Bz, M, i, j, lab); cr = pair_counts(Br, M, i, j, lab)
    print(f"{lab:22s} z: {cz[0]}/{cz[1]} models, {cz[2]}/{cz[3]} lin (p={binom2(cz[2],cz[3]):.3f}) | rel slope: {cr[0]}/17, {cr[2]}/11")
    out[lab] = {"z": cz, "rel": cr}
# robustness of binding<relational
for st in ["q25", "q50", "entropy", "top1"]:
    print(f"  {st}: b<r {pair_counts(back_t(Zdec(st)), M, Bi, R, '')}")
for name, (attn, var, w) in {"sink_head": ("eager", "sink_head", None), "sdpa": ("sdpa", "all", None), "w0": ("eager", "all", "w0"), "w1": ("eager", "all", "w1")}.items():
    Zv, ms = cells_Z(attn, var, w); print(f"  {name}: b<r {pair_counts(back_t(Zv), ms, Bi, R, '')}")
# accuracy
acc = np.array([[S0["acc"][m][t] for t in TK] for m in M])
print("mean accuracy by task:", dict(zip(TK, acc.mean(0).round(3))))
print("binding more accurate than relational:", int((acc[:, Bi] > acc[:, R]).sum()), "/17")
g_back = Bz[:, Bi] - Bz[:, R]; g_acc = acc[:, Bi] - acc[:, R]
r17 = spear(g_back, g_acc); bi = [M.index(m) for m in BASES]; rb = spear(g_back[bi], g_acc[bi])
print(f"Spearman(bind-rel back gap, bind-rel acc gap): 17 models {r17[0]:+.3f} p={r17[1]:.4f}; bases {rb[0]:+.3f} p={rb[1]:.4f}")
lg = lin_mean(dict(zip(M, g_back))); la = lin_mean(dict(zip(M, g_acc)))
rl = spear(list(lg.values()), list(la.values())); print(f"   lineage-mean (11): {rl[0]:+.3f} p={rl[1]:.4f}")
# same with rel slope
rr = spear(Br[:, Bi] - Br[:, R], g_acc); print(f"   rel-slope gap vs acc gap (17): {rr[0]:+.3f} p={rr[1]:.4f}")
# accuracy gap alternative confound: does binding have more items / higher baseline? check gap vs H
print("   gap vs H:", spear(g_back, [json.load(open(f'{SP}/model_diff2.json'))['features']['H'][m] for m in M]))
# T7: strongest / weakest task per model
for nm, B in [("z", Bz), ("rel", Br)]:
    s = np.argmin(B, 1); w_ = np.argmax(B, 1)
    print(f"{nm}: strongest counts", {t: int((s == j).sum()) for j, t in enumerate(TK)}, " weakest", {t: int((w_ == j).sum()) for j, t in enumerate(TK)})
    LB = np.array([B[[i for i, m in enumerate(M) if LINEAGE[m] == L]].mean(0) for L in LINS])
    print(f"   lineage-level strongest", {t: int((np.argmin(LB, 1) == j).sum()) for j, t in enumerate(TK)}, "weakest", {t: int((np.argmax(LB, 1) == j).sum()) for j, t in enumerate(TK)})
    for j, t in enumerate(TK):
        mu, lo, hi = boot_ci(LB[:, j]); print(f"   lineage mean back {nm} {t}: {mu:+.3f} [{lo:+.3f},{hi:+.3f}]   model mean {B[:, j].mean():+.3f}")
for st in ["q25", "q50", "entropy", "top1"]:
    B = back_t(Zdec(st)); LB = np.array([B[[i for i, m in enumerate(M) if LINEAGE[m] == L]].mean(0) for L in LINS])
    print(f"  {st}: binding strongest {int((np.argmin(B,1)==Bi).sum())}/17 lin {int((np.argmin(LB,1)==Bi).sum())}/11; arith weakest {int((np.argmax(B,1)==A).sum())}/17 lin {int((np.argmax(LB,1)==A).sum())}/11")
for name, (attn, var, w) in {"sink_head": ("eager", "sink_head", None), "sdpa": ("sdpa", "all", None), "w0": ("eager", "all", "w0"), "w1": ("eager", "all", "w1")}.items():
    Zv, ms = cells_Z(attn, var, w); B = back_t(Zv)
    lins = sorted(set(LINEAGE[m] for m in ms), key=M.index); LB = np.array([B[[i for i, m in enumerate(ms) if LINEAGE[m] == L]].mean(0) for L in lins])
    print(f"  {name}: binding strongest {int((np.argmin(B,1)==Bi).sum())}/{len(ms)} lin {int((np.argmin(LB,1)==Bi).sum())}/{len(lins)}; arith weakest {int((np.argmax(B,1)==A).sum())}/{len(ms)} lin {int((np.argmax(LB,1)==A).sum())}/{len(lins)}")
json.dump(out, open(f"{OUT}/s4_tasks.json", "w"), indent=1)
