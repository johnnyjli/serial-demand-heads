"""T13 final layer; T15 mean |z| back vs front (serial and control)."""
import json, numpy as np
from vo_lib import *
S0 = json.load(open(f"{OUT}/s0_build.json"))
LZ = layer_z("q10"); LP = layer_z("q10", "p")
sig_final = sig_other = n_other = 0; conc = spread = 0; fz = {}; allc = {}; alls = {}
top1 = top3 = 0; l0c = l0s = 0; l0mean = {}
for m in M:
    zs = np.array([LZ[m][t] for t in TK]); ps = np.array([LP[m][t] for t in TK])
    sig_final += int((ps[:, -1] < .05).sum()); sig_other += int((ps[:, :-1] < .05).sum()); n_other += ps[:, :-1].size
    conc += int(((ps[:, -1] < .05) & (zs[:, -1] > 0)).sum()); spread += int(((ps[:, -1] < .05) & (zs[:, -1] < 0)).sum())
    fz[m] = float(zs[:, -1].mean()); allc[m] = bool((zs[:, -1] > 0).all()); alls[m] = bool((zs[:, -1] < 0).all())
    ma = np.abs(zs).mean(0); rk = np.argsort(-ma); top1 += int(rk[0] == len(ma) - 1); top3 += int(len(ma) - 1 in rk[:3])
    l0c += int(((ps[:, 0] < .05) & (zs[:, 0] > 0)).sum()); l0s += int(((ps[:, 0] < .05) & (zs[:, 0] < 0)).sum()); l0mean[m] = float(zs[:, 0].mean())
print(f"final layer significant {sig_final}/68 ({sig_final/68:.2f}); other layers {sig_other}/{n_other} ({sig_other/n_other:.3f}); final sig conc {conc} spread {spread}")
print("task-mean final z:", {m: round(v, 2) for m, v in fz.items()})
q = [m for m in M if fam(m) == "Qwen"]; nq = [m for m in M if fam(m) != "Qwen"]
print(f"Qwen final<0: {sum(fz[m]<0 for m in q)}/{len(q)}; non-Qwen final>0: {sum(fz[m]>0 for m in nq)}/{len(nq)}; all final>0 {sum(v>0 for v in fz.values())}/17")
tab = [[sum(fz[m] < 0 for m in q), sum(fz[m] >= 0 for m in q)], [sum(fz[m] < 0 for m in nq), sum(fz[m] >= 0 for m in nq)]]
print("  Fisher models", tab, stats.fisher_exact(tab))
lf = lin_mean(fz); ql = [L for L in LINS if fam(L) == "Qwen"]; nql = [L for L in LINS if fam(L) != "Qwen"]
tabl = [[sum(lf[L] < 0 for L in ql), sum(lf[L] >= 0 for L in ql)], [sum(lf[L] < 0 for L in nql), sum(lf[L] >= 0 for L in nql)]]
print("  Fisher lineages", tabl, stats.fisher_exact(tabl), " lineage vals", {L[:14]: round(v, 2) for L, v in lf.items()})
print("  concentrate in all 4 tasks:", [m for m in M if allc[m]], " spread in all 4:", [m for m in M if alls[m]])
print(f"  final layer largest mean|z| in {top1} models, top-3 in {top3}")
print(f"  layer-0 sig cells conc {l0c} spread {l0s}; layer-0 task-mean z>0 {sum(v>0 for v in l0mean.values())}/17")
# final layer per task counts
for j, t in enumerate(TK): print(f"   {t}: final z>0 in {sum(LZ[m][t][-1] > 0 for m in M)}/17; Qwen<0 {sum(LZ[m][t][-1] < 0 for m in q)}/6; nonQwen>0 {sum(LZ[m][t][-1] > 0 for m in nq)}/11")
cz = {m: S0["ctrl_z"][m][-1] for m in M}
print(f"control final z<0: Qwen {sum(cz[m]<0 for m in q)}/6, others {sum(cz[m]<0 for m in nq)}/11", {m[:12]: round(v, 2) for m, v in cz.items()})
# other statistics: family split
for st in ["q25", "q50", "entropy", "top1"]:
    L2 = layer_z(st); f2 = {m: np.mean([L2[m][t][-1] for t in TK]) for m in M}
    print(f"  {st}: Qwen final<0 {sum(f2[m]<0 for m in q)}/6, nonQwen final>0 {sum(f2[m]>0 for m in nq)}/11")
# ===== T15 =====
print()
def halves(z):
    L = len(z); x = np.arange(L) / (L - 1); return np.abs(z[x >= .5]).mean(), np.abs(z[x < .5]).mean()
def dec_abs(z):
    d = deciles(np.abs(z)); return d[5:].mean(), d[:5].mean()
for nm, fn in [("layers x>=.5", halves), ("deciles of |z|", dec_abs)]:
    ser = {m: np.mean([fn(LZ[m][t])[0] - fn(LZ[m][t])[1] for t in TK]) for m in M}
    ctl = {m: fn(np.array(S0["ctrl_z"][m]))[0] - fn(np.array(S0["ctrl_z"][m]))[1] for m in M}
    ls, lc = lin_mean(ser), lin_mean(ctl)
    print(f"T15 [{nm}] serial back>front {sum(v>0 for v in ser.values())}/17, lin {sum(v>0 for v in ls.values())}/11, lin mean CI {np.round(boot_ci(list(ls.values())),2)} | control {sum(v>0 for v in ctl.values())}/17, lin {sum(v>0 for v in lc.values())}/11, CI {np.round(boot_ci(list(lc.values())),2)}")
    # serial minus control difference in the size of the late-early asymmetry
    dd = {m: ser[m] - ctl[m] for m in M}; print(f"    serial asym > control asym: {sum(v>0 for v in dd.values())}/17, lin {sum(v>0 for v in lin_mean(dd).values())}/11")
# per task
for j, t in enumerate(TK):
    v = {m: halves(LZ[m][t])[0] - halves(LZ[m][t])[1] for m in M}; print(f"   {t}: back|z|>front|z| {sum(x>0 for x in v.values())}/17")
# null expectation: are late layers more correlated (fewer effective tests)? mean |z| under null ~0.8 regardless.
