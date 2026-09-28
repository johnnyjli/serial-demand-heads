"""T10, T11, T12: accuracy vs back-half / 50-70% spreading across groups; within-model task-centred link."""
import json, itertools, numpy as np
from vo_lib import *
F2 = json.load(open(f"{SP}/model_diff2.json"))["features"]; S0 = json.load(open(f"{OUT}/s0_build.json"))
H = F2["H"]
Z = Zdec("q10")
V = {"back": dict(zip(M, Z[:, :, 5:].mean((1, 2)))), "d50_70": dict(zip(M, Z[:, :, 5:7].mean((1, 2))))}
acc = {m: np.mean([S0["acc"][m][t] for t in TK]) for m in M}; acck1 = {m: np.mean([S0["acc_k1"][m][t] for t in TK]) for m in M}
groups = {"all17": M, "bases": BASES, "H32": [m for m in M if H[m] == 32], "H32_bases": [m for m in BASES if H[m] == 32], "nonQwen": [m for m in M if fam(m) != "Qwen"],
          "nonQwen_bases": [m for m in BASES if fam(m) != "Qwen"]}
for accname, A in [("acc", acc), ("acc_k1", acck1)]:
    for reg in ["back", "d50_70"]:
        s = []
        for g, ms in groups.items():
            r = spear([A[m] for m in ms], [V[reg][m] for m in ms]); s.append(f"{g}({len(ms)}) {r[0]:+.2f} p={r[1]:.4f}")
        pr17 = partial_spear([A[m] for m in M], [V[reg][m] for m in M], [H[m] for m in M])
        prb = partial_spear([A[m] for m in BASES], [V[reg][m] for m in BASES], [H[m] for m in BASES])
        print(f"{accname}~{reg}: " + "; ".join(s) + f"; partial|H 17 {pr17[0]:+.2f} p={pr17[1]:.3f}; bases|H {prb[0]:+.2f} p={prb[1]:.3f}")
    print()
# H among bases vs back (T10 note) and within H32 set composition
print("H32 set:", groups["H32"])
# other statistics for T11 H32
for st in ["q25", "q50", "entropy", "top1"]:
    Zs = Zdec(st); v = dict(zip(M, Zs[:, :, 5:7].mean((1, 2))))
    r = spear([acc[m] for m in groups["H32"]], [v[m] for m in groups["H32"]]); rq = spear([acc[m] for m in groups["nonQwen"]], [v[m] for m in groups["nonQwen"]])
    print(f"  {st}: acc~d50_70 H32 {r[0]:+.2f} p={r[1]:.3f}; nonQwen {rq[0]:+.2f} p={rq[1]:.3f}")
# lineage-level for T11 (independent units): lineage means among H32 lineages
L32 = [L for L in LINS if H[L] == 32]; la = lin_mean(acc); lv = lin_mean(V["d50_70"])
print("H32 lineages:", L32, "rho", spear([la[L] for L in L32], [lv[L] for L in L32]))
LnQ = [L for L in LINS if fam(L) != "Qwen"]; print("nonQwen lineages rho", spear([la[L] for L in LnQ], [lv[L] for L in LnQ]))
# ===== T12 within model =====
Bt = Z[:, :, 5:].mean(2); At = np.array([[S0["acc"][m][t] for t in TK] for m in M])
Bc = Bt - Bt.mean(0); Ac = At - At.mean(0)       # centre each task across models
rh = np.array([stats.spearmanr(Ac[i], Bc[i]).correlation for i in range(17)])
print(f"\nT12 within-model rho: mean {np.nanmean(rh):+.3f}, negative {int((rh<0).sum())}/17, values {np.round(rh,2)}")
rng = np.random.default_rng(1); NB = 20000; null = np.empty(NB)
perms = list(itertools.permutations(range(4)))
for b in range(NB):
    null[b] = np.nanmean([stats.spearmanr(Ac[i][list(perms[rng.integers(24)])], Bc[i]).correlation for i in range(17)]) if b < 5000 else np.nan
null = null[:5000]
print(f"  task-label permutation (5000): one-sided p={((null <= np.nanmean(rh)).sum()+1)/(len(null)+1):.4f}; two-sided p={((np.abs(null) >= abs(np.nanmean(rh))).sum()+1)/(len(null)+1):.4f}")
# lineage-level: average centred values within lineage
LA = np.array([Ac[[i for i, m in enumerate(M) if LINEAGE[m] == L]].mean(0) for L in LINS]); LBm = np.array([Bc[[i for i, m in enumerate(M) if LINEAGE[m] == L]].mean(0) for L in LINS])
rl = np.array([stats.spearmanr(LA[j], LBm[j]).correlation for j in range(11)])
nl = np.array([np.nanmean([stats.spearmanr(LA[j][list(perms[rng.integers(24)])], LBm[j]).correlation for j in range(11)]) for _ in range(5000)])
print(f"  lineage-level mean rho {np.nanmean(rl):+.3f}, negative {int((rl<0).sum())}/11; one-sided p={((nl <= np.nanmean(rl)).sum()+1)/5001:.4f} two-sided p={((np.abs(nl) >= abs(np.nanmean(rl))).sum()+1)/5001:.4f}")
pooled = stats.spearmanr(LA.ravel(), LBm.ravel()); print(f"  pooled lineage x task (44): rho {pooled.correlation:+.3f} p={pooled.pvalue:.4f}")
# alt centring: also centre within model (double-centred)
Ad = Ac - Ac.mean(1, keepdims=True); Bd = Bc - Bc.mean(1, keepdims=True)
pooled17 = stats.spearmanr(Ad.ravel(), Bd.ravel()); print(f"  pooled model x task double-centred (68): rho {pooled17.correlation:+.3f} p={pooled17.pvalue:.4f}")
# per task across models
for j, t in enumerate(TK):
    r = spear(At[:, j], Bt[:, j]); bi = [M.index(m) for m in BASES]; rb = spear(At[bi, j], Bt[bi, j])
    pH = partial_spear(At[:, j], Bt[:, j], [H[m] for m in M])
    print(f"  {t:10s}: 17 models rho {r[0]:+.2f} p={r[1]:.3f}; bases {rb[0]:+.2f} p={rb[1]:.3f}; partial|H (17) {pH[0]:+.2f} p={pH[1]:.3f}")
