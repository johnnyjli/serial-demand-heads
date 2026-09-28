"""T8, T9: size trends among base models (log params), within-family pairs with exact within-family permutation."""
import json, itertools, numpy as np
from vo_lib import *
from s1_profiles import cells_Z
F2 = json.load(open(f"{SP}/model_diff2.json"))["features"]
MD = json.load(open(f"{SP}/model_diff.json"))
NOMINAL = {"Llama-3.2-1B":1,"Llama-3.2-3B":3,"Llama-3.1-8B":8,"Qwen2.5-0.5B":0.5,"Qwen2.5-1.5B":1.5,"Qwen2.5-3B":3,"Qwen2.5-7B":7,"Qwen2.5-14B":14,
           "OLMo-2-1124-7B":7,"Mistral-7B-v0.1":7,"Mistral-Small-24B-Base-2501":24}
def regs(Z):
    return {"d50_70": Z[:, :, 5:7].mean((1, 2)), "back": Z[:, :, 5:].mean((1, 2)), "front": Z[:, :, :5].mean((1, 2)), "d70_100": Z[:, :, 7:].mean((1, 2))}
def pca_pc1(Z):
    mu = Z.mean((0, 1)); a = Z.mean(1) - mu
    U, s, Vt = np.linalg.svd(a - a.mean(0), full_matrices=False); sc = U[:, 0] * s[0]
    return sc, Vt[0]
FAMS = {"Llama": ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B"], "Mistral": ["Mistral-7B-v0.1", "Mistral-Small-24B-Base-2501"],
        "Qwen": ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-14B"]}
def pairs_stat(v, sign=-1, fams=FAMS):
    """count within-family size-ordered pairs where larger has (sign * value) larger: sign=-1 -> larger more negative"""
    c = n = 0
    for f, ms in fams.items():
        for a, b in itertools.combinations(ms, 2):   # ordered small -> large
            n += 1; c += int(sign * (v[b] - v[a]) > 0)
    return c, n
def pairs_perm(v, sign=-1, fams=FAMS):
    obs, n = pairs_stat(v, sign, fams); vals = []
    fl = list(fams.items())
    for perms in itertools.product(*[list(itertools.permutations(ms)) for _, ms in fl]):
        vv = {}
        for (f, ms), p in zip(fl, perms):
            for m, mp in zip(ms, p): vv[m] = v[mp]
        vals.append(pairs_stat(vv, sign, fams)[0])
    vals = np.array(vals); return obs, n, float((vals >= obs).mean())
def report(name, v, sign=-1, size=PARAMS):
    b = BASES; x = [np.log(size[m]) for m in b]; y = [v[m] for m in b]
    nq = [m for m in b if fam(m) != "Qwen"]; q = [m for m in b if fam(m) == "Qwen"]
    r_all = spear(x, y); r_nq = spear([np.log(size[m]) for m in nq], [v[m] for m in nq]); r_q = spear([np.log(size[m]) for m in q], [v[m] for m in q])
    obs, n, p = pairs_perm(v, sign); onq, nnq, pnq = pairs_perm(v, sign, {k: vv for k, vv in FAMS.items() if k != "Qwen"})
    print(f"{name:14s} rho bases {r_all[0]:+.2f} (p={r_all[1]:.3f}) nonQwen {r_nq[0]:+.2f} (p={r_nq[1]:.3f}) Qwen {r_q[0]:+.2f} (p={r_q[1]:.3f}) | pairs {obs}/{n} p={p:.4f}; nonQwen {onq}/{nnq} p={pnq:.3f}")
    return r_all, r_nq, r_q, (obs, n, p), (onq, nnq, pnq)
Z = Zdec("q10"); R = regs(Z); V = {k: dict(zip(M, R[k])) for k in R}
print("base values:"); [print(f"  {m:28s} 50-70 {V['d50_70'][m]:+.2f} back {V['back'][m]:+.2f} front {V['front'][m]:+.2f} 70-100 {V['d70_100'][m]:+.2f}") for m in BASES]
out = {}
for k in ["d50_70", "back", "d70_100"]: out[k] = report(k, V[k])
bmf = {m: V["back"][m] - V["front"][m] for m in M}; out["back-front"] = report("back-front", bmf)
eml = {m: V["d50_70"][m] - V["d70_100"][m] for m in M}; out["early-late"] = report("early-late", eml)
print("with nominal sizes:"); report("d50_70 nom", V["d50_70"], size=NOMINAL); report("back nom", V["back"], size=NOMINAL)
# excluding Qwen2.5-7B
b2 = [m for m in BASES if m != "Qwen2.5-7B"]; print("d50_70 excl Q7B rho:", spear([np.log(PARAMS[m]) for m in b2], [V["d50_70"][m] for m in b2]))
# PC1
sc, load = pca_pc1(Z); md_sc = np.array([MD["pca"]["scores"][m][0] for m in M])
if np.corrcoef(sc, md_sc)[0, 1] < 0: sc = -sc
print("own PC1 vs model_diff.json PC1 r =", round(np.corrcoef(sc, md_sc)[0, 1], 4), " loadings", np.round(load * np.sign(np.corrcoef(sc, md_sc)[0,1]), 2))
pc = dict(zip(M, sc)); out["pc1"] = report("PC1 (+)", pc, sign=+1)
print("  PC1 vs d50_70 across 17:", spear(sc, R["d50_70"]), " vs back:", spear(sc, R["back"]))
# collinearity among bases
acc = F2["acc"]; H = F2["H"]; b = BASES
lx = [np.log(PARAMS[m]) for m in b]
print("bases: size~acc", spear(lx, [acc[m] for m in b]), " size~H", spear(lx, [H[m] for m in b]))
for k in ["d50_70", "back"]:
    y = [V[k][m] for m in b]
    print(f"  partial {k}~logN | H: {partial_spear(lx, y, [H[m] for m in b])}   | acc: {partial_spear(lx, y, [acc[m] for m in b])}   {k}~H | logN: {partial_spear([H[m] for m in b], y, lx)}")
# other statistics
for st in ["q25", "q50", "entropy", "top1"]:
    Zs = Zdec(st); Rs = regs(Zs); v = dict(zip(M, Rs["d50_70"])); vb = dict(zip(M, Rs["back"]))
    s2, _ = pca_pc1(Zs); s2 = s2 if np.corrcoef(s2, sc)[0, 1] > 0 else -s2
    r = spear([np.log(PARAMS[m]) for m in b], [v[m] for m in b]); pr = pairs_stat(v)
    rbk = spear([np.log(PARAMS[m]) for m in b], [vb[m] for m in b]); prb = pairs_perm(vb)
    print(f"  {st}: 50-70 rho {r[0]:+.2f} pairs {pr[0]}/14 | back rho {rbk[0]:+.2f} pairs {prb[0]}/14 p={prb[2]:.3f} | PC1 pairs {pairs_stat(dict(zip(M, s2)), +1)[0]}/14")
