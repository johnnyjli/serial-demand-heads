"""Reviewer checks on the q10 decomposition (reads consist.json, model_diff.json, vet3/cells.json)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, itertools, numpy as np
SP = str(paths.ANALYSIS)
sys.path.insert(0, SP)
from model_diff import ORDER, T, LIN, FAM, BASES, decompose, ss, dec_prof, from_consist
rng = np.random.default_rng(7)
Z = from_consist("q10"); mu, a, b, e = decompose(Z); S = ss(Z); ns = S["model"] + S["task"] + S["resid"]
nM, nT, nD = Z.shape
print("== (a) df-adjusted shares")
dfm, dft, dfr = (nM - 1) * nD, (nT - 1) * nD, (nM - 1) * (nT - 1) * nD
MSm, MSt, MSr = S["model"] / dfm, S["task"] / dft, S["resid"] / dfr
print(f"MS model {MSm:.2f} task {MSt:.2f} resid {MSr:.2f}; F_model {MSm/MSr:.2f} F_task {MSt/MSr:.2f}")
# variance components (random-effects style, per decile cell)
s2m = (MSm - MSr) / nT; s2t = (MSt - MSr) / nM; s2r = MSr
tot = s2m + s2t + s2r
print(f"variance components: model {s2m/tot:.2f} task {s2t/tot:.2f} resid {s2r/tot:.2f}")
print(f"omega^2-style: model {(S['model']-dfm*MSr)/(ns+MSr):.3f} task {(S['task']-dft*MSr)/(ns+MSr):.3f}")
# null for task share: shuffle task labels within model
def task_ss(Zx): return ss(Zx)["task"]
nt = [task_ss(np.stack([Z[i, rng.permutation(nT)] for i in range(nM)])) for _ in range(2000)]
print(f"task share {S['task']/ns:.3f}; task-label-shuffle null mean share {np.mean(nt)/ns:.3f}")
# ICC of model: excess over permutation null
MD = json.load(open(f"{SP}/model_diff.json"))
print(f"model share {S['model']/ns:.3f}, null {MD['primary']['model_ss_null_mean']/ns:.3f}, excess {(S['model']-MD['primary']['model_ss_null_mean'])/ns:.3f}")
print("== (b) 11 base models only")
bi = [ORDER.index(m) for m in BASES]; Zb = Z[bi]
Sb = ss(Zb); nsb = Sb["model"] + Sb["task"] + Sb["resid"]
def perm_model(Zx, NB=3000):
    obs = ss(Zx)["model"]; null = np.array([ss(np.stack([Zx[rng.permutation(len(Zx)), j] for j in range(4)], 1))["model"] for _ in range(NB)])
    return obs, null.mean(), (np.sum(null >= obs) + 1) / (NB + 1)
o, nm, p = perm_model(Zb)
def split_half(Zx):
    rs = []
    for Sx in [(0, 1), (0, 2), (0, 3)]:
        Sc = tuple(j for j in range(4) if j not in Sx)
        A1 = Zx[:, Sx].mean(1); A1 = A1 - A1.mean(0); A2 = Zx[:, Sc].mean(1); A2 = A2 - A2.mean(0)
        rs.append(np.corrcoef(A1.ravel(), A2.ravel())[0, 1])
    return rs
print(f"bases: model share {Sb['model']/nsb:.3f} task {Sb['task']/nsb:.3f} resid {Sb['resid']/nsb:.3f}; null model share {nm/nsb:.3f}, p {p:.4f}; split-half per split {np.round(split_half(Zb),3)} mean {np.mean(split_half(Zb)):.3f}")
print("== (c) split-half per split (all 17):", np.round(split_half(Z), 3), " (0,1)=pointer+arith vs binding+relational")
# task-by-task agreement of model deviations d_{m,t} = Z_mt - mean over models of Z_.t
D = Z - Z.mean(0, keepdims=True)
for j, k in itertools.combinations(range(4), 2):
    print(f"   r(dev {T[j]}, dev {T[k]}) = {np.corrcoef(D[:, j].ravel(), D[:, k].ravel())[0,1]:+.3f}")
print("   Spearman-Brown of split-half 0.36 to 4 tasks:", round(2 * .36 / 1.36, 3))
print("== (d) wording: signatures from w0 only vs w1 only (vet3/cells.json eager 'all')")
R = json.load(open(f"{SP}/vet3/cells.json")); Dd = {}
for r in R:
    if r["variant"] == "all" and r["fam"] in T: Dd[(r["model"], r["fam"], r["w"], r["attn"])] = np.array(r["z"], float)
def Zw(w):
    return np.array([[dec_prof(Dd[(m, t, w, "sdpa" if (m == "Qwen2.5-7B-Instruct" and (m, t, w, "sdpa") in Dd) else "eager")]) for t in T] for m in ORDER])
Z0, Z1 = Zw("w0"), Zw("w1"); a0, a1 = decompose(Z0)[1], decompose(Z1)[1]
print(f"r(a_w0, a_w1) pooled {np.corrcoef(a0.ravel(), a1.ravel())[0,1]:.3f}; per-model mean {np.mean([np.corrcoef(a0[i], a1[i])[0,1] for i in range(17)]):.3f}")
print("   per-model:", {m: round(float(np.corrcoef(a0[i], a1[i])[0, 1]), 2) for i, m in enumerate(ORDER)})
print(f"   r(Z_w0, Z_w1) per cell mean {np.mean([np.corrcoef(Z0[i,j], Z1[i,j])[0,1] for i in range(17) for j in range(4)]):.3f}")
for j, t in enumerate(T):
    print(f"   {t}: per-cell r(w0,w1) mean {np.mean([np.corrcoef(Z0[i,j], Z1[i,j])[0,1] for i in range(17)]):.3f}")
# does a model's w0-signature match its own w1-signature better than other models' (identification)?
Rw = np.corrcoef(np.vstack([a0, a1]))[:17, 17:]
ident = np.mean([np.argmax(Rw[i]) == i for i in range(17)])
print(f"   identification w0->w1 top-1 accuracy {ident:.2f}")
print("== (e) effect-size scale: signatures from 'rel' (100*slope/mean) instead of z")
C = {r["model"]: r for r in json.load(open(f"{SP}/consist.json")) if r["stat"] == "q10"}
Zr = np.array([[dec_prof(np.array(C[m]["families"][t]["rel"])) for t in T] for m in ORDER])
mur, ar, br, er = decompose(Zr); Sr = ss(Zr); nsr = Sr["model"] + Sr["task"] + Sr["resid"]
o, nm, p = perm_model(Zr)
print(f"rel: shared frac {Sr['shared']/sum(Sr.values()):.3f}; model {Sr['model']/nsr:.3f} (null {nm/nsr:.3f}, p {p:.4f}) task {Sr['task']/nsr:.3f}; split-half {np.mean(split_half(Zr)):.3f}; r(a_rel, a_z) {np.corrcoef(ar.ravel(), a.ravel())[0,1]:.3f}")
Rr = np.corrcoef(ar)
lin = np.mean([Rr[i, j] for i, j in itertools.combinations(range(17), 2) if LIN[ORDER[i]] == LIN[ORDER[j]]])
oth = np.mean([Rr[i, j] for i, j in itertools.combinations(range(17), 2) if LIN[ORDER[i]] != LIN[ORDER[j]]])
print(f"   rel lineage r {lin:.3f} other {oth:.3f}")
# null sd of z vs H: does |z| scale with H? per-model mean |z| and mean |rel|
H = {m: None for m in ORDER}
print("== (f) lineage similarity weighting")
Ra = np.array(MD["similarity"]["R"])
pairs = {}
for i, j in itertools.combinations(range(17), 2):
    if LIN[ORDER[i]] == LIN[ORDER[j]]: pairs.setdefault(LIN[ORDER[i]], []).append(Ra[i, j])
per_lin = {k: float(np.mean(v)) for k, v in pairs.items()}
print("per-lineage mean r:", {k: round(v, 3) for k, v in per_lin.items()}, " lineage-weighted mean", round(np.mean(list(per_lin.values())), 3))
base_post = [Ra[ORDER.index(LIN[m]), ORDER.index(m)] for m in ORDER if LIN[m] != m]
print("base-post pairs only (6):", np.round(base_post, 3), "mean", round(np.mean(base_post), 3))
nonolmo = [v for k, vs in pairs.items() if "OLMo" not in k for v in vs]
print("non-OLMo lineage pairs:", np.round(nonolmo, 3), "mean", round(np.mean(nonolmo), 3))
print("== (g) rank of base among 16 others for each post-trained model")
for m in ORDER:
    if LIN[m] == m: continue
    i = ORDER.index(m); others = [(Ra[i, j], ORDER[j]) for j in range(17) if j != i]; others.sort(reverse=True)
    rank = [o[1] for o in others].index(LIN[m]) + 1
    print(f"   {m:26s} r(base) {Ra[i, ORDER.index(LIN[m])]:+.2f} rank {rank}; top3 {[(n, round(v,2)) for v, n in others[:3]]}")
print("   excluding same-lineage siblings, best non-lineage match per post-trained model:")
for m in ORDER:
    if LIN[m] == m: continue
    i = ORDER.index(m); others = sorted([(Ra[i, j], ORDER[j]) for j in range(17) if LIN[ORDER[j]] != LIN[m]], reverse=True)
    print(f"   {m:26s} best other {others[0][1]} {others[0][0]:+.2f}")
print("== (h) centering-induced expected pairwise r among signatures: -1/(n-1) =", round(-1 / 16, 3))
# random-vector check
null_r = []
for _ in range(2000):
    X = rng.standard_normal((17, 10)); X = X - X.mean(0); null_r.append(np.mean([np.corrcoef(X[i], X[j])[0, 1] for i, j in itertools.combinations(range(17), 2)]))
print("   mean pairwise r of centered iid signatures:", round(np.mean(null_r), 3))
print("== (i) Qwen2.5-14B split-half and Qwen7B eager")
print("   per-model split-half:", {m: round(v, 2) for m, v in MD["primary"]["split_half_per_model"].items()})
