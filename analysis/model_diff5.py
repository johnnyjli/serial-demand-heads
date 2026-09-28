"""Checks raised in review of the model-signature analysis (top-10% share).
(a) split-half by split and model-deviation agreement for each pair of tasks; task-share chance level;
(b) item-split reliability: disjoint halves of instances/bodies, per-wording exact permutation z (vet3 cache), wording-averaged;
    reliability of profiles, signatures a and residual e;
(c) lineage detail: per-lineage similarity, base-vs-post pairs, lineage-weighted mean, centring baseline;
(d) heads per layer: Spearman links across 17 models and across the 11 base models; signatures among H=32 models;
(e) lineage-level bootstrap of the shared decile profile; (f) w0-only vs w1-only signatures;
(g) extra robustness rows: 5 bins, 20-point interpolation, base models only, no OLMo-2 exclusion."""
import sys, os, json, itertools, numpy as np
from scipy.stats import spearmanr
SP = sys.argv[1]; sys.path.insert(0, SP)
os.environ["VPRIM"] = "y10"; sys.path.insert(0, f"{SP}/vet3")
from v2lib import load as vload, zprof as vz
from model_diff import ORDER, T, LIN, decompose, ss, perm_model, split_half, from_consist, dec_prof
rng = np.random.default_rng(7); out = {}
Z = from_consist("q10"); mu, a, b, e = decompose(Z)
# (a)
spl = {}
for S in [(0, 1), (0, 2), (0, 3)]:
    Sc = tuple(j for j in range(4) if j not in S)
    A1 = Z[:, S].mean(1); A1 -= A1.mean(0); A2 = Z[:, Sc].mean(1); A2 -= A2.mean(0)
    spl["+".join(T[j] for j in S) + " vs " + "+".join(T[j] for j in Sc)] = float(np.corrcoef(A1.ravel(), A2.ravel())[0, 1])
dev = Z - Z.mean(0, keepdims=True)                       # model deviation within each task
pair = {f"{T[i]}-{T[j]}": float(np.corrcoef(dev[:, i].ravel(), dev[:, j].ravel())[0, 1]) for i, j in itertools.combinations(range(4), 2)}
s = ss(Z); ns = s["model"] + s["task"] + s["resid"]
tnull = []
for _ in range(2000):
    Zp = np.stack([Z[m][rng.permutation(4)] for m in range(17)]); tnull.append(ss(Zp)["task"] / ns)
nM, nT, nD = 17, 4, 10
dfm, dft, dfe = (nM - 1) * nD, (nT - 1) * nD, (nM - 1) * (nT - 1) * nD
msm, mst, mse = s["model"] / dfm, s["task"] / dft, s["resid"] / dfe
vc_m, vc_t = max(0, (msm - mse) / nT), max(0, (mst - mse) / nM)
out["a"] = {"splits": spl, "task_pairs": pair, "task_share": s["task"] / ns, "task_share_chance": float(np.mean(tnull)),
            "var_comp_model": vc_m / (vc_m + vc_t + mse), "var_comp_task": vc_t / (vc_m + vc_t + mse), "var_comp_resid": mse / (vc_m + vc_t + mse)}
print("(a) splits", {k: round(v, 2) for k, v in spl.items()}); print("    task pairs", {k: round(v, 2) for k, v in pair.items()})
print(f"    task share {s['task']/ns:.3f} chance {np.mean(tnull):.3f}; variance components model {out['a']['var_comp_model']:.2f} task {out['a']['var_comp_task']:.2f} resid {out['a']['var_comp_resid']:.2f}")
# (b) item-split reliability
def prof_half(m, t, half, seed):
    zs = []
    for w in ("w0", "w1"):
        d = vload(m, t, w, "sdpa" if m == "Qwen2.5-7B-Instruct" else "eager", True)
        ids = d["body"]; u = np.unique(ids); r = np.random.default_rng(seed + hash(t) % 1000); pick = set(r.permutation(u)[: len(u) // 2])
        sel = np.array([(i in pick) == (half == 0) for i in ids])
        zs.append(vz(d["y25"][sel], d["k"][sel], d["body"][sel], d["within"])[2])
    return dec_prof(0.5 * (zs[0] + zs[1]))
rel = {"profile": [], "a": [], "e": [], "per_task": {t: [] for t in T}}
for seed in range(4):
    Z1 = np.array([[prof_half(m, t, 0, seed) for t in T] for m in ORDER]); Z2 = np.array([[prof_half(m, t, 1, seed) for t in T] for m in ORDER])
    d1, d2 = decompose(Z1), decompose(Z2)
    rel["profile"].append(np.mean([np.corrcoef(Z1[i, j], Z2[i, j])[0, 1] for i in range(17) for j in range(4)]))
    for j, t in enumerate(T): rel["per_task"][t].append(np.mean([np.corrcoef(Z1[i, j], Z2[i, j])[0, 1] for i in range(17)]))
    rel["a"].append(np.corrcoef(d1[1].ravel(), d2[1].ravel())[0, 1]); rel["e"].append(np.corrcoef(d1[3].ravel(), d2[3].ravel())[0, 1])
out["b"] = {"profile_r": float(np.mean(rel["profile"])), "signature_r": float(np.mean(rel["a"])), "resid_r": float(np.mean(rel["e"])),
            "per_task": {t: float(np.mean(v)) for t, v in rel["per_task"].items()}}
print("(b) item-split reliability", {k: (round(v, 2) if isinstance(v, float) else {kk: round(vv, 2) for kk, vv in v.items()}) for k, v in out["b"].items()})
# (c) lineage
R = np.corrcoef(a); idx = {m: i for i, m in enumerate(ORDER)}
lins = sorted({LIN[m] for m in ORDER if sum(LIN[x] == LIN[m] for x in ORDER) > 1})
per = {}
for L_ in lins:
    ms = [m for m in ORDER if LIN[m] == L_]; per[L_] = float(np.mean([R[idx[x], idx[y]] for x, y in itertools.combinations(ms, 2)]))
bp = {m: float(R[idx[m], idx[LIN[m]]]) for m in ORDER if LIN[m] != m}
closest = {m: ORDER[int(np.argsort(-R[idx[m]])[1])] for m in bp}
out["c"] = {"per_lineage": per, "lineage_weighted": float(np.mean(list(per.values()))), "base_post": bp, "base_post_mean": float(np.mean(list(bp.values()))),
            "centring_baseline": -1 / 16, "closest_to_post": closest}
print("(c) per lineage", {k: round(v, 2) for k, v in per.items()}, "weighted", round(out["c"]["lineage_weighted"], 2), "| base-post", {k: round(v, 2) for k, v in bp.items()}, "mean", round(out["c"]["base_post_mean"], 2))
print("    closest signature to each post-trained model:", closest)
# (d) heads per layer
MD2 = json.load(open(f"{SP}/model_diff2.json"))["features"]; MD = json.load(open(f"{SP}/model_diff.json"))
back = np.array([MD["per_model"][m]["back"] for m in ORDER]); pc1 = np.array([MD["pca"]["scores"][m][0] for m in ORDER])
bases = [i for i, m in enumerate(ORDER) if LIN[m] == m]
dd = {}
for f in ["acc", "acc_k1", "ent_slope", "H"]:
    v = np.array([MD2[f][m] for m in ORDER])
    dd[f] = {"all_back": spearmanr(v, back)[:2], "all_pc1": spearmanr(v, pc1)[:2], "bases_back": spearmanr(v[bases], back[bases])[:2], "bases_pc1": spearmanr(v[bases], pc1[bases])[:2]}
h32 = [i for i, m in enumerate(ORDER) if MD2["H"][m] == 32]; h32b = [i for i in h32 if i in bases]
def sub(ix):
    Zs = Z[ix]; o, nm, p = perm_model(Zs, 2000); sv = ss(Zs); n_ = sv["model"] + sv["task"] + sv["resid"]
    return {"n": len(ix), "model_share": sv["model"] / n_, "chance": nm / n_, "p": p, "split_half": split_half(Zs)[0]}
out["d"] = {"links": {k: {kk: [float(vv[0]), float(vv[1])] for kk, vv in v.items()} for k, v in dd.items()}, "H32": sub(h32), "H32_bases": sub(h32b), "bases": sub(bases)}
for k, v in dd.items(): print(f"(d) {k:10s} " + " ".join(f"{kk} {vv[0]:+.2f} (p={vv[1]:.3f})" for kk, vv in v.items()))
print("    H=32 models", {k: round(v, 3) for k, v in out["d"]["H32"].items()}, "\n    H=32 bases", {k: round(v, 3) for k, v in out["d"]["H32_bases"].items()}, "\n    11 bases", {k: round(v, 3) for k, v in out["d"]["bases"].items()})
# (e) lineage-level bootstrap of shared deciles
Lm = sorted(set(LIN.values())); ZL = np.array([Z[[idx[m] for m in ORDER if LIN[m] == L_]].mean((0, 1)) for L_ in Lm])
bt = np.array([ZL[rng.integers(0, len(Lm), len(Lm))].mean(0) for _ in range(5000)]); lo, hi = np.quantile(bt, [.025, .975], 0)
out["e"] = {"mean": ZL.mean(0).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
print("(e) lineage-level deciles", " ".join(f"{m:+.2f}[{l:+.2f},{h:+.2f}]" for m, l, h in zip(ZL.mean(0), lo, hi)))
# lineage-level back<front and back<0
fz = np.array([Z[[idx[m] for m in ORDER if LIN[m] == L_]][:, :, :5].mean() for L_ in Lm]); bz = np.array([Z[[idx[m] for m in ORDER if LIN[m] == L_]][:, :, 5:].mean() for L_ in Lm])
out["e"]["lineages_back_lt_front"] = int((bz < fz).sum()); out["e"]["lineages_back_lt0"] = int((bz < 0).sum())
print("    lineages: back<front", int((bz < fz).sum()), "back<0", int((bz < 0).sum()), "of", len(Lm))
# (f) wording
C = json.load(open(f"{SP}/vet3/cells.json")); D = {(r["model"], r["fam"], r["w"]): np.array(r["z"]) for r in C if r["variant"] == "all" and r["fam"] in T and
                                                   r["attn"] == ("sdpa" if r["model"] == "Qwen2.5-7B-Instruct" else "eager")}
Zw = {w: np.array([[dec_prof(D[(m, t, w)]) for t in T] for m in ORDER]) for w in ("w0", "w1")}
a0, a1 = decompose(Zw["w0"])[1], decompose(Zw["w1"])[1]
out["f"] = {"sig_w0_w1": float(np.corrcoef(a0.ravel(), a1.ravel())[0, 1]), "per_model": {m: float(np.corrcoef(a0[i], a1[i])[0, 1]) for i, m in enumerate(ORDER)},
            "profile_w0_w1": float(np.mean([np.corrcoef(Zw["w0"][i, j], Zw["w1"][i, j])[0, 1] for i in range(17) for j in range(4)]))}
print(f"(f) signatures w0 vs w1 r {out['f']['sig_w0_w1']:.2f}; per-cell profile w0 vs w1 {out['f']['profile_w0_w1']:.2f}; per model range {min(out['f']['per_model'].values()):.2f}-{max(out['f']['per_model'].values()):.2f}")
# (g) extra rows
def row(Zv, ref=a):
    mu_, av, bv, ev = decompose(Zv); sv = ss(Zv); n_ = sv["model"] + sv["task"] + sv["resid"]; o, nm, p = perm_model(Zv, 2000)
    Rv = np.corrcoef(av); li = np.mean([Rv[i, j] for i, j in itertools.combinations(range(len(Zv)), 2) if LIN[ORDER[i]] == LIN[ORDER[j]]]) if len(Zv) == 17 else float("nan")
    ot = np.mean([Rv[i, j] for i, j in itertools.combinations(range(len(Zv)), 2) if len(Zv) != 17 or LIN[ORDER[i]] != LIN[ORDER[j]]])
    return {"model_share": sv["model"] / n_, "chance": nm / n_, "p": p, "split_half": split_half(Zv)[0], "r_primary": float(np.corrcoef(av.ravel(), ref.ravel())[0, 1]) if ref is not None else None,
            "lineage_r": float(li), "other_r": float(ot)}
Cq = {r["model"]: r for r in json.load(open(f"{SP}/consist.json")) if r["stat"] == "q10"}
def prof_bins(z, nb):
    z = np.asarray(z); L = len(z); x = np.arange(L) / (L - 1); d = np.minimum((x * nb).astype(int), nb - 1); return np.array([z[d == i].mean() for i in range(nb)])
def prof_interp(z, n):
    z = np.asarray(z); L = len(z); return np.interp(np.linspace(0, 1, n), np.arange(L) / (L - 1), z)
g = {}
g["5 depth bins"] = row(np.array([[prof_bins(Cq[m]["families"][t]["z"], 5) for t in T] for m in ORDER]), ref=None)
g["20-point interpolation"] = row(np.array([[prof_interp(Cq[m]["families"][t]["z"], 20) for t in T] for m in ORDER]), ref=None)
Zb = Z[bases]; mb = decompose(Zb); sv = ss(Zb); n_ = sv["model"] + sv["task"] + sv["resid"]; o, nm, p = perm_model(Zb, 2000)
g["base models only (11)"] = {"model_share": sv["model"] / n_, "chance": nm / n_, "p": p, "split_half": split_half(Zb)[0]}
Dn = {(r["model"], r["fam"], r["w"]): np.array(r["z"]) for r in C if r["variant"] == "noclean" and r["fam"] in T}
Zn = np.array([[dec_prof(0.5 * (Dn[(m, t, "w0")] + Dn[(m, t, "w1")])) if (m, t, "w0") in Dn else dec_prof(0.5 * (D[(m, t, "w0")] + D[(m, t, "w1")])) for t in T] for m in ORDER])
Zp = np.array([[dec_prof(0.5 * (D[(m, t, "w0")] + D[(m, t, "w1")])) for t in T] for m in ORDER]); ap = decompose(Zp)[1]
g["no OLMo-2 exclusion"] = row(Zn, ref=ap)
out["g"] = g
for k, v in g.items(): print("(g)", k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
json.dump(out, open(f"{SP}/model_diff5.json", "w"), indent=1, default=float)
