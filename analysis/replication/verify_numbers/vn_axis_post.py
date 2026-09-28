"""Verify: paper PC1 projection of E2b signatures; correlates; post-training changes (layer- and decile-based,
relative slope, same-item window slope), OLMo without exclusion, k=1 accuracy of the pairs."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, glob, os, itertools, numpy as np
from scipy.stats import spearmanr
from vn_lib import *
OUT = {}
ORDER17 = list(CQ.keys())
Z17 = Zprof_e2(ORDER17); a17 = Z17.mean(1) - Z17.mean((0, 1))
U, s, Vt = np.linalg.svd(a17 - a17.mean(0), full_matrices=False); v1 = Vt[0]; sc = (U * s)[:, 0]
sig5070 = a17[:, 5:7].mean(1)
if r_(sc, sig5070) > 0: v1 = -v1; sc = -sc
print("paper PC1 explained %.3f; r(PC1, 50-70 sig) %.3f; Q3B %.2f OLMo %.2f" % (s[0] ** 2 / (s ** 2).sum(), r_(sc, sig5070), sc[ORDER17.index("Qwen2.5-3B")], sc[ORDER17.index("OLMo-2-1124-7B")]))
paper_sc = {m: sc[i] for i, m in enumerate(ORDER17)}

# E2 accuracy (strict, both wordings, Q7I SDPA) and E2b strict accuracy
E2_ROOT = str(paths.E2_SWEEP)
def e2_file(m, f, w, attn):
    fs = [p for p in glob.glob(f"{E2_ROOT}/bf16_{attn}/*__{f}_{w}_*.npz") if os.path.basename(p).split("__")[0].split("_", 1)[1] == m]
    assert len(fs) == 1, (m, f, w, attn, fs); return fs[0]
def e2_acc(m, attn=None):
    attn = attn or ("sdpa" if m == "Qwen2.5-7B-Instruct" else "eager")
    return float(np.mean([np.load(e2_file(m, f, w, attn))["correct_strict"].mean() for f in T for w in ("w0", "w1")]))
def e2b_acc(m, kind="strict", k1=False):
    v = []
    for t in T:
        its = CACHE[(m, "direct", t)]["items"]
        if k1: its = [x for x in its if x["k"] == 1]
        v.append(np.mean([(x["correct"] and x["correct_constrained"]) if kind == "strict" else x["correct"] for x in its]))
    return float(np.mean(v))
H = {m: CACHE[(m, "direct", "pointer")]["H"] for m in M8}

for tag, models in (("M8", M8), ("M6", M6)):
    Zb = Zprof(models); Za = Zprof_e2(models)
    aB = Zb.mean(1) - Zb.mean((0, 1)); aA = Za.mean(1) - Za.mean((0, 1))
    pB = aB @ v1; paper = np.array([paper_sc[m] for m in models])
    o = {"proj": dict(zip(models, np.round(pB, 3).tolist())), "paper": dict(zip(models, np.round(paper, 3).tolist())),
         "pearson": exact_perm_corr(pB, paper), "spearman": exact_perm_corr(pB, paper, spearman=True),
         "z5070": exact_perm_corr(Za[:, :, 5:7].mean((1, 2)), Zb[:, :, 5:7].mean((1, 2))),
         "post_further": {f"{b}->{p}": bool(pB[models.index(p)] > pB[models.index(b)]) for b, p in PAIRS if b in models},
         "post_further_E2": {f"{b}->{p}": bool(paper_sc[p] > paper_sc[b]) for b, p in PAIRS if b in models}}
    if "Qwen2.5-7B" in models: o["Q14_beyond_Q7"] = bool(pB[models.index("Qwen2.5-14B")] > pB[models.index("Qwen2.5-7B")])
    accb = [e2b_acc(m) for m in models]; acca = [e2_acc(m) for m in models]; hh = [H[m] for m in models]
    o["rho_proj_H"] = exact_perm_corr(pB, hh, spearman=True); o["rho_proj_acc"] = exact_perm_corr(pB, accb, spearman=True)
    o["rho_paperPC1_E2acc"] = exact_perm_corr(paper, acca, spearman=True)
    o["E2_acc"] = dict(zip(models, np.round(acca, 3).tolist())); o["E2b_strict_acc"] = dict(zip(models, np.round(accb, 3).tolist()))
    # PCA on these models alone
    for nm, a in (("E2b", aB), ("E2", aA)):
        U_, s_, Vt_ = np.linalg.svd(a - a.mean(0), full_matrices=False); sc_ = (U_ * s_)[:, 0]
        o[f"pca8_{nm}_pc1_r_sig5070"] = r_(sc_, a[:, 5:7].mean(1))
    OUT[tag] = o
    print("\n==", tag); print(json.dumps(o, default=lambda x: round(float(x), 4)))

# post-training
def back_layer(z, L): return z[rel_depth(L) >= .5].mean()
post = {}
for b, p in PAIRS:
    L = CACHE[(b, "direct", "pointer")]["L"]
    e2b_l = np.mean([back_layer(zlayers(p, "direct", t), L) for t in T]) - np.mean([back_layer(zlayers(b, "direct", t), L) for t in T])
    e2_l = np.mean([back_layer(e2_z(p, t), L) for t in T]) - np.mean([back_layer(e2_z(b, t), L) for t in T])
    e2b_d = np.mean([dec_prof(zlayers(p, "direct", t))[5:].mean() for t in T]) - np.mean([dec_prof(zlayers(b, "direct", t))[5:].mean() for t in T])
    e2_d = np.mean([dec_prof(e2_z(p, t))[5:].mean() for t in T]) - np.mean([dec_prof(e2_z(b, t))[5:].mean() for t in T])
    d = {"E2b_layer": e2b_l, "E2_layer": e2_l, "E2b_dec": e2b_d, "E2_dec": e2_d}
    if b == "Qwen2.5-7B":
        zq = {t: 0.5 * (e2_cells_z(p, t, "w0", "eager") + e2_cells_z(p, t, "w1", "eager")) for t in T}
        d["E2_layer_both_eager"] = np.mean([back_layer(zq[t], L) for t in T]) - np.mean([back_layer(e2_z(b, t), L) for t in T])
        zq0 = {t: 0.5 * (e2_cells_z(b, t, "w0", "eager") + e2_cells_z(b, t, "w1", "eager")) for t in T}
        d["E2_layer_both_eager_cells"] = np.mean([back_layer(zq[t], L) for t in T]) - np.mean([back_layer(zq0[t], L) for t in T])
    if b.startswith("OLMo"):
        d["E2b_layer_noexcl"] = np.mean([back_layer(zlayers(p, "direct", t, excl=False), L) for t in T]) - np.mean([back_layer(zlayers(b, "direct", t, excl=False), L) for t in T])
        d["E2b_dec_noexcl"] = np.mean([dec_prof(zlayers(p, "direct", t, excl=False))[5:].mean() for t in T]) - np.mean([dec_prof(zlayers(b, "direct", t, excl=False))[5:].mean() for t in T])
    # relative slope, back-half deciles
    def relb(m, src):
        v = []
        for t in T:
            if src == "E2b":
                Y, k, _, _ = cell(m, "direct", t); rel = 100 * zprof(Y, k)[0] / np.abs(Y.mean(0))
            else: rel = e2_z(m, t, "rel")
            v.append(dec_prof(rel)[5:].mean())
        return np.mean(v)
    d["rel_E2b"] = relb(p, "E2b") - relb(b, "E2b"); d["rel_E2"] = relb(p, "E2") - relb(b, "E2")
    # same-item window slope
    w = slice(int(.4 * (L - 1)), int(.8 * (L - 1)) + 1)
    rng = np.random.default_rng(11); sl = []; nulls = []
    for t in T:
        Yb, kb, ib, _ = cell(b, "direct", t); Yp, kp, ip, _ = cell(p, "direct", t)
        idb = {x["id"]: j for j, x in enumerate(ib)}; idp = {x["id"]: j for j, x in enumerate(ip)}
        com = sorted(set(idb) & set(idp)); jb = [idb[c] for c in com]; jp = [idp[c] for c in com]
        dl = 100 * (Yp[jp][:, w].mean(1) - Yb[jb][:, w].mean(1)); k = kb[jb]; kc = k - k.mean(); dc = dl - dl.mean()
        sl.append(kc @ dc / (kc @ kc)); nulls.append(np.array([rng.permutation(kc) @ dc for _ in range(20000)]) / (kc @ kc))
    obs = np.mean(sl); nm = np.mean(nulls, 0)
    d["same_item_slope"] = obs; d["same_item_p"] = float((np.sum(np.abs(nm) >= abs(obs)) + 1) / (len(nm) + 1)); d["same_item_per_task"] = dict(zip(T, np.round(sl, 4).tolist()))
    d["k1_strict"] = [e2b_acc(b, "strict", True), e2b_acc(p, "strict", True)]; d["k1_written"] = [e2b_acc(b, "written", True), e2b_acc(p, "written", True)]
    post[f"{b}->{p}"] = d
    print(f"\n{b}->{p}", json.dumps(d, default=lambda x: round(float(x), 4)))
OUT["post"] = post
# E2 OLMo checkpoints per paper
for c in ["OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct"]:
    print(c, "E2 layer-based change %.3f" % (np.mean([back_layer(e2_z(c, t), 32) for t in T]) - np.mean([back_layer(e2_z("OLMo-2-1124-7B", t), 32) for t in T])))
json.dump(OUT, open(f"{HERE}/vn_axis_post.json", "w"), indent=1, default=float)
