"""Model differences in how head activity is redistributed under serial demand, and whether each model's
pattern is consistent across tasks.

Profile: per model m x task t, the per-layer standardised slope z of the concentration statistic on k
(+ = concentrating, - = spreading), averaged within relative-depth deciles d -> Z[m, t, d] (17 x 4 x 10).
Two-way decomposition  Z = mu_d + a_{m,d} + b_{t,d} + e:  mu = profile shared by all models and tasks,
a = model signature (a model's deviation, common to its four tasks), b = task effect, e = model x task residual.
Tests: (1) permutation of model labels within each task (breaks model identity across tasks) for the size
of the model signature; (2) split-half across tasks: signature from two tasks vs the other two;
(3) lineage / family similarity of signatures; (4) PCA of signatures; (5) robustness across statistics,
sink removal, attention kernel.  Primary: consist.json stat q10 (wording-averaged items; Qwen2.5-7B-Instruct SDPA)."""
import sys, json, itertools, numpy as np
SP = sys.argv[1] if len(sys.argv) > 1 else '.'
T = ["pointer", "arith", "binding", "relational"]
ORDER = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.2-3B-Instruct", "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
         "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
         "OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct",
         "Mistral-7B-v0.1", "Mistral-Small-24B-Base-2501"]
FAM = {m: m.split("-")[0].replace("Qwen2.5", "Qwen").replace("OLMo", "OLMo") for m in ORDER}
LIN = {"Llama-3.2-3B-Instruct": "Llama-3.2-3B", "Llama-3.1-8B-Instruct": "Llama-3.1-8B", "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
       "OLMo-2-1124-7B-SFT": "OLMo-2-1124-7B", "OLMo-2-1124-7B-DPO": "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct": "OLMo-2-1124-7B"}
LIN = {m: LIN.get(m, m) for m in ORDER}
BASES = [m for m in ORDER if LIN[m] == m]
ND = 10; rng = np.random.default_rng(0)
def dec_prof(z):
    z = np.asarray(z, float); L = len(z); x = np.arange(L) / (L - 1); d = np.minimum((x * ND).astype(int), ND - 1)
    return np.array([np.nanmean(z[d == i]) for i in range(ND)])
def from_consist(stat):
    C = {r["model"]: r for r in json.load(open(f"{SP}/consist.json")) if r["stat"] == stat}
    sign = 1.0
    return np.array([[dec_prof(sign * np.array(C[m]["families"][t]["z"])) for t in T] for m in ORDER])
def from_cells(attn, variant, models=ORDER):
    R = json.load(open(f"{SP}/vet3/cells.json")); D = {}
    for r in R:
        if r["attn"] == attn and r["variant"] == variant and r["fam"] in T: D[(r["model"], r["fam"], r["w"])] = np.array(r["z"], float)
    out = []
    for m in models:
        if not all((m, t, w) in D for t in T for w in ("w0", "w1")): return None, m
        out.append([dec_prof(0.5 * (D[(m, t, "w0")] + D[(m, t, "w1")])) for t in T])
    return np.array(out), None
def decompose(Z):
    mu = Z.mean((0, 1)); a = Z.mean(1) - mu; b = Z.mean(0) - mu; e = Z - mu - a[:, None] - b[None]
    return mu, a, b, e
def ss(Z):
    mu, a, b, e = decompose(Z); nM, nT = Z.shape[:2]
    return {"shared": nM * nT * (mu ** 2).sum(), "model": nT * (a ** 2).sum(), "task": nM * (b ** 2).sum(), "resid": (e ** 2).sum()}
def perm_model(Z, NB=5000):
    obs = ss(Z)["model"]; nM = Z.shape[0]; null = np.empty(NB)
    for i in range(NB):
        Zp = np.stack([Z[rng.permutation(nM), j] for j in range(Z.shape[1])], 1); null[i] = ss(Zp)["model"]
    return obs, float(null.mean()), float((np.sum(null >= obs) + 1) / (NB + 1))
def split_half(Z):
    """model signature from tasks S vs the complement: pooled r over (model, decile) and per-model r"""
    res = []
    for S in [(0, 1), (0, 2), (0, 3)]:
        Sc = tuple(j for j in range(4) if j not in S)
        A1 = Z[:, S].mean(1); A1 = A1 - A1.mean(0); A2 = Z[:, Sc].mean(1); A2 = A2 - A2.mean(0)
        res.append((np.corrcoef(A1.ravel(), A2.ravel())[0, 1], [np.corrcoef(A1[i], A2[i])[0, 1] for i in range(len(A1))]))
    pooled = float(np.mean([r[0] for r in res])); per_model = np.nanmean([r[1] for r in res], 0)
    nullr = []
    for _ in range(2000):
        Zp = np.stack([Z[rng.permutation(Z.shape[0]), j] for j in range(4)], 1)
        nullr.append(np.mean([np.corrcoef((Zp[:, S].mean(1) - Zp[:, S].mean((0, 1))).ravel(),
                                          (Zp[:, [j for j in range(4) if j not in S]].mean(1) - Zp[:, [j for j in range(4) if j not in S]].mean((0, 1))).ravel())[0, 1]
                              for S in [(0, 1), (0, 2), (0, 3)]]))
    return pooled, float(np.mean(nullr)), float((np.sum(np.array(nullr) >= pooled) + 1) / (len(nullr) + 1)), per_model.tolist()
def task_consistency(Z):
    """per model: mean pairwise correlation of its four task profiles (full profile, incl. the shared part)
    vs. the same statistic for profiles of different models in different tasks"""
    within = [np.mean([np.corrcoef(Z[i, a], Z[i, b])[0, 1] for a, b in itertools.combinations(range(4), 2)]) for i in range(len(Z))]
    between = [np.corrcoef(Z[i, a], Z[j, b])[0, 1] for i in range(len(Z)) for j in range(len(Z)) if i != j for a in range(4) for b in range(4) if a != b]
    same_task_between = [np.corrcoef(Z[i, a], Z[j, a])[0, 1] for i in range(len(Z)) for j in range(i + 1, len(Z)) for a in range(4)]
    return within, float(np.mean(between)), float(np.mean(same_task_between))
if __name__ == '__main__':
    out = {}
    Z = from_consist("q10"); mu, a, b, e = decompose(Z); S = ss(Z); tot = sum(S.values())
    obs, nullm, p = perm_model(Z)
    sh = split_half(Z); within, between, stb = task_consistency(Z)
    out["primary"] = {"ss": {k: float(v) for k, v in S.items()}, "ss_frac": {k: float(v / tot) for k, v in S.items()},
                      "ss_frac_nonshared": {k: float(S[k] / (tot - S["shared"])) for k in ("model", "task", "resid")},
                      "model_ss_null_mean": nullm, "p_model": p, "split_half_r": sh[0], "split_half_null": sh[1], "p_split_half": sh[2],
                      "split_half_per_model": dict(zip(ORDER, sh[3])), "within_model_cross_task_r": dict(zip(ORDER, map(float, within))),
                      "between_model_cross_task_r": between, "between_model_same_task_r": stb,
                      "mu": mu.tolist(), "a": {m: a[i].tolist() for i, m in enumerate(ORDER)}, "b": {t: b[j].tolist() for j, t in enumerate(T)},
                      "Z": {m: {t: Z[i, j].tolist() for j, t in enumerate(T)} for i, m in enumerate(ORDER)}}
    print("variance of Z[m,t,d]: " + ", ".join(f"{k} {v / tot:.2f}" for k, v in S.items()) + f"  | non-shared: model {S['model']/(tot-S['shared']):.2f} task {S['task']/(tot-S['shared']):.2f} resid {S['resid']/(tot-S['shared']):.2f}")
    print(f"model-signature SS {obs:.1f} vs label-shuffled null {nullm:.1f} (p={p:.4f});  split-half r {sh[0]:.2f} (null {sh[1]:.2f}, p={sh[2]:.4f})")
    print(f"full-profile r: same model, different task {np.mean(within):.2f};  different model, same task {stb:.2f};  different model, different task {between:.2f}")
    # lineage / family similarity of model signatures a_m
    Ra = np.corrcoef(a)
    def mean_pairs(pred):
        v = [Ra[i, j] for i in range(17) for j in range(i + 1, 17) if pred(ORDER[i], ORDER[j])]; return float(np.mean(v)), len(v)
    lin = mean_pairs(lambda x, y: LIN[x] == LIN[y]); fam_ = mean_pairs(lambda x, y: LIN[x] != LIN[y] and FAM[x] == FAM[y]); oth = mean_pairs(lambda x, y: FAM[x] != FAM[y])
    print(f"signature r: same lineage {lin[0]:.2f} (n={lin[1]}), same family other lineage {fam_[0]:.2f} (n={fam_[1]}), different family {oth[0]:.2f} (n={oth[1]})")
    # permutation for lineage: shuffle model labels of the signature matrix
    def lin_stat(R, labs): return np.mean([R[i, j] for i in range(17) for j in range(i + 1, 17) if LIN[labs[i]] == LIN[labs[j]]])
    nl = [lin_stat(Ra, list(np.array(ORDER)[rng.permutation(17)])) for _ in range(5000)]
    p_lin = float((np.sum(np.array(nl) >= lin[0]) + 1) / 5001)
    # family among base models only (one per lineage)
    bi = [ORDER.index(m) for m in BASES]; Rb = Ra[np.ix_(bi, bi)]; fb = [FAM[m] for m in BASES]
    def fam_stat(labs): return np.mean([Rb[i, j] for i in range(len(bi)) for j in range(i + 1, len(bi)) if labs[i] == labs[j]])
    fo = fam_stat(fb); nf = [fam_stat(list(np.array(fb)[rng.permutation(len(fb))])) for _ in range(5000)]
    p_fam = float((np.sum(np.array(nf) >= fo) + 1) / 5001)
    print(f"lineage similarity p={p_lin:.4f};  family similarity among 11 base models: within {fo:.2f} vs null {np.mean(nf):.2f}, p={p_fam:.4f}")
    out["similarity"] = {"same_lineage": lin, "same_family_other_lineage": fam_, "different_family": oth, "p_lineage": p_lin,
                         "family_within_bases": float(fo), "family_null": float(np.mean(nf)), "p_family_bases": p_fam, "R": Ra.tolist()}
    # PCA of signatures
    U, s, Vt = np.linalg.svd(a - a.mean(0), full_matrices=False); ev = s ** 2 / (s ** 2).sum()
    sc = U * s
    print("PCA of signatures: explained " + " ".join(f"{v:.2f}" for v in ev[:4]))
    print("  PC1 loadings by decile:", np.round(Vt[0], 2), "\n  PC2 loadings:", np.round(Vt[1], 2))
    for i, m in enumerate(ORDER): print(f"    {m:28s} PC1 {sc[i, 0]:+6.2f}  PC2 {sc[i, 1]:+6.2f}")
    out["pca"] = {"explained": ev.tolist(), "loadings": Vt[:3].tolist(), "scores": {m: sc[i, :3].tolist() for i, m in enumerate(ORDER)}}
    # per-model region summary (from the full profile) and per-task sign consistency of the model's deviation
    reg = {"front": slice(0, 5), "40-50": slice(4, 5), "50-70": slice(5, 7), "back": slice(5, 10), "70-100": slice(7, 10)}
    pm = {}
    for i, m in enumerate(ORDER):
        pm[m] = {k: float(Z[i, :, s_].mean()) for k, s_ in reg.items()}
        pm[m].update({f"{k}_per_task": Z[i, :, s_].mean(1).tolist() for k, s_ in reg.items()})
        pm[m]["sig_back_dev_per_task"] = (Z[i, :, 5:10].mean(1) - Z[:, :, 5:10].mean((0, 2))).tolist()
    out["per_model"] = pm
    print("\nper model: back-half mean z per task (P A B R) | mean | deviation from the task mean in every task?")
    for i, m in enumerate(ORDER):
        bt = Z[i, :, 5:].mean(1); dev = bt - Z[:, :, 5:].mean((0, 2))
        print(f"  {m:28s} " + " ".join(f"{v:+5.2f}" for v in bt) + f" | {bt.mean():+5.2f} | dev signs {''.join('+' if v > 0 else '-' for v in dev)}")
    # robustness of the signatures
    rob = {}
    for name, Zv in [("q25", from_consist("q25")), ("q50", from_consist("q50")), ("entropy", from_consist("entropy")), ("top1", from_consist("top1"))]:
        av = decompose(Zv)[1]; o, nm, pp = perm_model(Zv, 2000)
        rob[name] = {"r_with_primary": float(np.corrcoef(av.ravel(), a.ravel())[0, 1]), "p_model": pp, "ss_ratio_to_null": float(o / nm),
                     "per_model_r": [float(np.corrcoef(av[i], a[i])[0, 1]) for i in range(17)]}
    for name, (attn, var) in [("sink_head_removed", ("eager", "sink_head")), ("eager_per_wording", ("eager", "all"))]:
        Zv, miss = from_cells(attn, var)
        if Zv is None: rob[name] = {"missing": miss}; continue
        av = decompose(Zv)[1]; o, nm, pp = perm_model(Zv, 2000)
        rob[name] = {"r_with_primary": float(np.corrcoef(av.ravel(), a.ravel())[0, 1]), "p_model": pp, "ss_ratio_to_null": float(o / nm),
                     "per_model_r": [float(np.corrcoef(av[i], a[i])[0, 1]) for i in range(17)]}
    # SDPA: models with SDPA for all four tasks; compare with eager on the same models
    ms = [m for m in ORDER if from_cells("sdpa", "all", [m])[0] is not None]
    Zs, _ = from_cells("sdpa", "all", ms); Ze, _ = from_cells("eager", "all", ms)
    as_, ae = decompose(Zs)[1], decompose(Ze)[1]; o, nm, pp = perm_model(Zs, 2000)
    rob["sdpa_vs_eager_same_models"] = {"models": ms, "r": float(np.corrcoef(as_.ravel(), ae.ravel())[0, 1]), "p_model_sdpa": pp, "ss_ratio_to_null_sdpa": float(o / nm)}
    out["robustness"] = rob
    print("\nrobustness of model signatures (r with primary, model-effect p, SS/null):")
    for k, v in rob.items(): print("  ", k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ("per_model_r", "models")}, len(v.get("models", [])) or "")
    json.dump(out, open(f"{SP}/model_diff.json", "w"), indent=1)
