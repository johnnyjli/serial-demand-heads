"""Do the model signatures just track accuracy, answer uncertainty, head count or baseline concentration?
Per model: mean strict accuracy (4 tasks, all k, both wordings), accuracy at k=1, slope of answer entropy on k,
heads per layer H, baseline top-10% share; correlated (Spearman, 17 models) with the signature PC1 score and with
the back-half mean z.  Also: the Qwen2.5-7B eager vs SDPA z-profile on the one cell captured under both kernels."""
import sys, json, numpy as np
from scipy.stats import spearmanr
SP = sys.argv[1]; sys.path.insert(0, f"{SP}/vet3"); import os; os.environ["VPRIM"] = "y10"
from v2lib import load, ORDER, TASKS, zprof
MD = json.load(open(f"{SP}/model_diff.json"))
pc1 = np.array([MD["pca"]["scores"][m][0] for m in ORDER]); pc2 = np.array([MD["pca"]["scores"][m][1] for m in ORDER])
back = np.array([MD["per_model"][m]["back"] for m in ORDER]); b5070 = np.array([MD["per_model"][m]["50-70"] for m in ORDER])
feat = {k: [] for k in ["acc", "acc_k1", "acc_k2", "ent_slope", "H", "L", "base_share_back", "base_share_all"]}
for m in ORDER:
    acc, a1, a2, es, bs, ba = [], [], [], [], [], []
    for t in TASKS:
        for w in ("w0", "w1"):
            d = load(m, t, w, "sdpa" if m == "Qwen2.5-7B-Instruct" else "eager", True)
            acc.append(d["correct"].mean()); a1.append(d["correct"][d["k"] == 1].mean()); a2.append(d["correct"][d["k"] == 2].mean())
            k = d["k"]; kc = k - k.mean(); es.append(kc @ (d["ent"] - d["ent"].mean()) / (kc @ kc))
            bs.append(d["y25"][:, d["x"] >= .5].mean()); ba.append(d["y25"].mean()); H = int(d["H"]); L = d["L"]
    for k_, v in zip(["acc", "acc_k1", "acc_k2", "ent_slope", "base_share_back", "base_share_all"], [acc, a1, a2, es, bs, ba]): feat[k_].append(float(np.mean(v)))
    feat["H"].append(H); feat["L"].append(L)
out = {"features": {k: dict(zip(ORDER, v)) for k, v in feat.items()}, "spearman": {}}
print(f"{'feature':16s} {'rho(PC1)':>9s} {'p':>6s} {'rho(back z)':>11s} {'p':>6s} {'rho(50-70 z)':>12s} {'p':>6s}")
for k, v in feat.items():
    r1 = spearmanr(v, pc1); r2 = spearmanr(v, back); r3 = spearmanr(v, b5070)
    out["spearman"][k] = {"pc1": [float(r1.correlation), float(r1.pvalue)], "back": [float(r2.correlation), float(r2.pvalue)], "b5070": [float(r3.correlation), float(r3.pvalue)]}
    print(f"{k:16s} {r1.correlation:+9.2f} {r1.pvalue:6.3f} {r2.correlation:+11.2f} {r2.pvalue:6.3f} {r3.correlation:+12.2f} {r3.pvalue:6.3f}")
for i, m in enumerate(ORDER):
    print(f"  {m:28s} acc {feat['acc'][i]:.2f} k1 {feat['acc_k1'][i]:.2f} ent/k {feat['ent_slope'][i]:+.3f} H {feat['H'][i]:2d} base10(back) {feat['base_share_back'][i]:.3f} | PC1 {pc1[i]:+.2f} back {back[i]:+.2f}")
# Qwen2.5-7B: the cell with both kernels
for m in ["Qwen2.5-7B", "Llama-3.1-8B", "OLMo-2-1124-7B"]:
    for t in TASKS + ["fixed_digit"]:
        for w in ("w0", "w1"):
            s = load(m, t, w, "sdpa", True)
            if s is None: continue
            e = load(m, t, w, "eager", True)
            ze = zprof(e["y25"], e["k"], e["body"], e["within"])[2]; zs = zprof(s["y25"], s["k"], s["body"], s["within"])[2]
            agree = float((e["correct"] == s["correct"]).mean())
            print(f"{m} {t}/{w}: eager vs SDPA per-layer z r = {np.corrcoef(ze, zs)[0, 1]:.2f}; back z eager {ze[e['x'] >= .5].mean():+.2f} sdpa {zs[s['x'] >= .5].mean():+.2f}; correctness agreement {agree:.2f}")
            out.setdefault("kernel_check", []).append({"model": m, "cell": f"{t}/{w}", "r": float(np.corrcoef(ze, zs)[0, 1]), "back_eager": float(ze[e['x'] >= .5].mean()), "back_sdpa": float(zs[s['x'] >= .5].mean()), "correct_agree": agree})
json.dump(out, open(f"{SP}/model_diff2.json", "w"), indent=1)
