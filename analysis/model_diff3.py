"""Model signatures after item-level covariate adjustment (vet3/altexp_layers.csv: per-layer t of the k-slope,
fits 'k', 'k+ent', 'k+ent+corr'); wording-averaged decile profiles, same decomposition as model_diff.py."""
import sys, json, numpy as np, pandas as pd
SP = sys.argv[1]; sys.path.insert(0, SP)
from model_diff import ORDER, T, decompose, ss, perm_model, split_half, LIN
import model_diff as md
A = pd.read_csv(f"{SP}/vet3/altexp_layers.csv"); A["dec"] = np.minimum((A.x * 10).astype(int), 9)
MD = json.load(open(f"{SP}/model_diff.json")); a0 = np.array([MD["primary"]["a"][m] for m in ORDER])
out = {}
for fit in ["k", "k+ent", "k+ent+corr"]:
    S = A[A.fit == fit].groupby(["model", "task", "w", "dec"]).t.mean().groupby(["model", "task", "dec"]).mean()
    Z = np.array([[[S[(m, t, d)] for d in range(10)] for t in T] for m in ORDER])
    mu, a, b, e = decompose(Z); s = ss(Z); tot = sum(s.values()); o, nm, p = perm_model(Z, 2000); sh = split_half(Z)
    Ra = np.corrcoef(a); lin = np.mean([Ra[i, j] for i in range(17) for j in range(i + 1, 17) if LIN[ORDER[i]] == LIN[ORDER[j]]])
    oth = np.mean([Ra[i, j] for i in range(17) for j in range(i + 1, 17) if LIN[ORDER[i]] != LIN[ORDER[j]]])
    back = Z[:, :, 5:].mean((1, 2))
    out[fit] = {"ss_frac": {k: float(v / tot) for k, v in s.items()}, "p_model": p, "ss_ratio": float(o / nm), "split_half": sh[0], "p_split": sh[2],
                "r_with_primary_signature": float(np.corrcoef(a.ravel(), a0.ravel())[0, 1]), "lineage_r": float(lin), "other_r": float(oth),
                "back_mean_t": dict(zip(ORDER, back.tolist())), "a": {m: a[i].tolist() for i, m in enumerate(ORDER)}}
    print(f"{fit:12s} model {s['model']/tot:.2f} task {s['task']/tot:.2f} shared {s['shared']/tot:.2f} resid {s['resid']/tot:.2f} | model p {p:.4f} SS/null {o/nm:.2f} | split-half r {sh[0]:.2f} p {sh[2]:.4f} | r(signature, primary) {out[fit]['r_with_primary_signature']:.2f} | lineage r {lin:.2f} other {oth:.2f}")
    print("     back-half mean t per model: " + " ".join(f"{m.split('-')[0][:4]}{m.split('-')[1][:4] if '-' in m else ''}:{v:+.2f}" for m, v in zip(ORDER, back)))
json.dump(out, open(f"{SP}/model_diff3.json", "w"), indent=1)
