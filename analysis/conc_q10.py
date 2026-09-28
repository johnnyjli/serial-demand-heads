"""Concentrating and spreading layers under the top-10% head share (paper's primary statistic).
Inputs: consist.json (wording-averaged per model x family, stat q10; Qwen2.5-7B-Instruct under SDPA) and
vet3/layertab.csv (per wording, includes the fixed_digit numeral control).
(1) decile profile of z per task (model-bootstrap CIs) and fraction of significant concentrating / spreading layers;
(2) consistently concentrating / spreading layers (>= 3 of 4 tasks, none opposite): location, participating tasks,
    and a pooled circular-shift null for the count inside each depth band;
(3) numeral control decile profile (per wording, then averaged)."""
import sys, json, numpy as np, pandas as pd
SP = sys.argv[1]
T = ["pointer", "arith", "binding", "relational"]
C = [r for r in json.load(open(f"{SP}/consist.json")) if r["stat"] == "q10"]
rng = np.random.default_rng(0); out = {"deciles": {}, "consistent": {}, "control": {}}
def boot(Zd, n=4000):
    b = np.array([Zd[rng.integers(0, len(Zd), len(Zd))].mean(0) for _ in range(n)]); return np.quantile(b, [.025, .975], 0)
for t in T + ["all4"]:
    fams = T if t == "all4" else [t]; Zd, Cd, Sd = [], [], []
    for r in C:
        L = r["L"]; x = np.arange(L) / (L - 1); dec = np.minimum((x * 10).astype(int), 9)
        Z = np.stack([r["families"][f]["z"] for f in fams]).mean(0)
        M = np.stack([np.sign(r["families"][f]["z"]) * (np.array(r["families"][f]["p"]) < .05) for f in fams])
        Zd.append([Z[dec == d].mean() for d in range(10)]); Cd.append([(M[:, dec == d] > 0).mean() for d in range(10)]); Sd.append([(M[:, dec == d] < 0).mean() for d in range(10)])
    Zd = np.array(Zd); lo, hi = boot(Zd)
    out["deciles"][t] = {"mean_z": Zd.mean(0).tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "models_pos": (Zd > 0).sum(0).tolist(),
                         "frac_sig_conc": np.mean(Cd, 0).tolist(), "frac_sig_spread": np.mean(Sd, 0).tolist()}
# (2) consistent layers
def consistent(M):
    pos, neg = (M > 0).sum(0), (M < 0).sum(0)
    return ((pos >= 3) & (neg == 0)).astype(int) - ((neg >= 3) & (pos == 0)).astype(int)
bands = {"0-35": (0, .35), "35-50": (.35, .5), "50-90": (.5, .9), "90-100": (.9, 1.01)}
obs = {b: [0, 0] for b in bands}; rows = []; NB = 5000; null = np.zeros((NB, len(bands), 2))
part = {"conc": {t: 0 for t in T}, "spread": {t: 0 for t in T}}
for r in C:
    L = r["L"]; x = np.arange(L) / (L - 1)
    M = np.stack([np.sign(r["families"][f]["z"]) * (np.array(r["families"][f]["p"]) < .05) for f in T])
    c = consistent(M)
    for j, (b, (a, e)) in enumerate(bands.items()):
        sel = (x >= a) & (x < e); obs[b][0] += int((c[sel] > 0).sum()); obs[b][1] += int((c[sel] < 0).sum())
    for l in np.nonzero(c)[0]:
        kind = "conc" if c[l] > 0 else "spread"
        for i, t in enumerate(T):
            if M[i, l] == c[l]: part[kind][t] += 1
        rows.append({"model": r["model"], "layer": int(l), "x": float(x[l]), "kind": kind, "tasks": [t for i, t in enumerate(T) if M[i, l] == c[l]]})
    for s in range(NB):
        cs = consistent(np.stack([np.roll(M[i], rng.integers(L)) for i in range(4)]))
        for j, (b, (a, e)) in enumerate(bands.items()):
            sel = (x >= a) & (x < e); null[s, j, 0] += (cs[sel] > 0).sum(); null[s, j, 1] += (cs[sel] < 0).sum()
for j, b in enumerate(bands):
    out["consistent"][b] = {"conc": obs[b][0], "conc_null": float(null[:, j, 0].mean()), "p_conc": float((np.sum(null[:, j, 0] >= obs[b][0]) + 1) / (NB + 1)),
                            "spread": obs[b][1], "spread_null": float(null[:, j, 1].mean()), "p_spread": float((np.sum(null[:, j, 1] >= obs[b][1]) + 1) / (NB + 1))}
tot = null.sum(1); oc = sum(v[0] for v in obs.values()); osp = sum(v[1] for v in obs.values())
out["consistent"]["total"] = {"conc": oc, "conc_null": float(tot[:, 0].mean()), "p_conc": float((np.sum(tot[:, 0] >= oc) + 1) / (NB + 1)),
                              "spread": osp, "spread_null": float(tot[:, 1].mean()), "p_spread": float((np.sum(tot[:, 1] >= osp) + 1) / (NB + 1))}
out["consistent"]["participation"] = part; out["consistent"]["layers"] = rows
out["consistent"]["models_with_conc"] = len({r["model"] for r in rows if r["kind"] == "conc"})
out["consistent"]["models_with_spread"] = len({r["model"] for r in rows if r["kind"] == "spread"})
# per model: front-half mean z, 40-50% band mean z (all4)
pm = []
for r in C:
    L = r["L"]; x = np.arange(L) / (L - 1); Z = np.stack([r["families"][f]["z"] for f in T]).mean(0)
    pm.append({"model": r["model"], "front": float(Z[x < .5].mean()), "back": float(Z[x >= .5].mean()), "band40_50": float(Z[(x >= .4) & (x < .5)].mean()),
               "band50_70": float(Z[(x >= .5) & (x < .7)].mean())})
out["per_model"] = pm
# (3) numeral control, per wording (vet3 layertab: top-10% share, eager, clean)
LT = pd.read_csv(f"{SP}/vet3/layertab.csv"); LT["dec"] = np.minimum((LT.x * 10).astype(int), 9)
for t in ["fixed_digit"] + T:
    Cc = LT[LT.task == t].groupby(["model", "w", "dec"]).z.mean().unstack("dec").groupby("model").mean()
    lo, hi = boot(Cc.values)
    out["control"][t] = {"mean_z": Cc.mean().tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "models_pos": (Cc > 0).sum().tolist()}
json.dump(out, open(f"{SP}/conc_q10.json", "w"), indent=1)
for t, d in out["deciles"].items():
    print(f"{t:10s} z " + " ".join(f"{m:+5.2f}{'*' if (a > 0 or b < 0) else ' '}" for m, a, b in zip(d["mean_z"], d["lo"], d["hi"])))
for t, d in out["control"].items():
    print(f"ctl/{t[:9]:9s} z " + " ".join(f"{m:+5.2f}{'*' if (a > 0 or b < 0) else ' '}" for m, a, b in zip(d["mean_z"], d["lo"], d["hi"])) + "  models>0 " + " ".join(str(v) for v in d["models_pos"]))
for b, d in out["consistent"].items():
    if b in bands or b == "total": print(b, d)
print("participation", part, "models with conc", out["consistent"]["models_with_conc"], "spread", out["consistent"]["models_with_spread"])
f = np.array([p["front"] for p in pm]); bb = np.array([p["band40_50"] for p in pm]); b57 = np.array([p["band50_70"] for p in pm])
print("front>0", (f > 0).sum(), "band40-50>0", (bb > 0).sum(), "band50-70<0", (b57 < 0).sum(), "of 17")
