"""Verify: E2 wording benchmark (w0 vs w1), item-split reliabilities of E2 and E2b, noise ceilings,
task-deviation reliabilities and disattenuated task-pair agreement, E2 subsampled to the E2b design."""
import json, itertools, numpy as np
from vn_lib import *
OUT = {}
def sb(r): return 2 * r / (1 + r)
ORDER17 = list(CQ.keys())
def e2attn(m): return "sdpa" if m == "Qwen2.5-7B-Instruct" else "eager"

# (a) wording benchmark from cells.json
def wording(models):
    Z0 = np.array([[dec_prof(e2_cells_z(m, t, "w0", e2attn(m))) for t in T] for m in models])
    Z1 = np.array([[dec_prof(e2_cells_z(m, t, "w1", e2attn(m))) for t in T] for m in models])
    cellr = [r_(Z0[i, j], Z1[i, j]) for i in range(len(models)) for j in range(4)]
    a0 = Z0.mean(1) - Z0.mean((0, 1)); a1 = Z1.mean(1) - Z1.mean((0, 1))
    per = [r_(a0[i], a1[i]) for i in range(len(models))]
    return {"per_cell_mean": float(np.mean(cellr)), "sig_pooled": r_(a0, a1), "sig_per_model_mean": float(np.mean(per)),
            "sig_per_model_range": [float(min(per)), float(max(per))]}
for tag, models in (("M8", M8), ("M6", M6), ("ALL17", ORDER17)):
    OUT[f"wording_{tag}"] = wording(models); print("wording", tag, {k: np.round(v, 3) for k, v in OUT[f"wording_{tag}"].items()})

# (b) E2b item-split (stratified by k), 4 random splits
def e2b_halves(seed, models=M8):
    rng = np.random.default_rng(10_000 + seed); Z1, Z2 = [], []
    for m in models:
        r1, r2 = [], []
        for t in T:
            Y, k, _, _ = cell(m, "direct", t)
            sel = np.zeros(len(k), bool)
            for kk in np.unique(k):
                ix = np.nonzero(k == kk)[0]; sel[rng.choice(ix, len(ix) // 2, replace=False)] = True
            r1.append(dec_prof(zprof(Y[sel], k[sel])[2])); r2.append(dec_prof(zprof(Y[~sel], k[~sel])[2]))
        Z1.append(r1); Z2.append(r2)
    return np.array(Z1), np.array(Z2)
# (c) E2 item-split by prompt body / instance, per-wording z averaged over wordings
def e2_halves(seed, models=M8):
    rng = np.random.default_rng(20_000 + seed); Z1, Z2 = [], []
    for m in models:
        r1, r2 = [], []
        for t in T:
            d0 = vload(m, t, "w0", e2attn(m), True); d1 = vload(m, t, "w1", e2attn(m), True)
            u = np.union1d(np.unique(d0["body"]), np.unique(d1["body"])); pick = set(rng.choice(u, len(u) // 2, replace=False).tolist())
            zh = [[], []]
            for d in (d0, d1):
                s = np.array([b in pick for b in d["body"]])
                for h, sel in ((0, s), (1, ~s)): zh[h].append(zprof(d["y25"][sel], d["k"][sel], d["body"][sel], d["within"])[2])
            r1.append(dec_prof(0.5 * (zh[0][0] + zh[0][1]))); r2.append(dec_prof(0.5 * (zh[1][0] + zh[1][1])))
        Z1.append(r1); Z2.append(r2)
    return np.array(Z1), np.array(Z2)
HB = [e2b_halves(s) for s in range(4)]; HA = [e2_halves(s) for s in range(4)]
def relstats(H, ix):
    prof = []; sig = []; per_t = {t: [] for t in T}
    for Z1, Z2 in H:
        Z1, Z2 = Z1[ix], Z2[ix]
        rr = np.array([[r_(Z1[i, j], Z2[i, j]) for j in range(4)] for i in range(len(ix))])
        prof.append(rr.mean()); [per_t[t].append(rr[:, j].mean()) for j, t in enumerate(T)]
        sig.append(r_(Z1.mean(1) - Z1.mean((0, 1)), Z2.mean(1) - Z2.mean((0, 1))))
    # model profile (task-averaged) reliability
    mp = [np.mean([r_(Z1[ix][i].mean(0), Z2[ix][i].mean(0)) for i in range(len(ix))]) for Z1, Z2 in H]
    return {"profile_half": float(np.mean(prof)), "per_task_half": {t: float(np.mean(v)) for t, v in per_t.items()}, "sig_half": float(np.mean(sig)),
            "model_profile_half": float(np.mean(mp))}
def devrel(H, ix):
    out = {}
    for j, t in enumerate(T):
        rs = [r_(Z1[ix][:, j] - Z1[ix][:, j].mean(0), Z2[ix][:, j] - Z2[ix][:, j].mean(0)) for Z1, Z2 in H]
        out[t] = float(sb(np.mean(rs)))
    return out
for tag, models in (("M8", M8), ("M6", M6)):
    ix = [M8.index(m) for m in models]
    rb, ra = relstats(HB, ix), relstats(HA, ix)
    o = {"E2b": rb, "E2": ra,
         "ceiling_cell": float(np.sqrt(sb(rb["profile_half"]) * sb(ra["profile_half"]))),
         "ceiling_sig": float(np.sqrt(sb(rb["sig_half"]) * sb(ra["sig_half"])))}
    Zb = Zprof(models); Za = Zprof_e2(models)
    for nm, H, Z in (("E2b", HB, Zb), ("E2", HA, Za)):
        rel = devrel(H, ix); dev = Z - Z.mean(0, keepdims=True)
        br = r_(dev[:, 2], dev[:, 3]); pa = r_(dev[:, 0], dev[:, 1])
        o[f"devrel_{nm}"] = rel; o[f"br_disatt_{nm}"] = br / np.sqrt(rel["binding"] * rel["relational"]); o[f"pa_disatt_{nm}"] = pa / np.sqrt(rel["pointer"] * rel["arith"])
    OUT[f"rel_{tag}"] = o
    print("\nreliability", tag, json.dumps(o, indent=None, default=lambda x: round(float(x), 3)))

# (f) E2 subsampled to the E2b design: w0 only, 30 items per depth, k shuffled across items
Zfull = Zprof_e2(M8); Zb8 = Zprof(M8)
draws = []
for s in range(20):
    rng = np.random.default_rng(30_000 + s); Zs = []
    for m in M8:
        row = []
        for t in T:
            d = vload(m, t, "w0", e2attn(m), True); k = d["k"]; sel = []
            for kk in np.unique(k):
                ix = np.nonzero(k == kk)[0]; sel += rng.choice(ix, min(30, len(ix)), replace=False).tolist()
            sel = np.array(sorted(sel)); row.append(dec_prof(zprof(d["y25"][sel], k[sel])[2]))
        Zs.append(row)
    draws.append(np.array(Zs))
gb = [D[:, :, 5:].mean() for D in draws]
OUT["subsample"] = {"full_back": float(Zfull[:, :, 5:].mean()), "sub_back_mean": float(np.mean(gb)), "sub_back_sd": float(np.std(gb)), "E2b_back": float(Zb8[:, :, 5:].mean()),
                    "frac_expected": float((Zfull[:, :, 5:].mean() - np.mean(gb)) / (Zfull[:, :, 5:].mean() - Zb8[:, :, 5:].mean())),
                    "mean_abs_full": float(np.abs(Zfull).mean()), "mean_abs_sub": float(np.mean([np.abs(D).mean() for D in draws])), "mean_abs_E2b": float(np.abs(Zb8).mean()),
                    "per_task": {t: [float(Zfull[:, j, 5:].mean()), float(np.mean([D[:, j, 5:].mean() for D in draws])), float(Zb8[:, j, 5:].mean())] for j, t in enumerate(T)}}
print("\nsubsample", json.dumps(OUT["subsample"], default=lambda x: round(float(x), 3)))
# Qwen2.5-7B-Instruct binding cell vs E2 eager
qb = r_(dec_prof(0.5 * (e2_cells_z("Qwen2.5-7B-Instruct", "binding", "w0", "eager") + e2_cells_z("Qwen2.5-7B-Instruct", "binding", "w1", "eager"))), Zb8[3, 2])
OUT["Q7I_binding_vs_E2_eager"] = qb; print("Q7I binding vs E2 eager", round(qb, 3))
# paper 17-model binding-relational deviation agreement
Z17 = Zprof_e2(ORDER17); dev = Z17 - Z17.mean(0, keepdims=True)
OUT["paper17_br"] = r_(dev[:, 2], dev[:, 3]); OUT["paper17_pa"] = r_(dev[:, 0], dev[:, 1]); print("paper 17 b-r %.3f p-a %.3f" % (OUT["paper17_br"], OUT["paper17_pa"]))
json.dump(OUT, open(f"{HERE}/vn_reliability.json", "w"), indent=1, default=float)
