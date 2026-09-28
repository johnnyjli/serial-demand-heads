"""Statistical power / noise of E2b direct (30 items per depth, one wording) vs E2 (42 per depth x 2 wordings),
for the 8 models run in both.
(a) item-split reliability of E2b: disjoint halves of the items (15 per depth), 4 random splits;
    per-cell decile-profile r and signature r between halves; Spearman-Brown to full length.
(b) the same for E2 on the same 8 models (model_diff5 procedure: halves of instances / prompt bodies,
    per-wording exact z averaged over the two wordings; Qwen2.5-7B-Instruct SDPA).
(c) E2 subsampled to the E2b design (wording w0 only, 30 items per depth, k shuffled across items for every
    family): how much of the smaller |z| in E2b is expected from the design alone.  20 draws."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, json, numpy as np
SP8 = str(paths.ANALYSIS)
OUT = os.path.dirname(os.path.abspath(__file__))
os.environ["VPRIM"] = "y10"
sys.path.insert(0, SP8); sys.path.insert(0, f"{SP8}/vet3"); sys.path.insert(0, str(paths.REPLICATION))
from v2lib import zprof, load as vload
from model_diff import dec_prof, decompose
import e2b
T = ["pointer", "arith", "binding", "relational"]
M8 = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
      "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
M6 = [m for m in M8 if not m.startswith("Qwen2.5-7B")]
def q10(C):
    C = C.astype(np.float64); P = C / C.sum(-1, keepdims=True); m = max(1, int(np.ceil(0.1 * C.shape[-1]))); return -np.sort(-P, -1)[..., :m].sum(-1)
def sb(r): return 2 * r / (1 + r)
runs = {(r["model"].split("/")[-1], r["family"]): r for r in e2b.load_results(str(paths.E2B)) if r["mode"] == "direct"}
B = {}
for m in M8:
    for t in T:
        r = runs[(m, t)]; C = r["C"]; k = np.array([x["k"] for x in r["items"]], float); keep = np.ones(len(k), bool)
        if m.startswith("OLMo"): keep = ~((C.max(-1) / np.maximum(np.median(C, -1), 1e-9)) > 50).any(1)
        B[(m, t)] = (q10(C[keep]), k[keep])
out = {}
# (a) E2b item-split
def e2b_half(seed):
    Z1, Z2 = [], []
    for m in M8:
        r1, r2 = [], []
        for t in T:
            Y, k = B[(m, t)]; rng = np.random.default_rng(seed * 100 + T.index(t)); sel = np.zeros(len(k), bool)
            for kk in np.unique(k):
                ix = np.nonzero(k == kk)[0]; sel[rng.permutation(ix)[: len(ix) // 2]] = True
            r1.append(dec_prof(zprof(Y[sel], k[sel])[2])); r2.append(dec_prof(zprof(Y[~sel], k[~sel])[2]))
        Z1.append(r1); Z2.append(r2)
    return np.array(Z1), np.array(Z2)
def rel_stats(pairs, models_idx):
    prof_r, sig_r, per_task = [], [], {t: [] for t in T}
    for Z1, Z2 in pairs:
        Z1, Z2 = Z1[models_idx], Z2[models_idx]
        prof_r.append(np.mean([np.corrcoef(Z1[i, j], Z2[i, j])[0, 1] for i in range(len(Z1)) for j in range(4)]))
        for j, t in enumerate(T): per_task[t].append(np.mean([np.corrcoef(Z1[i, j], Z2[i, j])[0, 1] for i in range(len(Z1))]))
        sig_r.append(np.corrcoef(decompose(Z1)[1].ravel(), decompose(Z2)[1].ravel())[0, 1])
    pr, sr = float(np.mean(prof_r)), float(np.mean(sig_r))
    return {"profile_half_r": pr, "profile_full_sb": float(sb(pr)), "signature_half_r": sr, "signature_full_sb": float(sb(sr)),
            "per_task_half_r": {t: float(np.mean(v)) for t, v in per_task.items()}}
pa = [e2b_half(s) for s in range(4)]
# (b) E2 item-split on the same models
def e2_half(seed):
    Z1, Z2 = [], []
    for m in M8:
        r1, r2 = [], []
        for t in T:
            zs = {0: [], 1: []}
            for w in ("w0", "w1"):
                d = vload(m, t, w, "sdpa" if m == "Qwen2.5-7B-Instruct" else "eager", True)
                u = np.unique(d["body"]); rng = np.random.default_rng(seed * 100 + T.index(t)); pick = set(rng.permutation(u)[: len(u) // 2])
                s = np.array([b in pick for b in d["body"]])
                for h, sel in ((0, s), (1, ~s)): zs[h].append(zprof(d["y25"][sel], d["k"][sel], d["body"][sel], d["within"])[2])
            r1.append(dec_prof(0.5 * (zs[0][0] + zs[0][1]))); r2.append(dec_prof(0.5 * (zs[1][0] + zs[1][1])))
        Z1.append(r1); Z2.append(r2)
    return np.array(Z1), np.array(Z2)
pb = [e2_half(s) for s in range(4)]
for tag, models in (("M8", M8), ("M6_noQwen7B", M6)):
    ix = [M8.index(m) for m in models]
    ra, rb = rel_stats(pa, ix), rel_stats(pb, ix)
    out[tag] = {"E2b_item_split": ra, "E2_item_split": rb,
                "ceiling_per_cell_r_if_true_profiles_equal": float(np.sqrt(ra["profile_full_sb"] * rb["profile_full_sb"])),
                "ceiling_signature_r_if_true_signatures_equal": float(np.sqrt(ra["signature_full_sb"] * rb["signature_full_sb"]))}
# (c) E2 subsampled to the E2b design
CQ = {r["model"]: r for r in json.load(open(f"{SP8}/consist.json")) if r["stat"] == "q10"}
Zfull = np.array([[dec_prof(np.array(CQ[m]["families"][t]["z"])) for t in T] for m in M8])
Zb = np.array([[dec_prof(zprof(*B[(m, t)])[2]) for t in T] for m in M8])
draws = []
for s in range(20):
    rng = np.random.default_rng(1000 + s); Zs = []
    for m in M8:
        row = []
        for t in T:
            d = vload(m, t, "w0", "sdpa" if m == "Qwen2.5-7B-Instruct" else "eager", True); k = d["k"]; sel = []
            for kk in np.unique(k):
                ix = np.nonzero(k == kk)[0]; sel += list(rng.permutation(ix)[: min(30, len(ix))])
            sel = np.array(sorted(sel)); row.append(dec_prof(zprof(d["y25"][sel], k[sel])[2]))
        Zs.append(row)
    draws.append(np.array(Zs))
Dm = np.mean(draws, 0)
out["design_matched_E2"] = {
    "mu_full_E2": Zfull.mean((0, 1)).round(3).tolist(), "mu_E2_design_matched_mean_of_20": Dm.mean((0, 1)).round(3).tolist(), "mu_E2b": Zb.mean((0, 1)).round(3).tolist(),
    "back_mean_z_full_E2": dict(zip(M8, Zfull[:, :, 5:].mean((1, 2)).round(3).tolist())),
    "back_mean_z_E2_design_matched": dict(zip(M8, Dm[:, :, 5:].mean((1, 2)).round(3).tolist())),
    "back_mean_z_E2b": dict(zip(M8, Zb[:, :, 5:].mean((1, 2)).round(3).tolist())),
    "grand_back_mean_z": {"full_E2": float(Zfull[:, :, 5:].mean()), "E2_design_matched": float(Dm[:, :, 5:].mean()), "E2b": float(Zb[:, :, 5:].mean())},
    "grand_back_mean_z_per_task": {t: {"full_E2": float(Zfull[:, j, 5:].mean()), "E2_design_matched": float(Dm[:, j, 5:].mean()), "E2b": float(Zb[:, j, 5:].mean())} for j, t in enumerate(T)},
    "d4_40_50": {"full_E2": float(Zfull[:, :, 4].mean()), "E2_design_matched": float(Dm[:, :, 4].mean()), "E2b": float(Zb[:, :, 4].mean())},
    "d4_40_50_arith": {"full_E2": float(Zfull[:, 1, 4].mean()), "E2_design_matched": float(Dm[:, 1, 4].mean()), "E2b": float(Zb[:, 1, 4].mean())},
    "sd_of_back_mean_across_draws": float(np.std([D[:, :, 5:].mean() for D in draws])),
    "mean_abs_z_all_layers": {"full_E2": float(np.abs(Zfull).mean()), "E2_design_matched": float(np.mean([np.abs(D).mean() for D in draws])), "E2b": float(np.abs(Zb).mean())},
    "per_cell_r_design_matched_single_draw_vs_E2b_mean": float(np.mean([np.mean([np.corrcoef(D[i, j], Zb[i, j])[0, 1] for i in range(8) for j in range(4)]) for D in draws])),
}
json.dump(out, open(f"{OUT}/e2b_direct_power.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))

# (d) reliability of each task's model deviations (Z - mean over models, per task) between item halves, and the
#     task-pair agreement corrected for it (r / sqrt(rel_i * rel_j), full-length Spearman-Brown reliabilities)
import itertools
def dev_rel(pairs, ix):
    rel = {}
    for j, t in enumerate(T):
        rs = []
        for Z1, Z2 in pairs:
            A1 = Z1[ix][:, j] - Z1[ix][:, j].mean(0); A2 = Z2[ix][:, j] - Z2[ix][:, j].mean(0); rs.append(np.corrcoef(A1.ravel(), A2.ravel())[0, 1])
        rel[t] = float(sb(np.mean(rs)))
    return rel
Zfull_b = {tag: np.array([[dec_prof(zprof(*B[(m, t)])[2]) for t in T] for m in models]) for tag, models in (("M8", M8), ("M6_noQwen7B", M6))}
Zfull_a = {tag: np.array([[dec_prof(np.array(CQ[m]["families"][t]["z"])) for t in T] for m in models]) for tag, models in (("M8", M8), ("M6_noQwen7B", M6))}
for tag, models in (("M8", M8), ("M6_noQwen7B", M6)):
    ix = [M8.index(m) for m in models]; o = {}
    for nm, pairs, Z in (("E2b", pa, Zfull_b[tag]), ("E2", pb, Zfull_a[tag])):
        rel = dev_rel(pairs, ix); dev = Z - Z.mean(0, keepdims=True); pr = {}
        for i, j in itertools.combinations(range(4), 2):
            r = float(np.corrcoef(dev[:, i].ravel(), dev[:, j].ravel())[0, 1]); pr[f"{T[i]}-{T[j]}"] = {"r": r, "disattenuated": r / np.sqrt(rel[T[i]] * rel[T[j]])}
        o[nm] = {"deviation_reliability_full": rel, "task_pairs": pr}
    out[tag]["task_deviation_reliability"] = o
json.dump(out, open(f"{OUT}/e2b_direct_power.json", "w"), indent=1, default=float)
for tag in ("M8", "M6_noQwen7B"):
    for nm in ("E2b", "E2"):
        o = out[tag]["task_deviation_reliability"][nm]
        print(tag, nm, {k: round(v, 2) for k, v in o["deviation_reliability_full"].items()}, {k: (round(v["r"], 2), round(v["disattenuated"], 2)) for k, v in o["task_pairs"].items()})
