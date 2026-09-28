"""Completeness critic: paper claims the drafted items do not check against E2b direct mode.
Same pipeline as e2b_direct_replication.py (v2lib.zprof within=False, model_diff.dec_prof/decompose)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, json, math, numpy as np
from scipy.stats import t as tdist
SP8 = str(paths.ANALYSIS)
os.environ["VPRIM"] = "y10"
sys.path.insert(0, SP8); sys.path.insert(0, f"{SP8}/vet3"); sys.path.insert(0, str(paths.REPLICATION))
from v2lib import zprof
import model_diff as md
from model_diff import dec_prof, decompose, split_half
import e2b
md.rng = np.random.default_rng(0)
T = ["pointer", "arith", "binding", "relational"]
M8 = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
      "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
def q10(C):
    C = C.astype(np.float64); P = C / C.sum(-1, keepdims=True); H = C.shape[-1]; m = max(1, int(np.ceil(0.1 * H)))
    return -np.sort(-P, -1)[..., :m].sum(-1)
def tci(v):
    v = np.asarray(v, float); n = len(v); h = tdist.ppf(.975, n - 1) * v.std(ddof=1) / math.sqrt(n); return [round(float(v.mean() - h), 3), round(float(v.mean() + h), 3)]
def r_(a, b): return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])
runs = [r for r in e2b.load_results(str(paths.E2B)) if r["mode"] == "direct"]
RB = {(r["model"].split("/")[-1], r["family"]): r for r in runs}
IT = {}
it_json = json.load(open(str(paths.E2B / "items.json")))
for f in it_json: 
    for it in it_json[f]: IT[it["id"]] = it
D = {}
for m in M8:
    for t in T:
        r = RB[(m, t)]; C = r["C"].astype(np.float64); k = np.array([x["k"] for x in r["items"]], float)
        flag = ((C.max(-1) / np.maximum(np.median(C, -1), 1e-9)) > 50).any(1) if m.startswith("OLMo") else np.zeros(len(k), bool)
        L = r["L"]; x = np.arange(L) / (L - 1)
        # answer-set entropy from digit logits
        ent = []
        for it_r, dl in zip(r["items"], r["DL"]):
            aset = [int(a) for a in IT[it_r["id"]]["answer_set"]]; lg = dl[aset].astype(float); p = np.exp(lg - lg.max()); p /= p.sum()
            ent.append(float(-(p * np.log(p + 1e-30)).sum()))
        ent = np.array(ent)
        out = {}
        for var, keep in [("excl", ~flag), ("noexcl", np.ones(len(k), bool))]:
            Y = q10(C[keep]); kk = k[keep]; b, sd, z = zprof(Y, kk, within=False)
            tot = np.log(C[keep].sum(-1)); bt, _, zt = zprof(tot, kk, within=False)
            # entropy-adjusted t (k + ent), and k-only t, as vet3/altexp.py
            Xk = (kk - kk.mean())[:, None]; e = ent[keep] - ent[keep].mean(); Yc = Y - Y.mean(0)
            def ols_t(X):
                n, p = X.shape; XtX = np.linalg.pinv(X.T @ X); B = XtX @ X.T @ Yc; E = Yc - X @ B
                s2 = (E ** 2).sum(0) / (n - p - 1); se = np.sqrt(np.outer(np.diag(XtX), s2)); return (B / se)[0]
            out[var] = {"z": z, "bpp": 100 * b, "zt": zt, "t_k": ols_t(Xk), "t_kent": ols_t(np.hstack([Xk, e[:, None]]))}
        D[(m, t)] = dict(out, x=x, L=L)
res = {}
# ---- (1) cells: back<0, back<front; per model/task back-half
back = {(m, t): float(D[(m, t)]["excl"]["z"][D[(m, t)]["x"] >= .5].mean()) for m in M8 for t in T}
front = {(m, t): float(D[(m, t)]["excl"]["z"][D[(m, t)]["x"] < .5].mean()) for m in M8 for t in T}
res["cells"] = {"n": 32, "back_neg": sum(v < 0 for v in back.values()), "back_lt_front": sum(back[c] < front[c] for c in back)}
res["back_by_model_task"] = {m: {t: round(back[(m, t)], 2) for t in T} for m in M8}
res["binding_spreads_more_than_relational"] = f"{sum(back[(m,'binding')] < back[(m,'relational')] for m in M8)}/8"
res["arith_weakest"] = f"{sum(max(T, key=lambda t: back[(m, t)]) == 'arith' for m in M8)}/8"
res["weakest_task_per_model"] = {m: max(T, key=lambda t: back[(m, t)]) for m in M8}
res["strongest_task_per_model"] = {m: min(T, key=lambda t: back[(m, t)]) for m in M8}
res["per_task_back_mean_ci"] = {t: [round(float(np.mean([back[(m, t)] for m in M8])), 2), tci([back[(m, t)] for m in M8]), sum(back[(m, t)] < 0 for m in M8)] for t in T}
# ---- (2) final layer task-mean z
res["final_layer_taskmean_z_E2b"] = {m: round(float(np.mean([D[(m, t)]["excl"]["z"][-1] for t in T])), 2) for m in M8}
CQ = {r["model"]: r for r in json.load(open(f"{SP8}/consist.json")) if r["stat"] == "q10"}
res["final_layer_taskmean_z_E2"] = {m: round(float(np.mean([CQ[m]["families"][t]["z"][-1] for t in T])), 2) for m in M8}
# ---- (3) Qwen2.5-7B: back-half deviation from the 8-model task mean, per task
def devs(Zfun):
    Z = np.array([[dec_prof(Zfun(m, t)) for t in T] for m in M8])  # [8,4,10]
    dv = Z - Z.mean(0, keepdims=True)
    return {m: [round(float(dv[i, j, 5:].mean()), 2) for j in range(4)] for i, m in enumerate(M8)}
res["backhalf_deviation_from_8model_mean_E2b"] = devs(lambda m, t: D[(m, t)]["excl"]["z"])
res["backhalf_deviation_from_8model_mean_E2"] = devs(lambda m, t: np.array(CQ[m]["families"][t]["z"]))
# ---- (4) signatures; OLMo exclusion; entropy adjustment
def sig(key, var="excl"): return decompose(np.array([[dec_prof(D[(m, t)][var][key]) for t in T] for m in M8]))[1]
aE2 = decompose(np.array([[dec_prof(np.array(CQ[m]["families"][t]["z"])) for t in T] for m in M8]))[1]
a_ex, a_no = sig("z", "excl"), sig("z", "noexcl")
res["olmo_exclusion"] = {"r_sig_excl_vs_noexcl_pooled": r_(a_ex, a_no),
                         "own_r_vs_E2_excl": {m: round(r_(a_ex[i], aE2[i]), 3) for i, m in enumerate(M8) if m.startswith("OLMo")},
                         "own_r_vs_E2_noexcl": {m: round(r_(a_no[i], aE2[i]), 3) for i, m in enumerate(M8) if m.startswith("OLMo")},
                         "pooled_vs_E2_excl": r_(a_ex, aE2), "pooled_vs_E2_noexcl": r_(a_no, aE2)}
Zno = np.array([[dec_prof(D[(m, t)]["noexcl"]["z"]) for t in T] for m in M8])
res["olmo_exclusion"]["split_half_noexcl"] = float(split_half(Zno)[0])
a_tk, a_te = sig("t_k"), sig("t_kent")
res["entropy_adjusted"] = {"r_sig_adj_vs_unadj": r_(a_te, a_tk), "r_adj_vs_E2": r_(a_te, aE2),
                           "own_r_adj_vs_E2_mean": float(np.mean([r_(a_te[i], aE2[i]) for i in range(8)])),
                           "own_r_unadj_t_vs_E2_mean": float(np.mean([r_(a_tk[i], aE2[i]) for i in range(8)])),
                           "own_best_adj": int(sum(np.argmax([r_(a_te[i], aE2[j]) for j in range(8)]) == i for i in range(8)))}
Zte = np.array([[dec_prof(D[(m, t)]["excl"]["t_kent"]) for t in T] for m in M8])
Ztk = np.array([[dec_prof(D[(m, t)]["excl"]["t_k"]) for t in T] for m in M8])
res["entropy_adjusted"]["back_half_mean_t_unadj"] = float(Ztk[:, :, 5:].mean()); res["entropy_adjusted"]["back_half_mean_t_adj"] = float(Zte[:, :, 5:].mean())
res["entropy_adjusted"]["shared_50_80_adj"] = [round(float(Zte[:, :, 5:8].mean()), 2), tci(Zte[:, :, 5:8].mean((1, 2)))]
res["entropy_adjusted"]["shared_50_80_unadj_t"] = [round(float(Ztk[:, :, 5:8].mean()), 2), tci(Ztk[:, :, 5:8].mean((1, 2)))]
# ---- (5) magnitude: back-half slope in pp/step; total output
bpp = {m: np.mean([D[(m, t)]["excl"]["bpp"][D[(m, t)]["x"] >= .5].mean() for t in T]) for m in M8}
res["back_slope_pp_per_step"] = {"mean_over_models": round(float(np.mean(list(bpp.values()))), 3), "t_ci": tci(list(bpp.values())),
                                 "per_model": {m: round(float(v), 3) for m, v in bpp.items()}}
tb = {(m, t): float(D[(m, t)]["excl"]["zt"][D[(m, t)]["x"] >= .5].mean()) for m in M8 for t in T}
res["total_output_falls_back_half"] = {"n_cells_neg": sum(v < 0 for v in tb.values()), "n": 32}
json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
