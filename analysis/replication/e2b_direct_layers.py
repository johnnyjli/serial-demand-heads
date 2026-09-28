"""Layer-level claims of the paper's 'How the models differ' section, checked in E2b direct (top-10% share).
Per model x task: per-layer OLS slope on k, two-sided permutation p (B=2000, k shuffled across items, one shuffle
for all layers, as consist.py).  Consistent layer: significant (p<.05) in the same direction in >=3 of 4 tasks
and significant in the opposite direction in none.  Null count from independent per-task null draws (consist.py).
Compared with consist.json (E2) for the same models.  Also: per-model 40-50% decile z per task, and the decile
of strongest spreading in the task-averaged profile."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, json, numpy as np
SP8 = str(paths.ANALYSIS)
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP8); sys.path.insert(0, str(paths.REPLICATION))
from model_diff import dec_prof
import e2b
T = ["pointer", "arith", "binding", "relational"]; NB = 2000
M8 = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
      "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
def q10(C):
    C = C.astype(np.float64); P = C / C.sum(-1, keepdims=True); m = max(1, int(np.ceil(0.1 * C.shape[-1]))); return -np.sort(-P, -1)[..., :m].sum(-1)
def slopes(K, Y):
    Kc = K - K.mean(-1, keepdims=True); Yc = Y - Y.mean(0); return (Kc @ Yc) / (Kc ** 2).sum(-1, keepdims=True)
def consistent(ss):
    pos, neg = (ss > 0).sum(0), (ss < 0).sum(0); return ((pos >= 3) & (neg == 0)).astype(int) - ((neg >= 3) & (pos == 0)).astype(int)
runs = {(r["model"].split("/")[-1], r["family"]): r for r in e2b.load_results(str(paths.E2B)) if r["mode"] == "direct"}
CQ = {r["model"]: r for r in json.load(open(f"{SP8}/consist.json")) if r["stat"] == "q10"}
out = {}
for m in M8:
    rng = np.random.default_rng(0); fam = {}
    for t in T:
        r = runs[(m, t)]; C = r["C"]; k = np.array([x["k"] for x in r["items"]], float); keep = np.ones(len(k), bool)
        if m.startswith("OLMo"): keep = ~((C.max(-1) / np.maximum(np.median(C, -1), 1e-9)) > 50).any(1)
        Y = q10(C[keep]); k = k[keep]
        obs = slopes(k[None], Y)[0]; null = slopes(np.array([k[rng.permutation(len(k))] for _ in range(NB)]), Y)
        p = (np.sum(np.abs(null) >= np.abs(obs) - 1e-15, 0) + 1) / (NB + 1)
        A = np.abs(null); srt = np.sort(A, 0)
        pnull = (NB - np.array([np.searchsorted(srt[:, l], A[:, l], side="left") for l in range(A.shape[1])]).T + 1) / (NB + 1)
        fam[t] = {"sign": np.sign(obs), "p": p, "nsign": np.sign(null), "pnull": pnull, "z": obs / null.std(0)}
    L = len(fam[T[0]]["p"]); x = np.arange(L) / (L - 1)
    cons = consistent(np.stack([fam[t]["sign"] * (fam[t]["p"] < .05) for t in T]))
    nc = np.array([[(c := consistent(np.stack([fam[t]["nsign"][b] * (fam[t]["pnull"][b] < .05) for t in T]))).__gt__(0).sum(), (c < 0).sum()] for b in range(NB)])
    e2c = np.array(CQ[m]["consistent"])
    zb = np.array([dec_prof(fam[t]["z"]) for t in T]); za = np.array([dec_prof(np.array(CQ[m]["families"][t]["z"])) for t in T])
    out[m] = {"L": L, "E2b_conc": [f"{l} ({x[l]:.2f})" for l in np.nonzero(cons > 0)[0]], "E2b_spread": [f"{l} ({x[l]:.2f})" for l in np.nonzero(cons < 0)[0]],
              "E2_conc": [f"{l} ({x[l]:.2f})" for l in np.nonzero(e2c > 0)[0]], "E2_spread": [f"{l} ({x[l]:.2f})" for l in np.nonzero(e2c < 0)[0]],
              "shared_spread_layers": [int(l) for l in np.nonzero((cons < 0) & (e2c < 0))[0]], "shared_conc_layers": [int(l) for l in np.nonzero((cons > 0) & (e2c > 0))[0]],
              "E2b_n_spread": int((cons < 0).sum()), "E2b_n_conc": int((cons > 0).sum()),
              "E2b_null_spread_mean": float(nc[:, 1].mean()), "E2b_null_conc_mean": float(nc[:, 0].mean()),
              "E2_n_spread": int((e2c < 0).sum()), "E2_n_conc": int((e2c > 0).sum()),
              "d4_per_task_E2b": dict(zip(T, zb[:, 4].round(2).tolist())), "d4_per_task_E2": dict(zip(T, za[:, 4].round(2).tolist())),
              "min_decile_E2b": int(zb.mean(0).argmin()), "min_decile_E2": int(za.mean(0).argmin()),
              "taskavg_E2b": zb.mean(0).round(2).tolist(), "taskavg_E2": za.mean(0).round(2).tolist(),
              "sig_layers_per_task_E2b": {t: {"spread": int(((fam[t]["sign"] < 0) & (fam[t]["p"] < .05)).sum()), "conc": int(((fam[t]["sign"] > 0) & (fam[t]["p"] < .05)).sum())} for t in T},
              "sig_layers_per_task_E2": {t: {"spread": CQ[m]["families"][t]["n_sig_spread"], "conc": CQ[m]["families"][t]["n_sig_conc"]} for t in T}}
tot = {k: sum(out[m][k] for m in M8) for k in ("E2b_n_spread", "E2b_n_conc", "E2b_null_spread_mean", "E2b_null_conc_mean", "E2_n_spread", "E2_n_conc")}
tot.update({"E2_null_spread_mean_8": sum(CQ[m]["null_spread_mean"] for m in M8), "E2_null_conc_mean_8": sum(CQ[m]["null_conc_mean"] for m in M8)})
out["_totals"] = tot
json.dump(out, open(f"{OUT}/e2b_direct_layers.json", "w"), indent=1, default=float)
for m in M8:
    o = out[m]
    print(f"{m:28s} E2b spread {o['E2b_spread']} conc {o['E2b_conc']}\n{'':28s} E2  spread {o['E2_spread']} conc {o['E2_conc']}\n{'':28s} shared spread {o['shared_spread_layers']} conc {o['shared_conc_layers']} | min decile E2b {o['min_decile_E2b']} E2 {o['min_decile_E2']}\n{'':28s} d4 E2b {o['d4_per_task_E2b']}  E2 {o['d4_per_task_E2']}\n{'':28s} taskavg E2b {o['taskavg_E2b']}\n{'':28s} taskavg E2  {o['taskavg_E2']}")
print(tot)
