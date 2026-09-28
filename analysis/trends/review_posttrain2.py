"""Reviewer checks part 2: (a) C5 entropy adjustment in coefficient units vs t-units and k-ent collinearity;
(b) Qwen pointer/w0 kernel cells recomputed from the cache; (c) numeral-control accuracy/entropy vs k."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, sys, json, itertools, numpy as np, pandas as pd
os.environ["VPRIM"] = "y10"
SP = str(paths.ANALYSIS)
sys.path.insert(0, f"{SP}/vet3"); from v2lib import load as vload, zprof
T = ["pointer", "arith", "binding", "relational"]
PAIRS = [("Llama-3.2-3B", "Llama-3.2-3B-Instruct"), ("Llama-3.1-8B", "Llama-3.1-8B-Instruct"),
         ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT"),
         ("OLMo-2-1124-7B", "OLMo-2-1124-7B-DPO"), ("OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct")]
LIN = [[0], [1], [2], [3, 4, 5]]
def lw(v): v = np.asarray(v, float); return float(np.mean([v[i].mean() for i in LIN]))
O = {}
A = pd.read_csv(f"{SP}/vet3/altexp_layers.csv")
for fams, lab in ((T, "serial"), (["fixed_digit"], "control")):
    S = A[A.task.isin(fams) & (A.x >= .5)]
    g = S.groupby(["model", "fit", "task", "w"])[["b_pp", "t"]].mean().groupby(["model", "fit"]).mean()
    res = {}
    for fit in ("k", "k+ent", "k+ent+corr"):
        db = [g.loc[(p, fit), "b_pp"] - g.loc[(b, fit), "b_pp"] for b, p in PAIRS]
        dt = [g.loc[(p, fit), "t"] - g.loc[(b, fit), "t"] for b, p in PAIRS]
        res[fit] = {"b_pp_diff": np.round(db, 4).tolist(), "b_LW": round(lw(db), 4), "t_diff": np.round(dt, 2).tolist(), "t_LW": round(lw(dt), 3)}
    res["b_ratio_adj_over_k"] = round(res["k+ent"]["b_LW"] / res["k"]["b_LW"], 3)
    res["t_ratio_adj_over_k"] = round(res["k+ent"]["t_LW"] / res["k"]["t_LW"], 3)
    O[f"adj_{lab}"] = res
# k-entropy collinearity within (model, task, wording), demeaned within body for binding/relational
def corr_ke(m, f, w):
    d = vload(m, f, w, "eager", True)
    k, e, body = d["k"], d["ent"], d["body"]
    if f in ("binding", "relational"):
        for gg in np.unique(body):
            i = body == gg; k = k.copy(); e = e.copy(); k[i] -= k[i].mean(); e[i] -= e[i].mean()
    return float(np.corrcoef(k, e)[0, 1])
ms = sorted({m for pr in PAIRS for m in pr})
O["corr_k_ent"] = {lab: {m: round(np.mean([corr_ke(m, f, w) for f in fams for w in ("w0", "w1")]), 2) for m in ms}
                   for fams, lab in ((T, "serial"), (["fixed_digit"], "control"))}
# (b) Qwen pointer/w0 kernel from cache
def bz(m, attn):
    d = vload(m, "pointer", "w0", attn, True); z = zprof(d["y10"], d["k"])[2]; x = d["x"]; return float(z[x >= .5].mean())
q = {f"{m}_{a}": round(bz(m, a), 2) for m in ("Qwen2.5-7B", "Qwen2.5-7B-Instruct") for a in ("eager", "sdpa")}
q["diff_sdpa_sdpa"] = round(q["Qwen2.5-7B-Instruct_sdpa"] - q["Qwen2.5-7B_sdpa"], 2)
q["diff_eager_eager"] = round(q["Qwen2.5-7B-Instruct_eager"] - q["Qwen2.5-7B_eager"], 2)
q["diff_mixed"] = round(q["Qwen2.5-7B-Instruct_sdpa"] - q["Qwen2.5-7B_eager"], 2)
O["qwen_pointer_w0"] = q
# (c) control: accuracy and answer entropy by k, base vs post
cc = {}
for m in ms:
    rows = []
    for w in ("w0", "w1"):
        d = vload(m, "fixed_digit", w, "sdpa" if m == "Qwen2.5-7B-Instruct" else "eager", True)
        for kk in np.unique(d["k"]):
            i = d["k"] == kk; rows.append((w, int(kk), d["correct"][i].mean(), d["ent"][i].mean()))
    df = pd.DataFrame(rows, columns=["w", "k", "acc", "ent"]).groupby("k")[["acc", "ent"]].mean()
    cc[m] = {"acc_by_k": df.acc.round(2).tolist(), "ent_by_k": df.ent.round(2).tolist()}
O["control_acc_ent_by_k"] = cc
json.dump(O, open(f"{SP}/trends/review_posttrain2.json", "w"), indent=1, default=float)
for k, v in O.items(): print(k, json.dumps(v, default=float))
