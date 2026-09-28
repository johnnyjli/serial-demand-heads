"""E2 (same 8 models) context for per-task claims; exact p for the final-layer family pattern in E2b."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np, itertools
SP8 = str(paths.ANALYSIS)
T = ["pointer", "arith", "binding", "relational"]
M8 = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
      "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
CQ = {r["model"]: r for r in json.load(open(f"{SP8}/consist.json")) if r["stat"] == "q10"}
H = json.load(open("completeness/completeness_head_checks.json"))
out = {}
bk = {}
for m in M8:
    L = len(CQ[m]["families"]["pointer"]["z"]); x = np.arange(L) / (L - 1)
    for t in T: bk[(m, t)] = float(np.mean(np.array(CQ[m]["families"][t]["z"])[x >= .5]))
out["E2_same8_arith_weakest"] = f"{sum(max(T, key=lambda t: bk[(m, t)]) == 'arith' for m in M8)}/8"
out["E2_same8_binding_more_than_relational"] = f"{sum(bk[(m,'binding')] < bk[(m,'relational')] for m in M8)}/8"
out["E2_same8_per_task_back_mean"] = {t: round(float(np.mean([bk[(m, t)] for m in M8])), 2) for t in T}
out["E2_same8_weakest"] = {m: max(T, key=lambda t: bk[(m, t)]) for m in M8}
fl = H["final_layer_taskmean_z_E2b"]
neg = {m for m, v in fl.items() if v < 0}; qwen = {m for m in M8 if m.startswith("Qwen")}
out["final_layer_E2b_neg_models"] = sorted(neg)
out["final_layer_neg_equals_qwen"] = neg == qwen
out["final_layer_label_perm_p"] = 1 / len(list(itertools.combinations(range(8), 3)))
# per task: how many of the 32 cells follow the family sign
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
print(json.dumps(out, indent=1))
