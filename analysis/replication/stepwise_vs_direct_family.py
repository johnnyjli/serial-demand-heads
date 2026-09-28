"""Per-family stepwise-minus-direct differences in back-half mean z and slope (top-10% share), for all 8 models,
without the Qwen2.5-7B pair, and for the 4 models without Qwen2.5-7B or OLMo-2; plus slope differences without
OLMo-2-Instruct.  Writes stepwise_vs_direct_family.json."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np
from scipy import stats as sst
SPD = str(paths.ANALYSIS / "replication")
C = json.load(open(f"{SPD}/stepwise_vs_direct.json"))["cells"]
M = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B", "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
F = ["pointer", "arith", "binding", "relational"]
sets = {"all8": M, "no_qwen7": [m for m in M if "Qwen2.5-7B" not in m], "clean4": [m for m in M if "Qwen2.5-7B" not in m and "OLMo" not in m],
        "no_olmo_instruct": [m for m in M if m != "OLMo-2-1124-7B-Instruct"]}
out = {}
for sn, ms in sets.items():
    for sub in ("all", "correct"):
        for f in F:
            d = [C[f"q10|direct|{sub}|{m}|{f}"] for m in ms]; s = [C[f"q10|stepwise|{sub}|{m}|{f}"] for m in ms]
            dz = np.array([b["back"] - a["back"] for a, b in zip(d, s)])
            out[f"{sn}|{sub}|{f}"] = {"direct_mean_back": float(np.mean([a["back"] for a in d])), "step_mean_back": float(np.mean([b["back"] for b in s])),
                                      "direct_n_neg": int(sum(a["back"] < 0 for a in d)), "step_n_neg": int(sum(b["back"] < 0 for b in s)),
                                      "n_step_higher": int((dz > 0).sum()), "n": len(dz), "mean_diff": float(dz.mean()),
                                      "sign_p": float(sst.binomtest(int((dz > 0).sum()), len(dz)).pvalue)}
        dp = np.array([C[f"q10|stepwise|{sub}|{m}|{f}"]["back_slope_pp"] - C[f"q10|direct|{sub}|{m}|{f}"]["back_slope_pp"] for m in ms for f in F])
        out[f"{sn}|{sub}|slope_pp"] = {"n": len(dp), "n_step_higher": int((dp > 0).sum()), "mean": float(dp.mean()), "median": float(np.median(dp)),
                                      "sign_p": float(sst.binomtest(int((dp > 0).sum()), len(dp)).pvalue)}
json.dump(out, open(f"{SPD}/stepwise_vs_direct_family.json", "w"), indent=1)
for k, v in out.items(): print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
