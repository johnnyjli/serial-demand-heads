"""Benchmark for the E2 vs E2b agreement: agreement between E2's two wordings (w0 vs w1, per-wording exact z from
vet3/cells.json, Qwen2.5-7B-Instruct SDPA as in the paper) on the same 8 (and 6) models: per-cell decile-profile r,
signature r (pooled and per model), back-half mean z r.  Also per-model tie check of the E2b->E2 best match."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, numpy as np
SP8 = str(paths.ANALYSIS)
sys.path.insert(0, SP8)
from model_diff import dec_prof, decompose
T = ["pointer", "arith", "binding", "relational"]
M8 = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
      "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
M6 = [m for m in M8 if not m.startswith("Qwen2.5-7B")]
C = json.load(open(f"{SP8}/vet3/cells.json"))
D = {(r["model"], r["fam"], r["w"]): np.array(r["z"]) for r in C if r["variant"] == "all" and r["fam"] in T and r["attn"] == ("sdpa" if r["model"] == "Qwen2.5-7B-Instruct" else "eager")}
out = {}
for tag, models in (("M8", M8), ("M6_noQwen7B", M6)):
    Z0 = np.array([[dec_prof(D[(m, t, "w0")]) for t in T] for m in models]); Z1 = np.array([[dec_prof(D[(m, t, "w1")]) for t in T] for m in models])
    cell = [np.corrcoef(Z0[i, j], Z1[i, j])[0, 1] for i in range(len(models)) for j in range(4)]
    a0, a1 = decompose(Z0)[1], decompose(Z1)[1]
    b0, b1 = Z0[:, :, 5:].mean((1, 2)), Z1[:, :, 5:].mean((1, 2))
    out[tag] = {"per_cell_r_mean": float(np.mean(cell)), "per_cell_r_min": float(np.min(cell)), "per_cell_r_max": float(np.max(cell)),
                "signature_pooled_r": float(np.corrcoef(a0.ravel(), a1.ravel())[0, 1]),
                "signature_per_model_r": {m: float(np.corrcoef(a0[i], a1[i])[0, 1]) for i, m in enumerate(models)},
                "signature_per_model_r_mean": float(np.mean([np.corrcoef(a0[i], a1[i])[0, 1] for i in range(len(models))])),
                "back_mean_z_r": float(np.corrcoef(b0, b1)[0, 1])}
R = json.load(open("e2b_direct_replication.json")); X = np.array(R["M8"]["3_signatures"]["X"])
out["tie_check_Llama-3.1-8B_row"] = X[0].tolist()
json.dump(out, open("e2b_direct_wording_benchmark.json", "w"), indent=1)
print(json.dumps(out, indent=1))
