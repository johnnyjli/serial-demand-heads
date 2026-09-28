"""Per model, k>=3: stepwise vs direct on the same items (exact McNemar / binomial test on discordant pairs)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, json
import numpy as np
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(paths.REPLICATION))
import e2b
runs = e2b.load_results(str(paths.E2B))
R = {(r["model"].split("/")[-1], r["mode"], r["family"]): r for r in runs}
MODELS = ["Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B", "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
          "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
out = {}
for m in MODELS:
    for kset, sel in (("k>=3", lambda k: k >= 3), ("k=1", lambda k: k == 1)):
        b = c = n = 0
        for f in ("pointer", "arith", "binding", "relational"):
            for xd, xs in zip(R[(m, "direct", f)]["items"], R[(m, "stepwise", f)]["items"]):
                assert xd["id"] == xs["id"]
                if not sel(xd["k"]): continue
                n += 1; b += (xs["correct"] and not xd["correct"]); c += (xd["correct"] and not xs["correct"])
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        out.setdefault(m, {})[kset] = {"n": n, "stepwise_only": b, "direct_only": c, "p": p}
    print(f"{m:28s}", out[m])
json.dump(out, open(os.path.join(HERE, "e2b_mode_mcnemar.json"), "w"), indent=1)
