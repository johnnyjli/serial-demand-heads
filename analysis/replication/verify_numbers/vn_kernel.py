"""E2 Qwen2.5-7B-Instruct: share of generated outputs not starting with a digit, eager vs SDPA, serial families only and incl. numeral control."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import glob, os, re, json, numpy as np
R = str(paths.E2_SWEEP)
out = {}
for attn in ("eager", "sdpa"):
    fs = sorted(glob.glob(f"{R}/bf16_{attn}/Qwen_Qwen2.5-7B-Instruct__*.npz"))
    per = {}
    for f in fs:
        fam = os.path.basename(f).split("__")[1].split("_w")[0]
        its = json.loads(str(np.load(f)["meta"]))["items"]
        per.setdefault(fam, []).extend([not re.match(r"\s*\d", it["gen_text"] or "") for it in its])
    out[attn] = {"serial4": float(np.mean(sum([per[f] for f in ("pointer", "arith", "binding", "relational")], []))),
                 "all_incl_control": float(np.mean(sum(per.values(), []))), "per_family": {k: float(np.mean(v)) for k, v in per.items()}}
print(json.dumps(out, indent=1)); json.dump(out, open("vn_kernel.json", "w"), indent=1)
