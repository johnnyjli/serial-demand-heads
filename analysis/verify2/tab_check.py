import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, json, numpy as np
SP=str(paths.ANALYSIS)
sys.path.insert(0, SP); import e2lib
from e2lib import ORDER, FAMS
def acc(m, c, attn, clean):
    if clean: os.environ["E2_CLEAN"]="1"
    else: os.environ.pop("E2_CLEAN", None)
    d = e2lib.load(m, c, attn); k = d["k"]; ok = d["correct"].astype(bool)
    return [float(ok[k==kk].mean()) for kk in sorted(set(k))]
D = {(r["model"], r["cell"]): r for r in json.load(open(f"{SP}/decomp_q10.json"))}
for name, fn in [("sdpa_all", lambda m,c: acc(m,c,"sdpa" if m=="Qwen2.5-7B-Instruct" else "eager", False)),
                 ("eager_all", lambda m,c: acc(m,c,"eager", False)),
                 ("decomp_q10 (fig01)", lambda m,c: D[(m,c)]["acc"])]:
    print(name)
    for f in FAMS:
        v = np.mean([np.mean([fn(m, f"{f}/{w}") for w in ("w0","w1")],0) for m in ORDER],0)
        print(f"  {f:12s}", " ".join(f"{x:.2f}" for x in v))
