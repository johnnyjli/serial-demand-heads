import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, json, numpy as np
SP=str(paths.ANALYSIS)
sys.path.insert(0, SP)
import e2lib
from e2lib import ORDER, CELLS, FAMS
def acc(m, c, attn="eager", clean=False):
    if clean: os.environ["E2_CLEAN"]="1"
    else: os.environ.pop("E2_CLEAN", None)
    d = e2lib.load(m, c, attn); k = d["k"]; ok = d["correct"].astype(bool)
    ks = sorted(set(k)); return [float(ok[k==kk].mean()) for kk in ks], [int((k==kk).sum()) for kk in ks]
D = {(r["model"], r["cell"]): r for r in json.load(open(f"{SP}/decomp_q10.json"))}
# 1. Does decomp_q10 match eager-all, eager-clean for each model?
mism = {}
for m in ORDER:
    for c in CELLS:
        a_all, n_all = acc(m, c, "eager", False)
        dq = D[(m,c)]["acc"]
        if not np.allclose(a_all, dq):
            a_cl, n_cl = acc(m, c, "eager", True)
            mism[(m,c)] = (np.round(dq,3).tolist(), np.round(a_all,3).tolist(), np.round(a_cl,3).tolist(), n_all, n_cl, np.allclose(a_cl, dq))
print("n per depth example", acc("Llama-3.2-1B","pointer/w0")[1], acc("Llama-3.2-1B","arith/w0")[1])
print("mismatches decomp_q10 vs eager-all:", len(mism))
for k,v in mism.items(): print(k, "decomp", v[0], "all", v[1], "clean", v[2], "n_all", v[3], "n_clean", v[4], "matches clean", v[5])
# Qwen2.5-7B-Instruct eager vs sdpa
m="Qwen2.5-7B-Instruct"
for c in CELLS:
    try:
        e,_=acc(m,c,"eager"); s,_=acc(m,c,"sdpa")
        print(c, "eager", np.round(e,2), "sdpa", np.round(s,2))
    except KeyError as ex: print(c, "missing", ex)
