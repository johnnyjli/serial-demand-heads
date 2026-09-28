import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, numpy as np
SP=str(paths.ANALYSIS)
sys.path.insert(0, f"{SP}/vet3"); os.environ["VPRIM"]="y10"
from v2lib import load, ORDER, TASKS
for m in ["Qwen2.5-7B","OLMo-2-1124-7B","Llama-3.1-8B","Qwen2.5-7B-Instruct","Qwen2.5-3B","Qwen2.5-1.5B"]:
    for attn in ("eager","sdpa"):
        a1=[]; cells=[]
        for t in TASKS:
            for w in ("w0","w1"):
                d=load(m,t,w,attn,True)
                if d is None: continue
                a1.append(d["correct"][d["k"]==1].mean()); cells.append(f"{t}/{w}")
                if t=="pointer" and w=="w0": print(m,attn,"pointer/w0 k1 acc",round(float(d["correct"][d["k"]==1].mean()),3), "all acc", round(float(d["correct"].mean()),3))
        if a1: print(m,attn,"mean acc_k1 over",len(a1),"cells",round(float(np.mean(a1)),4))
