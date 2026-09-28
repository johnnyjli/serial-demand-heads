import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np
SP=str(paths.ANALYSIS)
d=json.load(open(f"{SP}/model_diff.json")); p=d['primary']; T=["pointer","arith","binding","relational"]
M=list(p['Z']); Z=np.array([[p['Z'][m][t] for t in T] for m in M]); rng=np.random.default_rng(1)
def pm_split(Z,i):
    out=[]
    for S in [(0,1),(0,2),(0,3)]:
        Sc=[j for j in range(4) if j not in S]
        A1=Z[:,list(S)].mean(1); A1=A1-A1.mean(0); A2=Z[:,Sc].mean(1); A2=A2-A2.mean(0)
        out.append(np.corrcoef(A1[i],A2[i])[0,1])
    return np.mean(out)
for m in ["Qwen2.5-0.5B","Qwen2.5-1.5B","Qwen2.5-3B","Qwen2.5-7B","OLMo-2-1124-7B","Mistral-7B-v0.1","Llama-3.2-1B"]:
    i=M.index(m); obs=pm_split(Z,i)
    # null A: shuffle this model's deciles independently within each task (keeps amplitude, destroys own recurring shape)
    nA=[]
    for _ in range(2000):
        Zp=Z.copy(); Zp[i]=np.stack([rng.permutation(Z[i,t]) for t in range(4)]); nA.append(pm_split(Zp,i))
    # null B: flat model
    Zf=Z.copy(); Zf[i]=0; flat=pm_split(Zf,i)
    nA=np.array(nA)
    print(f"{m:28s} obs {obs:.2f} | shuffled-decile null mean {nA.mean():.2f} (p>=obs {np.mean(nA>=obs):.2f}) | flat-profile {flat:.2f}")
