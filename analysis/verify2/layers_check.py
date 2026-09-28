import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np
SP=str(paths.ANALYSIS)
T=["pointer","arith","binding","relational"]
C={r["model"]:r for r in json.load(open(f"{SP}/consist.json")) if r["stat"]=="q10"}
for m in ["OLMo-2-1124-7B","OLMo-2-1124-7B-SFT","OLMo-2-1124-7B-DPO","OLMo-2-1124-7B-Instruct","Llama-3.1-8B-Instruct","Llama-3.1-8B","Mistral-7B-v0.1","Llama-3.2-1B","Qwen2.5-7B-Instruct"]:
    r=C[m]; L=r["L"]; x=np.arange(L)/(L-1)
    z=np.array([r["families"][f]["z"] for f in T]); p=np.array([r["families"][f]["p"] for f in T])
    zm=z.mean(0)
    print(m, "L",L)
    print("  layers mean z<=-2:", [(l, round(x[l],2), round(zm[l],1)) for l in range(L) if zm[l]<=-2])
    print("  layers mean z>=+2:", [(l, round(x[l],2), round(zm[l],1)) for l in range(L) if zm[l]>=2])
    back=x>=.5; neg=np.clip(zm[back],None,0)
    srt=np.sort(neg); 
    print(f"  back-half neg mass {neg.sum():.1f}; top4 layers share {srt[:4].sum()/neg.sum():.2f}; nlayers<=-1 in back {int((zm[back]<=-1).sum())}/{int(back.sum())}")
    nsig=(np.sign(z)*(p<.05)<0).sum(0)
    print("  #tasks sig spread per layer (back half):", {int(l):int(nsig[l]) for l in range(L) if back[l] and nsig[l]>0})
