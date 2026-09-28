import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, sys, json, numpy as np
SP = str(paths.ANALYSIS)
os.environ["VPRIM"] = "y10"
sys.path.insert(0, SP); sys.path.insert(0, f"{SP}/vet3")
import model_diff as MDm
MDm.SP = SP
from model_diff import ORDER, T, LIN, from_consist, dec_prof
Z = from_consist("q10")   # 17x4x10
LINS = sorted(set(LIN.values()), key=lambda x: ORDER.index(x))
def lin_mean(X): return np.array([np.mean([X[ORDER.index(m)] for m in ORDER if LIN[m]==L],0) for L in LINS])
Za = Z.mean(1)  # 17x10
Zl = lin_mean(Za)
print("d40-50>d50-60 models", (Za[:,4]>Za[:,5]).sum(), "lineages", (Zl[:,4]>Zl[:,5]).sum())
print("d40-50>0 and d50-60<0 models", ((Za[:,4]>0)&(Za[:,5]<0)).sum(), "lineages", ((Zl[:,4]>0)&(Zl[:,5]<0)).sum())
print(" fail:", [m for m,a,b in zip(ORDER,Za[:,4],Za[:,5]) if not (a>0 and b<0)])
print("d40-50>0 models", (Za[:,4]>0).sum(), "d50-60<0 models", (Za[:,5]<0).sum())
print("adjacent d_i>d_i+1 models", [(Za[:,i]>Za[:,i+1]).sum() for i in range(9)])
print("adjacent lineages", [(Zl[:,i]>Zl[:,i+1]).sum() for i in range(9)])
print("adjacent d_i<d_i+1 models (rising)", [(Za[:,i]<Za[:,i+1]).sum() for i in range(9)])
# largest drop location
drops = Za[:,:-1]-Za[:,1:]
print("argmax drop at boundary 4 (40-50|50-60):", (drops.argmax(1)==4).sum())
for m,row in zip(ORDER,Za): print(f"{m:32s}", " ".join(f"{v:+.2f}" for v in row))
# PC1
MD = json.load(open(f"{SP}/model_diff.json"))
PC1 = {m: MD["pca"]["scores"][m][0] for m in ORDER}
print("PC1", {m: round(v,2) for m,v in PC1.items()})
pairs=[("Llama-3.2-3B","Llama-3.2-3B-Instruct"),("Llama-3.1-8B","Llama-3.1-8B-Instruct"),("Qwen2.5-7B","Qwen2.5-7B-Instruct"),
       ("OLMo-2-1124-7B","OLMo-2-1124-7B-SFT"),("OLMo-2-1124-7B","OLMo-2-1124-7B-DPO"),("OLMo-2-1124-7B","OLMo-2-1124-7B-Instruct")]
print("PC1 diffs", [round(PC1[b]-PC1[a],2) for a,b in pairs])
# back half layer-averaged from consist
C = {r["model"]: r for r in json.load(open(f"{SP}/consist.json")) if r["stat"] == "q10"}
def back(m):
    vals=[]
    for t in T:
        z=np.array(C[m]["families"][t]["z"]); L=len(z); x=np.arange(L)/(L-1)
        vals.append(z[x>=0.5].mean())
    return np.mean(vals)
print("back diffs", [round(back(b)-back(a),3) for a,b in pairs])
print("back values", {m: round(back(m),2) for m in ORDER})
