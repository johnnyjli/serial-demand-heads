import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, os, numpy as np
SP=str(paths.ANALYSIS)
sys.path.insert(0, SP); sys.argv=[None, SP]
import backhalf as bh
from e2lib import ORDER, load
from concurrent.futures import ProcessPoolExecutor
QI="Qwen2.5-7B-Instruct"
def stat(S,name):
    P=S/S.sum(-1,keepdims=True); H=S.shape[-1]
    if name.startswith("q"):
        m=max(1,int(np.ceil(int(name[1:])/100*H))); return -np.sort(-P,-1)[...,:m].sum(-1)
    if name=="top1": return P.max(-1)
    if name=="neff": return -np.exp(-(P*np.log(np.maximum(P,1e-300))).sum(-1))/H
    if name=="gini":
        s=np.sort(P,-1); i=np.arange(1,H+1); return ((2*i-H-1)*s).sum(-1)/(H-1)
def one(args):
    m,c=args; attn="sdpa" if m==QI else "eager"
    d=load(m,c,attn); S=d["contribution"].astype(float); k=d["k"].astype(float); L=d["L"]
    body=np.array([int(t["id"].split("/")[-1]) for t in d["items"]]); x=np.arange(L)/(L-1)
    out={}
    for n in ["q10","q25","q50","top1","neff","gini"]:
        seed=abs(hash((m,c,attn,"all")))%(2**32)
        z=bh.zprof(stat(S,n),k,body,c.split("/")[0],np.random.default_rng(seed))
        out[n]=(float(z[x<.5].mean()),float(z[x>=.5].mean()))
    return out
if __name__=="__main__":
    jobs=[(m,f"{f}/{w}") for m in ORDER for f in ["pointer","arith","binding","relational"] for w in ("w0","w1")]
    with ProcessPoolExecutor(10) as ex: R=list(ex.map(one,jobs))
    for n in ["q10","q25","q50","top1","neff","gini"]:
        a=np.array([r[n] for r in R]); print(n, len(a), round(100*np.mean(a[:,1]<0),1), round(100*np.mean(a[:,1]<a[:,0]),1))
