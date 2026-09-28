import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np
SP=str(paths.ANALYSIS)
T=["pointer","arith","binding","relational"]
C=[r for r in json.load(open(f"{SP}/consist.json")) if r["stat"]=="q10"]
lit_fail=0; tot=0; lit_sign_only=0
for r in C:
    L=r["L"]; x=np.arange(L)/(L-1)
    z=np.array([r["families"][f]["z"] for f in T]); p=np.array([r["families"][f]["p"] for f in T])
    M=np.sign(z)*(p<.05)
    pos,neg=(M>0).sum(0),(M<0).sum(0)
    c=((pos>=3)&(neg==0)).astype(int)-((neg>=3)&(pos==0)).astype(int)
    conc=[f"{l} ({x[l]:.2f})" for l in np.nonzero(c>0)[0]]; spr=[f"{l} ({x[l]:.2f})" for l in np.nonzero(c<0)[0]]
    # literal reading: same sign (sig) in >=3, and no task with opposite sign z at all
    for l in np.nonzero(c)[0]:
        tot+=1
        if np.any(np.sign(z[:,l])==-c[l]): lit_fail+=1
    # sign-only reading (no significance): >=3 same sign, none opposite -> count
    S=np.sign(z); sp_,sn_=(S>0).sum(0),(S<0).sum(0)
    cs=((sp_>=3)&(sn_==0)).astype(int)-((sn_>=3)&(sp_==0)).astype(int)
    lit_sign_only+=int((cs!=0).sum())
    print(f"{r['model']:30s} L={L} conc: {', '.join(conc) or '--'} | spread: {', '.join(spr) or '--'}")
print("consistent layers", tot, "of which some task has nonsignificant opposite-sign z:", lit_fail)
print("layers under sign-only reading (>=3 same sign, 0 opposite sign, no significance):", lit_sign_only)
