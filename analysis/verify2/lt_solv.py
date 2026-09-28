import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, os, numpy as np
os.environ['E2_STAT']='q10'
SP=str(paths.ANALYSIS)
sys.argv=[None,SP]; sys.path.insert(0,SP)
from e2lib import *
import pairs
P=json.load(open(f'{SP}/pairs_q10.json')); ref={(r['base'],r['post'],r['cell']):r for r in P}
res=[]
for a,b in pairs.PAIRS:
    for c in CELLS:
        A,B=load(a,c),load(b,c)
        ia,ib=[t["id"] for t in A["items"]],[t["id"] for t in B["items"]]
        common=sorted(set(ia)&set(ib),key=ia.index)
        def sub(X,ids):
            pos={i:j for j,i in enumerate(ids)}; sel=np.array([pos[i] for i in common]); n=len(ids)
            for key,v in list(X.items()):
                if isinstance(v,np.ndarray) and v.ndim>=1 and v.shape[0]==n: X[key]=v[sel]
        sub(A,ia); sub(B,ib)
        k=A["k"].astype(float); ya,yb=pairs.ystat(A,"mean"),pairs.ystat(B,"mean")
        d=yb/np.nanmean(yb)-ya/np.nanmean(ya)
        o_all=perm_p(d,k,1000)
        sel=A["correct"]|B["correct"]
        o=perm_p(d[sel],k[sel],1000) if len(set(k[sel]))>1 else (np.nan,np.nan)
        res.append((a,b,c,100*o_all[0],o_all[1],100*o[0],o[1],int(sel.sum())))
        r=ref[(a,b,c)]['mean_all']
        if abs(r[0]-100*o_all[0])>1e-6: print("MISMATCH",a,b,c,r,100*o_all[0])
ser=[x for x in res if not x[2].startswith('fixed')]; fx=[x for x in res if x[2].startswith('fixed')]
for nm,S in (('serial',ser),('fixed',fx)):
    print(nm,'either-correct: n',len(S),'neg',sum(x[5]<0 for x in S),'sig neg',sum(x[5]<0 and x[6]<.05 for x in S),'sig pos',sum(x[5]>0 and x[6]<.05 for x in S),'median n items',np.median([x[7] for x in S]))
