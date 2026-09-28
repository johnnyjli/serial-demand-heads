import json, numpy as np, sys
T = ["pointer", "arith", "binding", "relational"]
def consistent(M):
    pos, neg = (M > 0).sum(0), (M < 0).sum(0)
    return ((pos >= 3) & (neg == 0)).astype(int) - ((neg >= 3) & (pos == 0)).astype(int)
R=[r for r in json.load(open('consist.json')) if r['stat']=='q25']
Ms=[(r['L'],np.stack([np.sign(r["families"][f]["rel"]) * (np.array(r["families"][f]["p"]) < .05) for f in T])) for r in R]
for seed in (1,12345,7):
    NB=20000; rng=np.random.default_rng(seed); Sp=np.zeros(NB); Sc=np.zeros(NB)
    for L,M in Ms:
        # vectorized shifts
        sh=rng.integers(L,size=(NB,4))
        idx=(np.arange(L)[None,None,:]-sh[:,:,None])%L  # roll: out[j]=M[(j-s)%L]
        Mr=np.take_along_axis(np.broadcast_to(M,(NB,4,L)),idx,axis=2)
        pos=(Mr>0).sum(1); neg=(Mr<0).sum(1)
        Sp+=((neg>=3)&(pos==0)).sum(1); Sc+=((pos>=3)&(neg==0)).sum(1)
    print(seed,'spread null',Sp.mean(),'sd',Sp.std(),'p(>=64)',(np.sum(Sp>=64)+1)/(NB+1),'conc null',Sc.mean(),'p(>=13)',(np.sum(Sc>=13)+1)/(NB+1))
