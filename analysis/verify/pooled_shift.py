import json, numpy as np
T = ["pointer", "arith", "binding", "relational"]; NB = 5000
def consistent(M):
    pos, neg = (M > 0).sum(0), (M < 0).sum(0)
    return ((pos >= 3) & (neg == 0)).astype(int) - ((neg >= 3) & (pos == 0)).astype(int)
R=[r for r in json.load(open('consist.json')) if r['stat']=='q25']
Sp=np.zeros(NB); Sc=np.zeros(NB); Sp2=np.zeros(NB); Sc2=np.zeros(NB); obs_s=obs_c=0
for r in R:
    L=r['L']; rng=np.random.default_rng(0)
    M = np.stack([np.sign(r["families"][f]["rel"]) * (np.array(r["families"][f]["p"]) < .05) for f in T])
    c=consistent(M); obs_s+=(c<0).sum(); obs_c+=(c>0).sum()
    for b in range(NB):
        cc=consistent(np.stack([np.roll(M[i], rng.integers(L)) for i in range(4)]))
        Sp[b]+=(cc<0).sum(); Sc[b]+=(cc>0).sum()
    for b in range(NB):
        cc=consistent(np.stack([M[i][rng.permutation(L)] for i in range(4)]))
        Sp2[b]+=(cc<0).sum(); Sc2[b]+=(cc>0).sum()
print('obs spread',obs_s,'conc',obs_c)
print('shift null mean spread',Sp.mean(),'p',(np.sum(Sp>=obs_s)+1)/(NB+1),'ratio',obs_s/Sp.mean())
print('shift null mean conc',Sc.mean(),'p',(np.sum(Sc>=obs_c)+1)/(NB+1),'ratio',obs_c/Sc.mean())
print('perm null mean spread',Sp2.mean(),'p',(np.sum(Sp2>=obs_s)+1)/(NB+1))
print('perm null mean conc',Sc2.mean(),'p',(np.sum(Sc2>=obs_c)+1)/(NB+1))
