import json, itertools, numpy as np
T=["pointer","arith","binding","relational"]
C={r['model']:r for r in json.load(open('consist.json')) if r['stat']=='q10'}
ORDER=list(json.load(open('model_diff.json'))['primary']['Z'].keys())
print({m:C[m].get('attn') for m in ORDER})
def dec(z):
    z=np.asarray(z,float); L=len(z); x=np.arange(L)/(L-1); d=np.minimum((x*10).astype(int),9)
    return np.array([z[d==i].mean() for i in range(10)])
Z=np.array([[dec(C[m]['families'][t]['z']) for t in T] for m in ORDER])
mu=Z.mean((0,1)); a=Z.mean(1)-mu; b=Z.mean(0)-mu; e=Z-mu-a[:,None]-b[None]
S=dict(shared=68*(mu**2).sum(), model=4*(a**2).sum(), task=17*(b**2).sum(), resid=(e**2).sum())
tot=sum(S.values()); ns=tot-S['shared']
print('total SS vs sum Z^2', tot, (Z**2).sum())
print({k:round(v/tot,4) for k,v in S.items()}, {k:round(S[k]/ns,4) for k in ('model','task','resid')})
# centered variance version
tc=((Z-Z.mean())**2).sum(); print('centered total', tc, 'shared frac (centered mu)', 68*((mu-mu.mean())**2).sum()/tc)
rng=np.random.default_rng(1); null=[]
for _ in range(2000):
    Zp=np.stack([Z[rng.permutation(17),j] for j in range(4)],1); ap=Zp.mean(1)-Zp.mean((0,1)); null.append(4*(ap**2).sum())
print('null model frac of nonshared', np.mean(null)/ns, 'p', (np.sum(np.array(null)>=S['model'])+1)/2001)
Zm=np.stack([Z[i].mean(0) for i in range(17)])
wr=[np.mean([np.corrcoef(Z[i,x],Z[i,y])[0,1] for x,y in itertools.combinations(range(4),2)]) for i in range(17)]
print('within', np.mean(wr))
U,s,Vt=np.linalg.svd(a-a.mean(0),full_matrices=False); ev=s**2/(s**2).sum(); print('PCA', np.round(ev[:3],4)); print(np.round(Vt[:2],2))
