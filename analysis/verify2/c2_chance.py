import sys, json, numpy as np, pandas as pd
SP = sys.argv[1]; sys.path.insert(0, SP)
from model_diff import ORDER, T, from_consist, from_cells
rng = np.random.default_rng(12345)
def parts(Z):
    mu = Z.mean((0,1)); a = Z.mean(1)-mu; b = Z.mean(0)-mu; e = Z-mu-a[:,None]-b[None]
    nM,nT = Z.shape[:2]
    return nT*(a**2).sum(), nM*(b**2).sum(), (e**2).sum()
def chance(Z, NB=3000):
    m,t,r = parts(Z); ns = m+t+r
    null=[]
    for _ in range(NB):
        Zp = np.stack([Z[rng.permutation(Z.shape[0]), j] for j in range(Z.shape[1])],1)
        null.append(parts(Zp)[0])
    nM=Z.shape[0]
    analytic = (1/Z.shape[1])*(m+r)/ns   # E[model SS] under within-task shuffle = (model+resid)/nT
    return m/ns, t/ns, np.mean(null)/ns, analytic
rows = {}
for st in ["q10","q25","q50","entropy","top1"]: rows[st]=chance(from_consist(st))
rows["sink"]=chance(from_cells("eager","sink_head")[0])
rows["perword"]=chance(from_cells("eager","all")[0])
ms=[m for m in ORDER if from_cells("sdpa","all",[m])[0] is not None]
rows["sdpa(%d)"%len(ms)]=chance(from_cells("sdpa","all",ms)[0])
A = pd.read_csv(f"{SP}/vet3/altexp_layers.csv"); A["dec"] = np.minimum((A.x * 10).astype(int), 9)
for fit in ["k+ent","k+ent+corr"]:
    S = A[A.fit == fit].groupby(["model","task","w","dec"]).t.mean().groupby(["model","task","dec"]).mean()
    rows[fit]=chance(np.array([[[S[(m,t,d)] for d in range(10)] for t in T] for m in ORDER]))
for k,v in rows.items(): print(f"{k:12s} model {v[0]:.3f} task {v[1]:.3f} chance(perm) {v[2]:.3f} chance(analytic) {v[3]:.3f}")
