import json, numpy as np, sys
R=json.load(open(sys.argv[1]))
a=lambda k,s=None: np.array([r[k] for r in R])[s] if s is not None else np.array([r[k] for r in R])
bY=a("bY_back"); bF=a("bF_back"); bI=a("bI_back"); fam=a("fam")
print("n cells", len(R), "check bY=bF+bI max abs err", np.max(np.abs(bY-bF-bI)))
sp=bY<0
print("spreading cells", sp.sum())
fr=bF[sp]/bY[sp]
print("ALL: median frac %.3f pooled %.3f  mean-of-ratios %.3f  IQR %s" % (np.median(fr), bF[sp].sum()/bY[sp].sum(), fr.mean(), np.round(np.quantile(fr,[.25,.75]),2)))
print("ALL cells (not only spreading): pooled %.3f" % (bF.sum()/bY.sum()))
print("frac spreading cells with fixed-set part >= 50%%: %.2f; with fixed-set part <=0: %.2f" % ((fr>=.5).mean(), (fr<=0).mean()))
for t in ["pointer","arith","binding","relational"]:
    s=sp&(fam==t); f=bF[s]/bY[s]
    print(f"{t:10s} n_sp {s.sum():2d} pooled {bF[s].sum()/bY[s].sum():+.2f} median {np.median(f):+.2f}  sum bY {bY[s].sum():+.4f} sum bF {bF[s].sum():+.4f} sum bI {bI[s].sum():+.4f}; cells bF<0 {np.mean(bF[s]<0):.2f}")
# model-cluster bootstrap for pooled fraction
rng=np.random.default_rng(0); models=a("model"); ms=np.unique(models)
bs=[]
for _ in range(4000):
    pick=rng.choice(ms,len(ms)); idx=np.concatenate([np.nonzero((models==m)&sp)[0] for m in pick])
    bs.append(bF[idx].sum()/bY[idx].sum())
print("pooled frac model-boot CI", np.round(np.quantile(bs,[.025,.975]),2))
