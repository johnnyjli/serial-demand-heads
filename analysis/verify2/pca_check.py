import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np
SP=str(paths.ANALYSIS)
d=json.load(open(f"{SP}/model_diff.json")); p=d['primary']
M=list(p['a']); a=np.array([p['a'][m] for m in M])
ac=a-a.mean(0)
U,s,Vt=np.linalg.svd(ac,full_matrices=False); ev=s**2/(s**2).sum(); sc=U*s
print("explained", np.round(ev[:5],3), "json", np.round(d['pca']['explained'][:5],3))
for k in range(4): print(f"PC{k+1} loadings", np.round(Vt[k],2))
print("json loadings PC1", np.round(d['pca']['loadings'][0],2)); print("json PC2",np.round(d['pca']['loadings'][1],2)); print("json PC3",np.round(d['pca']['loadings'][2],2))
# correlation of scores with signature per decile
for dd in range(10):
    print(f"decile {dd*10}-{dd*10+10}: r(PC1) {np.corrcoef(sc[:,0],ac[:,dd])[0,1]:+.2f}  r(PC2) {np.corrcoef(sc[:,1],ac[:,dd])[0,1]:+.2f}  r(PC3) {np.corrcoef(sc[:,2],ac[:,dd])[0,1]:+.2f}  var share {ac[:,dd].var()/ac.var(0).sum():.2f}")
# variance of decile 4 (40-50) explained by PC1+PC2 and by PC3
for k in [1,2,3]:
    rec=sc[:,:k]@Vt[:k]
    print(f"first {k} PCs: R2 of 40-50% signature {1-((ac[:,4]-rec[:,4])**2).sum()/(ac[:,4]**2).sum():.2f}")
# regional feature correlations
f5070=ac[:,5:7].mean(1); f7080=ac[:,7]; f80100=ac[:,8:10].mean(1)
print("r(PC1, 50-70 sig)", np.corrcoef(sc[:,0],f5070)[0,1], "r(PC2, 70-80 minus 80-100)", np.corrcoef(sc[:,1],f7080-f80100)[0,1])
for i,m in enumerate(M): print(f"{m:30s} PC1 {sc[i,0]:+.2f} PC2 {sc[i,1]:+.2f} PC3 {sc[i,2]:+.2f} a40-50 {ac[i,4]:+.2f}")
