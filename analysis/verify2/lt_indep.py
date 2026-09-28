import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np, sys
from scipy.stats import spearmanr
SP=str(paths.ANALYSIS)
sys.path.insert(0,SP)
from model_diff import ORDER, LIN
M1=json.load(open(f'{SP}/model_diff.json')); M2=json.load(open(f'{SP}/model_diff2.json')); M3=json.load(open(f'{SP}/model_diff3.json'))
F=M2['features']; acc=[F['acc'][m] for m in ORDER]; acc1=[F['acc_k1'][m] for m in ORDER]
zb=[M1['per_model'][m]['back'] for m in ORDER]
print('unadj z back vs acc', spearmanr(zb,acc), 'vs acc_k1', spearmanr(zb,acc1))
p0=np.array([M1['pca']['scores'][m][0] for m in ORDER])
print('primary PC1 vs acc', spearmanr(p0,acc))
for fit in ['k','k+ent','k+ent+corr']:
    b=[M3[fit]['back_mean_t'][m] for m in ORDER]
    print(fit,'back t vs acc',[round(x,3) for x in spearmanr(b,acc)],'vs acc_k1',[round(x,3) for x in spearmanr(b,acc1)])
    a=np.array([M3[fit]['a'][m] for m in ORDER]); U,s,Vt=np.linalg.svd(a-a.mean(0),full_matrices=False); sc=U*s
    pc=sc[:,0]; 
    if np.corrcoef(pc,p0)[0,1]<0: pc=-pc
    print('   PC1 r with primary PC1 %.2f; expl %.2f; PC1 vs acc'%(np.corrcoef(pc,p0)[0,1], s[0]**2/(s**2).sum()), [round(x,4) for x in spearmanr(pc,acc)], 'vs acc_k1',[round(x,4) for x in spearmanr(pc,acc1)])
    # also PC1 of adjusted projected on primary loadings
    L0=np.array(M1['pca']['loadings'][0])
    proj=(a-a.mean(0))@L0
    print('   proj on primary PC1 axis vs acc',[round(x,4) for x in spearmanr(proj,acc)])
    for m in ORDER:
        if LIN[m]!=m:
            i,j=ORDER.index(m),ORDER.index(LIN[m])
            print('     %-26s back t diff %+.2f  PC1(own) diff %+.2f  proj-primary-axis diff %+.2f'%(m,b[i]-b[j],pc[i]-pc[j],proj[i]-proj[j]))
