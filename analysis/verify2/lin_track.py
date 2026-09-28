import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np, sys
from scipy.stats import spearmanr
sys.path.insert(0,'..')
SP=str(paths.ANALYSIS)
sys.path.insert(0,SP)
from model_diff import ORDER, LIN
M3=json.load(open(f'{SP}/model_diff3.json')); M2=json.load(open(f'{SP}/model_diff2.json')); M1=json.load(open(f'{SP}/model_diff.json'))
F=M2['features']
back_z=np.array([M1['per_model'][m]['back'] for m in ORDER])
for fit in ['k','k+ent','k+ent+corr']:
    b=np.array([M3[fit]['back_mean_t'][m] for m in ORDER])
    for f in ['acc','acc_k1','ent_slope','H']:
        r=spearmanr(b,[F[f][m] for m in ORDER]); print(fit,f,'rho %.3f p %.4f'%(r.correlation,r.pvalue))
    # PC1 of adjusted signatures
    a=np.array([M3[fit]['a'][m] for m in ORDER]); U,s,Vt=np.linalg.svd(a-a.mean(0),full_matrices=False); pc1=(U*s)[:,0]
    # orient sign to primary pc1
    p0=np.array([M1['pca']['scores'][m][0] for m in ORDER]); 
    if np.corrcoef(pc1,p0)[0,1]<0: pc1=-pc1
    r=spearmanr(pc1,[F['acc'][m] for m in ORDER]); print(fit,'PC1 vs acc rho %.3f p %.4f; r(pc1,pc1_primary)=%.2f'%(r.correlation,r.pvalue,np.corrcoef(pc1,p0)[0,1]))
    print('   post-base pc1:', {m:round(pc1[ORDER.index(m)]-pc1[ORDER.index(LIN[m])],2) for m in ORDER if LIN[m]!=m})
print()
print('%-28s %7s %7s %7s %7s'%('model','z_back','t_k','t_kent','t_kec'))
for m in ORDER:
    print('%-28s %+7.2f %+7.2f %+7.2f %+7.2f'%(m,M1['per_model'][m]['back'],M3['k']['back_mean_t'][m],M3['k+ent']['back_mean_t'][m],M3['k+ent+corr']['back_mean_t'][m]))
print('\npost minus base back-half:')
for m in ORDER:
    if LIN[m]!=m:
        b=LIN[m]
        print('%-28s z %+6.2f  t_k %+6.2f  t_kent %+6.2f  t_kec %+6.2f'%(m, M1['per_model'][m]['back']-M1['per_model'][b]['back'], *[M3[f]['back_mean_t'][m]-M3[f]['back_mean_t'][b] for f in ['k','k+ent','k+ent+corr']]))
# r between adjusted and unadjusted item-level signatures
a0=np.array([M3['k']['a'][m] for m in ORDER])
for fit in ['k+ent','k+ent+corr']:
    a=np.array([M3[fit]['a'][m] for m in ORDER]); print(fit,'r with fit k signature %.3f'%np.corrcoef(a.ravel(),a0.ravel())[0,1])
