import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np, sys
SP=str(paths.ANALYSIS)
sys.path.insert(0,SP); sys.argv=[None,SP]
import model_diff as md
from model_diff import ORDER, LIN, decompose, from_consist, from_cells
M1=json.load(open(f'{SP}/model_diff.json'))
L0=np.array(M1['pca']['loadings'][0]); p0=np.array([M1['pca']['scores'][m][0] for m in ORDER])
a0=decompose(from_consist('q10'))[1]
print('check primary proj == scores:', np.allclose((a0-a0.mean(0))@L0, p0))
posts=[m for m in ORDER if LIN[m]!=m]
def rep(name,a):
    proj=(a-a.mean(0))@L0
    U,s,Vt=np.linalg.svd(a-a.mean(0),full_matrices=False); pc=(U*s)[:,0]
    if np.corrcoef(pc,p0)[0,1]<0: pc=-pc
    dp=[proj[ORDER.index(m)]-proj[ORDER.index(LIN[m])] for m in posts]
    do=[pc[ORDER.index(m)]-pc[ORDER.index(LIN[m])] for m in posts]
    print('%-14s primary-axis: '%name+' '.join('%+.2f'%x for x in dp)+' | own PC1: '+' '.join('%+.2f'%x for x in do))
print('order:',posts)
rep('q10',a0)
for st in ['q25','q50','entropy','top1']:
    rep(st, decompose(from_consist(st))[1])
for name,(attn,var) in [('sink',('eager','sink_head')),('perword',('eager','all'))]:
    Z,miss=from_cells(attn,var); rep(name, decompose(Z)[1])
