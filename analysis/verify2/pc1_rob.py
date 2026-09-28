import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np, sys
SP=str(paths.ANALYSIS)
sys.path.insert(0,SP)
from model_diff import ORDER, LIN, from_consist, from_cells, decompose
M1=json.load(open(f'{SP}/model_diff.json')); p0=np.array([M1['pca']['scores'][m][0] for m in ORDER])
def pc1(a):
    U,s,Vt=np.linalg.svd(a-a.mean(0),full_matrices=False); v=(U*s)[:,0]
    return v if np.corrcoef(v,p0)[0,1]>0 else -v
V={}
for st in ['q10','q25','q50','entropy','top1']: V[st]=decompose(from_consist(st))[1]
V['sink']=decompose(from_cells('eager','sink_head')[0])[1]; V['perword_eager']=decompose(from_cells('eager','all')[0])[1]
for k,a in V.items():
    p=pc1(a); print('%-14s r(pc1,primary)=%.2f '%(k,np.corrcoef(p,p0)[0,1]) + ' '.join('%s:%+.2f'%(m.replace('OLMo-2-1124-7B','O').replace('Llama-','L').replace('Qwen2.5-','Q').replace('-Instruct','I'),p[ORDER.index(m)]-p[ORDER.index(LIN[m])]) for m in ORDER if LIN[m]!=m))
print('--- projection on primary PC1 loadings')
v1=np.array(M1['pca']['loadings'][0])
for k,a in V.items():
    p=(a-a.mean(0))@v1
    print('%-14s '%k + ' '.join('%s:%+.2f'%(m.replace('OLMo-2-1124-7B','O').replace('Llama-','L').replace('Qwen2.5-','Q').replace('-Instruct','I'),p[ORDER.index(m)]-p[ORDER.index(LIN[m])]) for m in ORDER if LIN[m]!=m))
M3=json.load(open(f'{SP}/model_diff3.json'))
for fit in ['k','k+ent','k+ent+corr']:
    a=np.array([M3[fit]['a'][m] for m in ORDER]); p=(a-a.mean(0))@v1
    print('%-14s '%('item '+fit) + ' '.join('%s:%+.2f'%(m.replace('OLMo-2-1124-7B','O').replace('Llama-','L').replace('Qwen2.5-','Q').replace('-Instruct','I'),p[ORDER.index(m)]-p[ORDER.index(LIN[m])]) for m in ORDER if LIN[m]!=m))
