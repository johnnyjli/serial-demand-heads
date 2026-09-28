import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np, sys
SP=str(paths.ANALYSIS)
sys.path.insert(0,SP)
from model_diff import ORDER, T, from_consist, FAM
from scipy.stats import spearmanr
C={}
for r in json.load(open(f'{SP}/consist.json')): C[(r['model'],r['stat'])]=r
res={}
for st in ['q10','q25','q50','entropy','top1']:
    Z=from_consist(st); dec_back=Z[:,:,5:].mean((1,2))
    lay_back=[]
    for m in ORDER:
        r=C[(m,st)]; L=r['L']; x=np.arange(L)/(L-1); z=np.stack([r['families'][t]['z'] for t in T]).mean(0); lay_back.append(z[x>=.5].mean())
    res[st]=(dec_back,np.array(lay_back))
print('%-28s'%'model'+''.join('%16s'%s for s in res))
for i,m in enumerate(ORDER): print('%-28s'%m+''.join('  %+6.2f/%+6.2f '%(res[s][0][i],res[s][1][i]) for s in res))
ent_d,ent_l=res['entropy']
fam=np.array([FAM[m] for m in ORDER])
for nm,v in [('decile',ent_d),('layer',ent_l)]:
    nonq=[i for i,m in enumerate(ORDER) if not m.startswith('Qwen')]
    print(nm,'entropy back mean: non-Qwen %.2f'%v[nonq].mean(), ' non-Qwen bases %.2f'%np.mean([v[i] for i in nonq if ORDER[i] in ('Llama-3.2-1B','Llama-3.2-3B','Llama-3.1-8B','OLMo-2-1124-7B','Mistral-7B-v0.1','Mistral-Small-24B-Base-2501')]),
          'Qwen3B %.2f Qwen7B %.2f Q0.5 %.2f Q1.5 %.2f Q14 %.2f Q7I %.2f'%tuple(v[ORDER.index(m)] for m in ['Qwen2.5-3B','Qwen2.5-7B','Qwen2.5-0.5B','Qwen2.5-1.5B','Qwen2.5-14B','Qwen2.5-7B-Instruct']))
    print('   all other models (excluding Q3B,Q7B,Q0.5,Q1.5):', '%.2f'%np.mean([v[i] for i,m in enumerate(ORDER) if m not in ('Qwen2.5-3B','Qwen2.5-7B','Qwen2.5-0.5B','Qwen2.5-1.5B')]))
