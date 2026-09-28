import sys, numpy as np, os
os.environ.pop('E2_CLEAN', None)
sys.path.insert(0,'.')
import e2lib
from scipy.stats import binomtest
for f in ['binding','relational']:
    for w in ('w0','w1'):
        accs=[];sig=0
        for m in e2lib.ORDER:
            attn='sdpa' if m=='Qwen2.5-7B-Instruct' else 'eager'
            d=e2lib.load(m,f'{f}/{w}',attn); k=np.array([it['k'] for it in d['items']]); c=np.array(d['correct_strict'],bool)[k==2]
            accs.append(c.mean()); sig+= binomtest(int(c.sum()),len(c),1/3,alternative='greater').pvalue<.05
        print(f,w,'mean acc k=2 %.3f'%np.mean(accs),'models p<.05 above 1/3:',sig,'/17')
