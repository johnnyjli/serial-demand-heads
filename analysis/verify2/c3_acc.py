import sys, numpy as np, os
os.environ.pop('E2_CLEAN', None)
sys.path.insert(0,'.')
import e2lib
from scipy.stats import binomtest
F=['pointer','arith','binding','relational','fixed_digit']
res={f:{} for f in F}
for m in e2lib.ORDER:
    attn='sdpa' if m=='Qwen2.5-7B-Instruct' else 'eager'
    for f in F:
        c=[];k=[];ch=None
        for w in ('w0','w1'):
            d=e2lib.load(m,f'{f}/{w}',attn)
            c+= [bool(x) for x in d['correct_strict']]; k+=[it['k'] for it in d['items']]; ch=d['meta']['chance']
        res[f][m]=(np.array(c),np.array(k),ch)
for f in F:
    ks=sorted(set(res[f][e2lib.ORDER[0]][1]))
    means=[np.mean([res[f][m][0][res[f][m][1]==kk].mean() for m in e2lib.ORDER]) for kk in ks]
    ch=res[f][e2lib.ORDER[0]][2]
    # per-model binomial at k=2 (two wordings pooled; note wordings share instances)
    sig=0; sigw0=0; accs=[]
    for m in e2lib.ORDER:
        c,k,_=res[f][m]; x=c[k==2]; accs.append(x.mean())
        if binomtest(int(x.sum()),len(x),ch,alternative='greater').pvalue<.05: sig+=1
        n=len(x)//2
        if binomtest(int(x[:n].sum()),n,ch,alternative='greater').pvalue<.05: sigw0+=1
    print(f"{f:12s} chance {ch:.3f} mean acc by k {[round(v,3) for v in means]}  k=2: models above chance p<.05 (pooled 2 wordings, n={len(x)}) {sig}/17; w0 only {sigw0}/17; min {min(accs):.2f} median {np.median(accs):.2f} max {max(accs):.2f}; ratio to chance {means[1]/ch:.2f}")
