import sys, json, collections, numpy as np
sys.path.insert(0,'..')
import e2lib
def get(model,cell,attn=None):
    attn=attn or ('sdpa' if model=='Qwen2.5-7B-Instruct' else 'eager')
    return e2lib.load(model,cell,attn)
FAM=["pointer","arith","binding","relational","fixed_digit"]
res={}; permodel={}
for f in FAM:
    for key in ('correct_strict','correct_constrained'):
        tab=collections.defaultdict(list)
        for M in e2lib.ORDER:
            ks=[];c=[]
            for w in ('w0','w1'):
                d=get(M,f+'/'+w); ks.append(d['k']); c.append(d[key])
            ks=np.concatenate(ks); c=np.concatenate(c)
            for k in sorted(set(ks)):
                tab[int(k)].append(c[ks==k].mean())
            if key=='correct_strict': permodel[(M,f)]={int(k):float(c[ks==k].mean()) for k in sorted(set(ks))}
        print(f,key,' '.join('k%d=%.3f'%(k,np.mean(v)) for k,v in sorted(tab.items())))
json.dump({f"{m}|{f}":v for (m,f),v in permodel.items()},open('acc_permodel.json','w'),indent=0)
# Qwen7B-Instruct under eager for comparison
for f in FAM[:1]:
    pass
