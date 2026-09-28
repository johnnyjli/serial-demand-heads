import sys, re, collections, numpy as np
sys.path.insert(0,'.')
import e2lib
M='Llama-3.2-1B'
for cell in ['pointer/w0','arith/w0','fixed_digit/w0','binding/w0','relational/w0']:
    d=e2lib.load(M,cell); ks=sorted(set(it['k'] for it in d['items']))
    rows=[]
    for k in ks:
        its=[it for it in d['items'] if it['k']==k]
        A=np.array([[re.search(r' (\d)\.$',b).group(1)==it['gold'] for b in it['text'].split('\n\n')[:-1]] for it in its])
        rows.append(f"k{k}: pos {'/'.join(str(int(x)) for x in A.sum(0))} any {int(A.any(1).sum())}")
    print(cell, '; '.join(rows), '(of 42)')
