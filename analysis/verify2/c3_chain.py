import sys, re, collections
sys.path.insert(0,'.')
import e2lib
M='Llama-3.2-1B'
for cell in ['binding/w0','binding/w1','relational/w0','relational/w1']:
    d=e2lib.load(M,cell); lens=collections.Counter(); nroots=collections.Counter(); order=collections.Counter()
    for it in d['items']:
        if it['k']!=1: continue
        fin=it['text'].split('\n\n')[-1]
        if cell.startswith('binding'):
            if cell.endswith('w0'): pairs=re.findall(r'(\w) = (\w)\.', fin)
            else: pairs=re.findall(r'Let (\w) be (\w)\.', fin)
        else:
            if cell.endswith('w0'): pairs=re.findall(r'The boss of (\w) is (\w)\.', fin)
            else: pairs=re.findall(r'(\w) reports to (\w)\.', fin)
        par={a:b for a,b in pairs}
        roots=[a for a,b in pairs if b.isdigit()]; nroots[len(roots)]+=1
        # chain length from each root: count nodes whose chain reaches root
        def depth(x):
            n=0
            while not x.isdigit(): x=par[x]; n+=1
            return n,x
        ch=collections.Counter(depth(a)[1] for a,_ in pairs)
        lens[tuple(sorted(ch.values()))]+=1
        # is each chain's order in file monotone (interleaved) or shuffled?
        pos={a:i for i,(a,_) in enumerate(pairs)}
        mono=all(all(pos[a] < pos[c] for c,b in pairs if b==a) for a,_ in pairs)
        order[mono]+=1
    print(cell,'n statements per chain',dict(lens),'roots',dict(nroots),'defn-before-use in all chains',dict(order))
for cell in ['pointer/w0','arith/w0','relational/w0','fixed_digit/w0']:
    d=e2lib.load(M,cell); it=d['items'][0]; print('---',cell,'k',it['k'],'gold',it['gold']); print(it['text'].split('\n\n')[-1][:500])
