import sys, re, collections
sys.path.insert(0,'.')
import e2lib
M='Llama-3.2-1B'
for cell in ['pointer/w0','arith/w0','fixed_digit/w0','binding/w0','relational/w0']:
    d=e2lib.load(M,cell); dep=collections.Counter(); per=collections.Counter()
    for it in d['items']:
        ex=it['text'].split('\n\n')[:-1]; ds=[]
        for b in ex:
            if cell.startswith(('pointer','fixed')): ds.append(int(re.search(r'Take (\d+) steps',b).group(1)))
            elif cell.startswith('arith'): ds.append(int(re.search(r'Apply the rule (\d+) times',b).group(1)))
            else:
                if cell.startswith('binding'):
                    pairs=re.findall(r'(\w) = (\w)\.', b.split('Therefore')[0]); q=re.search(r'Therefore (\w) =',b).group(1)
                else:
                    pairs=re.findall(r'The boss of (\w) is (\w)\.', b); q=re.search(r'The top boss above (\w) is',b).group(1)
                par=dict(pairs); n=0; x=q
                while not x.isdigit(): x=par[x]; n+=1
                ds.append(n)
        per[tuple(sorted(ds))]+=1
    print(cell, 'sorted example depths per item', dict(per))
