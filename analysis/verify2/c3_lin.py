import json, numpy as np, sys
sys.path.insert(0,'.')
from model_diff import ORDER, LIN, FAM, from_consist, decompose
a = decompose(from_consist("q10"))[1]
R = np.corrcoef(a)
R2 = np.array(json.load(open('model_diff.json'))['similarity']['R'])
print('max |R-R2|', np.abs(R-R2).max())
print(FAM)
g = {'lin':[], 'samefam':[], 'difffam':[]}
for i in range(17):
    for j in range(i+1,17):
        x,y=ORDER[i],ORDER[j]
        if LIN[x]==LIN[y]: g['lin'].append(R[i,j])
        elif FAM[x]==FAM[y]: g['samefam'].append(R[i,j])
        else: g['difffam'].append(R[i,j])
for k,v in g.items(): print(k, len(v), round(np.mean(v),4))
v = g['samefam']+g['difffam']; print('non-lineage', len(v), round(np.mean(v),4))
# base models only
B=[m for m in ORDER if LIN[m]==m]
bi=[ORDER.index(m) for m in B]
sf=[R[bi[i],bi[j]] for i in range(len(bi)) for j in range(i+1,len(bi)) if FAM[B[i]]==FAM[B[j]]]
df=[R[bi[i],bi[j]] for i in range(len(bi)) for j in range(i+1,len(bi)) if FAM[B[i]]!=FAM[B[j]]]
print('bases samefam', len(sf), np.mean(sf), 'difffam', len(df), np.mean(df), 'all', np.mean(sf+df))
