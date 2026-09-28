import json, numpy as np
d=json.load(open('model_diff.json'))
p=d['primary']; T=["pointer","arith","binding","relational"]
ORDER=list(p['Z'].keys())
Z=np.array([[p['Z'][m][t] for t in T] for m in ORDER])
mu=np.array(p['mu'])
print('mu', np.round(mu,2))
print('within mean', np.mean(list(p['within_model_cross_task_r'].values())))
print('%-28s'%'model', ' '.join(f'{i*10:>3d}-' for i in range(10)), '| back(d5-9) | front')
for i,m in enumerate(ORDER):
    zm=Z[i].mean(0)
    print('%-28s'%m, ' '.join(f'{v:+5.1f}' for v in zm), f'| {zm[5:].mean():+5.2f} | {zm[:5].mean():+5.2f}')
print()
print('per task decile 4 (40-50%) and signs:')
for i,m in enumerate(ORDER):
    print('%-28s'%m, ' '.join(f'{v:+5.2f}' for v in Z[i,:,4]), '| d5', ' '.join(f'{v:+5.2f}' for v in Z[i,:,5]), '| d6', ' '.join(f'{v:+5.2f}' for v in Z[i,:,6]))
