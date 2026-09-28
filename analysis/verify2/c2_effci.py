import sys, numpy as np, pandas as pd
V3=sys.argv[1]
T=pd.read_csv(f"{V3}/layertab.csv"); T=T[T.task.isin(["pointer","arith","binding","relational"])]
print(T.attn.unique() if "attn" in T else "", T.shape, T.model.nunique())
B=T[T.x>=.5].groupby(["model","task","w"]).b_pp.mean().reset_index()
print("cells",len(B),"mean",B.b_pp.mean())
M=B.groupby("model").b_pp.mean(); print("model-level mean", M.mean())
for seed in range(6):
    rng=np.random.default_rng(seed)
    g=B.model.values; v=B.b_pp.values; gs=np.unique(g); idx={x:np.nonzero(g==x)[0] for x in gs}
    s=[np.mean(np.concatenate([v[idx[x]] for x in rng.choice(gs,len(gs))])) for _ in range(5000)]
    print(seed, np.round(np.quantile(s,[.025,.975]),4))
