import sys, json, numpy as np
SP=sys.argv[1]; sys.path.insert(0,SP)
from model_diff import ORDER, T, from_consist
Z=from_consist("q10")  # 17x4x10
rng=np.random.default_rng(7)
C={r["model"]: r for r in json.load(open(f"{SP}/consist.json")) if r["stat"]=="q10"}
print("attn fields:", {m: C[m].get("attn") for m in ORDER if C[m].get("attn")})
for j,t in enumerate(T+["all4"]):
    if t=="all4": bk=Z[:,:,5:].mean((1,2)); fr=Z[:,:,:5].mean((1,2))
    else: bk=Z[:,j,5:].mean(1); fr=Z[:,j,:5].mean(1)
    bs=[bk[rng.integers(0,17,17)].mean() for _ in range(10000)]
    # layer based
    lay=[]
    for m in ORDER:
        L=C[m]["L"]; x=np.arange(L)/(L-1)
        zz=np.mean([np.array(C[m]["families"][f]["z"]) for f in (T if t=="all4" else [t])],0)
        lay.append(zz[x>=.5].mean())
    lay=np.array(lay)
    print(f"{t:10s} back(dec) {bk.mean():+.2f} CI[{np.quantile(bs,.025):+.2f},{np.quantile(bs,.975):+.2f}] n<0 {(bk<0).sum()}/17  back<front {(bk<fr).sum()} | back(layers x>=.5) {lay.mean():+.2f} n<0 {(lay<0).sum()}")
    if t!="all4": print("     per-model:", " ".join(f"{v:+.1f}" for v in bk))
