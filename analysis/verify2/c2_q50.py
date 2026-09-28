import sys, json, numpy as np
SP=sys.argv[1]; sys.path.insert(0,SP)
from model_diff import ORDER, T, from_consist, from_cells
def sig(Z):
    mu=Z.mean((0,1)); return Z.mean(1)-mu
S={s:sig(from_consist(s)) for s in ["q10","q25","q50","entropy","top1"]}
S["sink"]=sig(from_cells("eager","sink_head")[0]); S["perword"]=sig(from_cells("eager","all")[0])
ks=list(S)
print("      "+" ".join(f"{k:>8s}" for k in ks))
for a in ks: print(f"{a:8s}"+" ".join(f"{np.corrcoef(S[a].ravel(),S[b].ravel())[0,1]:8.2f}" for b in ks))
# mean r of q50 with the other unadjusted
o=[np.corrcoef(S["q50"].ravel(),S[b].ravel())[0,1] for b in ["q10","q25","entropy","top1"]]
print("q50 mean r with q10,q25,entropy,top1:", np.round(o,2), np.mean(o))
# per-model correlation q50 vs q10
pm=[np.corrcoef(S["q50"][i],S["q10"][i])[0,1] for i in range(17)]
print("per-model r q50 vs q10 median", np.median(pm))
