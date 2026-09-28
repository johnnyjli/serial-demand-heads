import sys, numpy as np; SP=sys.argv[1]; sys.path.insert(0, SP)
from model_diff import from_consist, decompose
A = {s: decompose(from_consist(s))[1].ravel() for s in ["q10","q25","q50","entropy","top1"]}
for s in A:
    print(s, " ".join(f"{t}:{np.corrcoef(A[s],A[t])[0,1]:.2f}" for t in A))
