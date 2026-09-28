import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import glob, os, json, numpy as np, collections
ROOT=str(paths.E2_SWEEP)
bad=[]; n=0; rng=collections.defaultdict(set)
for attn in ("eager","sdpa"):
    for f in glob.glob(f"{ROOT}/bf16_{attn}/*.npz"):
        z=np.load(f, allow_pickle=False); k=z["k"]; nt=z["n_tokens"]
        b=os.path.basename(f); fam=b.split("__")[1].split("_w")[0]
        n+=1
        for kk in np.unique(k):
            pass
        u=np.unique(nt)
        if len(u)>1:
            # constant within k?
            per_k={int(kk):tuple(np.unique(nt[k==kk])) for kk in np.unique(k)}
            bad.append((attn,b[:60],len(u),{kk:(min(v),max(v)) for kk,v in per_k.items()}))
        rng[fam].add(tuple(u.tolist())[:3])
print("files",n,"with >1 distinct n_tokens:",len(bad))
for x in bad[:20]: print(x)
