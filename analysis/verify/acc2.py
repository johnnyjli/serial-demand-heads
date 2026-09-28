import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, numpy as np, os
from scipy.stats import binomtest
SP = str(paths.ANALYSIS)
sys.path.insert(0, SP); from e2lib import *
CH = {"pointer": .1, "arith": 1/7, "binding": 1/3, "relational": 1/3}
for q7 in ("eager","sdpa"):
  print("Q7I", q7, "clean", os.environ.get("E2_CLEAN"))
  for fam in CH:
    rows = []; sig = 0; sigw = [0, 0]; raw = 0
    for m in ORDER:
        a = q7 if m == "Qwen2.5-7B-Instruct" else "eager"
        c2 = []; 
        for j, w in enumerate(("w0", "w1")):
            d = load(m, f"{fam}/{w}", a); c = d["correct_strict"][d["k"] == 2]; c2.append(c)
            if binomtest(int(c.sum()), len(c), CH[fam], alternative="greater").pvalue < .05: sigw[j] += 1
        c = np.concatenate(c2)
        if binomtest(int(c.sum()), len(c), CH[fam], alternative="greater").pvalue < .05: sig += 1
        if c.mean() > CH[fam]: raw += 1
    print(f"  {fam:11s} k=2: raw>chance {raw}/17, binom(pooled 84) sig {sig}/17, per wording sig w0 {sigw[0]} w1 {sigw[1]}")
  # exact means
  for fam in CH:
    M = []
    for m in ORDER:
        a = q7 if m == "Qwen2.5-7B-Instruct" else "eager"
        pm = []
        for w in ("w0","w1"):
            d = load(m, f"{fam}/{w}", a); pm.append([d["correct_strict"][d["k"]==kk].mean() for kk in sorted(set(d["k"].tolist()))])
        M.append(np.mean(pm, 0))
    print("  ", fam, np.round(np.mean(M, 0), 4))
