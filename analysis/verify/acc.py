import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, numpy as np, os
import os; os.environ.setdefault("X","1")
SP = str(paths.ANALYSIS)
sys.path.insert(0, SP); from e2lib import *
IE, IS = index("eager"), index("sdpa")
for q7 in ("eager", "sdpa"):
    print("Qwen2.5-7B-Instruct attn:", q7)
    for fam in ["pointer", "arith", "binding", "relational", "fixed_digit"]:
        per_model = {}
        for m in ORDER:
            a = q7 if m == "Qwen2.5-7B-Instruct" else "eager"
            accs = []
            for w in ("w0", "w1"):
                d = load(m, f"{fam}/{w}", a)
                k = d["k"]; c = d["correct_strict"]
                accs.append({int(kk): c[k == kk].mean() for kk in np.unique(k)})
            per_model[m] = {kk: np.mean([ac[kk] for ac in accs]) for kk in accs[0]}
        ks = sorted(per_model[ORDER[0]])
        print(f"  {fam:12s}", " ".join(f"k{kk}={np.mean([per_model[m][kk] for m in ORDER]):.3f}" for kk in ks))
        if fam in ("binding",):
            print("    models above chance at k=2:", sum(per_model[m][2] > 1/3 for m in ORDER), "/17")
        if fam in ("relational","pointer","arith"):
            ch = {"pointer": .1, "arith": 1/7, "relational": 1/3}[fam]
            print(f"    models above chance at k=2:", sum(per_model[m][2] > ch for m in ORDER), "/17")
