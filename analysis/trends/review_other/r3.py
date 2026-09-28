"""Entropy/correctness-adjusted back-half k-slope t for Qwen2.5-7B-Instruct under SDPA (altexp_layers.csv uses eager for it).
Per wording, per layer OLS y10 ~ k [+ ent + correct]; t of k; decile profile; wording-averaged; task-averaged back-half mean."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, sys, numpy as np
SP = str(paths.ANALYSIS)
os.environ["VPRIM"] = "y10"; sys.path.insert(0, SP); sys.path.insert(0, f"{SP}/vet3")
from model_diff import dec_prof
from v2lib import load
def tk(Y, X):
    X = np.c_[np.ones(len(X)), X]; XtXi = np.linalg.inv(X.T @ X); B = XtXi @ X.T @ Y; E = Y - X @ B
    s2 = (E ** 2).sum(0) / (len(Y) - X.shape[1]); return B[1] / np.sqrt(s2 * XtXi[1, 1])
for m, attn in [("Qwen2.5-7B-Instruct", "sdpa"), ("Qwen2.5-7B-Instruct", "eager"), ("Qwen2.5-7B", "eager")]:
    for fam in ["pointer", "arith", "binding", "relational", "fixed_digit"][:5]:
        pass
    res = {}
    for fit in ["k", "k+ent+corr"]:
        backs = []
        for fam in ["pointer", "arith", "binding", "relational"]:
            prof = []
            for w in ("w0", "w1"):
                d = load(m, fam, w, attn, True); k = d["k"]
                X = k[:, None] if fit == "k" else np.c_[k, d["ent"], d["correct"].astype(float)]
                prof.append(dec_prof(tk(d["y10"], X)))
            backs.append(np.mean(prof, 0)[5:].mean())
        res[fit] = round(float(np.mean(backs)), 2)
    print(m, attn, res)
