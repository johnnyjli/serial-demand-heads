"""Statistic sensitivity with the exact (analytic) permutation z: top-10/25/50% share, top-1 share, and
negative normalised effective number of heads (exp entropy / H, sign flipped so + = concentrating)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json
import numpy as np
sys.path.insert(0, str(paths.ANALYSIS / "vet3"))
from myload import catalogue, read, MODELS, SERIAL
from repro import topq_share, groups_of
QI = "Qwen2.5-7B-Instruct"


def stat(S, name):
    if name.startswith("q"): return topq_share(S, q=float(name[1:]) / 100)
    P = S / S.sum(-1, keepdims=True)
    if name == "top1": return P.max(-1)
    if name == "neff": return -np.exp(-(P * np.log(np.maximum(P, 1e-300))).sum(-1)) / S.shape[-1]
    if name == "gini":
        s = np.sort(P, -1); H = S.shape[-1]; i = np.arange(1, H + 1)
        return ((2 * i - H - 1) * s).sum(-1) / (H - 1)
    raise ValueError


def zx(Y, k, groups):
    kc = k - k.mean(); Yc = Y - Y.mean(0); num = kc @ Yc; var = np.zeros(Y.shape[1]); mu = np.zeros(Y.shape[1])
    for g in groups:
        if len(g) < 2: continue
        kg = k[g] - k[g].mean(); yg = Y[g] - Y[g].mean(0)
        var += (kg @ kg) * (yg * yg).sum(0) / (len(g) - 1); mu += len(g) * (k[g].mean() - k.mean()) * (Y[g].mean(0) - Y.mean(0))
    return (num - mu) / np.sqrt(var)


names = ["q10", "q25", "q50", "top1", "neff", "gini"]
res = {n: [] for n in names}
for m in MODELS:
    cat = catalogue("sdpa" if m == QI else "eager")
    for f in SERIAL:
        for w in ("w0", "w1"):
            d = read(cat[(m, f, w)]); L = d["L"]; x = np.arange(L) / (L - 1)
            for n in names:
                z = zx(stat(d["S"], n), d["k"], groups_of(f, d["k_i"]))
                res[n].append((f, z[x < .5].mean(), z[x >= .5].mean()))
for n in names:
    a = np.array([(r[1], r[2]) for r in res[n]])
    line = f"{n:5s} all: back<0 {100*np.mean(a[:,1]<0):5.1f}%  back<front {100*np.mean(a[:,1]<a[:,0]):5.1f}%  mean front {a[:,0].mean():+.2f} back {a[:,1].mean():+.2f} |"
    for f in SERIAL:
        b = np.array([(r[1], r[2]) for r in res[n] if r[0] == f])
        line += f" {f[:4]} {100*np.mean(b[:,1]<0):3.0f}/{100*np.mean(b[:,1]<b[:,0]):3.0f}"
    print(line)
