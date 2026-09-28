import numpy as np
from ilib import *
rng = np.random.default_rng(0)
for m, f in [("Llama-3.2-3B", "pointer"), ("Llama-3.2-3B", "binding"), ("Llama-3.2-3B", "arith")]:
    d = wavg(m, f); n = len(d["k"]); x = xrel(d["L"])
    Z = []; Zs = []
    b0, z0 = zexact(d["Y"], d["k"], d["body"], within(f)); sd0 = b0 / z0
    for _ in range(400):
        i = rng.integers(0, n, n); b, z = zexact(d["Y"][i], d["k"][i], d["body"][i], within(f)); Z.append(z); Zs.append(b / sd0)
    Z = np.array(Z); Zs = np.array(Zs)
    print(m, f, "per-layer boot SD of z (recomputed null)", Z.std(0).mean().round(2), "(fixed null SD)", Zs.std(0).mean().round(2),
          "| back-mean SD", Z[:, x >= .5].mean(1).std().round(2), Zs[:, x >= .5].mean(1).std().round(2), "| mean pairwise layer corr", np.corrcoef(Z[:, x>=.5].T)[np.triu_indices((x>=.5).sum(),1)].mean().round(2))
