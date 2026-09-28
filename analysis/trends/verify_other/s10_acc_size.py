"""T11 confound: accuracy vs size ranks among H=32 bases; post-training pairs (size fixed): accuracy change vs 50-70% change."""
import json, numpy as np
from vo_lib import *
F2 = json.load(open(f"{SP}/model_diff2.json"))["features"]; S0 = json.load(open(f"{OUT}/s0_build.json"))
Z = Zdec("q10"); d57 = dict(zip(M, Z[:, :, 5:7].mean((1, 2)))); acc = {m: np.mean([S0["acc"][m][t] for t in TK]) for m in M}
h32b = [m for m in BASES if F2["H"][m] == 32]
print("H=32 bases sorted by size:", [(m, PARAMS[m], round(acc[m], 3), round(d57[m], 2)) for m in sorted(h32b, key=lambda m: PARAMS[m])])
print("  rho(size, acc) H32 bases:", spear([PARAMS[m] for m in h32b], [acc[m] for m in h32b]), " rho(size, d50_70):", spear([PARAMS[m] for m in h32b], [d57[m] for m in h32b]))
nq = [m for m in M if fam(m) != "Qwen"]
print("nonQwen 11 models: rho(size, d50_70)", spear([PARAMS[m] for m in nq], [d57[m] for m in nq]), " partial acc|size", partial_spear([acc[m] for m in nq], [d57[m] for m in nq], [PARAMS[m] for m in nq]))
print("post-training pairs (size fixed): d_acc, d_50_70")
for p, b in POST2BASE.items(): print(f"  {p:28s} dacc {acc[p]-acc[b]:+.3f}  d50_70 {d57[p]-d57[b]:+.2f}")
