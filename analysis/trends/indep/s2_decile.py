import json, numpy as np
from ilib import *
np.set_printoptions(precision=2, suppress=True)
CJ = json.load(open(f"{SP}/consist.json"))
def Zc(stat):
    Cs = {r["model"]: r for r in CJ if r["stat"] == stat}
    return np.array([[dec(Cs[m]["families"][f]["z"]) for f in TASKS] for m in ORDER])
Z = Zc("q10")
mu = Z.mean((0, 1)); a = Z.mean(1) - mu
U, s, Vt = np.linalg.svd(a - a.mean(0), full_matrices=False); ev = s**2 / (s**2).sum(); sc = U * s
# orient so that + = more back-half spreading
if Vt[0][5:].mean() > 0: Vt[0] *= -1; sc[:, 0] *= -1
print("PC1 explained", ev[:3], "loadings", Vt[0])
MD = json.load(open(f"{SP}/model_diff.json")); print("model_diff PC1 explained", MD["pca"]["explained"][:3])
idx = {m: i for i, m in enumerate(ORDER)}
dpc = np.array([sc[idx[p], 0] - sc[idx[b], 0] for b, p in PAIRS])
print("PC1 change per pair", dpc, "LW", lw(dpc), "lineages>0", (lineages(dpc) > 0).sum())
print("PC1 scores:", {m: round(sc[i, 0], 2) for i, m in enumerate(ORDER)})
# decile LW changes
D = np.array([Z[idx[p]] - Z[idx[b]] for b, p in PAIRS])     # 6 x 4 x 10
Dm = D.mean(1)                                               # 6 x 10
print("decile per pair:\n", Dm)
print("decile LW", lw(Dm)); print("lineages<0", (lineages(Dm) < 0).sum(0), "pairs<0", (Dm < 0).sum(0))
R = {"front(0-50)": slice(0, 5), "40-50": slice(4, 5), "50-70": slice(5, 7), "70-100": slice(7, 10), "back(50-100)": slice(5, 10)}
for k, s_ in R.items():
    pt = D[:, :, s_].mean(2); pp = pt.mean(1)
    print(f"{k:12s} per pair {pp}  LW {lw(pp):+.3f}  pairs<0 {(pp<0).sum()} lineages<0 {(lineages(pp)<0).sum()} pairxtask<0 {(pt<0).sum()}/24 lineage {lineages(pp)}")
# lineage x task hierarchical bootstrap: resample 4 lineages and 4 tasks with replacement
rng = np.random.default_rng(1)
L4 = np.array([lineages(D[:, t, :]) for t in range(4)]).transpose(1, 0, 2)   # 4 lineages x 4 tasks x 10
for k, s_ in R.items():
    v = L4[:, :, s_].mean(2)   # 4 x 4
    bs = []
    for _ in range(10000):
        li = rng.integers(0, 4, 4); ti = rng.integers(0, 4, 4); bs.append(v[np.ix_(li, ti)].mean())
    print(f"  lineage x task boot {k:12s} {v.mean():+.3f} CI [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]  P(>=0) {np.mean(np.array(bs)>=0):.3f}")
    # task-only and lineage-only bootstraps
    bl = [v[rng.integers(0,4,4)].mean() for _ in range(10000)]; bt = [v[:, rng.integers(0,4,4)].mean() for _ in range(10000)]
    print(f"      lineage-only CI [{np.percentile(bl,2.5):+.2f}, {np.percentile(bl,97.5):+.2f}] task-only CI [{np.percentile(bt,2.5):+.2f}, {np.percentile(bt,97.5):+.2f}]")
# peak concentration OLMo decile 4
for m in ["OLMo-2-1124-7B","OLMo-2-1124-7B-SFT","OLMo-2-1124-7B-DPO","OLMo-2-1124-7B-Instruct","Llama-3.2-3B","Llama-3.2-3B-Instruct","Llama-3.1-8B","Llama-3.1-8B-Instruct","Qwen2.5-7B","Qwen2.5-7B-Instruct"]:
    print(f"  {m:28s} decile4 {Z[idx[m], :, 4].mean():+.2f}  max front decile {Z[idx[m], :, :5].mean(0).max():+.2f}")
mf = np.array([Z[idx[p], :, :5].mean(0).max() - Z[idx[b], :, :5].mean(0).max() for b, p in PAIRS]); print("max front decile change", mf, "LW", lw(mf))
# per-task 50-70 for Qwen and others
for stat in ["q25","q50","entropy","top1"]:
    Zs = Zc(stat); Ds = np.array([Zs[idx[p]] - Zs[idx[b]] for b, p in PAIRS]).mean(1)
    for k in ["50-70", "front(0-50)", "70-100"]:
        pp = Ds[:, R[k]].mean(1); print(f"  {stat:8s} {k:12s} LW {lw(pp):+.3f} lineages<0 {(lineages(pp)<0).sum()} pairs<0 {(pp<0).sum()} {pp}")
