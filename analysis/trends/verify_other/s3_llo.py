"""T5: leave-lineage-out profile correlation; circular-shift null; per task; variants."""
import json, numpy as np
from vo_lib import *
from s1_profiles import cells_Z
def llo(Z, ms, weight="lineage", shift=0, task=None):
    P = Z.mean(1) if task is None else Z[:, task]
    lins = sorted(set(LINEAGE[m] for m in ms), key=M.index)
    LP = {L: P[[i for i, m in enumerate(ms) if LINEAGE[m] == L]].mean(0) for L in lins}
    rm, rl = {}, {}
    for i, m in enumerate(ms):
        if weight == "lineage": ref = np.mean([LP[L] for L in lins if L != LINEAGE[m]], 0)
        else: ref = P[[j for j, mm in enumerate(ms) if LINEAGE[mm] != LINEAGE[m]]].mean(0)
        rm[m] = float(np.corrcoef(np.roll(P[i], shift), ref)[0, 1])
    for L in lins:
        ref = np.mean([LP[K] for K in lins if K != L], 0); rl[L] = float(np.corrcoef(np.roll(LP[L], shift), ref)[0, 1])
    return rm, rl
out = {}
Z = Zdec("q10")
for wt in ["lineage", "model"]:
    rm, rl = llo(Z, M, wt)
    v = np.array(list(rm.values()))
    print(f"[{wt}-weighted ref] r>0 models {int((v>0).sum())}/17 (p={binom2(int((v>0).sum()),17):.4f}) lineages {sum(x>0 for x in rl.values())}/11 (p={binom2(sum(x>0 for x in rl.values()),11):.4f}); median r {np.median(v):.3f}; r>0.5 {int((v>0.5).sum())}/17")
    print("   per model:", {m[:14]: round(x, 2) for m, x in rm.items()})
    print("   lineage r:", {m[:14]: round(x, 2) for m, x in rl.items()})
    sh = [sum(x > 0 for x in llo(Z, M, wt, s)[0].values()) for s in range(1, 10)]
    shl = [sum(x > 0 for x in llo(Z, M, wt, s)[1].values()) for s in range(1, 10)]
    print(f"   shift null counts s=1..9 models {sh} mean {np.mean(sh):.1f} max {max(sh)}; lineages {shl} mean {np.mean(shl):.1f}")
    out[wt] = {"rm": rm, "rl": rl, "shift_models": sh, "shift_lin": shl}
# per task
cnt = np.zeros(17, int)
for j in range(4):
    rm, rl = llo(Z, M, "lineage", task=j); cnt += np.array([rm[m] > 0 for m in M])
    print(f"  task {TK[j]}: r>0 {sum(x>0 for x in rm.values())}/17, lin {sum(x>0 for x in rl.values())}/11")
# lineage version for >=3 tasks
lincnt = {}
for j in range(4):
    _, rl = llo(Z, M, "lineage", task=j)
    for L, x in rl.items(): lincnt[L] = lincnt.get(L, 0) + (x > 0)
print(f"  r>0 in >=3 of 4 tasks: models {(cnt>=3).sum()}/17, lineages (lineage profile) {sum(v>=3 for v in lincnt.values())}/11")
# per-model-per-task counted at lineage level by lineage-mean count? alt: lineage passes if its base passes
# variants
for st in ["q25", "q50", "entropy", "top1"]:
    rm, rl = llo(Zdec(st), M); print(f"  {st}: r>0 {sum(x>0 for x in rm.values())}/17 lin {sum(x>0 for x in rl.values())}/11 median {np.median(list(rm.values())):.2f}")
for name, (attn, var) in {"sink_head": ("eager", "sink_head"), "sdpa": ("sdpa", "all")}.items():
    Zv, ms = cells_Z(attn, var); rm, rl = llo(Zv, ms)
    print(f"  {name}: r>0 {sum(x>0 for x in rm.values())}/{len(ms)} lin {sum(x>0 for x in rl.values())}/{len(rl)} median {np.median(list(rm.values())):.2f}")
# Q7B under SDPA: only pointer w0 exists; compare Q7B pointer profile eager vs sdpa r with others' pointer profile
json.dump(out, open(f"{OUT}/s3_llo.json", "w"), indent=1)
