import json, numpy as np, sys
sys.path.insert(0, "..")
from e2lib import ORDER
from scipy.stats import binomtest
R = json.load(open("../consist.json"))
T = ["pointer","arith","binding","relational"]
BASE = ["Llama-3.2-1B","Llama-3.2-3B","Llama-3.1-8B","Qwen2.5-0.5B","Qwen2.5-1.5B","Qwen2.5-3B","Qwen2.5-7B","Qwen2.5-14B","OLMo-2-1124-7B","Mistral-7B-v0.1","Mistral-Small-24B-Base-2501"]
print("stats in consist.json:", sorted({r["stat"] for r in R}), "attn:", {r["model"]: r["attn"] for r in R if r["stat"]=="q25" and r["attn"]!="eager"})
for stat in sorted({r["stat"] for r in R}):
    C = {r["model"]: r for r in R if r["stat"] == stat}
    res = {}
    for m in ORDER:
        r = C[m]; L = r["L"]; x = np.arange(L)/(L-1)
        fr = np.mean([np.mean(np.array(r["families"][f]["z"])[x<.5]) for f in T])
        bk = np.mean([np.mean(np.array(r["families"][f]["z"])[x>=.5]) for f in T])
        nfam = sum(np.mean(np.array(r["families"][f]["z"])[x>=.5])<0 for f in T)
        res[m] = (fr, bk, nfam)
    sgn = -1 if stat in ("entropy",) else 1
    n_contrast = sum(res[m][1] < res[m][0] for m in ORDER); n_base = sum(res[m][1] < res[m][0] for m in BASE)
    n_out = sum(res[m][1] < 0 for m in ORDER); n_front_pos = sum(res[m][0] > 0 for m in ORDER)
    n3 = sum(res[m][2] >= 3 for m in ORDER)
    print(f"{stat:8s} back<front {n_contrast}/17 base {n_base}/11  back<0 {n_out}/17 not: {[m for m in ORDER if res[m][1]>=0]}  front>0 {n_front_pos}/17  >=3 fams back<0: {n3}/17",
          "sign p", binomtest(n_contrast, 17, .5, alternative='greater').pvalue if stat=='q25' else '', binomtest(n_base, 11, .5, alternative='greater').pvalue if stat=='q25' else '')
