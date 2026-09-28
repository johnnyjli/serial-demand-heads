import json, numpy as np
C = [r for r in json.load(open("consist.json")) if r["stat"] == "q10"]
T = ["pointer", "arith", "binding", "relational"]; rng = np.random.default_rng(0)
print(len(C), sorted(set(r.get("attn","?") for r in C)))
for t in T + ["all4"]:
    fams = T if t == "all4" else [t]
    back, front, backdec, b4050 = [], [], [], []
    for r in C:
        L = r["L"]; x = np.arange(L) / (L - 1); dec = np.minimum((x * 10).astype(int), 9)
        Z = np.stack([np.array(r["families"][f]["z"]) for f in fams]).mean(0)
        back.append(Z[x >= .5].mean()); front.append(Z[x < .5].mean())
        backdec.append(np.mean([Z[dec == d].mean() for d in range(5, 10)]))
    back = np.array(back); front = np.array(front); backdec = np.array(backdec)
    bs = [back[rng.integers(0, 17, 17)].mean() for _ in range(4000)]
    bsd = [backdec[rng.integers(0, 17, 17)].mean() for _ in range(4000)]
    print(f"{t:10s} back(layers) {back.mean():+.2f} CI [{np.quantile(bs,.025):+.2f},{np.quantile(bs,.975):+.2f}] n<0 {(back<0).sum()} back<front {(back<front).sum()} | back(deciles) {backdec.mean():+.2f} CI [{np.quantile(bsd,.025):+.2f},{np.quantile(bsd,.975):+.2f}]")
d = json.load(open("conc_q10.json"))["deciles"]["all4"]
print("all4 decile hi", d["hi"][5:], "lo", d["lo"][4])
