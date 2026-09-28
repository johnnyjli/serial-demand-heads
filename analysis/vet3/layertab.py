"""Layer-level table: per model x task x wording x layer, the OLS slope of top-25% share on k (pp per step),
its permutation z, the slope of effective number of heads (heads per step), and baselines.  Writes layertab.npz / .csv."""
import sys, numpy as np, csv
sys.path.insert(0, '.'); from v2lib import *
rows = []
for attn, q7 in (("eager", False),):
    for m, f, w, a in cells(attn, TASKS + ["fixed_digit"], q7i_sdpa=q7):
        d = load(m, f, w, a, True)
        b, sd, z = zprof(d["y25"], d["k"], d["body"], d["within"])
        bn = zprof(d["neff"], d["k"], d["body"], d["within"])[0]
        base = d["y25"].mean(0); base1 = d["y25"][d["k"] == d["k"].min()].mean(0); isd = d["y25"].std(0)
        nb = d["neff"].mean(0); H = int(d["H"])
        for l in range(d["L"]):
            rows.append(dict(model=m, task=f, w=w, attn=a, cell=f"{m}|{f}|{w}", layer=l, L=d["L"], x=d["x"][l], H=H,
                             b_pp=100 * b[l], z=z[l], sd_pp=100 * sd[l], neff_b=bn[l], neff_base=nb[l], base_pp=100 * base[l],
                             base_k1_pp=100 * base1[l], item_sd_pp=100 * isd[l], kmax=d["k"].max(), kmin=d["k"].min(),
                             family=FAMILY[m], base_model=BASE[m]))
keys = list(rows[0])
with open("layertab.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, keys); wr.writeheader(); wr.writerows(rows)
print(len(rows))
