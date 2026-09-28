"""Verify layer-level claims: consistently responding layers in E2b direct (permutation p, 2000 shuffles, k shuffled
across items), overlap with the paper's consistent layers, strongest-spreading decile, 40-50% band per model and task."""
import json, numpy as np
from vn_lib import *
B = 2000
OUT = {}
def perm_p(Y, k, rng):
    kc = k - k.mean(); Yc = Y - Y.mean(0); obs = kc @ Yc / (kc @ kc)
    null = np.array([rng.permutation(kc) @ Yc for _ in range(B)]) / (kc @ kc)
    p = (np.sum(np.abs(null) >= np.abs(obs) - 1e-15, 0) + 1) / (B + 1)
    return obs, p
cons_b = {}
for m in M8:
    rng = np.random.default_rng(7)
    sig = []
    for t in T:
        Y, k, _, _ = cell(m, "direct", t); obs, p = perm_p(Y, k, rng); sig.append(np.sign(obs) * (p < .05))
    sig = np.array(sig); pos = (sig > 0).sum(0); neg = (sig < 0).sum(0)
    conc = np.nonzero((pos >= 3) & (neg == 0))[0].tolist(); spr = np.nonzero((neg >= 3) & (pos == 0))[0].tolist()
    L = sig.shape[1]
    e2c = np.array(CQ[m]["consistent"]); e2conc = np.nonzero(e2c > 0)[0].tolist(); e2spr = np.nonzero(e2c < 0)[0].tolist()
    cons_b[m] = {"E2b_conc": conc, "E2b_spread": spr, "E2_conc": e2conc, "E2_spread": e2spr, "L": L,
                 "spread_overlap": sorted(set(spr) & set(e2spr)), "conc_overlap": sorted(set(conc) & set(e2conc))}
    print(f"{m:30s} E2b spread {spr} conc {conc} | E2 spread {e2spr} conc {e2conc} | overlap spread {cons_b[m]['spread_overlap']} conc {cons_b[m]['conc_overlap']}")
tot = lambda key: sum(len(v[key]) for v in cons_b.values())
print("totals: E2b spread %d conc %d; E2 spread %d conc %d; spread overlap %d; conc overlap %d" % (tot("E2b_spread"), tot("E2b_conc"), tot("E2_spread"), tot("E2_conc"), tot("spread_overlap"), tot("conc_overlap")))
OUT["consistent"] = cons_b
Zb = Zprof(M8); Za = Zprof_e2(M8)
dec = {}
for i, m in enumerate(M8):
    pb, pa = Zb[i].mean(0), Za[i].mean(0)
    dec[m] = {"E2b_argmin": int(pb.argmin()), "E2_argmin": int(pa.argmin()), "E2b_d4_per_task": np.round(Zb[i, :, 4], 2).tolist(), "E2_d4_per_task": np.round(Za[i, :, 4], 2).tolist(),
              "E2b_50_80": float(pb[5:8].mean()), "E2_50_80": float(pa[5:8].mean())}
    print(f"{m:30s} strongest-spreading decile E2b {dec[m]['E2b_argmin']} E2 {dec[m]['E2_argmin']} | d4 per task E2b {dec[m]['E2b_d4_per_task']} E2 {dec[m]['E2_d4_per_task']}")
print("same strongest decile:", sum(v["E2b_argmin"] == v["E2_argmin"] for v in dec.values()))
OUT["deciles"] = dec
json.dump(OUT, open(f"{HERE}/vn_layers.json", "w"), indent=1, default=float)
