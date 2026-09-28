"""Build: control (fixed_digit) per-layer z from cache (exact z), validate rebuild vs consist.json serial z, accuracies."""
import json, numpy as np
from vo_lib import *
out = {"ctrl_z": {}, "ctrl_rel": {}, "serial_rebuild_r": {}, "acc": {}, "acc_k1": {}, "decile_impl_diff": []}
LZ = layer_z("q10")
minr = 1; maxdz = 0
for m in M:
    for t in TK:
        z, rel = rebuild_z(m, t)
        r = np.corrcoef(z, LZ[m][t])[0, 1]; minr = min(minr, r); maxdz = max(maxdz, np.max(np.abs(z - LZ[m][t])))
        out["serial_rebuild_r"][f"{m}|{t}"] = float(r)
        out["acc"].setdefault(m, {})[t] = accuracy(m, t); out["acc_k1"].setdefault(m, {})[t] = accuracy(m, t, k1=True)
    zc, relc = rebuild_z(m, "fixed_digit")
    out["ctrl_z"][m] = zc.tolist(); out["ctrl_rel"][m] = relc.tolist()
    out["acc"][m]["fixed_digit"] = accuracy(m, "fixed_digit")
    # decile impl check vs int(x*10)
    L = len(LZ[m]["pointer"]); x = np.arange(L) / (L - 1)
    d1 = np.minimum((x * 10).astype(int), 9); d2 = np.minimum(np.floor(x * 10 + 1e-9).astype(int), 9)
    if (d1 != d2).any(): out["decile_impl_diff"].append([m, int((d1 != d2).sum())])
print("serial rebuild vs consist: min layer-profile r", round(minr, 5), "max |dz|", round(maxdz, 3))
print("decile implementation differences (int(x*10) vs floor(x*10+eps)):", out["decile_impl_diff"])
MD2 = json.load(open(f"{SP}/model_diff2.json"))["features"]
for m in M:
    a = np.mean([out["acc"][m][t] for t in TK]); print(f"{m:28s} acc {a:.4f} (md2 {MD2['acc'][m]:.4f}) H {MD2['H'][m]} L {MD2['L'][m]} ctrl acc {out['acc'][m]['fixed_digit']:.3f}")
json.dump(out, open(f"{OUT}/s0_build.json", "w"), indent=1)
