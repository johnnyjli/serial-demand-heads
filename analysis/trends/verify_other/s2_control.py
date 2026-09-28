"""T2: numeral control back half, per model; other statistics; kernels for the post-trained pairs."""
import json, numpy as np
from vo_lib import *
S1 = json.load(open(f"{OUT}/s1_profiles.json"))
ctl = S1["control"]
print(f"{'model':28s} {'ser_back':>8s} {'ctl_back':>8s} {'ctl_front':>9s} diff")
for m in M:
    sb, cb, cf = ctl["serial_back"][m], ctl["ctrl_back"][m], ctl["ctrl_front"][m]
    print(f"{m:28s} {sb:+8.2f} {cb:+8.2f} {cf:+9.2f} {sb-cb:+.2f} {'<-- exception' if sb-cb>=0 else ''}")
# other statistics for the control (cache stats) and serial (cache rebuild, same stat)
STATS = {"q10": "y10", "q25": "y25", "q50": "y50", "top1": "top1", "neff": "neff"}
res = {}
for sname, key in STATS.items():
    sign = -1 if key == "neff" else 1          # neff: + = more heads = spreading; flip so + = concentrating
    cb, sb = {}, {}
    for m in M:
        z, _ = rebuild_z(m, "fixed_digit", stat=key); cb[m] = float(sign * deciles(z)[5:].mean())
        sb[m] = float(np.mean([sign * deciles(rebuild_z(m, t, stat=key)[0])[5:].mean() for t in TK]))
    d = {m: sb[m] - cb[m] for m in M}; ld = lin_mean(d); lc = lin_mean(cb)
    res[sname] = {"ctrl_back": cb, "serial_back": sb}
    exc = [m for m in M if d[m] >= 0]
    print(f"{sname:5s} serial<ctrl {sum(v<0 for v in d.values())}/17 lin {sum(v<0 for v in ld.values())}/11 | ctrl back<0 {sum(v<0 for v in cb.values())}/17 lin {sum(v<0 for v in lc.values())}/11 | exceptions {exc}")
    print("      post vs base ctrl back: " + "; ".join(f"{p.split('-')[0][:5]}{p[-12:]} {cb[p]:+.2f} vs base {cb[b]:+.2f}" for p, b in POST2BASE.items()))
# kernels: SDPA control for pairs where both exist
print("\nSDPA control back (where both wordings exist):")
for m in M:
    r = rebuild_z(m, "fixed_digit", attn="sdpa")
    e = rebuild_z(m, "fixed_digit", attn="eager")
    s = f"{deciles(r[0])[5:].mean():+.2f}" if r is not None else "  n/a"
    print(f"  {m:28s} sdpa {s}  eager {deciles(e[0])[5:].mean():+.2f}")
json.dump(res, open(f"{OUT}/s2_control.json", "w"), indent=1)
