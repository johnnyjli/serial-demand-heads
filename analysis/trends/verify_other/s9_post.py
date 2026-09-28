"""Context for T2 / the user's post-training question: post-minus-base back-half z, serial vs numeral control, all statistics;
plus kernel-matched (eager) version for the Qwen2.5-7B pair."""
import json, numpy as np
from vo_lib import *
S2 = json.load(open(f"{OUT}/s2_control.json"))
print(f"{'pair':34s} " + " ".join(f"{s:>16s}" for s in S2))
for p, b in POST2BASE.items():
    row = []
    for s in S2:
        ds = S2[s]["serial_back"][p] - S2[s]["serial_back"][b]; dc = S2[s]["ctrl_back"][p] - S2[s]["ctrl_back"][b]
        row.append(f"ser{ds:+5.2f} ctl{dc:+5.2f}")
    print(f"{p:34s} " + " ".join(row))
# counts
for s in S2:
    ds = [S2[s]["serial_back"][p] - S2[s]["serial_back"][b] for p, b in POST2BASE.items()]
    dc = [S2[s]["ctrl_back"][p] - S2[s]["ctrl_back"][b] for p, b in POST2BASE.items()]
    print(f"  {s}: serial post<base {sum(x<0 for x in ds)}/6, control post<base {sum(x<0 for x in dc)}/6, serial shift more negative than control shift {sum(a<c for a,c in zip(ds,dc))}/6")
# kernel-matched Qwen2.5-7B pair (both eager)
for s, key in {"q10": "y10", "q25": "y25", "q50": "y50", "top1": "top1", "neff": "neff"}.items():
    sg = -1 if key == "neff" else 1
    qi_ser = np.mean([sg * deciles(rebuild_z("Qwen2.5-7B-Instruct", t, attn="eager", stat=key)[0])[5:].mean() for t in TK])
    qi_ctl = sg * deciles(rebuild_z("Qwen2.5-7B-Instruct", "fixed_digit", attn="eager", stat=key)[0])[5:].mean()
    print(f"  eager-eager Qwen2.5-7B pair {s}: serial post-base {qi_ser - S2[s]['serial_back']['Qwen2.5-7B']:+.2f}, control post-base {qi_ctl - S2[s]['ctrl_back']['Qwen2.5-7B']:+.2f}")
