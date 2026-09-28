"""(1) Split sensitivity + threshold-free trend + variants.  Writes splits.json."""
import sys, json, numpy as np
sys.path.insert(0, '.'); from v2lib import *
from scipy.stats import spearmanr
def regions(z, x):
    g = lambda msk: np.nanmean(z[msk]) if msk.any() else np.nan
    r = {"whole": g(x >= 0)}
    for t in (.4, .5, .6):
        r[f"back{int(t*100)}"] = g(x >= t); r[f"front{int(t*100)}"] = g(x < t)
    r["back3"] = g(x >= 2/3); r["mid3"] = g((x >= 1/3) & (x < 2/3)); r["front3"] = g(x < 1/3)
    r["back50_no_last10"] = g((x >= .5) & (x < .9)); r["last10"] = g(x >= .9); r["front50_no_first10"] = g((x < .5) & (x >= .1))
    ok = ~np.isnan(z); r["rho"] = spearmanr(x[ok], z[ok]).correlation
    return r
def run(stat="y25", attn="eager", clean=True, q7i_sdpa=False, tasks=TASKS + ["fixed_digit"], sign=1):
    rows = []
    for m, f, w, a in cells(attn, tasks, q7i_sdpa=q7i_sdpa):
        d = load(m, f, w, a, clean)
        if d is None: continue
        Y = sign * d[stat]
        if np.isnan(Y).all(): continue
        z = zprof(Y, d["k"], d["body"], d["within"])[2]
        rows.append({"model": m, "task": f, "w": w, "attn": a, **{k: float(v) for k, v in regions(z, d["x"]).items()}})
    return rows
def summarize(rows, label):
    out = {}
    for T in [TASKS, ["pointer"], ["arith"], ["binding"], ["relational"], ["pointer", "binding", "relational"], ["binding", "relational"], ["fixed_digit"]]:
        R = [r for r in rows if r["task"] in T]
        if not R: continue
        a = lambda key: np.array([r[key] for r in R])
        s = {"n": len(R),
             "whole<0": (a("whole") < 0).mean(), "front50>0": (a("front50") > 0).mean(),
             "back40<0": (a("back40") < 0).mean(), "back40<front40": (a("back40") < a("front40")).mean(),
             "back50<0": (a("back50") < 0).mean(), "back50<front50": (a("back50") < a("front50")).mean(),
             "back60<0": (a("back60") < 0).mean(), "back60<front60": (a("back60") < a("front60")).mean(),
             "back3<0": (a("back3") < 0).mean(), "back3<front3": (a("back3") < a("front3")).mean(), "mid3<0": (a("mid3") < 0).mean(),
             "b50x10<0": (a("back50_no_last10") < 0).mean(), "b50x10<f50": (a("back50_no_last10") < a("front50")).mean(),
             "last10<0": (a("last10") < 0).mean(),
             "rho<0": (a("rho") < 0).mean(), "rho_mean": a("rho").mean(),
             "front50_mean": a("front50").mean(), "back50_mean": a("back50").mean(), "whole_mean": a("whole").mean(),
             "back3_mean": a("back3").mean(), "mid3_mean": a("mid3").mean(), "front3_mean": a("front3").mean()}
        out["+".join(T) if len(T) < 4 or T != TASKS else "all4"] = {k: round(float(v), 3) for k, v in s.items()}
    return out
if __name__ == "__main__":
    V = {"primary(eager,clean,y25)": dict(),
         "q7i_sdpa": dict(q7i_sdpa=True),
         "sdpa_all_available": dict(attn="sdpa"),
         "no_clean": dict(clean=False),
         "nosink": dict(stat="y25ns"),
         "top10pct": dict(stat="y10"), "top25pct": dict(stat="y25orig"), "top50pct": dict(stat="y50"),
         "neg_eff_heads": dict(stat="neff", sign=-1), "top1": dict(stat="top1")}
    res = {}; allrows = {}
    for name, kw in V.items():
        rows = run(**kw); allrows[name] = rows; res[name] = summarize(rows, name)
        s = res[name]["all4"]
        print(f"{name:26s} n={int(s['n']):3d} whole<0 {s['whole<0']:.2f} | b40 {s['back40<0']:.2f}/{s['back40<front40']:.2f} | b50 {s['back50<0']:.2f}/{s['back50<front50']:.2f} | b60 {s['back60<0']:.2f}/{s['back60<front60']:.2f} | thirds {s['back3<0']:.2f}/{s['back3<front3']:.2f} mid3<0 {s['mid3<0']:.2f} | b50-no-last10 {s['b50x10<0']:.2f}/{s['b50x10<f50']:.2f} last10<0 {s['last10<0']:.2f} | rho<0 {s['rho<0']:.2f} rho {s['rho_mean']:+.2f} | front>0 {s['front50>0']:.2f} means F/B {s['front50_mean']:+.2f}/{s['back50_mean']:+.2f}")
    json.dump({"summary": res, "rows": allrows}, open("splits.json", "w"), indent=1)
    print()
    for T, s in res["primary(eager,clean,y25)"].items():
        print(f"{T:28s} n={int(s['n']):3d} whole<0 {s['whole<0']:.2f} b40 {s['back40<0']:.2f}/{s['back40<front40']:.2f} b50 {s['back50<0']:.2f}/{s['back50<front50']:.2f} b60 {s['back60<0']:.2f}/{s['back60<front60']:.2f} thirds {s['back3<0']:.2f}/{s['back3<front3']:.2f} b50x10 {s['b50x10<0']:.2f}/{s['b50x10<f50']:.2f} rho<0 {s['rho<0']:.2f} ({s['rho_mean']:+.2f}) front>0 {s['front50>0']:.2f} F/M/B3 {s['front3_mean']:+.2f}/{s['mid3_mean']:+.2f}/{s['back3_mean']:+.2f}")
