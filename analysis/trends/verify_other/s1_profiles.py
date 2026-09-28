"""T1, T3, T4, T14 (+ variants): profile-region counts at model and lineage level, sign tests, lineage bootstrap CIs."""
import json, numpy as np
from vo_lib import *
S0 = json.load(open(f"{OUT}/s0_build.json"))
def cells_Z(attn, variant, w=None, q7i_sdpa=True):
    R = json.load(open(f"{SP}/vet3/cells.json")); D = {}
    for r in R:
        if r["variant"] == variant and r["fam"] in TK + ["fixed_digit"]: D[(r["model"], r["fam"], r["w"], r["attn"])] = np.array(r["z"], float)
    Z, ms = [], []
    for m in M:
        a = "sdpa" if (q7i_sdpa and m == "Qwen2.5-7B-Instruct" and attn == "eager") else attn
        if variant != "all" and a == "sdpa" and attn == "eager": continue   # sink variants are eager only: drop Q7B-Instruct
        ws = [w] if w else ["w0", "w1"]
        if not all((m, t, ww, a) in D for t in TK for ww in ws): continue
        Z.append([deciles(np.mean([D[(m, t, ww, a)] for ww in ws], 0)) for t in TK]); ms.append(m)
    return np.array(Z), ms
def summarize(vals, ms, pred):
    """vals: per-model values (list aligned with ms). model count, lineage count (lineage mean)."""
    v = dict(zip(ms, vals)); lins = sorted(set(LINEAGE[m] for m in ms), key=M.index)
    lv = {L: np.mean([v[m] for m in ms if LINEAGE[m] == L]) for L in lins}
    nm = count(vals, pred); nl = count(lv.values(), pred)
    mu, lo, hi = boot_ci(list(lv.values()))
    return {"models": nm, "n_models": len(ms), "lineages": nl, "n_lin": len(lins), "p_models": binom2(nm, len(ms)), "p_lin": binom2(nl, len(lins)),
            "lin_mean": mu, "ci": [lo, hi], "fails_models": [m for m in ms if not pred(v[m])], "fails_lin": [L for L in lins if not pred(lv[L])]}
def regions(Z):
    return {"back": Z[:, :, 5:].mean((1, 2)), "front": Z[:, :, :5].mean((1, 2)), "d4": Z[:, :, 4].mean(1), "d5": Z[:, :, 5].mean(1),
            "d3": Z[:, :, 3].mean(1), "d0": Z[:, :, 0].mean(1), "d9": Z[:, :, 9].mean(1),
            "back_t": Z[:, :, 5:].mean(2), "front_t": Z[:, :, :5].mean(2), "d4_t": Z[:, :, 4], "d5_t": Z[:, :, 5], "d0_t": Z[:, :, 0]}
neg = lambda x: x < 0; pos = lambda x: x > 0
def full(Z, ms, tag):
    R = regions(Z); o = {}
    o["back_minus_front<0"] = summarize(R["back"] - R["front"], ms, neg)
    o["back<0"] = summarize(R["back"], ms, neg)
    nt = (R["back_t"] < 0).sum(1); o["back<0_ge3tasks_models"] = int((nt >= 3).sum())
    # lineage version: lineage-mean per task, then count tasks
    lins = sorted(set(LINEAGE[m] for m in ms), key=M.index)
    lbt = np.array([R["back_t"][[i for i, m in enumerate(ms) if LINEAGE[m] == L]].mean(0) for L in lins])
    o["back<0_ge3tasks_lin"] = int(((lbt < 0).sum(1) >= 3).sum()); o["back<0_ge3tasks_fails"] = [m for i, m in enumerate(ms) if nt[i] < 3]
    o["back<0_ge3tasks_fails_lin"] = [L for j, L in enumerate(lins) if (lbt[j] < 0).sum() < 3]
    for j, t in enumerate(TK):
        o[f"back<0|{t}"] = summarize(R["back_t"][:, j], ms, neg)
        o[f"back<front|{t}"] = summarize(R["back_t"][:, j] - R["front_t"][:, j], ms, neg)
        o[f"d4>d5|{t}"] = summarize(R["d4_t"][:, j] - R["d5_t"][:, j], ms, pos)
        o[f"d4>0|{t}"] = summarize(R["d4_t"][:, j], ms, pos)
        o[f"d0>0|{t}"] = summarize(R["d0_t"][:, j], ms, pos)
    o["d4>d5"] = summarize(R["d4"] - R["d5"], ms, pos)
    o["d4>back"] = summarize(R["d4"] - R["back"], ms, pos)
    o["d4>0"] = summarize(R["d4"], ms, pos)
    o["d4>0_no_arith"] = summarize(Z[:, [0, 2, 3], 4].mean(1), ms, pos)
    o["d4>0_ptr_arith"] = summarize(Z[:, [0, 1], 4].mean(1), ms, pos)
    o["d4>0_bind_rel"] = summarize(Z[:, [2, 3], 4].mean(1), ms, pos)
    o["d4_ptrarith_minus_bindrel"] = summarize(Z[:, [0, 1], 4].mean(1) - Z[:, [2, 3], 4].mean(1), ms, pos)
    o["d3>0"] = summarize(R["d3"], ms, pos)
    o["d0>0"] = summarize(R["d0"], ms, pos)
    o["d9<0"] = summarize(R["d9"], ms, neg)
    o["per_model"] = {m: {"back": float(R["back"][i]), "front": float(R["front"][i]), "d4": float(R["d4"][i]), "d5": float(R["d5"][i]),
                          "back_t": R["back_t"][i].tolist(), "d0": float(R["d0"][i]), "d9": float(R["d9"][i])} for i, m in enumerate(ms)}
    return o
if __name__ == "__main__":
    res = {}
    Zp = Zdec("q10"); res["primary"] = full(Zp, M, "primary")
    for st in ["q25", "q50", "entropy", "top1"]: res[st] = full(Zdec(st), M, st)
    for name, (attn, var, w) in {"sink_head": ("eager", "sink_head", None), "sdpa": ("sdpa", "all", None), "w0": ("eager", "all", "w0"),
                                 "w1": ("eager", "all", "w1"), "eager_per_wording_avg": ("eager", "all", None)}.items():
        Z, ms = cells_Z(attn, var, w); res[name] = full(Z, ms, name); res[name]["models_list"] = ms
    # bases only
    bi = [M.index(m) for m in BASES]; res["bases_only"] = full(Zp[bi], BASES, "bases")
    # control
    C = np.array([deciles(S0["ctrl_z"][m]) for m in M])
    ser_back = Zp[:, :, 5:].mean((1, 2)); c_back = C[:, 5:].mean(1); c_front = C[:, :5].mean(1)
    res["control"] = {"serial_back_minus_ctrl_back<0": summarize(ser_back - c_back, M, neg), "ctrl_back<0": summarize(c_back, M, neg),
                      "ctrl_back<front": summarize(c_back - c_front, M, neg), "ctrl_d4>d5": summarize(C[:, 4] - C[:, 5], M, pos),
                      "ctrl_d3>0": summarize(C[:, 3], M, pos), "ctrl_d4>0": summarize(C[:, 4], M, pos),
                      "ctrl_back": dict(zip(M, c_back.tolist())), "ctrl_front": dict(zip(M, c_front.tolist())), "serial_back": dict(zip(M, ser_back.tolist())),
                      "ctrl_profile": {m: C[i].tolist() for i, m in enumerate(M)}}
    json.dump(res, open(f"{OUT}/s1_profiles.json", "w"), indent=1, default=float)
    def show(o, keys):
        for k in keys:
            v = o[k]
            if isinstance(v, dict) and "models" in v:
                print(f"  {k:32s} models {v['models']}/{v['n_models']} (p={v['p_models']:.3f}) lin {v['lineages']}/{v['n_lin']} (p={v['p_lin']:.3f}) linmean {v['lin_mean']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] fails {v['fails_models']} | linfails {v['fails_lin']}")
            else: print(f"  {k:32s} {v}")
    K1 = ["back_minus_front<0", "back<0", "back<0_ge3tasks_models", "back<0_ge3tasks_lin", "back<0_ge3tasks_fails", "back<0_ge3tasks_fails_lin"] + [f"back<0|{t}" for t in TK] + [f"back<front|{t}" for t in TK]
    print("=== T1 primary"); show(res["primary"], K1)
    K3 = ["d4>d5", "d4>back"] + [f"d4>d5|{t}" for t in TK]
    print("=== T3 primary"); show(res["primary"], K3)
    K4 = ["d4>0"] + [f"d4>0|{t}" for t in TK] + ["d4>0_no_arith", "d4>0_ptr_arith", "d4>0_bind_rel", "d4_ptrarith_minus_bindrel", "d3>0"]
    print("=== T4 primary"); show(res["primary"], K4)
    K14 = ["d0>0"] + [f"d0>0|{t}" for t in TK] + ["d9<0"]
    print("=== T14 primary"); show(res["primary"], K14)
    print("=== control"); show(res["control"], ["serial_back_minus_ctrl_back<0", "ctrl_back<0", "ctrl_back<front", "ctrl_d4>d5", "ctrl_d3>0", "ctrl_d4>0"])
    print("=== robustness (models/lineages)")
    for name in ["q25", "q50", "entropy", "top1", "sink_head", "sdpa", "w0", "w1", "eager_per_wording_avg", "bases_only"]:
        o = res[name]; f = lambda k: f"{o[k]['models']}/{o[k]['n_models']},{o[k]['lineages']}/{o[k]['n_lin']}"
        print(f"  {name:22s} b<f {f('back_minus_front<0')}  b<0 {f('back<0')}  ptr b<0 {f('back<0|pointer')}  d4>d5 {f('d4>d5')}  d4>0 {f('d4>0')}  arith d4>0 {f('d4>0|arith')}  d0>0 {f('d0>0')}")
