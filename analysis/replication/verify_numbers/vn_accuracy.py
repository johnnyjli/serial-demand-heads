"""Verify accuracy / behaviour numbers (regime, limitations, reliability-kernel, appendix-written-steps)."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, glob, os, re, collections, numpy as np
from scipy.stats import binomtest
from vn_lib import *
OUT = {}
ITEMS = json.load(open(str(paths.E2B / "items.json")))
ITEM = {it["id"]: it for f in T for it in ITEMS[f]}
E2_ROOT = str(paths.E2_SWEEP)
ORDER17 = list(CQ.keys())
def e2_file(m, f, w, attn):
    fs = [p for p in glob.glob(f"{E2_ROOT}/bf16_{attn}/*__{f}_{w}_*.npz") if os.path.basename(p).split("__")[0].split("_", 1)[1] == m]
    return fs[0] if fs else None
def e2attn(m): return "sdpa" if m == "Qwen2.5-7B-Instruct" else "eager"
_e2c = {}
def e2load(m, f, w, attn):
    key = (m, f, w, attn)
    if key not in _e2c:
        z = np.load(e2_file(m, f, w, attn)); meta = json.loads(str(z["meta"]))
        _e2c[key] = {"k": z["k"].astype(int), "strict": z["correct_strict"].astype(bool), "pc": z["pred_constrained"].astype(int), "items": meta["items"]}
    return _e2c[key]
def boot(v, B=10000, seed=0):
    v = np.asarray(v, float); r = np.random.default_rng(seed); bs = v[r.integers(0, len(v), (B, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def acc_e2(m, sel, attn=None):
    attn = attn or e2attn(m); v = []
    for f in T:
        a = [e2load(m, f, w, attn) for w in ("w0", "w1")]
        v.append(np.mean([d["strict"][sel(d["k"])].mean() for d in a]))
    return float(np.mean(v))
def acc_e2b(m, mode, sel, kind="written"):
    v = []
    for f in T:
        its = [x for x in CACHE[(m, mode, f)]["items"] if sel(np.array([x["k"]]))[0]]
        v.append(np.mean([(x["correct"] and x["correct_constrained"]) if kind == "strict" else bool(x["correct"]) for x in its]))
    return float(np.mean(v))
ALL = lambda k: k >= 1; GE3 = lambda k: k >= 3; K1 = lambda k: k == 1
tab = {}
for m in M8:
    tab[m] = {"E2": [acc_e2(m, s) for s in (ALL, GE3, K1)], "direct": [acc_e2b(m, "direct", s) for s in (ALL, GE3, K1)],
              "direct_strict": [acc_e2b(m, "direct", s, "strict") for s in (ALL, GE3, K1)], "step": [acc_e2b(m, "stepwise", s) for s in (ALL, GE3, K1)]}
    print(f"{m:30s} E2 {np.round(tab[m]['E2'], 3)} direct {np.round(tab[m]['direct'], 3)} strict {np.round(tab[m]['direct_strict'], 3)} step {np.round(tab[m]['step'], 3)}")
for tag, models in (("M8", M8), ("M6", M6)):
    o = {}
    for src in ("E2", "direct", "direct_strict", "step"):
        for j, nm in enumerate(("all", "ge3", "k1")): o[f"{src}_{nm}"] = boot([tab[m][src][j] for m in models])
    for j, nm in enumerate(("all", "ge3")):
        o[f"direct_minus_E2_{nm}"] = boot([tab[m]["direct"][j] - tab[m]["E2"][j] for m in models])
        o[f"step_minus_direct_{nm}"] = boot([tab[m]["step"][j] - tab[m]["direct"][j] for m in models])
    OUT[f"acc_{tag}"] = o
    print(tag, {k: np.round(v, 3).tolist() for k, v in o.items()})
OUT["tab"] = tab
# arith k=1
ar_e2 = np.mean([np.mean([e2load(m, "arith", w, e2attn(m))["strict"][e2load(m, "arith", w, e2attn(m))["k"] == 1].mean() for w in ("w0", "w1")]) for m in M8])
ar_b = np.mean([np.mean([x["correct"] for x in CACHE[(m, "direct", "arith")]["items"] if x["k"] == 1]) for m in M8])
ar_lower = sum(np.mean([x["correct"] for x in CACHE[(m, "direct", "arith")]["items"] if x["k"] == 1]) < np.mean([e2load(m, "arith", w, e2attn(m))["strict"][e2load(m, "arith", w, e2attn(m))["k"] == 1].mean() for w in ("w0", "w1")]) for m in M8)
print("arith k=1: E2 %.3f E2b direct %.3f; lower in %d/8" % (ar_e2, ar_b, ar_lower)); OUT["arith_k1"] = [ar_e2, ar_b, ar_lower]

# McNemar per model at k>=3
mc = {}
for m in M8:
    g = l = 0
    for f in T:
        D = {x["id"]: x for x in CACHE[(m, "direct", f)]["items"]}
        for x in CACHE[(m, "stepwise", f)]["items"]:
            if x["k"] < 3: continue
            a, b = bool(D[x["id"]]["correct"]), bool(x["correct"]); g += (b and not a); l += (a and not b)
    mc[m] = (g, l, binomtest(g, g + l, .5).pvalue if g + l else 1.0)
print("McNemar k>=3 (gained, lost, p):", {m: (v[0], v[1], float("%.2g" % v[2])) for m, v in mc.items()}); OUT["mcnemar"] = mc

# one-step answers at k=2 (pointer, arithmetic), pred_constrained
def ptr_parse(text):
    q = text.split("\n\n")[-1]
    tbl = dict((int(a), int(b)) for a, b in re.findall(r"(\d) (?:goes to|leads to) (\d)\.", q))
    s = re.search(r"(?:Start at|From) (\d)", q).group(1); return tbl, int(s)
def ar_parse(text):
    q = text.split("\n\n")[-1]
    m1 = re.search(r"add (\d) and keep the remainder after dividing by 7\. Start at (\d)\.", q)
    if m1: return int(m1.group(1)), int(m1.group(2))
    m2 = re.search(r"x = (\d)\. Repeat \d+ times: x = \(x \+ (\d)\) mod 7", q); return int(m2.group(2)), int(m2.group(1))
def onestep_e2b(m, f):
    v = []
    for x in CACHE[(m, "direct", f)]["items"]:
        if x["k"] != 2: continue
        txt = ITEM[x["id"]]["text_direct"]
        if f == "pointer": tbl, s = ptr_parse(txt); one = tbl[s]
        else: d, s = ar_parse(txt); one = (s + d) % 7
        v.append(int(x["pred_constrained"]) == one)
    return np.mean(v)
def onestep_e2(m, f):
    v = []
    for w in ("w0", "w1"):
        d = e2load(m, f, w, e2attn(m))
        for it, pc, k in zip(d["items"], d["pc"], d["k"]):
            if k != 2: continue
            if f == "pointer": tbl, s = ptr_parse(it["text"]); one = tbl[s]
            else: dd, s = ar_parse(it["text"]); one = (s + dd) % 7
            v.append(pc == one)
    return np.mean(v)
os_ = {f: {"E2b": float(np.mean([onestep_e2b(m, f) for m in M8])), "E2_8": float(np.mean([onestep_e2(m, f) for m in M8])),
           "E2_17": float(np.mean([onestep_e2(m, f) for m in ORDER17]))} for f in ("pointer", "arith")}
os_["arith_diff_boot"] = boot([onestep_e2b(m, "arith") - onestep_e2(m, "arith") for m in M8])
print("one-step at k=2:", os_); OUT["onestep"] = os_
# pointer accuracy at k=2 in E2b direct
print("E2b direct pointer k=2 accuracy %.3f" % np.mean([np.mean([x["correct"] for x in CACHE[(m, "direct", "pointer")]["items"] if x["k"] == 2]) for m in M8]))

# trace fidelity (stepwise)
def trace_elems(gen): body = gen.split(".")[0]; return [e.strip() for e in re.split(r"->|=", body)]
tq = {"parsed": 0, "last_eq": 0, "wrong_parsed": 0, "wrong_inexact": 0, "wrong_eq_last": 0, "exact": 0, "exact_correct": 0, "n": 0}
per = {}
tq6 = collections.Counter()
for m in M8:
    ne = nexact = nparse = 0
    for f in T:
        for x in CACHE[(m, "stepwise", f)]["items"]:
            tq["n"] += 1; ne += 1
            nexact += bool(x["trace_exact"]); nparse += bool(x["parsed"])
            if x["trace_exact"]:
                tq["exact"] += 1; tq["exact_correct"] += bool(x["correct"])
                if not m.startswith("Qwen2.5-7B"): tq6["exact"] += 1; tq6["exact_correct"] += bool(x["correct"])
            if not x["parsed"]: continue
            tq["parsed"] += 1; last = trace_elems(x["gen"])[-1]; tq["last_eq"] += (last == x["pred_gen"])
            if not m.startswith("Qwen2.5-7B"): tq6["parsed"] += 1; tq6["last_eq"] += (last == x["pred_gen"])
            if not x["correct"]:
                tq["wrong_parsed"] += 1; tq["wrong_inexact"] += (not x["trace_exact"]); tq["wrong_eq_last"] += (last == x["pred_gen"])
                if not m.startswith("Qwen2.5-7B"): tq6["wrong_parsed"] += 1; tq6["wrong_inexact"] += (not x["trace_exact"])
    per[m] = {"exact_rate": nexact / ne, "parse_rate": nparse / ne}
print("trace:", tq, {k: round(tq[a] / tq[b], 4) for k, (a, b) in {"last_eq": ("last_eq", "parsed"), "wrong_inexact": ("wrong_inexact", "wrong_parsed"), "exact_correct": ("exact_correct", "exact"), "wrong_eq_last": ("wrong_eq_last", "wrong_parsed")}.items()})
print("trace no-Q7:", dict(tq6), {k: round(tq6[a] / tq6[b], 4) for k, (a, b) in {"last_eq": ("last_eq", "parsed"), "wrong_inexact": ("wrong_inexact", "wrong_parsed"), "exact_correct": ("exact_correct", "exact")}.items()})
print("per model exact/parse:", {m: (round(v["exact_rate"], 3), round(v["parse_rate"], 3)) for m, v in per.items()})
print("parse rate all %.3f, no Q7 %.3f" % (np.mean([per[m]["parse_rate"] for m in M8]), np.mean([per[m]["parse_rate"] for m in M6])))
OUT["trace"] = {"all": tq, "noQ7": dict(tq6), "per_model": per}

# malformed stepwise traces and direct non-digit outputs
mal = {}
for m in M8:
    n = bad = 0
    for f in T:
        for x in CACHE[(m, "stepwise", f)]["items"]:
            n += 1; el = trace_elems(x["gen"])
            ok = (lambda e: re.fullmatch(r"\d", e) is not None) if f in ("pointer", "arith") else (lambda e: re.fullmatch(r"[A-Za-z0-9]", e) is not None)
            bad += not all(ok(e) for e in el)
    nd = [not re.match(r"^\s*\d", x["gen"]) for f in T for x in CACHE[(m, "direct", f)]["items"]]
    mal[m] = {"malformed_step": bad / n, "n": n, "bad": bad, "direct_nondigit": float(np.mean(nd))}
print("malformed:", {m: (round(v["malformed_step"], 4), round(v["direct_nondigit"], 4)) for m, v in mal.items()})
print("pooled malformed other six: %.4f" % (sum(mal[m]["bad"] for m in M6) / sum(mal[m]["n"] for m in M6)))
OUT["malformed"] = mal
# E2 Qwen2.5-7B-Instruct: non-digit generated outputs and accuracy by kernel
for attn in ("eager", "sdpa"):
    gens = [it.get("gen_text", "") for f in T for w in ("w0", "w1") for it in e2load("Qwen2.5-7B-Instruct", f, w, attn)["items"]]
    nd = np.mean([not re.match(r"^\s*\d", g or "") for g in gens])
    print("E2 Q7I %s: non-digit %.4f, strict acc %.3f" % (attn, nd, acc_e2("Qwen2.5-7B-Instruct", ALL, attn)))
    OUT[f"E2_Q7I_{attn}"] = [float(nd), acc_e2("Qwen2.5-7B-Instruct", ALL, attn)]

# relational k=1 over-long traces and reversed lookups
rev = {}
for m in M8:
    over = acc_step = 0; firsts = collections.Counter(); its = [x for x in CACHE[(m, "stepwise", "relational")]["items"] if x["k"] == 1]
    for x in its:
        el = trace_elems(x["gen"]); hops = len(el) - 1
        q = ITEM[x["id"]]["text_stepwise"].split("\n\n")[-1].split("Question:")[0]
        boss = dict(re.findall(r"The boss of (\w) is (\w)\.", q))
        if hops > 1:
            over += 1
            first = None
            for a, b in zip(el[:-1], el[1:]):
                if boss.get(a) == b: continue
                first = "reverse" if boss.get(b) == a else "other"; break
            firsts[first] += 1
    acc_d = np.mean([x["correct"] for x in CACHE[(m, "direct", "relational")]["items"] if x["k"] == 1])
    rev[m] = {"overlong": over / len(its), "n_over": over, "first_error": dict(firsts), "acc_step_k1": float(np.mean([x["correct"] for x in its])), "acc_direct_k1": float(acc_d)}
    print(f"{m:30s} relational k=1 over-long {over}/{len(its)} first errors {dict(firsts)} acc step {rev[m]['acc_step_k1']:.2f} direct {acc_d:.2f}")
OUT["relational_k1"] = rev
json.dump(OUT, open(f"{HERE}/vn_accuracy.json", "w"), indent=1, default=float)
