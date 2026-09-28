"""E2b accuracy and behaviour (task: accuracy/behaviour for the Regime paragraph and Limitations).

Outputs e2b_accuracy_behaviour.json next to this script.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, os, re, json, glob, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(paths.REPLICATION))
import e2b

E2B_ROOT = str(paths.E2B)
E2_ROOT = str(paths.E2_SWEEP)
FAMS = ["pointer", "arith", "binding", "relational"]
KR = {"pointer": range(1, 6), "arith": range(1, 7), "binding": range(1, 7), "relational": range(1, 7)}
MODELS = ["Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B", "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
          "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
Q7 = ["Qwen2.5-7B", "Qwen2.5-7B-Instruct"]
SUBSETS = {"all8": MODELS, "no_q7": [m for m in MODELS if m not in Q7]}
B = 10000
rng = np.random.default_rng(0)
OUT = {}


def short(name):
    return name.split("/")[-1]


def boot(vals, B=B, seed=0):
    """mean and 95% percentile bootstrap CI over models"""
    v = np.asarray(vals, float)
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(v), size=(B, len(v)))
    bs = v[idx].mean(1)
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ───────────────────────── load E2b ─────────────────────────
runs = e2b.load_results(E2B_ROOT)
R = {(short(r["model"]), r["mode"], r["family"]): r for r in runs}
assert set(m for m, _, _ in R) == set(MODELS), set(m for m, _, _ in R)
items_json = json.load(open(os.path.join(E2B_ROOT, "items.json")))
ITEM = {it["id"]: it for f in FAMS for it in items_json[f]}

# sanity: direct and stepwise records are the same items in the same order
for m in MODELS:
    for f in FAMS:
        a = [x["id"] for x in R[(m, "direct", f)]["items"]]
        b = [x["id"] for x in R[(m, "stepwise", f)]["items"]]
        assert a == b
        for x in R[(m, "direct", f)]["items"]:
            assert ITEM[x["id"]]["gold"] == x["gold"] and int(ITEM[x["id"]]["k"]) == x["k"]


def e2b_correct(m, mode, f, kind):
    its = R[(m, mode, f)]["items"]
    k = np.array([x["k"] for x in its])
    if kind == "written":
        c = np.array([bool(x["correct"]) for x in its])
    elif kind == "constrained":
        c = np.array([bool(x["correct_constrained"]) for x in its])
    elif kind == "strict":
        c = np.array([bool(x["correct"]) and bool(x["correct_constrained"]) for x in its])
    return k, c


# ───────────────────────── load E2 ─────────────────────────
def e2_index(attn):
    out = {}
    for fp in glob.glob(f"{E2_ROOT}/bf16_{attn}/*.npz"):
        b = os.path.basename(fp); model = b.split("__")[0].split("_", 1)[1]
        suite = b.split("__")[1]; fam = suite.split("_w")[0]; w = "w" + suite.split("_w")[1][0]
        out[(model, fam, w)] = fp
    return out


IDX = {"eager": e2_index("eager"), "sdpa": e2_index("sdpa")}
_cache = {}


def e2_load(m, f, w, attn):
    key = (m, f, w, attn)
    if key not in _cache:
        z = np.load(IDX[attn][(m, f, w)], allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        _cache[key] = {"k": z["k"].astype(int), "strict": z["correct_strict"].astype(bool),
                       "constrained": z["correct_constrained"].astype(bool),
                       "pred_constrained": z["pred_constrained"].astype(int), "items": meta["items"],
                       "attn": meta["attn_impl"]}
    return _cache[key]


def e2_attn(m):  # the paper's choice
    return "sdpa" if m == "Qwen2.5-7B-Instruct" else "eager"


def e2_acc_by_k(m, f, attn=None, kind="strict"):
    attn = attn or e2_attn(m)
    out = []
    for k in KR[f]:
        vals = []
        for w in ("w0", "w1"):
            d = e2_load(m, f, w, attn)
            vals.append(d[kind][d["k"] == k].mean())
        out.append(float(np.mean(vals)))
    return out


def e2b_acc_by_k(m, mode, f, kind):
    k, c = e2b_correct(m, mode, f, kind)
    return [float(c[k == kk].mean()) for kk in KR[f]]


# ───────────────────── (1) accuracy by family x k ─────────────────────
ACC = {}  # ACC[source][m][f] = list over k
for m in MODELS:
    for src in ("E2", "E2b_direct_written", "E2b_direct_constrained", "E2b_direct_strict",
                "E2b_stepwise_written", "E2b_stepwise_constrained"):
        ACC.setdefault(src, {})[m] = {}
    for f in FAMS:
        ACC["E2"][m][f] = e2_acc_by_k(m, f)
        ACC["E2b_direct_written"][m][f] = e2b_acc_by_k(m, "direct", f, "written")
        ACC["E2b_direct_constrained"][m][f] = e2b_acc_by_k(m, "direct", f, "constrained")
        ACC["E2b_direct_strict"][m][f] = e2b_acc_by_k(m, "direct", f, "strict")
        ACC["E2b_stepwise_written"][m][f] = e2b_acc_by_k(m, "stepwise", f, "written")
        ACC["E2b_stepwise_constrained"][m][f] = e2b_acc_by_k(m, "stepwise", f, "constrained")
OUT["acc_by_model_family_k"] = ACC


def model_summary(src, m, kmin=1):
    """mean over families of the mean over k>=kmin (equal weight per depth; items are balanced)."""
    return float(np.mean([np.mean([a for kk, a in zip(KR[f], ACC[src][m][f]) if kk >= kmin]) for f in FAMS]))


tab = {}
for sub, ms in SUBSETS.items():
    tab[sub] = {}
    for src in ACC:
        tab[sub][src] = {}
        for f in FAMS:
            tab[sub][src][f] = [float(np.mean([ACC[src][m][f][i] for m in ms])) for i in range(len(KR[f]))]
        tab[sub][src]["overall"] = boot([model_summary(src, m, 1) for m in ms])
        tab[sub][src]["k>=3"] = boot([model_summary(src, m, 3) for m in ms])
        tab[sub][src]["k=1"] = boot([np.mean([ACC[src][m][f][0] for f in FAMS]) for m in ms])
    # paired differences
    tab[sub]["diffs"] = {}
    for a, b_ in (("E2b_direct_strict", "E2"), ("E2b_direct_written", "E2"),
                  ("E2b_stepwise_written", "E2b_direct_written"), ("E2b_stepwise_written", "E2b_direct_strict"),
                  ("E2b_stepwise_written", "E2")):
        for kmin in (1, 3):
            tab[sub]["diffs"][f"{a} - {b_} (k>={kmin})"] = boot(
                [model_summary(a, m, kmin) - model_summary(b_, m, kmin) for m in ms])
OUT["acc_table"] = tab

# ───────────────────── (2) per model ─────────────────────
per = {}
for m in MODELS:
    d = {}
    for src in ACC:
        d[src] = {"overall": model_summary(src, m, 1), "k>=3": model_summary(src, m, 3),
                  "k=1": float(np.mean([ACC[src][m][f][0] for f in FAMS]))}
    d["stepwise_min_cell_k>=3"] = float(min(a for f in FAMS for kk, a in zip(KR[f], ACC["E2b_stepwise_written"][m][f]) if kk >= 3))
    d["stepwise_min_cell_all"] = float(min(a for f in FAMS for a in ACC["E2b_stepwise_written"][m][f]))
    d["stepwise_by_family_k>=3"] = {f: float(np.mean([a for kk, a in zip(KR[f], ACC["E2b_stepwise_written"][m][f]) if kk >= 3])) for f in FAMS}
    d["stepwise_by_family_all"] = {f: float(np.mean(ACC["E2b_stepwise_written"][m][f])) for f in FAMS}
    d["direct_by_family_k>=3"] = {f: float(np.mean([a for kk, a in zip(KR[f], ACC["E2b_direct_written"][m][f]) if kk >= 3])) for f in FAMS}
    per[m] = d
OUT["per_model"] = per

# ───────────────────── (3) trace quality ─────────────────────
SEP = {"pointer": "->", "arith": "->", "binding": "=", "relational": "->"}


def trace_elems(gen, fam):
    body = gen.split(".")[0] if "." in gen else gen
    return [x.strip() for x in body.split(SEP[fam])]


def is_state(tok, fam):
    if fam in ("pointer", "arith"):
        return bool(re.fullmatch(r"\d", tok))
    return bool(re.fullmatch(r"[A-Za-z]|\d", tok))


tq = {}
tq_items = {}
for m in MODELS:
    tq[m] = {}
    for f in FAMS:
        its = R[(m, "stepwise", f)]["items"]
        rec = []
        for x in its:
            el = trace_elems(x["gen"], f)
            last = el[-1] if el else None
            gold_el = ITEM[x["id"]]["gold_trace"].split(SEP[f])
            gold_el = [g.strip() for g in gold_el]
            anomalous = any(not is_state(e, f) for e in el)
            rec.append(dict(k=x["k"], parsed=bool(x["parsed"]), exact=bool(x["trace_exact"]),
                            steps=float(x["trace_steps_correct"]), correct=bool(x["correct"]),
                            pred=x.get("pred_gen"), last=last, n_hops=len(el) - 1,
                            gold_hops=len(gold_el) - 1, anomalous=anomalous,
                            first_ok=(el[0] == gold_el[0]) if el else False,
                            ntok=x.get("n_tokens_measured")))
        tq_items[(m, f)] = rec
        n = len(rec)
        parsed = [r for r in rec if r["parsed"]]
        wrong = [r for r in parsed if not r["correct"]]
        right = [r for r in parsed if r["correct"]]
        exact = [r for r in rec if r["exact"]]
        tq[m][f] = {
            "n": n,
            "parse_rate": np.mean([r["parsed"] for r in rec]),
            "exact_rate": np.mean([r["exact"] for r in rec]),
            "steps_correct_mean": np.mean([r["steps"] for r in rec]),
            "answer_eq_last_state_among_parsed": np.mean([r["pred"] == r["last"] for r in parsed]) if parsed else None,
            "n_wrong_parsed": len(wrong),
            "wrong_with_inexact_trace": np.mean([not r["exact"] for r in wrong]) if wrong else None,
            "wrong_answer_eq_last_state": np.mean([r["pred"] == r["last"] for r in wrong]) if wrong else None,
            "correct_given_exact": np.mean([r["correct"] for r in exact]) if exact else None,
            "correct_with_inexact_trace": np.mean([not r["exact"] for r in right]) if right else None,
            "anomalous_token_rate": np.mean([r["anomalous"] for r in rec]),
            "exact_by_k": [float(np.mean([r["exact"] for r in rec if r["k"] == kk])) for kk in KR[f]],
            "parse_by_k": [float(np.mean([r["parsed"] for r in rec if r["k"] == kk])) for kk in KR[f]],
        }
        tq[m][f] = {kk: (float(v) if isinstance(v, (np.floating, float)) else v) for kk, v in tq[m][f].items()}

# pooled over models (equal weight per model x family)
def pooled(ms, key):
    vals = [tq[m][f][key] for m in ms for f in FAMS if tq[m][f][key] is not None]
    return float(np.mean(vals))


def pooled_items(ms, fn_sel, fn_val):
    v = [fn_val(r) for m in ms for f in FAMS for r in tq_items[(m, f)] if fn_sel(r)]
    return float(np.mean(v)), len(v)


tq_sum = {}
for sub, ms in SUBSETS.items():
    s = {key: pooled(ms, key) for key in ("parse_rate", "exact_rate", "steps_correct_mean",
                                          "answer_eq_last_state_among_parsed", "anomalous_token_rate")}
    s["items_wrong_parsed_with_inexact_trace"] = pooled_items(ms, lambda r: r["parsed"] and not r["correct"], lambda r: not r["exact"])
    s["items_wrong_parsed_answer_eq_last"] = pooled_items(ms, lambda r: r["parsed"] and not r["correct"], lambda r: r["pred"] == r["last"])
    s["items_correct_given_exact"] = pooled_items(ms, lambda r: r["exact"], lambda r: r["correct"])
    s["items_exact_given_correct"] = pooled_items(ms, lambda r: r["correct"], lambda r: r["exact"])
    s["items_parsed_answer_eq_last"] = pooled_items(ms, lambda r: r["parsed"], lambda r: r["pred"] == r["last"])
    s["items_unparsed_share"] = pooled_items(ms, lambda r: True, lambda r: not r["parsed"])
    s["items_unparsed_with_anomalous_token"] = pooled_items(ms, lambda r: not r["parsed"], lambda r: r["anomalous"])
    tq_sum[sub] = s
OUT["trace_quality"] = {"per_model_family": tq, "summary": tq_sum}

# per-model trace summary over families
tq_model = {}
for m in MODELS:
    tq_model[m] = {key: float(np.mean([tq[m][f][key] for f in FAMS])) for key in
                   ("parse_rate", "exact_rate", "steps_correct_mean", "anomalous_token_rate")}
    tq_model[m]["answer_eq_last_state_among_parsed"] = pooled_items([m], lambda r: r["parsed"], lambda r: r["pred"] == r["last"])
    tq_model[m]["wrong_parsed_with_inexact_trace"] = pooled_items([m], lambda r: r["parsed"] and not r["correct"], lambda r: not r["exact"]) \
        if any(r["parsed"] and not r["correct"] for f in FAMS for r in tq_items[(m, f)]) else None
OUT["trace_quality_per_model"] = tq_model

# ───────────────────── (4) failure modes ─────────────────────
# trace length vs true k
tl = {}
for m in MODELS:
    tl[m] = {}
    for f in FAMS:
        rec = tq_items[(m, f)]
        tl[m][f] = {}
        for kk in KR[f]:
            rr = [r for r in rec if r["k"] == kk]
            nh = np.array([r["n_hops"] for r in rr])
            tl[m][f][kk] = {"mean_hops": float(nh.mean()), "over": float(np.mean(nh > kk)),
                            "under": float(np.mean(nh < kk)), "equal": float(np.mean(nh == kk)),
                            "acc": float(np.mean([r["correct"] for r in rr])),
                            "parsed": float(np.mean([r["parsed"] for r in rr]))}
OUT["trace_length"] = tl
# overshoot summary over models
over = {}
for sub, ms in SUBSETS.items():
    over[sub] = {f: {kk: {"over": float(np.mean([tl[m][f][kk]["over"] for m in ms])),
                          "under": float(np.mean([tl[m][f][kk]["under"] for m in ms])),
                          "mean_hops": float(np.mean([tl[m][f][kk]["mean_hops"] for m in ms]))}
                     for kk in KR[f]} for f in FAMS}
OUT["trace_length_summary"] = over

# relational k=1 detail: among overshooting traces, where does the extra hop come from?
rel1 = {}
for m in MODELS:
    rr = [(r, x) for r, x in zip(tq_items[(m, "relational")], R[(m, "stepwise", "relational")]["items"]) if r["k"] == 1]
    rel1[m] = {"acc": float(np.mean([r["correct"] for r, _ in rr])),
               "parsed": float(np.mean([r["parsed"] for r, _ in rr])),
               "over": float(np.mean([r["n_hops"] > 1 for r, _ in rr])),
               "first_elem_ok": float(np.mean([r["first_ok"] for r, _ in rr])),
               "examples": [x["gen"][:120] for r, x in rr if r["n_hops"] > 1][:3]}
OUT["relational_k1_stepwise"] = rel1


# hops done, pointer / arith
def hops_done(fam, text, pred):
    b = text.split("\n\n")[-1]
    if pred is None or not str(pred).isdigit():
        return None
    pred = int(pred)
    if fam == "pointer":
        tbl = dict((int(a), int(c)) for a, c in re.findall(r"(\d) (?:goes|leads) to (\d)\.", b))
        s = int(re.search(r"(?:Start at|From) (\d)", b).group(1)); path = [s]
        for _ in range(9): path.append(tbl[path[-1]])
        return path.index(pred) if pred in path else None
    mm = re.search(r"add (\d) and keep the remainder after dividing by (\d+)\. Start at (\d)", b)
    if mm:
        d, mod, s = map(int, mm.groups())
    else:
        s, d, mod = map(int, re.search(r"x = (\d)\. Repeat \d+ times: x = \(x \+ (\d)\) mod (\d+)", b).groups())
    if pred >= mod: return None
    return ((pred - s) * pow(d, -1, mod)) % mod


hops = {"E2b_direct": {}, "E2b_stepwise": {}, "E2_8models": {}, "E2_17models": {}}
for f in ("pointer", "arith"):
    for mode in ("direct", "stepwise"):
        for key in ("pred_constrained", "pred_gen"):
            per_k = {}
            for kk in KR[f]:
                h = []
                for m in MODELS:
                    for x in R[(m, mode, f)]["items"]:
                        if x["k"] != kk: continue
                        h.append(hops_done(f, ITEM[x["id"]]["text_direct"], x.get(key)))
                h = np.array([np.nan if v is None else v for v in h], float)
                per_k[kk] = {"n": len(h), "one_step": float(np.mean(h == 1)), "correct": float(np.mean(h == kk)),
                             "k_minus_1": float(np.mean(h == kk - 1)), "k_plus_1": float(np.mean(h == kk + 1)),
                             "zero": float(np.mean(h == 0)), "unmapped": float(np.mean(np.isnan(h)))}
            hops[f"E2b_{mode}"][f"{f}/{key}"] = per_k
    # E2, same computation (paper: pred_constrained, all items, eager)
    all17 = sorted({m for (m, _, _) in IDX["eager"]})
    for label, ms in (("E2_8models", MODELS), ("E2_17models", all17)):
        for key in ("pred_constrained", "pred_first_digit"):
            for attn_rule in ("paper", "all_eager"):
                h = []
                for m in ms:
                    attn = e2_attn(m) if attn_rule == "paper" else "eager"
                    for w in ("w0", "w1"):
                        d = e2_load(m, f, w, attn)
                        for it in d["items"]:
                            if it["k"] != 2: continue
                            h.append(hops_done(f, it["text"], it.get(key)))
                h = np.array([np.nan if v is None else v for v in h], float)
                hops[label][f"{f}/{key}/{attn_rule}/k=2"] = {"n": len(h), "one_step": float(np.mean(h == 1)),
                                                             "correct": float(np.mean(h == 2))}
OUT["hops_done"] = hops

# per model one-step share at k=2 direct (pred_gen)
os_pm = {}
for f in ("pointer", "arith"):
    for m in MODELS:
        h = [hops_done(f, ITEM[x["id"]]["text_direct"], x.get("pred_gen")) for x in R[(m, "direct", f)]["items"] if x["k"] == 2]
        os_pm.setdefault(f, {})[m] = float(np.mean([v == 1 for v in h]))
OUT["one_step_k2_direct_per_model"] = os_pm

# ───────────────────── (5) direct-mode continuation ─────────────────────
cont = {}
for m in MODELS:
    cont[m] = {}
    for f in FAMS:
        its = R[(m, "direct", f)]["items"]
        agree = [x["pred_gen"] == x["pred_constrained"] for x in its]
        first_is_digit = [bool(re.match(r"\s*\d", x["gen"])) for x in its]
        nodigit = [x["pred_gen"] is None for x in its]
        # later digits in the continuation that differ from the first one
        digits = [re.findall(r"\d", x["gen"]) for x in its]
        later_diff = [len(set(d)) > 1 for d in digits]
        # among disagreements, which is right?
        dis = [x for x in its if x["pred_gen"] != x["pred_constrained"]]
        cont[m][f] = {"n": len(its), "agree": float(np.mean(agree)), "first_token_digit": float(np.mean(first_is_digit)),
                      "no_digit": float(np.mean(nodigit)), "multiple_distinct_digits": float(np.mean(later_diff)),
                      "n_disagree": len(dis),
                      "disagree_gen_correct": int(sum(x["correct"] for x in dis)),
                      "disagree_constrained_correct": int(sum(x["correct_constrained"] for x in dis)),
                      "late_digit_items": int(sum((not fd) and (pg is not None) for fd, pg in zip(first_is_digit, [x["pred_gen"] for x in its]))),
                      "late_digit_correct": int(sum((not fd) and x["correct"] for fd, x in zip(first_is_digit, its))),
                      "examples_nonfirst": [x["gen"] for x, fd in zip(its, first_is_digit) if not fd][:4]}
OUT["direct_continuation"] = cont
cont_sum = {}
for sub, ms in SUBSETS.items():
    cont_sum[sub] = {key: float(np.mean([cont[m][f][key] for m in ms for f in FAMS]))
                     for key in ("agree", "first_token_digit", "no_digit", "multiple_distinct_digits")}
    cont_sum[sub]["n_disagree"] = int(sum(cont[m][f]["n_disagree"] for m in ms for f in FAMS))
    cont_sum[sub]["n_items"] = int(sum(cont[m][f]["n"] for m in ms for f in FAMS))
    cont_sum[sub]["disagree_gen_correct"] = int(sum(cont[m][f]["disagree_gen_correct"] for m in ms for f in FAMS))
    cont_sum[sub]["disagree_constrained_correct"] = int(sum(cont[m][f]["disagree_constrained_correct"] for m in ms for f in FAMS))
    cont_sum[sub]["late_digit_items"] = int(sum(cont[m][f]["late_digit_items"] for m in ms for f in FAMS))
    cont_sum[sub]["late_digit_correct"] = int(sum(cont[m][f]["late_digit_correct"] for m in ms for f in FAMS))
OUT["direct_continuation_summary"] = cont_sum

# ───────────────────── (6) kernel check ─────────────────────
kern = {}
for m in sorted({m for (m, _, _) in IDX["sdpa"]}):
    cells = [(f, w) for (mm, f, w) in IDX["sdpa"] if mm == m and (m, f, w) in IDX["eager"]]
    acc_e, acc_s, agree, nd_e, nd_s, n = [], [], [], [], [], 0
    for f, w in cells:
        de, ds = e2_load(m, f, w, "eager"), e2_load(m, f, w, "sdpa")
        assert [it["id"] for it in de["items"]] == [it["id"] for it in ds["items"]]
        acc_e.append(de["strict"].mean()); acc_s.append(ds["strict"].mean())
        agree += list(de["pred_constrained"] == ds["pred_constrained"])
        nd_e += [not re.match(r"\s*\d", it["gen_text"] or "") for it in de["items"]]
        nd_s += [not re.match(r"\s*\d", it["gen_text"] or "") for it in ds["items"]]
    serial = [i for i, (f, w) in enumerate(cells) if f != "fixed_digit"]
    kern[m] = {"cells": [f"{f}/{w}" for f, w in cells],
               "strict_eager_all_cells": float(np.mean(acc_e)), "strict_sdpa_all_cells": float(np.mean(acc_s)),
               "strict_eager_serial": float(np.mean([acc_e[i] for i in serial])) if serial else None,
               "strict_sdpa_serial": float(np.mean([acc_s[i] for i in serial])) if serial else None,
               "pred_constrained_agreement": float(np.mean(agree)),
               "gen_first_not_digit_eager": float(np.mean(nd_e)), "gen_first_not_digit_sdpa": float(np.mean(nd_s))}
OUT["E2_kernel_eager_vs_sdpa"] = kern

# E2b: Qwen-7B pair against E2 eager and E2 SDPA (serial families, strict, overall)
q7cmp = {}
for m in Q7:
    q7cmp[m] = {"E2b_direct_strict": model_summary("E2b_direct_strict", m, 1),
                "E2b_direct_written": model_summary("E2b_direct_written", m, 1),
                "E2b_stepwise_written": model_summary("E2b_stepwise_written", m, 1),
                "E2_eager_strict": float(np.mean([np.mean(e2_acc_by_k(m, f, "eager")) for f in FAMS]))}
    if all((m, f, w) in IDX["sdpa"] for f in FAMS for w in ("w0", "w1")):
        q7cmp[m]["E2_sdpa_strict"] = float(np.mean([np.mean(e2_acc_by_k(m, f, "sdpa")) for f in FAMS]))
OUT["q7_accuracy_by_kernel"] = q7cmp

# direct-mode anomalous outputs per model (first generated token not a digit)
OUT["direct_first_not_digit_per_model"] = {m: float(np.mean([1 - cont[m][f]["first_token_digit"] for f in FAMS])) for m in MODELS}

# stepwise context length at measured position
ctx = {}
for m in MODELS:
    for f in FAMS:
        rec = tq_items[(m, f)]
        ctx.setdefault(m, {})[f] = {kk: float(np.nanmean([r["ntok"] if r["ntok"] else np.nan for r in rec if r["k"] == kk])) for kk in KR[f]}
OUT["stepwise_measured_ntokens"] = ctx


def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


json.dump(conv(OUT), open(os.path.join(HERE, "e2b_accuracy_behaviour.json"), "w"), indent=1)
print("saved")
