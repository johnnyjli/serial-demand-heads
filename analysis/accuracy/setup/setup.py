"""Setup checks and error structure for the E2 accuracy (research_outline.tex, tab:acc).

Reads items.pkl (load_items.py: every E2 run, eager and SDPA, no head arrays) and writes setup.json.
Main convention (as tab:acc): all 17 models, eager, except Qwen2.5-7B-Instruct under SDPA; both
wordings; all items (no OLMo-2 exclusion); strict correctness.
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, os, pickle, re, sys, itertools, collections
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = str(paths.ANALYSIS)
sys.path.insert(0, OLD)
from solve import blocks, solve_block          # independent solver (earlier session)

ORDER = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.2-3B-Instruct", "Llama-3.1-8B", "Llama-3.1-8B-Instruct",
         "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
         "OLMo-2-1124-7B", "OLMo-2-1124-7B-SFT", "OLMo-2-1124-7B-DPO", "OLMo-2-1124-7B-Instruct",
         "Mistral-7B-v0.1", "Mistral-Small-24B-Base-2501"]
SERIAL = ["pointer", "arith", "binding", "relational"]
FAMS = SERIAL + ["fixed_digit"]
CHANCE = {"pointer": .1, "arith": 1 / 7, "binding": 1 / 3, "relational": 1 / 3, "fixed_digit": 1 / 7}
PAPER_ACC = {"pointer": [.57, .16, .09, .07, .06], "arith": [.63, .15, .16, .13, .12, .11],
             "binding": [.86, .61, .44, .34, .32, .28], "relational": [.76, .45, .25, .27, .23, .27],
             "fixed_digit": [.41, .30, .17, .15, .14, .03]}
PARAMS_B = {"Llama-3.2-1B": 1.24, "Llama-3.2-3B": 3.21, "Llama-3.1-8B": 8.03, "Qwen2.5-0.5B": .49,
            "Qwen2.5-1.5B": 1.54, "Qwen2.5-3B": 3.09, "Qwen2.5-7B": 7.62, "Qwen2.5-14B": 14.7,
            "OLMo-2-1124-7B": 7.30, "Mistral-7B-v0.1": 7.24, "Mistral-Small-24B-Base-2501": 23.6}
BASE = list(PARAMS_B)

D = pickle.load(open(os.path.join(HERE, "items.pkl"), "rb"))
A = pd.DataFrame(D["rows"]); RUNS = D["runs"]
OUT = {}
r3 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def main_table(q7="sdpa"):
    """17 models; eager except Qwen2.5-7B-Instruct under q7."""
    keep = ((A.attn == "eager") & (A.model != "Qwen2.5-7B-Instruct")) | \
           ((A.model == "Qwen2.5-7B-Instruct") & (A.attn == q7))
    return A[keep].copy()


M = main_table("sdpa")
assert sorted(M.model.unique()) == sorted(ORDER) and len(M) == 17 * 2436, len(M)

# ─────────────────────────────── 1. answer verification ───────────────────────────────
def walk_pointer(b):
    tbl = dict((int(a), int(c)) for a, c in re.findall(r"(\d) (?:goes|leads) to (\d)\.", b))
    s = int(re.search(r"(?:Start at|From) (\d)", b).group(1))
    kw = int(re.search(r"(?:Take|make) (\d+) (?:steps|moves)", b).group(1))
    return tbl, s, kw


def arith_params(w, b):
    if w == "w0":
        d, mod, s, kw = map(int, re.search(r"add (\d) and keep the remainder after dividing by (\d+)\. "
                                           r"Start at (\d)\. Apply the rule (\d+) times", b).groups())
    else:
        s, kw, d, mod = map(int, re.search(r"x = (\d)\. Repeat (\d+) times: x = \(x \+ (\d)\) mod (\d+)", b).groups())
    return s, d, mod, kw


QMARK = {("binding", "w0"): "Therefore ", ("binding", "w1"): "The value of ",
         ("relational", "w0"): "The top boss above ", ("relational", "w1"): None}


def split_query(fam, w, b):
    """(statement text, query sentence).  The query sentence is cut off before the statements are
    parsed, so a shown answer (e.g. 'Therefore y = 3.') can never be read as a statement."""
    b = b.strip()
    if (fam, w) == ("relational", "w1"):
        i = b.rindex(" ultimately reports to"); j = b.rindex(". ", 0, i) + 2
    else:
        j = b.rindex(QMARK[(fam, w)])
    return b[:j], b[j:]


def chains(fam, w, b):
    """Statements in text order and the query entity; also the shown answer (or None)."""
    body, q = split_query(fam, w, b)
    if fam == "binding":
        st = re.findall(r"(\w) = (\w)\." if w == "w0" else r"Let (\w) be (\w)\.", body)
        mq = re.fullmatch(r"(?:Therefore|The value of) (\w) (?:=|is)(?: (\d)\.)?", q)
    else:
        st = re.findall(r"The boss of (\w) is (\w)\." if w == "w0" else r"(\w) reports to (\w)\.", body)
        mq = re.fullmatch(r"(?:The top boss above (\w) is|(\w) ultimately reports to)(?: (\d)\.)?", q)
    g = [x for x in mq.groups() if x is not None]
    qv = g[0]; shown = int(g[1]) if len(g) > 1 else None
    assert len(st) == body.count(".") == 21, (len(st), body.count("."))   # 3 chains x 7 statements, none missed
    return st, qv, shown


def solve2(fam, w, b):
    """Second independent solver (fixes solve.py's binding-w0 regex, which also reads the shown
    answer 'Therefore y = 3.' as a statement and so re-derives any shown answer)."""
    if fam in ("pointer", "fixed_digit"):
        tbl, s, kw = walk_pointer(b)
        assert sorted(tbl) == sorted(set(tbl.values())) == list(range(len(tbl)))   # a permutation
        for _ in range(kw): s = tbl[s]
        m = re.search(r"(?:end at|finish at) (\d)\.$", b.strip()); return s, (int(m.group(1)) if m else None)
    if fam == "arith":
        s, d, mod, kw = arith_params(w, b); x = s
        for _ in range(kw): x = (x + d) % mod
        m = re.search(r"(?:end at|Now x =) (\d)\.$", b.strip()); return x, (int(m.group(1)) if m else None)
    st, qv, shown = chains(fam, w, b)
    lhs = [a for a, _ in st]; assert len(lhs) == len(set(lhs)), "entity defined twice"
    return int(hops_to_root(st, qv)[0]), shown


def hops_to_root(st, v):
    nxt = dict(st); n = 0
    while not v.isdigit():
        v = nxt[v]; n += 1
    return v, n


def depth_of_block(fam, w, b, shot=False):
    """Serial depth as written/implied by the block text."""
    if fam == "fixed_digit" and shot:
        return walk_pointer(b)[2]
    if fam in ("pointer", "fixed_digit"):
        tbl, s, kw = walk_pointer(b)
        if fam == "pointer":
            return kw
        # cycle length through the start - 1 (the item's k in the numeral control)
        c, x = 1, tbl[s]
        while x != s:
            x = tbl[x]; c += 1
        return c - 1
    if fam == "arith":
        return arith_params(w, b)[3]
    st, qv, _ = chains(fam, w, b)
    return hops_to_root(st, qv)[1]


ver = {"prompts": 0, "worked_examples": 0, "gold_mismatch": 0, "shown_mismatch": 0, "k_mismatch": 0,
       "solver2_gold_mismatch": 0, "solver2_shown_mismatch": 0, "solvepy_binding_w0_shot_check_circular": None,
       "texts_identical_across_runs": True, "n_distinct_texts_per_id_max": 0,
       "shots_per_prompt": collections.Counter(), "shot_depths": collections.Counter(),
       "shot_equals_query": 0, "fixed_digit_written_k": collections.Counter(),
       "by_cell": {}}
g = A.groupby(["cell", "id"])
bal = collections.defaultdict(collections.Counter)
shot_ans_has_gold = collections.defaultdict(list)
item_shots = {}
for (cell, iid), grp in g:
    texts = grp.text.unique(); ver["n_distinct_texts_per_id_max"] = max(ver["n_distinct_texts_per_id_max"], len(texts))
    if len(texts) != 1: ver["texts_identical_across_runs"] = False
    assert grp.model.nunique() == 17
    t = texts[0]; fam, w = cell.split("/"); r0 = grp.iloc[0]
    bl = blocks(t); q = bl[-1]; shots = bl[:-1]
    ver["prompts"] += 1; ver["by_cell"][cell] = ver["by_cell"].get(cell, 0) + 1
    ans, shown = solve_block(fam, w, q)
    assert shown is None
    if ans != r0.gold: ver["gold_mismatch"] += 1
    a2q, sh2q = solve2(fam, w, q); assert sh2q is None
    if a2q != r0.gold: ver["solver2_gold_mismatch"] += 1
    if depth_of_block(fam, w, q) != r0.k: ver["k_mismatch"] += 1
    if fam == "fixed_digit": ver["fixed_digit_written_k"][walk_pointer(q)[2]] += 1
    ver["shots_per_prompt"][len(shots)] += 1
    sh_ans = []
    for sb in shots:
        ver["worked_examples"] += 1
        a2, s2 = solve_block(fam, w, sb)
        if s2 is None or a2 != s2: ver["shown_mismatch"] += 1
        a3, s3 = solve2(fam, w, sb)
        if s3 is None or a3 != s3 or s3 != s2: ver["solver2_shown_mismatch"] += 1
        ver["shot_depths"][(fam, depth_of_block(fam, w, sb, shot=True))] += 1
        # the worked example's question must not be the query (compare text before the answer)
        if sb.strip().rsplit(" ", 1)[0] == q.strip(): ver["shot_equals_query"] += 1
        sh_ans.append(s2)
    item_shots[(cell, iid)] = sh_ans
    bal[(cell, r0.k)][r0.gold] += 1
ver["shots_per_prompt"] = dict(ver["shots_per_prompt"])
_b = ("Assignments: u = e. w = g. k = j. z = 3. r = y. j = s. q = z. b = k. s = u. c = 2. p = m. x = l. g = r. "
      "t = p. m = x. n = v. l = 6. y = q. e = c. v = t. d = w. Therefore y = 9.")   # true answer 3
ver["solvepy_binding_w0_shot_check_circular"] = {"solve.py(answer, shown) on a corrupted shown answer 9": list(solve_block("binding", "w0", _b)),
                                                 "solver2": list(solve2("binding", "w0", _b))}
ver["shot_depths"] = {f"{f}:{d}": n for (f, d), n in sorted(ver["shot_depths"].items())}
ver["fixed_digit_written_k"] = dict(ver["fixed_digit_written_k"])
# answer balance within depth: max/min count of gold digits per (cell, k)
balance = {}
for (cell, k), c in sorted(bal.items()):
    balance.setdefault(cell, {})[k] = {"n_distinct_gold": len(c), "min": min(c.values()), "max": max(c.values())}
ver["gold_balance_by_cell_k"] = balance
# gold never equals the written numeral (pointer/arith)
PA = M[M.fam.isin(["pointer", "arith"])]
ver["gold_equals_echo"] = int((PA.gold == PA.echo).sum())
# prompt length constant across k within family (meta n_tokens_by_k, per run)
nt_var = []
for r in RUNS:
    lens = {k: sorted(set(v)) for k, v in r["n_tokens_by_k"].items()}
    allv = sorted(set(x for v in lens.values() for x in v))
    nt_var.append((r["model"], r["fam"], r["w"], len(allv), allv[0], allv[-1]))
ver["runs_with_prompt_length_varying_over_k"] = sum(1 for x in nt_var if x[3] > 1)
ver["prompt_length_ranges_example"] = [x for x in nt_var if x[3] > 1][:10]
# pointer min_hops == k (no cycle shortcut on the 10-cycle with k<=5)
P0 = M[M.fam == "pointer"]
ver["pointer_min_hops_equals_k"] = bool((P0.min_hops == P0.k).all())
OUT["1_answer_verification"] = ver
print("1:", {k: v for k, v in ver.items() if k not in ("gold_balance_by_cell_k", "by_cell")})

# ─────────────────────────────── 2. measurement checks ───────────────────────────────
meas = {}
for attn in ("eager", "sdpa"):
    R = [r for r in RUNS if r["attn"] == attn]
    ch = pd.DataFrame([dict(model=r["model"], cell=f'{r["fam"]}/{r["w"]}', **{k: v for k, v in r["checks"].items()
                       if isinstance(v, (int, float))}) for r in R])
    maxrow = ch.loc[ch.split_err.idxmax()]
    meas[attn] = {"n_runs": len(R), "n_items": int(sum(r["n_items"] for r in R)),
                  "max_split_err": r3(ch.split_err.max()), "max_split_err_where": f'{maxrow.model} {maxrow.cell}',
                  "max_split_exact_err": float(ch.split_exact_err.max()),
                  "max_post_norm_err": r3(ch.post_norm_err.max()),
                  "max_post_norm_input_err": r3(ch.post_norm_input_err.max()),
                  "max_attn_row_err": r3(ch.attn_row_err.max()),
                  "runs_with_split_warn(>1e-2)": int((ch.split_warn > 0).sum()),
                  "split_checks_total": int(ch.split_checks.sum()),
                  "all_verified": bool(all(r["checks"].get("verified") for r in R)),
                  "split_err_by_model_max": {m: r3(v) for m, v in ch.groupby("model").split_err.max().items()},
                  "runs_appending_a_token": int(sum(1 for r in R if any(len(a) for a in r["appended_forms"]))),
                  "runs_appending_only_whitespace": int(sum(1 for r in R if all(
                      len(a) == 1 for a in r["appended_forms"]))),
                  "appended_forms_by_model": {m: sorted({tuple(a) for r in R if r["model"] == m
                                                         for a in r["appended_forms"]}) for m in ORDER
                                              if any(r["model"] == m for r in R)},
                  "items_appended": int(sum(sum(r["measured_tokens"].values()) for r in R
                                            if any(len(a) for a in r["appended_forms"])))}
    meas[attn]["appended_forms_by_model"] = {m: [list(a) for a in v] for m, v in meas[attn]["appended_forms_by_model"].items()}
meas["items_with_appended_token_main_table"] = int((M.appended.map(len) > 0).sum())
meas["items_main_table"] = len(M)
# digit probability mass at the measured position
dm = M.groupby("model").digit_mass.agg(["min", "median", "mean"])
meas["digit_mass_by_model"] = {m: {k: r3(v) for k, v in dm.loc[m].items()} for m in ORDER}
meas["digit_mass_overall"] = {"mean": r3(M.digit_mass.mean()), "median": r3(M.digit_mass.median()),
                              "min_model_mean": r3(dm["mean"].min()),
                              "frac_items_below_0.5": r3((M.digit_mass < .5).mean())}
# answer-set mass: probability of the admissible digits relative to all digits is not stored; report
# ties at the maximum (bf16) and how many were scored wrong only because of a tie
meas["ties"] = {"items_tied_at_max": int((M.n_tied > 1).sum()), "frac_tied": r3((M.n_tied > 1).mean()),
                "gold_among_tied": int(M.gold_tied.sum()), "frac_gold_among_tied": r3(M.gold_tied.mean()),
                "gold_tied_by_model": {m: int(v) for m, v in M.groupby("model").gold_tied.sum().items()}}
OUT["2_measurement_checks"] = meas
print("2:", {a: {k: v for k, v in meas[a].items() if k not in ("split_err_by_model_max", "appended_forms_by_model")}
             for a in ("eager", "sdpa")})
print("   ties", meas["ties"]); print("   digit mass", meas["digit_mass_overall"])

# ─────────────────────────────── 3. output format ───────────────────────────────
def fmt(T):
    gt = T.gen_text.fillna("")
    first_is_digit = gt.map(lambda s: len(s) > 0 and s[0].isdigit())
    first_nonspace_digit = gt.map(lambda s: len(s.lstrip()) > 0 and s.lstrip()[0].isdigit())
    has_digit = T.pred_fd >= 0
    agree = (T.pred_c == T.pred_fd)
    notie = T.n_tied <= 1
    # argmax over the admissible set, ties broken by lowest digit (not against gold)
    def argmax_adm(r):
        dl = r.digit_logits; return max(r.answer_set, key=lambda d: (dl[d], -d))
    am = T.apply(argmax_adm, axis=1)
    return {"n": len(T), "first_char_digit": r3(first_is_digit.mean()),
            "first_nonspace_char_digit": r3(first_nonspace_digit.mean()),
            "any_digit_generated": r3(has_digit.mean()),
            "agree_constrained_vs_first_digit": r3(agree.mean()),
            "agree_excl_ties": r3(agree[notie].mean()),
            "agree_argmax_admissible_vs_first_digit": r3((am == T.pred_fd).mean()),
            "first_digit_outside_answer_set": r3(T.apply(lambda r: r.pred_fd >= 0 and r.pred_fd not in r.answer_set,
                                                         axis=1).mean()),
            "acc_constrained": r3(T.corr_c.mean()), "acc_strict": r3(T.corr_s.mean()),
            "acc_first_digit": r3((T.pred_fd == T.gold).mean()),
            "constrained_minus_strict": r3(T.corr_c.mean() - T.corr_s.mean())}


of = {"main_17_models": fmt(M)}
of["by_model"] = {m: fmt(M[M.model == m]) for m in ORDER}
Q7 = A[A.model == "Qwen2.5-7B-Instruct"]
of["Qwen2.5-7B-Instruct_eager"] = fmt(Q7[Q7.attn == "eager"])
of["Qwen2.5-7B-Instruct_sdpa"] = fmt(Q7[Q7.attn == "sdpa"])
of["16_models_excl_Qwen7BInstruct"] = fmt(M[M.model != "Qwen2.5-7B-Instruct"])
of["by_family"] = {f: fmt(M[M.fam == f]) for f in FAMS}
# largest per model x cell gap strict vs constrained
gap = M.groupby(["model", "cell"]).apply(lambda T: T.corr_c.mean() - T.corr_s.mean())
of["max_model_cell_gap_constrained_minus_strict"] = {"value": r3(gap.max()), "where": " ".join(gap.idxmax())}
of["min_model_cell_gap"] = {"value": r3(gap.min()), "where": " ".join(gap.idxmin())}
of["first_char_nondigit_by_model"] = {m: r3(1 - of["by_model"][m]["first_char_digit"]) for m in ORDER}
# what non-digit starts look like
nd = M[~M.gen_text.fillna("").map(lambda s: len(s) > 0 and s[0].isdigit())]
of["nondigit_start_examples"] = collections.Counter(nd.gen_text.fillna("<none>").map(lambda s: s[:12])).most_common(12)
of["nondigit_start_by_model"] = nd.model.value_counts().to_dict()
OUT["3_output_format"] = of
print("3:", of["main_17_models"]); print("   Q7 eager", of["Qwen2.5-7B-Instruct_eager"]); print("   Q7 sdpa", of["Qwen2.5-7B-Instruct_sdpa"])
print("   gap max", of["max_model_cell_gap_constrained_minus_strict"], of["nondigit_start_by_model"], of["nondigit_start_examples"][:6])

# ─────────────────────────────── 4. accuracy by k ───────────────────────────────
def acc_model_k(T, col="corr_s"):
    """model x (fam, k) accuracy, both wordings pooled (equal n per wording)."""
    return T.groupby(["model", "fam", "k"])[col].mean().unstack(["fam", "k"])


acc = acc_model_k(M)
acc4 = {}
for f in FAMS:
    ks = sorted(M[M.fam == f].k.unique())
    mean_k = {int(k): r3(acc[(f, k)].mean()) for k in ks}
    acc4[f] = {"chance": r3(CHANCE[f]), "mean_over_17_by_k": mean_k,
               "paper_tab_acc": dict(zip(map(int, ks), PAPER_ACC[f])),
               "max_abs_diff_vs_paper": r3(max(abs(mean_k[int(k)] - p) for k, p in zip(ks, PAPER_ACC[f]))),
               "median_over_17_by_k": {int(k): r3(acc[(f, k)].median()) for k in ks},
               "k1_range": {"min": r3(acc[(f, 1)].min()), "min_model": acc[(f, 1)].idxmin(),
                            "max": r3(acc[(f, 1)].max()), "max_model": acc[(f, 1)].idxmax()},
               "k1_by_model": {m: r3(acc.loc[m, (f, 1)]) for m in ORDER},
               "k2_by_model": {m: r3(acc.loc[m, (f, 2)]) for m in ORDER},
               "constrained_mean_by_k": {int(k): r3(M[(M.fam == f) & (M.k == k)].groupby("model").corr_c.mean().mean())
                                         for k in ks}}
    # models above chance at k=1 and k=2 (one-sided binomial on 84 items = 42 x 2 wordings; wordings
    # share instances, so this overstates n; also reported with 42 items of w0 only)
    for kk in (1, 2, 3):
        n_abv = 0; n_abv_w0 = 0
        for m in ORDER:
            T = M[(M.model == m) & (M.fam == f) & (M.k == kk)]
            n_abv += stats.binomtest(int(T.corr_s.sum()), len(T), CHANCE[f], alternative="greater").pvalue < .05
            T0 = T[T.w == "w0"]
            n_abv_w0 += stats.binomtest(int(T0.corr_s.sum()), len(T0), CHANCE[f], alternative="greater").pvalue < .05
        acc4[f][f"n_models_above_chance_k{kk}(p<.05,n=84)"] = int(n_abv)
        acc4[f][f"n_models_above_chance_k{kk}(p<.05,w0 n=42)"] = int(n_abv_w0)
# pooled over the four serial families at k=1 (mean over families, per model)
k1_serial = pd.DataFrame({f: acc[(f, 1)] for f in SERIAL})
acc4["serial_k1_mean_over_families_by_model"] = {m: r3(k1_serial.loc[m].mean()) for m in ORDER}
acc4["serial_k1_min_over_families_by_model"] = {m: r3(k1_serial.loc[m].min()) for m in ORDER}
# Accuracy at k=2,3 (depths shown in the worked examples) vs k>=4
for f in SERIAL:
    acc4[f]["mean_k2_k3"] = r3(np.mean([acc[(f, k)].mean() for k in (2, 3)]))
    acc4[f]["mean_k_ge4"] = r3(np.mean([acc[(f, k)].mean() for k in sorted(M[M.fam == f].k.unique()) if k >= 4]))
# all-eager version (Qwen2.5-7B-Instruct eager) for reference
acc_e = acc_model_k(main_table("eager"))
acc4["all_eager_mean_by_k"] = {f: {int(k): r3(acc_e[(f, k)].mean()) for k in sorted(M[M.fam == f].k.unique())} for f in FAMS}
# w0 vs w1
acc4["by_wording_mean_by_k"] = {w: {f: {int(k): r3(v) for k, v in
                                        M[(M.w == w) & (M.fam == f)].groupby(["model", "k"]).corr_s.mean().groupby("k").mean().items()}
                                    for f in FAMS} for w in ("w0", "w1")}
OUT["4_accuracy_by_k"] = acc4
for f in FAMS:
    print("4:", f, acc4[f]["mean_over_17_by_k"], "paper", acc4[f]["paper_tab_acc"], "k1 range", acc4[f]["k1_range"],
          "above-chance k1/k2:", acc4[f]["n_models_above_chance_k1(p<.05,n=84)"], acc4[f]["n_models_above_chance_k2(p<.05,n=84)"])

pickle.dump(dict(M=M, item_shots=item_shots), open(os.path.join(HERE, "stage.pkl"), "wb"))
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    return o

# ─────────────────────────────── 3b. scoring: ties and written answers ───────────────────────────────
def scoring(T):
    tie_split = T.corr_c.astype(float) + np.where(T.gold_tied, 1.0 / T.n_tied.clip(lower=1), 0.0)
    return {"strict": r3(T.corr_s.mean()), "constrained": r3(T.corr_c.mean()),
            "written_first_digit": r3((T.pred_fd == T.gold).mean()),
            "constrained_ties_split": r3(tie_split.mean()),
            "constrained_ties_counted_correct": r3((T.corr_c | T.gold_tied).mean()),
            "frac_gold_tied": r3(T.gold_tied.mean())}


sc = {}
for f in FAMS:
    sc[f] = {int(k): scoring(M[(M.fam == f) & (M.k == k)]) for k in sorted(M[M.fam == f].k.unique())}
sc["all"] = scoring(M)
sc["serial_k_ge3"] = scoring(M[M.fam.isin(SERIAL) & (M.k >= 3)])
sc["serial_k1"] = scoring(M[M.fam.isin(SERIAL) & (M.k == 1)])
OUT["3b_scoring_variants"] = sc
print("3b:", sc["all"], sc["serial_k_ge3"], sc["serial_k1"])

# ─────────────────────────────── 5. error structure ───────────────────────────────
def ptr_path(b, n=10):
    tbl, s, kw = walk_pointer(b); path = [s]
    for _ in range(n - 1): path.append(tbl[path[-1]])
    return tbl, path


def err_ptr_arith(fam, predcol):
    T = M[M.fam == fam].copy()
    js, starts, incs = [], [], []
    for _, r in T.iterrows():
        q = blocks(r.text)[-1]; pred = getattr(r, predcol)
        if fam == "pointer":
            tbl, path = ptr_path(q)
            assert len(set(path)) == 10 and path[r.k] == r.gold
            js.append(path.index(pred) if pred in path else -1)
        else:
            s0, d, mod, kw = arith_params(r.w, q); assert (s0 + kw * d) % mod == r.gold
            js.append(((pred - s0) * pow(d, -1, mod)) % mod if 0 <= pred < mod else -1)
            starts.append(s0); incs.append(d)
    T["j"] = js
    if fam == "arith": T["s0"] = starts; T["d"] = incs
    return T


err = {}
for fam in ("pointer", "arith"):
    nA = 9 if fam == "pointer" else 6      # other admissible answers
    for predcol in ("pred_c", "pred_fd"):
        T = err_ptr_arith(fam, predcol)
        res = {}
        for k in sorted(T.k.unique()):
            Tk = T[T.k == k]; wrong = Tk[Tk.j != k]
            jd = wrong.j.value_counts(normalize=True).sort_index()
            inter = wrong.j.between(1, k - 1)
            # per-model fraction of wrong answers that are intermediate states
            pm = wrong.groupby("model").apply(lambda x: x.j.between(1, k - 1).mean())
            pm_one = wrong.groupby("model").apply(lambda x: (x.j == 1).mean())
            echo_w = (wrong[predcol] == wrong.echo)
            ent = {"n_items": len(Tk), "n_wrong": len(wrong),
                   "frac_of_all_predictions_by_steps": {int(j): r3(v) for j, v in Tk.j.value_counts(normalize=True).sort_index().items()},
                   "frac_of_wrong_by_steps": {int(j): r3(v) for j, v in jd.items()},
                   "wrong_intermediate(1..k-1)": r3(inter.mean()) if k >= 2 else None,
                   "expected_uniform_intermediate": r3((k - 1) / nA) if k >= 2 else None,
                   "wrong_one_step(j=1)": r3((wrong.j == 1).mean()), "expected_uniform_one_step": r3(1 / nA),
                   "wrong_start(j=0)": r3((wrong.j == 0).mean()),
                   "wrong_one_past(j=k+1)": r3((wrong.j == k + 1).mean()),
                   "wrong_equals_written_k": r3(echo_w.mean()), "expected_uniform_written_k": r3(1 / nA),
                   "most_common_wrong_steps": int(jd.idxmax()), "most_common_wrong_share": r3(jd.max()),
                   "models_with_intermediate_above_uniform": int((pm > (k - 1) / nA).sum()) if k >= 2 else None,
                   "per_model_wrong_intermediate": {m: r3(v) for m, v in pm.items()} if k >= 2 else None,
                   "per_model_wrong_one_step_range": [r3(pm_one.min()), r3(pm_one.max())],
                   "pred_one_step_share_of_all": r3((Tk.j == 1).mean()), "pred_correct_share": r3((Tk.j == k).mean())}
            if fam == "arith":
                # alternative arithmetic errors among wrong answers
                w2 = wrong
                ent["wrong_equals_s_plus_k"] = r3(((w2.s0 + w2.k) % 7 == w2[predcol]).mean())
                ent["wrong_equals_increment_d"] = r3((w2.d == w2[predcol]).mean())
                ent["wrong_equals_k_times_d_mod7"] = r3(((w2.k * w2.d) % 7 == w2[predcol]).mean())
            if predcol == "pred_fd":
                ent["first_digit_outside_answer_set"] = r3((Tk.j == -1).mean())
                if fam == "arith":
                    unred = Tk.s0 + Tk.k * Tk.d
                    o = Tk[Tk.j == -1]
                    ent["outside_set_equals_first_digit_of_unreduced_s+kd"] = r3(
                        (o[predcol] == (o.s0 + o.k * o.d).astype(str).str[0].astype(int)).mean()) if len(o) else None
            res[int(k)] = ent
        # pooled k>=2
        T2 = T[(T.k >= 2) & (T.j != T.k)]
        obs = T2.apply(lambda r: 1 <= r.j <= r.k - 1, axis=1).mean()
        exp_ = T2.k.map(lambda k: (k - 1) / nA).mean()
        res["k>=2_pooled"] = {"n_wrong": len(T2), "wrong_intermediate": r3(obs), "expected_uniform": r3(exp_),
                              "ratio": r3(obs / exp_),
                              "wrong_one_step": r3((T2.j == 1).mean()), "expected_one_step": r3(1 / nA)}
        err[f"{fam}/{predcol}"] = res
        if predcol == "pred_c":
            ERRT = T if fam == "pointer" else ERRT
            if fam == "arith": ERRA = T
    print("5:", fam, {k: (v["wrong_intermediate(1..k-1)"], v["expected_uniform_intermediate"], v["most_common_wrong_steps"],
                         v["most_common_wrong_share"], v["pred_one_step_share_of_all"], v["pred_correct_share"])
                      for k, v in err[f"{fam}/pred_c"].items() if k != "k>=2_pooled"}, err[f"{fam}/pred_c"]["k>=2_pooled"])

# worked-example answers copied?
def shot_copy(T, predcol="pred_c"):
    out = {}
    for fam in FAMS:
        Tf = T[(T.fam == fam) & (T.k >= 2) & (T[predcol] != T.gold)]
        obs, exp_ = [], []
        for _, r in Tf.iterrows():
            sh = set(item_shots[(r.cell, r.id)]); others = [a for a in r.answer_set if a != r.gold]
            if not others: continue
            obs.append(r[predcol] in sh); exp_.append(sum(a in sh for a in others) / len(others))
        out[fam] = {"n_wrong": len(obs), "wrong_equals_a_worked_example_answer": r3(np.mean(obs)),
                    "expected_uniform": r3(np.mean(exp_))}
    return out
err["worked_example_answer_copy_k>=2"] = shot_copy(M)
print("5 copy:", err["worked_example_answer_copy_k>=2"])

# binding / relational: which wrong root?
def br_features(r):
    q = blocks(r.text)[-1]; st, qv, _ = chains(r.fam, r.w, q)
    pos = {a: i for i, (a, b) in enumerate(st)}
    nxt = dict(st)
    root_of, chain_pos = {}, collections.defaultdict(list)
    for a, _ in st:
        v, path = a, [a]
        while not v.isdigit(): v = nxt[v]; path.append(v)
        root_of[a] = int(v)
    for a, _ in st: chain_pos[root_of[a]].append(pos[a])
    root_stmt = {root_of[a]: pos[a] for a, b in st if b.isdigit()}
    qp = pos[qv]; gold = root_of[qv]; assert gold == r.gold
    alts = [x for x in r.answer_set if x != gold]; assert len(alts) == 2 and set(alts) <= set(root_stmt)
    feat = {}
    feat["later_root_statement"] = {a: root_stmt[a] for a in alts}                           # larger = later
    feat["root_statement_nearer_query_stmt"] = {a: -abs(root_stmt[a] - qp) for a in alts}      # larger = nearer
    feat["chain_mean_position_nearer_query_stmt"] = {a: -abs(np.mean(chain_pos[a]) - qp) for a in alts}
    feat["chain_has_nearest_statement_to_query"] = {a: -min(abs(p - qp) for p in chain_pos[a]) for a in alts}
    feat["chain_of_last_statement_in_body"] = {a: max(chain_pos[a]) for a in alts}
    sh = set(item_shots[(r.cell, r.id)])
    feat["shown_as_worked_example_answer"] = {a: int(a in sh) for a in alts}
    feat["smaller_digit"] = {a: -a for a in alts}
    # the gold chain's own root statement position relative to the other roots (for accuracy analysis)
    near_all = min(root_stmt, key=lambda a: abs(root_stmt[a] - qp))
    return alts, feat, near_all


brres = {}
for fam in ("binding", "relational"):
    T = M[M.fam == fam]
    recs = []
    for _, r in T.iterrows():
        alts, feat, near_all = br_features(r)
        recs.append((r.model, r.k, r.corr_c, r.pred_c, r.gold, alts, feat, near_all, r.pred_fd))
    fam_out = {"n_items": len(recs)}
    wrong = [x for x in recs if not x[2] and x[3] in x[5]]
    fam_out["n_wrong_constrained"] = sum(1 for x in recs if not x[2])
    fam_out["wrong_is_other_root"] = r3(len(wrong) / max(1, fam_out["n_wrong_constrained"]))
    for fname in wrong[0][6]:
        tot, fav = 0, 0; per_model = collections.defaultdict(lambda: [0, 0]); byk = collections.defaultdict(lambda: [0, 0])
        for (m, k, c, pred, gold, alts, feat, _, _) in wrong:
            v = feat[fname]
            if v[alts[0]] == v[alts[1]]: continue
            best = max(alts, key=lambda a: v[a]); tot += 1; fav += pred == best
            per_model[m][0] += pred == best; per_model[m][1] += 1; byk[k][0] += pred == best; byk[k][1] += 1
        pm = {m: a / b for m, (a, b) in per_model.items() if b}
        fam_out[fname] = {"n_informative_wrong": tot, "frac_choosing_favoured_alt": r3(fav / max(tot, 1)),
                          "binom_p_two_sided(pooled; items shared across models)": float(stats.binomtest(fav, tot, .5).pvalue) if tot else None,
                          "models_above_0.5": int(sum(v > .5 for v in pm.values())), "n_models": len(pm),
                          "per_model_range": [r3(min(pm.values())), r3(max(pm.values()))],
                          "by_k": {int(k): r3(a / b) for k, (a, b) in sorted(byk.items())}}
    # accuracy when the gold root is / is not the root statement nearest the query statement
    for k in range(1, 7):
        a1 = [x[2] for x in recs if x[1] == k and x[7] == x[4]]; a0 = [x[2] for x in recs if x[1] == k and x[7] != x[4]]
        fam_out.setdefault("acc_constrained_gold_root_nearest_vs_not", {})[k] = [r3(np.mean(a1)) if a1 else None, len(a1),
                                                                                r3(np.mean(a0)) if a0 else None, len(a0)]
    brres[fam] = fam_out
    print("5:", fam, {k: (v["frac_choosing_favoured_alt"], v["n_informative_wrong"], v["models_above_0.5"]) for k, v in fam_out.items()
                      if isinstance(v, dict) and "frac_choosing_favoured_alt" in v})
    print("   acc near vs not", fam_out["acc_constrained_gold_root_nearest_vs_not"])
err["binding_relational"] = brres

# numeral control: position of the prediction on the cycle
nc = {}
T = M[M.fam == "fixed_digit"]
for k in range(1, 7):
    Tk = T[T.k == k]; cats = collections.Counter()
    for _, r in Tk.iterrows():
        tbl, s, kw = walk_pointer(blocks(r.text)[-1]); cyc = [s]
        while tbl[cyc[-1]] != s: cyc.append(tbl[cyc[-1]])
        assert len(cyc) == k + 1 and cyc[7 % (k + 1)] == r.gold
        p = r.pred_c
        if p == r.gold: cats["correct"] += 1
        elif p == s: cats["start"] += 1
        elif p in cyc: cats[f"cycle+{cyc.index(p)}"] += 1
        else: cats["off_cycle_fixed_point"] += 1
    n = len(Tk)
    nc[k] = {"n": n, "answer_steps_along_cycle": 7 % (k + 1), "fewest_lookups": min(7 % (k + 1), k + 1 - 7 % (k + 1)),
             "acc_strict": r3(Tk.corr_s.mean()), "categories": {c: r3(v / n) for c, v in cats.most_common()},
             "pred_one_step_state": r3(cats.get("cycle+1", 0) / n + (Tk.corr_c.mean() if 7 % (k + 1) == 1 else 0)),
             "off_cycle_expected_uniform_among_wrong": r3((6 - k) / 6)}
err["numeral_control"] = nc
print("5 numeral:", {k: (v["acc_strict"], v["answer_steps_along_cycle"], v["pred_one_step_state"], v["categories"]) for k, v in nc.items()})
OUT["5_error_structure"] = err

# ─────────────────────────────── 6. accuracy vs size ───────────────────────────────
def acc_by(T, fams, k):
    return {m: r3(np.mean([T[(T.model == m) & (T.fam == f) & (T.k == k)].corr_s.mean() for f in fams])) for m in T.model.unique()}


sz = {"params_B": PARAMS_B}
LINEAGES = {"Qwen2.5 (base)": ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-14B"],
            "Llama (base)": ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B"],
            "Mistral (base)": ["Mistral-7B-v0.1", "Mistral-Small-24B-Base-2501"]}
for lin, ms in LINEAGES.items():
    d = {}
    for k in (1, 2, 3):
        d[f"k{k}_serial_mean"] = {m: acc_by(M, SERIAL, k)[m] for m in ms}
        d[f"k{k}_by_family"] = {f: {m: r3(M[(M.model == m) & (M.fam == f) & (M.k == k)].corr_s.mean()) for m in ms} for f in FAMS}
    d["k>=3_serial_mean"] = {m: r3(np.mean([M[(M.model == m) & (M.fam == f) & (M.k >= 3)].groupby("k").corr_s.mean().mean() for f in SERIAL])) for m in ms}
    sz[lin] = d
# Spearman across the 11 base models
for k in (1, 2, 3):
    a = acc_by(M, SERIAL, k); x = [PARAMS_B[m] for m in BASE]; y = [a[m] for m in BASE]
    rho, p = stats.spearmanr(x, y); sz[f"spearman_params_vs_serial_acc_k{k}_11_base"] = [r3(rho), r3(p)]
    for f in SERIAL:
        yy = [M[(M.model == m) & (M.fam == f) & (M.k == k)].corr_s.mean() for m in BASE]
        rho, p = stats.spearmanr(x, yy); sz.setdefault(f"spearman_by_family_k{k}", {})[f] = [r3(rho), r3(p)]
# Qwen under SDPA where available (eager degrades Qwen2.5-7B; paper app:reliability)
S = A[A.attn == "sdpa"]
sz["Qwen_sdpa_k1_k2_serial_mean"] = {m: [acc_by(S[S.model == m], SERIAL, 1)[m], acc_by(S[S.model == m], SERIAL, 2)[m]]
                                     for m in ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B"]}
q7 = A[(A.model == "Qwen2.5-7B") & (A.cell == "pointer/w0")]
sz["Qwen2.5-7B_pointer_w0_eager_vs_sdpa_by_k"] = {a: {int(k): r3(v) for k, v in q7[q7.attn == a].groupby("k").corr_s.mean().items()}
                                                   for a in ("eager", "sdpa")}
# kernel: eager vs SDPA accuracy for the models with full SDPA runs
kern = {}
for m in ORDER:
    E, Sd = A[(A.model == m) & (A.attn == "eager")], A[(A.model == m) & (A.attn == "sdpa")]
    if Sd.cell.nunique() == 10:
        j = E.merge(Sd, on=["cell", "id"], suffixes=("_e", "_s"))
        kern[m] = {"acc_eager": r3(E.corr_s.mean()), "acc_sdpa": r3(Sd.corr_s.mean()),
                   "pred_agreement": r3((j.pred_c_e == j.pred_c_s).mean())}
sz["kernel_eager_vs_sdpa_all_cells"] = kern
# k>=3 accuracy relative to chance (mean over families of per-family mean over k>=3)
sz["k>=3_minus_chance_by_model"] = {m: r3(np.mean([M[(M.model == m) & (M.fam == f) & (M.k >= 3)].groupby("k").corr_s.mean().mean() - CHANCE[f]
                                                   for f in SERIAL])) for m in ORDER}
sz["k>=3_by_family_mean_over_models"] = {f: {"acc": r3(np.mean([acc[(f, k)].mean() for k in sorted(M[M.fam == f].k.unique()) if k >= 3])),
                                             "chance": r3(CHANCE[f])} for f in SERIAL}
OUT["6_accuracy_vs_size"] = sz

# ─────────────────────────────── 7. one step vs two: models that do k=1 reliably ───────────────────────────────
two = {}
for f in SERIAL:
    rows_ = []
    for m in ORDER:
        p1, p2 = acc.loc[m, (f, 1)], acc.loc[m, (f, 2)]
        if p1 >= .9:
            # if a 2-step item were two independent k=1-like steps: p1^2, plus chance on failures
            exp2 = p1 ** 2 + (1 - p1 ** 2) * CHANCE[f]
            rows_.append({"model": m, "k1": r3(p1), "k2": r3(p2), "k2_if_two_independent_k1_steps": r3(exp2),
                          "k3": r3(acc.loc[m, (f, 3)])})
    two[f] = rows_
OUT["7_reliable_single_step_models"] = two
for f in SERIAL: print("7:", f, two[f])
agree_pm = {m: of["by_model"][m]["agree_constrained_vs_first_digit"] for m in ORDER}
OUT["3_output_format"]["agree_constrained_vs_first_digit_by_model_range"] = [min(agree_pm.values()), min(agree_pm, key=agree_pm.get),
                                                                             max(agree_pm.values()), max(agree_pm, key=agree_pm.get)]
X = M[M.model != "Qwen2.5-7B"]
OUT["3_output_format"]["excl_Qwen2.5-7B_eager"] = fmt(X)
for lin in LINEAGES: print("6:", lin, sz[lin]["k1_serial_mean"], sz[lin]["k2_serial_mean"], sz[lin]["k>=3_serial_mean"])
print("6: spearman", {k: v for k, v in sz.items() if k.startswith("spearman")})
print("6: qwen sdpa", sz["Qwen_sdpa_k1_k2_serial_mean"], sz["Qwen2.5-7B_pointer_w0_eager_vs_sdpa_by_k"])
print("6: kernel", kern)
json.dump(clean(OUT), open(os.path.join(HERE, "setup.json"), "w"), indent=1, default=str)
print("wrote setup.json")
