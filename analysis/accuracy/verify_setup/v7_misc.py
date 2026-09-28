"""Appendix-B parenthetical (main run on the 8 replication models, strict vs written), nearest-root tie
sensitivity, and a few extra checks."""
import collections, json, os, pickle, sys
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v_parse import parse
D = pickle.load(open(os.path.join(HERE, "v_items.pkl"), "rb")); D["cell"] = D.fam + "/" + D.w
D["wr"] = D.pfd == D.gold
M = pickle.load(open(os.path.join(HERE, "v_main.pkl"), "rb"))
SERIAL = ["pointer", "arith", "binding", "relational"]
M8 = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B",
      "OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct", "Mistral-Small-24B-Base-2501"]
R = {}


def agg(T, col, sel):
    # mean over models of mean over families of mean over wordings of item mean
    return float(np.mean([np.mean([np.mean([T[(T.model == m) & (T.fam == f) & (T.w == w) & sel(T)][col].mean()
                                            for w in ("w0", "w1")]) for f in SERIAL]) for m in M8]))


for q7 in ("sdpa", "eager"):
    T = D[((D.kernel == "eager") & (D.model != "Qwen2.5-7B-Instruct")) | ((D.model == "Qwen2.5-7B-Instruct") & (D.kernel == q7))]
    T = T[T.model.isin(M8) & T.fam.isin(SERIAL)]
    R[f"main8_Q7I_{q7}"] = {c: [round(agg(T, c, lambda X: X.k > 0), 4), round(agg(T, c, lambda X: X.k >= 3), 4)]
                            for c in ("cs", "wr", "cc")}
    R[f"main8_Q7I_{q7}_w0only"] = {c: [round(float(np.mean([np.mean([T[(T.model == m) & (T.fam == f) & (T.w == "w0")][c].mean() for f in SERIAL]) for m in M8])), 4),
                                         round(float(np.mean([np.mean([T[(T.model == m) & (T.fam == f) & (T.w == "w0") & (T.k >= 3)][c].mean() for f in SERIAL]) for m in M8])), 4)]
                                   for c in ("cs", "wr")}

# nearest-root sensitivity to tie-breaking (binding / relational)
out = {}
for fam in ("binding", "relational"):
    T = M[M.fam == fam]
    res = collections.defaultdict(lambda: collections.defaultdict(list)); nties = collections.Counter(); ntot = collections.Counter()
    for c, t, k, g, cc in zip(T.cell, T.text, T.k, T.gold, T.cc):
        fam_, w = c.split("/"); q = parse(fam_, w, t.split("\n\n")[-1]); st = q["stm"]
        pos = {a: n for n, (a, b) in enumerate(st)}; rpos = {int(b): n for n, (a, b) in enumerate(st) if b.isdigit()}
        qp = pos[q["qvar"]]; d = {r: abs(p - qp) for r, p in rpos.items()}; dm = min(d.values())
        tied = [r for r in d if d[r] == dm]
        ntot[k] += 1; nties[k] += len(tied) > 1
        text_first = min(tied, key=lambda r: rpos[r])     # earliest root statement among the tied
        text_last = max(tied, key=lambda r: rpos[r])
        res[k]["first"].append((g == text_first, cc)); res[k]["last"].append((g == text_last, cc))
        if len(tied) == 1: res[k]["strict"].append((g == tied[0], cc))
    o = {}
    for k in sorted(res):
        row = {}
        for v, L in res[k].items():
            a = [cc for n, cc in L if n]; b = [cc for n, cc in L if not n]
            row[v] = [round(float(np.mean(a)), 4) if a else None, len(a), round(float(np.mean(b)), 4) if b else None, len(b)]
        row["frac_items_equidistant_tie"] = round(nties[k] / ntot[k], 4)
        o[int(k)] = row
    out[fam] = o
R["near_root_tiebreak"] = out

# arithmetic: wrong answers equal to 6 vs equal to the written step count, by k
A = M[(M.fam == "arith") & (M.pc != M.gold)]
R["arith_wrong_6_vs_written_k"] = {int(k): [round(float((X.pc == 6).mean()), 4), round(float((X.pc == X.echo).mean()), 4), len(X)]
                                   for k, X in A.groupby("k")}
# gold-tied items: every one scored wrong under constrained and strict?
R["gold_tied_scored_wrong"] = [int(M[M.gtie_meta].cc.sum()), int(M[M.gtie_meta].cs.sum()), int(M.gtie_meta.sum())]
# Spearman exact p for k=1
PB = {"Llama-3.2-1B": 1.24, "Llama-3.2-3B": 3.21, "Llama-3.1-8B": 8.03, "Qwen2.5-0.5B": .49, "Qwen2.5-1.5B": 1.54,
      "Qwen2.5-3B": 3.09, "Qwen2.5-7B": 7.62, "Qwen2.5-14B": 14.7, "OLMo-2-1124-7B": 7.30, "Mistral-7B-v0.1": 7.24,
      "Mistral-Small-24B-Base-2501": 23.6}
acc = M.groupby(["model", "fam", "k"]).cs.mean()
y1 = [np.mean([acc[(m, f, 1)] for f in SERIAL]) for m in PB]
R["spearman_k1_exact_p"] = float(stats.spearmanr(list(PB.values()), y1).pvalue)
# pointer k=1 per model below thresholds
p1 = {m: float(acc[(m, "pointer", 1)]) for m in M.model.unique()}
R["pointer_k1_below_0.5"] = sorted([(round(v, 3), m) for m, v in p1.items() if v < .5])
R["pointer_k1_below_0.7"] = len([v for v in p1.values() if v < .7])
R["pointer_k1_at_least_0.9"] = sorted([(round(v, 3), m) for m, v in p1.items() if v >= .9])
# wording difference (instances shared): pointer/relational k=1,2 by wording, mean over models
R["by_wording_mean"] = {f: {w: M[(M.fam == f) & (M.w == w)].groupby(["model", "k"]).cs.mean().groupby("k").mean().round(3).to_dict()
                            for w in ("w0", "w1")} for f in SERIAL}
json.dump(R, open(os.path.join(HERE, "v7_misc.json"), "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
