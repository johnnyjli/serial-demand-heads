"""Section 5: error structure (pointer, arithmetic, worked-example copying, binding/relational roots, numeral)."""
import collections, json, os, pickle, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v_parse import parse
M = pickle.load(open(os.path.join(HERE, "v_main.pkl"), "rb"))
SH = pickle.load(open(os.path.join(HERE, "v_shot_answers.pkl"), "rb"))
R = {}
Q = {}
for (cell, iid), t in M.groupby(["cell", "iid"]).text.first().items():
    fam, w = cell.split("/"); Q[(cell, iid)] = parse(fam, w, t.split("\n\n")[-1])

# ---------------- pointer: steps along the 10-cycle from the start to the constrained prediction
P = M[M.fam == "pointer"].copy()
P["j"] = [Q[(c, i)]["cycle"].index(p) for c, i, p in zip(P.cell, P.iid, P.pc)]
assert (P[P.cc].j == P[P.cc].k).all()
pr = {}
for k in range(1, 6):
    T = P[P.k == k]; W = T[T.pc != T.gold]
    e = dict(n=len(T), n_wrong=len(W), share_all_j=T.j.value_counts(normalize=True).sort_index().round(4).to_dict(),
             wrong_j=W.j.value_counts(normalize=True).sort_index().round(4).to_dict(),
             wrong_intermediate=float(W.j.between(1, k - 1).mean()), exp_inter=(k - 1) / 9,
             wrong_one_step=float((W.j == 1).mean()), start_pred_all=float((T.j == 0).mean()),
             wrong_eq_written_k=float((W.pc == W.echo).mean()),
             all_eq_written_k=float((T.pc == T.echo).mean()))
    pm1 = W.groupby("model").apply(lambda x: (x.j == 1).mean())
    pmi = W.groupby("model").apply(lambda x: x.j.between(1, k - 1).mean())
    e["models_one_step_above_1/9"] = int((pm1 > 1 / 9).sum())
    e["models_intermediate_above_uniform"] = int((pmi > (k - 1) / 9).sum()) if k >= 2 else None
    e["most_common_wrong_j"] = int(W.j.value_counts().idxmax())
    pr[k] = e
W2 = P[(P.k >= 2) & (P.pc != P.gold)]
pr["pooled_k>=2"] = dict(n_wrong=len(W2), inter=float((W2.j >= 1).__and__(W2.j <= W2.k - 1).mean()),
                         expected=float(((W2.k - 1) / 9).mean()), one_step=float((W2.j == 1).mean()))
R["pointer"] = pr

# ---------------- arithmetic
A = M[M.fam == "arith"].copy()
A["s0"] = [Q[(c, i)]["start"] for c, i in zip(A.cell, A.iid)]
A["d"] = [Q[(c, i)]["inc"] for c, i in zip(A.cell, A.iid)]
A["j"] = [((p - s) * pow(d, -1, 7)) % 7 for p, s, d in zip(A.pc, A.s0, A.d)]
assert (A[A.cc].j == A[A.cc].k).all()
ar = {}
for k in range(1, 7):
    T = A[A.k == k]; W = T[T.pc != T.gold]
    ar[k] = dict(n_wrong=len(W), wrong_intermediate=float(W.j.between(1, k - 1).mean()), exp=(k - 1) / 6,
                 wrong_one_step=float((W.j == 1).mean()), most_common_wrong_j=int(W.j.value_counts().idxmax()),
                 one_step_share_all=float((T.j == 1).mean()),
                 wrong_is_6=float((W.pc == 6).mean()), wrong_eq_written_k=float((W.pc == W.echo).mean()),
                 gold_is_6=float((T.gold == 6).mean()), wrong_is_0=float((W.pc == 0).mean()))
W2 = A[(A.k >= 2) & (A.pc != A.gold)]
ar["pooled_k>=2"] = dict(n_wrong=len(W2), inter=float(((W2.j >= 1) & (W2.j <= W2.k - 1)).mean()),
                         expected=float(((W2.k - 1) / 6).mean()))
# k=1 written answers 7-9
T1 = A[(A.k == 1) & (A.pfd != A.gold)]
o = T1[T1.pfd.between(7, 9)]
ar["k1_written_wrong_7to9"] = [len(o), len(T1), float(len(o) / len(T1)),
                                float((o.pfd == (o.s0 + o.d).astype(str).str[0].astype(int)).mean()),
                                float((o.pfd == (o.s0 + o.d)).mean())]
R["arith"] = ar

# ---------------- worked-example copying (k>=2, constrained wrong answers)
cp = {}
for fam in ["pointer", "arith", "binding", "relational", "fixed_digit"]:
    T = M[(M.fam == fam) & (M.k >= 2) & (M.pc != M.gold)]
    obs, exp = [], []
    for c, i, p, g, aset in zip(T.cell, T.iid, T.pc, T.gold, T.aset):
        sh = set(SH[(c, i)]); oth = [a for a in aset if a != g]
        obs.append(p in sh); exp.append(np.mean([a in sh for a in oth]))
    cp[fam] = [len(obs), round(float(np.mean(obs)), 4), round(float(np.mean(exp)), 4)]
    # also all k
    T = M[(M.fam == fam) & (M.pc != M.gold)]
    obs, exp = [], []
    for c, i, p, g, aset in zip(T.cell, T.iid, T.pc, T.gold, T.aset):
        sh = set(SH[(c, i)]); oth = [a for a in aset if a != g]
        obs.append(p in sh); exp.append(np.mean([a in sh for a in oth]))
    cp[fam + "_allk"] = [len(obs), round(float(np.mean(obs)), 4), round(float(np.mean(exp)), 4)]
R["copy"] = cp

# ---------------- binding / relational wrong roots
br = {}
for fam in ("binding", "relational"):
    T = M[M.fam == fam]
    res = collections.defaultdict(lambda: [0, 0]); pm = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    nearacc = collections.defaultdict(lambda: [[], []]); n_wrong = 0; n_wrong_root = 0
    for m, c, i, k, p, g, cc in zip(T.model, T.cell, T.iid, T.k, T.pc, T.gold, T.cc):
        q = Q[(c, i)]; st = q["stm"]; pos = {a: n for n, (a, b) in enumerate(st)}
        rpos = {int(b): n for n, (a, b) in enumerate(st) if b.isdigit()}
        qp = pos[q["qvar"]]
        near = min(rpos, key=lambda r: (abs(rpos[r] - qp), r))
        nearacc[k][0 if near == g else 1].append(cc)
        if cc: continue
        n_wrong += 1; n_wrong_root += p in rpos
        alts = [r for r in rpos if r != g]
        sh = set(SH[(c, i)])
        feats = {"later_root_stmt": {a: rpos[a] for a in alts},
                 "root_stmt_nearer_query": {a: -abs(rpos[a] - qp) for a in alts},
                 "shown_in_worked_example": {a: int(a in sh) for a in alts}}
        for fn, v in feats.items():
            if v[alts[0]] == v[alts[1]]: continue
            best = max(alts, key=lambda a: v[a])
            res[fn][0] += (p == best); res[fn][1] += 1
            pm[fn][m][0] += (p == best); pm[fn][m][1] += 1
    out = {"n_wrong": n_wrong, "wrong_is_other_root": n_wrong_root}
    for fn, (a, b) in res.items():
        out[fn] = [round(a / b, 4), b, int(sum(1 for mm, (x, y) in pm[fn].items() if x / y > .5))]
    out["acc_near_vs_not"] = {int(k): [round(float(np.mean(v[0])), 4) if v[0] else None, len(v[0]),
                                     round(float(np.mean(v[1])), 4) if v[1] else None, len(v[1])] for k, v in sorted(nearacc.items())}
    br[fam] = out
R["br"] = br

# ---------------- numeral control at k=6
F = M[(M.fam == "fixed_digit") & (M.k == 6)]
cats = collections.Counter()
for c, i, p, g in zip(F.cell, F.iid, F.pc, F.gold):
    cyc = Q[(c, i)]["cycle"]; assert len(cyc) == 7 and cyc[0] == g
    cats["correct" if p == g else f"+{cyc.index(p)}"] += 1
R["numeral_k6"] = {c: round(v / len(F), 4) for c, v in sorted(cats.items())}
json.dump(R, open(os.path.join(HERE, "v5_errors.json"), "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
