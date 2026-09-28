"""Task 2: grouped cross-validated decoding of correctness at fixed depth, per model x task x wording cell.
Features: (a) activity = the L per-layer top-10% shares; (b) prompt features (decode2.prompt_feats: gold-digit
one-hot plus family-specific surface features); (c) prompt + activity.  All features residualised on k
(demeaned within depth), as in decode2.py.  Depths with >= 8 correct and >= 8 incorrect items; cells with >= 40
such items.  StratifiedGroupKFold(5) with the prompt body as group (strata = depth x correctness), so all items of
a body share a fold; StandardScaler + L2 logistic regression (C = 0.5); AUC of the pooled out-of-fold predictions.
Nulls (NPERM each): (a), (b) correctness shuffled within depth; (c) vs (b): activity rows shuffled within depth
(keeps the prompt-correctness link, breaks the activity-correctness link).
Also, as a setup check: whether the gold digit is the answer of one of the worked examples, and accuracy by it."""
import json, sys, re, zlib, numpy as np, warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from clib import *
sys.path.insert(0, OLD)
from solve import blocks, solve_block     # noqa: E402
def prompt_feats(fam, w, text, gold):      # copied from decode2.py (the baseline)
    b = blocks(text)[-1]; f = [int(gold == str(d)) for d in range(10)]
    if fam in ("binding", "relational"):
        from parse import parse
        P = parse(f"{fam}/{w}", text); f += [P["q_pos"], P["root_pos"], P["span"], P["forward"], P["q_from_end"]]
    elif fam in ("pointer", "fixed_digit"):
        tbl = dict((int(a), int(c)) for a, c in re.findall(r"(\d) (?:goes|leads) to (\d)\.", b))
        s = int(re.search(r"(?:Start at|From) (\d)", b).group(1)); f += [int(s == d) for d in range(10)]
        path = [s]; k = int(re.search(r"(?:Take|make) (\d+)", b).group(1))
        for _ in range(k): path.append(tbl[path[-1]])
        f += [sum(1 for a, c in zip(path, path[1:]) if c > a), sum(path) / len(path)]
    elif fam == "arith":
        nums = [int(x) for x in re.findall(r"\d+", b)]; f += nums[:4]
    return f
MINC, NPERM = 8, 100
def oof(X, y, strat, groups):
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, strat, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)); clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)
def resid(X, k):
    X = np.asarray(X, float).copy()
    for kk in np.unique(k): X[k == kk] -= X[k == kk].mean(0)
    return X
def one(key):
    m, f, w = key; d = load_cell(m, f, w); k, ok = d["k"], d["correct"].astype(int)
    shown = [[solve_block(f, w, b)[1] for b in blocks(t)[:-1]] for t in d["text"]]
    gold_shown = np.array([int(g in s) for g, s in zip(d["gold"], shown)]); pred_shown = np.array([int(p in s) for p, s in zip(d["pred"], shown)])
    out = {"model": m, "fam": f, "w": w, "n": int(len(k)), "acc": float(ok.mean()),
           "frac_gold_shown": float(gold_shown.mean()),
           "acc_gold_shown": float(ok[gold_shown == 1].mean()) if (gold_shown == 1).any() else None,
           "acc_gold_notshown": float(ok[gold_shown == 0].mean()) if (gold_shown == 0).any() else None,
           "frac_wrong_pred_shown": float(pred_shown[ok == 0].mean()) if (ok == 0).any() else None,
           "frac_wrong_gold_shown": float(gold_shown[ok == 0].mean()) if (ok == 0).any() else None}
    depths = [kk for kk in np.unique(k) if min(ok[k == kk].sum(), (1 - ok[k == kk]).sum()) >= MINC]
    use = np.isin(k, depths); out.update({"depths": [int(x) for x in depths], "n_used": int(use.sum())})
    if use.sum() < 40: return out
    ku, yu, gu = k[use], ok[use], d["body"][use]; strat = ku * 2 + yu
    rng = np.random.default_rng(zlib.crc32(f"{m}|{f}|{w}".encode()))
    Xa = resid(d["y10"][use], ku)
    F = np.array([prompt_feats(f, w, t, str(g)) for t, g in zip(d["text"][use], d["gold"][use])], float)
    Xp = resid(F, ku); Xp = Xp[:, Xp.std(0) > 1e-9]
    def perm_y():
        yp = yu.copy()
        for kk in depths: i = np.nonzero(ku == kk)[0]; yp[i] = rng.permutation(yp[i])
        return yp
    def perm_rows(X):
        Xq = X.copy()
        for kk in depths: i = np.nonzero(ku == kk)[0]; Xq[i] = X[i][rng.permutation(len(i))]
        return Xq
    for name, X in (("activity", Xa), ("prompt", Xp)):
        a = oof(X, yu, strat, gu); null = []
        for _ in range(NPERM):
            yp = perm_y(); null.append(oof(X, yp, ku * 2 + yp, gu))
        out[name] = {"auc": a, "null_mean": float(np.mean(null)), "null_95": float(np.percentile(null, 95)),
                     "p": float((np.sum(np.array(null) >= a) + 1) / (NPERM + 1))}
    a = oof(np.c_[Xp, Xa], yu, strat, gu); null = [oof(np.c_[Xp, perm_rows(Xa)], yu, strat, gu) for _ in range(NPERM)]
    out["prompt+activity"] = {"auc": a, "null_mean": float(np.mean(null)), "null_95": float(np.percentile(null, 95)),
                              "p_increment": float((np.sum(np.array(null) >= a) + 1) / (NPERM + 1))}
    xs = resid(gold_shown[use][:, None].astype(float), ku)
    out["gold_shown_only_auc"] = oof(xs, yu, strat, gu) if xs.std() > 0 else None
    print(m, f, w, "done", flush=True)
    return out
if __name__ == "__main__":
    keys = [(m, f, w) for m in ORDER for f in FAMS for w in ("w0", "w1")]
    with ProcessPoolExecutor(12) as ex: R = list(ex.map(one, keys))
    json.dump(R, open("decode.json", "w")); print(len(R), sum("activity" in r for r in R))
