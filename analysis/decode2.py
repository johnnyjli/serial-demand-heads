import sys, json, re, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *; from solve import blocks
from concurrent.futures import ProcessPoolExecutor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
MINC, NPERM = 8, 20
def oof(X, y, strat, groups, pca=True):
    p = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, strat, groups):
        steps = [StandardScaler()] + ([PCA(min(20, len(tr) - 2, X.shape[1]), random_state=0)] if pca else []) + [LogisticRegression(C=0.5, max_iter=2000)]
        clf = make_pipeline(*steps); clf.fit(X[tr], y[tr]); p[te] = clf.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)
def resid(X, k):
    X = X.copy()
    for kk in np.unique(k): X[k == kk] -= X[k == kk].mean(0)
    return X
def prompt_feats(fam, w, text, gold):
    b = blocks(text)[-1]; f = [int(gold == str(d)) for d in range(10)]
    if fam in ("binding", "relational"):
        from parse import parse
        P = parse(f"{fam}/{w}", text); f += [P["q_pos"], P["root_pos"], P["span"], P["forward"], P["q_from_end"]]
    elif fam in ("pointer", "fixed_digit"):
        tbl = dict((int(a), int(c)) for a, c in re.findall(r"(\d) (?:goes|leads) to (\d)\.", b))
        s = int(re.search(r"(?:Start at|From) (\d)", b).group(1)); f += [int(s == d) for d in range(10)]
        # how often the path revisits earlier-written rules (forward order in the rule list)
        path = [s]; k = int(re.search(r"(?:Take|make) (\d+)", b).group(1))
        for _ in range(k): path.append(tbl[path[-1]])
        f += [sum(1 for a, c in zip(path, path[1:]) if c > a), sum(path) / len(path)]
    elif fam == "arith":
        nums = [int(x) for x in re.findall(r"\d+", b)]; f += nums[:4]
    return f
def one(key):
    m, c = key; fam, w = c.split("/"); d = load(m, c); S = d["contribution"].astype(float); L = d["L"]
    k = d["k"]; ok = d["correct"].astype(int); x = np.arange(L) / (L - 1)
    grp = np.array([int(t["id"].split("/")[-1]) for t in d["items"]])
    use = np.isin(k, [kk for kk in np.unique(k) if min(ok[k == kk].sum(), (1 - ok[k == kk]).sum()) >= MINC])
    out = {"model": m, "cell": c}
    if use.sum() < 40: return out
    ku, yu, gu = k[use], ok[use], grp[use]; strat = ku * 2 + yu; rng = np.random.default_rng(1)
    first = np.log(S[use][:, x <= 0.15] + 1e-6).reshape(use.sum(), -1)
    F = np.array([prompt_feats(fam, w, t["text"], t["gold"]) for t, u in zip(d["items"], use) if u], float)
    full = np.log(S[use] + 1e-6).reshape(use.sum(), -1)
    for name, X, pca in (("first 15% layers", resid(first, ku), True), ("prompt features", resid(F, ku), False)):
        a = oof(X, yu, strat, gu, pca); null = []
        for _ in range(NPERM):
            yp = yu.copy()
            for kk in np.unique(ku): idx = np.nonzero(ku == kk)[0]; yp[idx] = rng.permutation(yp[idx])
            null.append(oof(X, yp, ku * 2 + yp, gu, pca))
        out[name] = {"auc": a, "p": float((np.sum(np.array(null) >= a) + 1) / (NPERM + 1))}
    # does activity add to prompt features?  PCA(activity) + prompt features together
    Xa = resid(full, ku); pcs = PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(Xa))
    out["activity"] = oof(Xa, yu, strat, gu, True)
    out["prompt+activity"] = oof(np.c_[resid(F, ku), pcs], yu, strat, gu, False)
    return out
if __name__ == "__main__":
    keys = [(m, c) for m in ORDER for c in CELLS]
    with ProcessPoolExecutor(14) as ex: R = list(ex.map(one, keys))
    json.dump(R, open(sys.argv[2], "w"))
