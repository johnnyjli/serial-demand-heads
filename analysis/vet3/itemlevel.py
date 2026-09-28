"""Item-level pooled model: top-25% share (pp) ~ k + k:depth [+ ent + ent:depth] with cell x layer (x body for
binding/relational) intercepts absorbed by demeaning; cluster-robust SE by model (CR1, G=17) and by base (G=11)."""
import sys, json, numpy as np
sys.path.insert(0, '.'); from v2lib import *
def dm(v, body, within):
    if not within: return v - v.mean(0)
    o = v.astype(float).copy()
    for g in np.unique(body): i = body == g; o[i] = v[i] - v[i].mean(0)
    return o
def run(tasks, with_ent=False, xmax=1.01):
    Xs, ys, gm, gb = [], [], [], []
    for m, f, w, a in cells("eager", tasks):
        d = load(m, f, w, a, True); x = d["x"]; keep = x < xmax
        Y = 100 * dm(d["y25"], d["body"], d["within"])[:, keep]; k = dm(d["k"][:, None], d["body"], d["within"]); xx = x[keep] - .5
        n, L = Y.shape; K = np.repeat(k, L, 1)
        cols = [K.ravel(), (K * xx[None]).ravel()]
        if with_ent:
            e = dm(d["ent"][:, None], d["body"], d["within"]); E = np.repeat(e, L, 1); cols += [E.ravel(), (E * xx[None]).ravel()]
        Xs.append(np.stack(cols, 1)); ys.append(Y.ravel()); gm += [m] * (n * L); gb += [BASE[m]] * (n * L)
    X = np.concatenate(Xs); y = np.concatenate(ys); XtX = np.linalg.inv(X.T @ X); b = XtX @ X.T @ y; e = y - X @ b
    res = {"coef": b.tolist()}
    for lab, g in (("model", np.array(gm)), ("base", np.array(gb))):
        codes = np.unique(g, return_inverse=True)[1]; G = codes.max() + 1
        S = np.zeros((G, X.shape[1]))
        for j in range(X.shape[1]): S[:, j] = np.bincount(codes, weights=X[:, j] * e, minlength=G)
        V = XtX @ (S.T @ S) @ XtX * G / (G - 1); se = np.sqrt(np.diag(V))
        from scipy.stats import t as tdist
        q = tdist.ppf(.975, G - 1)
        res[lab] = {"se": se.tolist(), "ci": [[float(bb - q * s), float(bb + q * s)] for bb, s in zip(b, se)], "p": [float(2 * tdist.sf(abs(bb / s), G - 1)) for bb, s in zip(b, se)]}
    return res
out = {}
names = ["k (at depth .5)", "k x depth", "ent (at .5)", "ent x depth"]
for label, tasks, ent, xmax in [("all4", TASKS, False, 1.01), ("all4+ent", TASKS, True, 1.01), ("no_arith", ["pointer", "binding", "relational"], False, 1.01),
                                ("binding+relational", ["binding", "relational"], False, 1.01), ("binding+relational+ent", ["binding", "relational"], True, 1.01),
                                ("all4_x<.9", TASKS, False, .9), ("fixed_digit", ["fixed_digit"], False, 1.01)]:
    r = run(tasks, ent, xmax); out[label] = r
    s = " | ".join(f"{names[j]} {r['coef'][j]:+.4f} pp CI(model) [{r['model']['ci'][j][0]:+.4f},{r['model']['ci'][j][1]:+.4f}] p={r['model']['p'][j]:.2g} (base p={r['base']['p'][j]:.2g})" for j in range(len(r["coef"])))
    print(f"{label:24s} {s}")
json.dump(out, open("itemlevel.json", "w"), indent=1)
