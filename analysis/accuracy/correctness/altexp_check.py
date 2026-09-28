"""Task 3a: what the tab:robust rows 'Adjusted for answer entropy' / 'Adjusted for entropy and correctness' compute,
and a check of their values.  Re-implements vet3/altexp.py (per-layer item-level OLS) and the model_diff4.py rows
without writing into the earlier scratchpad, then adds the unadjusted route ('k' only) so the effect of the
adjustment can be separated from the change of response measure.  Writes altexp_check.json."""
import os, sys, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("VPRIM", "y10")
from clib import OLD
sys.path.insert(0, OLD + "/vet3"); sys.path.insert(0, OLD)
import v2lib
import model_diff; model_diff.SP = OLD
from model_diff import ORDER, T, decompose, ss, perm_model, split_half, from_consist
def demean(v, body, within):
    if not within: return v - v.mean(0)
    out = v.astype(float).copy()
    for g in np.unique(body):
        i = body == g; out[i] = v[i] - v[i].mean(0)
    return out
def ols_t(Y, X):
    n, p = X.shape; XtX = np.linalg.pinv(X.T @ X); B = XtX @ X.T @ Y; E = Y - X @ B
    dof = n - p - 1; s2 = (E ** 2).sum(0) / dof; se = np.sqrt(np.outer(np.diag(XtX), s2)); return B, B / se
FITS = ["k", "k+ent", "k+corr", "k+ent+corr"]
def layer_rows(q7i_sdpa=False):
    rows = []
    for m, f, w, a in v2lib.cells("eager", v2lib.TASKS, q7i_sdpa=q7i_sdpa):
        d = v2lib.load(m, f, w, a, True); body = d["body"]; wi = d["within"]; x = d["x"]
        Y = demean(d["y25"], body, wi); k = demean(d["k"][:, None], body, wi)
        e = demean(d["ent"][:, None], body, wi); c = demean(d["correct"].astype(float)[:, None], body, wi)
        X = {"k": k, "k+ent": np.hstack([k, e]), "k+corr": np.hstack([k, c]), "k+ent+corr": np.hstack([k, e, c])}
        for name, M in X.items():
            if np.linalg.matrix_rank(M) < M.shape[1]: continue
            B, t = ols_t(Y, M)
            for l in range(d["L"]):
                r = {"model": m, "task": f, "w": w, "x": x[l], "fit": name, "b_pp": 100 * B[0, l], "t": t[0, l]}
                if "corr" in name: r["t_corr"] = t[-1, l]; r["b_corr_pp"] = 100 * B[-1, l]
                if "ent" in name: r["t_ent"] = t[1, l]
                rows.append(r)
    A = pd.DataFrame(rows); A["dec"] = np.minimum((A.x * 10).astype(int), 9); return A
def Z_of(A, fit, col="t"):
    S = A[A.fit == fit].groupby(["model", "task", "w", "dec"])[col].mean().groupby(["model", "task", "dec"]).mean()
    return np.array([[[S[(m, t, d)] for d in range(10)] for t in T] for m in ORDER])
def row(Z, ref):
    mu, a, b, e = decompose(Z); s = ss(Z); ns = s["model"] + s["task"] + s["resid"]; _, nm, p = perm_model(Z, 2000)
    return {"model_share": s["model"] / ns, "p_model": p, "split_half": split_half(Z)[0], "r_with_ref": float(np.corrcoef(a.ravel(), ref.ravel())[0, 1]),
            "per_model_r_median": float(np.median([np.corrcoef(a[i], ref[i])[0, 1] for i in range(len(a))]))}
if __name__ == "__main__":
    out = {}
    old = pd.read_csv(f"{OLD}/vet3/altexp_layers.csv")
    A = layer_rows(False)
    for fit in ("k", "k+ent", "k+ent+corr"):
        o = old[(old.fit == fit) & old.task.isin(T)].reset_index(drop=True); n_ = A[A.fit == fit].reset_index(drop=True)
        out[f"reproduces_altexp_layers_{fit}"] = {"n_rows": [len(o), len(n_)], "max_abs_t_diff": float(np.abs(o.t.values - n_.t.values).max())}
    a0 = decompose(from_consist("q10"))[1]                               # primary signatures (tab:robust reference)
    Zk = Z_of(A, "k"); ak = decompose(Zk)[1]
    for fit, lab in (("k", "unadjusted, same route (OLS t, per wording)"), ("k+ent", "adjusted for answer entropy"),
                     ("k+corr", "adjusted for correctness only"), ("k+ent+corr", "adjusted for entropy and correctness")):
        Z = Z_of(A, fit); a = decompose(Z)[1]
        out[lab] = {"vs_primary": row(Z, a0), "r_with_unadjusted_route": float(np.corrcoef(a.ravel(), ak.ravel())[0, 1]),
                    "mu_back_mean": float(decompose(Z)[0][5:].mean()), "mu_r_with_unadjusted": float(np.corrcoef(decompose(Z)[0], decompose(Zk)[0])[0, 1])}
        print(lab, {k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in out[lab].items()})
    # same, with Qwen2.5-7B-Instruct from SDPA (as in the primary analysis; altexp.py used eager for it)
    A2 = layer_rows(True); ak2 = decompose(Z_of(A2, "k"))[1]
    for fit in ("k", "k+ent", "k+ent+corr"):
        a = decompose(Z_of(A2, fit))[1]
        out[f"Q7I_sdpa|{fit}"] = {"r_primary": float(np.corrcoef(a.ravel(), a0.ravel())[0, 1]), "r_unadjusted_route": float(np.corrcoef(a.ravel(), ak2.ravel())[0, 1])}
    print({k: v for k, v in out.items() if k.startswith("Q7I") or k.startswith("repro")})
    # size of the k coefficient before/after adjustment, and the correctness coefficient itself (back half = x >= .5)
    tab = {}
    for g, fs in {**{t: [t] for t in T}, "all4": T}.items():
        s = A[A.task.isin(fs)]; bk = s[s.x >= .5]
        r = {}
        for fit in ("k", "k+corr", "k+ent", "k+ent+corr"):
            q = bk[bk.fit == fit]
            r[fit] = {"back_mean_t_k": float(q.t.mean()), "back_mean_b_pp": float(q.b_pp.mean())}
            if "corr" in fit:
                r[fit].update({"back_mean_t_corr": float(q.t_corr.mean()), "frac_layers_abs_t_corr_gt_1.96": float((q.t_corr.abs() > 1.96).mean()),
                               "all_layers_frac_abs_t_corr_gt_1.96": float((s[s.fit == fit].t_corr.abs() > 1.96).mean()),
                               "all_layers_mean_t_corr": float(s[s.fit == fit].t_corr.mean()), "back_mean_b_corr_pp": float(q.b_corr_pp.mean())})
        ratio = (bk[bk.fit == "k+corr"].groupby(["model", "task", "w"]).b_pp.mean() / bk[bk.fit == "k"].groupby(["model", "task", "w"]).b_pp.mean())
        r["median_ratio_back_b_kcorr_over_k"] = float(ratio.median()); tab[g] = r
        print(g, json.dumps(r))
    out["coefficients"] = tab
    json.dump(out, open("altexp_check.json", "w"), indent=1)
