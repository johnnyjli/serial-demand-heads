"""Mass / additive-floor alternative to the back-half 'spreading' result.

Per model x serial task x wording (eager, E2_CLEAN=1), per layer:
  raw     : OLS slope of top-25% share on k, z vs permutation null (k shuffled; within body for binding/relational)
  (1) adj : same slope after adjusting for layer mass (FWL; Freedman-Lane null = permute k-residual):
            covariates  logM | 1/M | cubic(logM) | logM + k*logM (k effect at mean mass)
  (2) floor: share recomputed on max(S - f_h, 0) with per-head floors
            f = c*min_i S_ih (c = .5, 1: c = 1 is the largest additive floor admissible in the cell),
            f = 5th pct within cell, f = min over all 10 cells of the model (incl. fixed_digit control)
  (3) xs  : within-k (plus body FE for binding/relational) slope of share on logM, and the k-slope it predicts
            through the observed slope of logM on k
  (4) head: fixed top-25% set T (by mean norm in the cell), rest R: slopes of log T, log R, log M on k;
            per-head slopes (count significantly rising heads); additive-floor model prediction of the
            fixed-set share slope with the maximal floor (f_h = min_i S_ih) vs the observed slope.
"""
import sys, os, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
from concurrent.futures import ProcessPoolExecutor
B = 1000
TASKS = ["pointer", "arith", "binding", "relational"]

Q = float(os.environ.get("VQ", ".10"))
def topq(S, q=Q):
    tot = np.clip(S.sum(-1, keepdims=True), 1e-12, None); P = S / tot
    H = S.shape[-1]; m = max(1, int(np.ceil(q * H)))
    return -np.sort(-P, -1)[..., :m].sum(-1)

def perm_idx(k, body, fam, rng):
    n = len(k); P = np.empty((B, n), int)
    if fam in ("binding", "relational"):
        groups = [np.nonzero(body == g)[0] for g in np.unique(body)]
        for b in range(B):
            idx = np.arange(n)
            for g in groups: idx[g] = g[rng.permutation(len(g))]
            P[b] = idx
    else:
        for b in range(B): P[b] = rng.permutation(n)
    return P

def raw_z(Y, k, P):
    kc = k - k.mean(); ss = kc @ kc; Yc = Y - Y.mean(0)
    obs = kc @ Yc / ss; null = kc[P] @ Yc / ss
    return obs, obs / null.std(0)

def adj_z(Y, k, P, Zfun):
    """Y [n,L]; Zfun(l) -> covariate matrix [n,p] (no intercept) for layer l.  FWL slope of Y on k given Z,
    Freedman-Lane permutation null."""
    n, L = Y.shape; obs = np.empty(L); z = np.empty(L)
    for l in range(L):
        Z = np.column_stack([np.ones(n), Zfun(l)])
        Q, _ = np.linalg.qr(Z)
        kr = k - Q @ (Q.T @ k); e = Y[:, l] - Q @ (Q.T @ Y[:, l]); ss = kr @ kr
        obs[l] = kr @ e / ss; null = kr[P] @ e / ss; z[l] = obs[l] / null.std()
    return obs, z

def fe_resid(k, body, fam):
    """residual-maker for k-dummies (+ body dummies for binding/relational)"""
    ks = np.unique(k); D = [(k == v).astype(float) for v in ks]
    if fam in ("binding", "relational"):
        bs = np.unique(body); D += [(body == v).astype(float) for v in bs[1:]]
    D = np.column_stack(D); Q, R = np.linalg.qr(D); keep = np.abs(np.diag(R)) > 1e-8
    Q = Q[:, keep]
    return lambda X: X - Q @ (Q.T @ X)

def fb(v, x):
    return float(np.nanmean(v[x < .5])), float(np.nanmean(v[x >= .5]))

def model_job(args):
    model, attn = args
    idx = index(attn); out = []
    cells = {c: load(model, c, attn) for c in CELLS if (model, c) in idx}
    # model-wide per-head minimum over all items of all 10 cells
    Smin_global = np.min(np.stack([d["contribution"].astype(float).min(0) for d in cells.values()]), 0)
    for c, d in cells.items():
        fam = c.split("/")[0]
        if fam not in TASKS: continue
        S = d["contribution"].astype(float); k = d["k"].astype(float); n, L, H = S.shape
        body = np.array([int(t["id"].split("/")[-1]) for t in d["items"]])
        x = np.arange(L) / (L - 1); back = x >= .5
        rng = np.random.default_rng(0); P = perm_idx(k, body, fam, rng)
        Y = topq(S); M = S.sum(-1); lM = np.log(M); lMc = lM - lM.mean(0)
        r = {"model": model, "cell": c, "fam": fam, "attn": attn, "L": L, "H": H, "n": n}
        bY, zY = raw_z(Y, k, P); bM, zM = raw_z(lM, k, P)
        r["raw"] = fb(zY, x); r["logM"] = fb(zM, x)
        r["bY_back"] = float(bY[back].sum()); r["bY_front"] = float(bY[~back].sum())
        # (1) covariate adjustment
        covs = {"adj_logM": lambda l: lM[:, l:l+1],
                "adj_invM": lambda l: 1 / M[:, l:l+1],
                "adj_cubic": lambda l: np.column_stack([lMc[:, l], lMc[:, l]**2, lMc[:, l]**3]),
                "adj_int": lambda l: np.column_stack([lMc[:, l], (k - k.mean()) * lMc[:, l]])}
        for name, Zf in covs.items():
            b, z = adj_z(Y, k, P, Zf)
            r[name] = fb(z, x); r[name + "_ratio_back"] = float(b[back].sum() / bY[back].sum())
            r[name + "_ratio_front"] = float(b[~back].sum() / bY[~back].sum())
            # back - front contrast in slope units, relative to raw contrast
            r[name + "_ratio_contrast"] = float((b[back].mean() - b[~back].mean()) / (bY[back].mean() - bY[~back].mean()))
        # (2) floors
        floors = {"floor_min50": .5 * S.min(0), "floor_min100": S.min(0),
                  "floor_p5": np.percentile(S, 5, axis=0), "floor_global": Smin_global}
        for name, f in floors.items():
            Sf = np.clip(S - f[None], 0, None)
            b, z = raw_z(topq(Sf), k, P)
            r[name] = fb(z, x); r[name + "_ratio_back"] = float(b[back].sum() / bY[back].sum())
            r[name + "_floorfrac"] = float(np.median(f.sum(-1) / M.mean(0)))   # floor mass / mean layer mass
        # (3) cross-sectional: within-k (+ body) slope of share on logM
        Rz = fe_resid(k, body, fam); Yw, lMw = Rz(Y), Rz(lM)
        beta = (lMw * Yw).sum(0) / (lMw ** 2).sum(0)
        rcs = (lMw * Yw).sum(0) / np.sqrt((lMw ** 2).sum(0) * (Yw ** 2).sum(0))
        pred = beta * bM
        r["xs_beta_back_median"] = float(np.median(beta[back])); r["xs_r_back_mean"] = float(rcs[back].mean())
        r["xs_frac_pos_back"] = float((beta[back] > 0).mean())
        r["xs_pred_ratio_back"] = float(pred[back].sum() / bY[back].sum())
        r["bM_back"] = float(bM[back].mean())    # mean slope of log mass on k (per unit k)
        # (4) head-level
        Sbar = S.mean(0); m = max(1, int(np.ceil(Q * H)))
        Tset = np.argsort(-Sbar, -1)[:, :m]                          # [L, m]
        Tmask = np.zeros((L, H), bool); np.put_along_axis(Tmask, Tset, True, -1)
        T = (S * Tmask[None]).sum(-1); Rr = M - T
        bT, zT = raw_z(np.log(T), k, P); bR, zR = raw_z(np.log(Rr), k, P)
        bPT, zPT = raw_z(T / M, k, P)
        r["logT"] = fb(zT, x); r["logR"] = fb(zR, x); r["fixedshare"] = fb(zPT, x)
        r["dlogT_back"] = float(bT[back].mean()); r["dlogR_back"] = float(bR[back].mean()); r["dlogM_back"] = float(bM[back].mean())
        # per-head absolute slopes and z
        Sh = S.reshape(n, L * H); bh, zh = raw_z(Sh, k, P); bh = bh.reshape(L, H); zh = zh.reshape(L, H)
        r["frac_heads_rising_back"] = float((zh[back] > 2).mean()); r["frac_heads_falling_back"] = float((zh[back] < -2).mean())
        # in back layers whose mass falls (zM<-2): fraction of heads rising significantly, and of non-top heads
        fl = back & (zM < -2)
        r["n_backlayers_massfall"] = int(fl.sum())
        r["frac_heads_rising_in_massfall"] = float((zh[fl] > 2).mean()) if fl.any() else None
        r["frac_rest_rising_in_massfall"] = float((zh[fl][~Tmask[fl]] > 2).mean()) if fl.any() else None
        # additive-floor model prediction with maximal floor f_h = min_i S_ih (linearised slopes of the fixed-set share)
        f = S.min(0); F = f.sum(-1); Mbar = M.mean(0); Tbar = T.mean(0)
        bMabs = raw_z(M, k, P)[0]; bTabs = raw_z(T, k, P)[0]
        rho = bMabs / (Mbar - F)
        bT_pred = rho * ((Sbar - f) * Tmask).sum(-1)
        obs_dP = (bTabs * Mbar - Tbar * bMabs) / Mbar ** 2
        pred_dP = (bT_pred * Mbar - Tbar * bMabs) / Mbar ** 2
        r["floor_pred_ratio_back"] = float(pred_dP[back].sum() / obs_dP[back].sum())
        r["obs_dP_back"] = float(obs_dP[back].mean()); r["pred_dP_back"] = float(pred_dP[back].mean())
        # proportional (scale-only) share of the head changes: fit b_h = a + s*Sbar_h per layer; R^2 of pure scaling
        # share change explained by the within-layer pattern of head slopes relative to scaling
        # decomposition of fixed-set share slope: rising heads vs differential falls
        rise = np.clip(bh, 0, None); fall = np.clip(bh, None, 0)
        # contribution of rising non-top heads to -dP_T (they raise R): Tbar*sum(rise_R)/Mbar^2
        riseR = (rise * ~Tmask).sum(-1); obs_neg = -(obs_dP)
        r["share_of_spread_from_rising_rest_back"] = float(np.sum((Tbar * riseR / Mbar ** 2)[back]) / np.sum(obs_neg[back])) if np.sum(obs_neg[back]) > 0 else None
        # ---- (2b) floor-model simulation for the sorted statistic: keep each item's layer mass and the floor,
        # randomise the above-floor composition across items (so it is independent of k) -> predicted slope
        sd_raw = bY / zY
        Tm = Tmask
        mn = S.min(0); p5 = np.percentile(S, 5, axis=0)
        fvars = {"sim_min50": .5 * mn, "sim_min100": mn, "sim_p5": p5,
                 "sim_worst_min": np.where(Tm, 0.0, mn), "sim_worst_p5": np.where(Tm, 0.0, p5),
                 "sim_uniform": np.repeat(mn.min(-1, keepdims=True), H, -1), "sim_global": Smin_global}
        rs = np.random.default_rng(1); NS = 40; perms = [rs.permutation(n) for _ in range(NS)]
        for name, f in fvars.items():
            u = np.clip(S - f[None], 0, None); U = u.sum(-1, keepdims=True); C = u / np.clip(U, 1e-12, None)
            bp = np.zeros(L)
            for pi in perms:
                Ss = f[None] + U * C[pi]; bp += raw_z(topq(Ss), k, P[:1])[0]
            bp /= NS; zr = (bY - bp) / sd_raw
            r[name + "_pred_ratio_back"] = float(bp[back].sum() / bY[back].sum())
            r[name + "_resid"] = fb(zr, x)
            r[name + "_floorfrac"] = float(np.median(f.sum(-1) / M.mean(0)))
        # ---- (4c) best-case per-head floor fit to the head slope vector (0 <= f_h <= min_h, common rho)
        expl = np.zeros(L); vexp = np.zeros(L)
        for l in range(L):
            b_, Sb, m_ = bh[l], Sbar[l], mn[l]
            g = np.linspace(-1, 1, 801) * np.max(np.abs(b_) / Sb) * 1.5
            lo = np.minimum(g[:, None] * Sb, g[:, None] * (Sb - m_)); hi = np.maximum(g[:, None] * Sb, g[:, None] * (Sb - m_))
            bh_hat = np.clip(b_[None], lo, hi); sse = ((b_[None] - bh_hat) ** 2).sum(1); j = np.argmin(sse)
            bb = bh_hat[j]; vexp[l] = 1 - sse[j] / (b_ ** 2).sum()
            Tb, Rb = bb[Tm[l]].sum(), bb[~Tm[l]].sum()
            expl[l] = ((Mbar[l] - Tbar[l]) * Tb - Tbar[l] * Rb) / Mbar[l] ** 2
        r["floorfit_share_ratio_back"] = float(expl[back].sum() / obs_dP[back].sum())
        r["floorfit_R2_back"] = float(np.median(vexp[back]))
        r["obs_dP_layers_back"] = obs_dP[back].tolist(); r["floorfit_dP_layers_back"] = expl[back].tolist()
        # decomposition of the fixed-set share slope (linearised): top-set term and rest term
        topterm = (Mbar - Tbar) * bTabs / Mbar ** 2; restterm = -Tbar * (bMabs - bTabs) / Mbar ** 2
        r["topterm_back"] = float(topterm[back].sum()); r["restterm_back"] = float(restterm[back].sum())
        r["frac_back_layers_rest_rises"] = float(((bMabs - bTabs)[back] > 0).mean())
        r["frac_back_layers_top_falls_rest_rises"] = float((((bMabs - bTabs) > 0) & (bTabs < 0))[back].mean())
        out.append(r)
    return out

if __name__ == "__main__":
    attns = sys.argv[3].split(",") if len(sys.argv) > 3 else ["eager"]
    jobs = [(m, a) for a in attns for m in ORDER]
    with ProcessPoolExecutor(10) as ex: R = [r for rr in ex.map(model_job, jobs) for r in rr]
    json.dump(R, open(sys.argv[2], "w")); print(len(R))
