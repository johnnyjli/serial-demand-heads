"""(i) plausibility of large 'floors' on weak heads: stability (min/mean, within-k CV) of top-25% vs other heads, back half;
(ii) calibration of the per-head 'rising head' rate: same z computed with k replaced by one random permutation (null rate);
(iii) sorted top-25% share = fixed-set share + sorting term; k-slopes of each (back half)."""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
sys.path.insert(0, sys.argv[1] + "/vet3"); from mass import topq, perm_idx, raw_z, TASKS, Q
from concurrent.futures import ProcessPoolExecutor
def job(model):
    out = []
    for c in CELLS:
        fam = c.split("/")[0]
        if fam not in TASKS: continue
        d = load(model, c); S = d["contribution"].astype(float); k = d["k"].astype(float); n, L, H = S.shape
        body = np.array([int(t["id"].split("/")[-1]) for t in d["items"]])
        rng = np.random.default_rng(0); P = perm_idx(k, body, fam, rng); x = np.arange(L) / (L - 1); back = x >= .5
        Sbar = S.mean(0); m = max(1, int(np.ceil(Q * H)))
        Tm = np.zeros((L, H), bool); np.put_along_axis(Tm, np.argsort(-Sbar, -1)[:, :m], True, -1)
        stab = S.min(0) / Sbar
        # within-k CV
        cv = np.mean([S[k == v].std(0) / S[k == v].mean(0) for v in np.unique(k)], 0)
        Sh = S.reshape(n, -1); _, zh = raw_z(Sh, k, P); zh = zh.reshape(L, H)
        kperm = k[P[7]]; P2 = perm_idx(kperm, body, fam, np.random.default_rng(3)); _, zn = raw_z(Sh, kperm, P2); zn = zn.reshape(L, H)
        Y = topq(S); PT = (S * Tm[None]).sum(-1) / S.sum(-1)
        bY, zY = raw_z(Y, k, P); bF, zF = raw_z(PT, k, P); bI, zI = raw_z(Y - PT, k, P)
        out.append({"model": model, "cell": c, "fam": fam,
                    "stab_top": float(stab[back][Tm[back]].mean()), "stab_rest": float(stab[back][~Tm[back]].mean()),
                    "cv_top": float(cv[back][Tm[back]].mean()), "cv_rest": float(cv[back][~Tm[back]].mean()),
                    "rise": float((zh[back] > 2).mean()), "fall": float((zh[back] < -2).mean()),
                    "rise_null": float((zn[back] > 2).mean()), "fall_null": float((zn[back] < -2).mean()),
                    "rise_front": float((zh[~back] > 2).mean()), "fall_front": float((zh[~back] < -2).mean()),
                    "bY_back": float(bY[back].sum()), "bF_back": float(bF[back].sum()), "bI_back": float(bI[back].sum()),
                    "zF_back": float(zF[back].mean()), "zI_back": float(zI[back].mean())})
    return out
if __name__ == "__main__":
    with ProcessPoolExecutor(10) as ex: R = [r for rr in ex.map(job, ORDER) for r in rr]
    json.dump(R, open(sys.argv[2], "w"))
    a = lambda key: np.array([r[key] for r in R])
    print("back half: min/mean top-25% heads median", np.median(a("stab_top")).round(3), " other heads", np.median(a("stab_rest")).round(3),
          " frac cells top more stable:", np.mean(a("stab_top") > a("stab_rest")).round(2))
    print("back half: within-k CV top", np.median(a("cv_top")).round(3), " other", np.median(a("cv_rest")).round(3), " frac cells top CV < rest CV:", np.mean(a("cv_top") < a("cv_rest")).round(2))
    print("heads z>2 (rising) back: obs", a("rise").mean().round(3), " null(permuted k)", a("rise_null").mean().round(3), "| z<-2 obs", a("fall").mean().round(3), "null", a("fall_null").mean().round(3))
    print("heads rising front:", a("rise_front").mean().round(3), "falling front", a("fall_front").mean().round(3))
    sp = a("bY_back") < 0
    print("sorted top-25% slope = fixed-set part + sorting part (back sums, spreading cells): fixed-set fraction median",
          np.median(a("bF_back")[sp] / a("bY_back")[sp]).round(2), " pooled", (a("bF_back")[sp].sum() / a("bY_back")[sp].sum()).round(2))
    print("fixed-set share back z<0:", np.mean(a("zF_back") < 0).round(2), " sorting term back z<0:", np.mean(a("zI_back") < 0).round(2))
    for t in ["pointer", "arith", "binding", "relational"]:
        s = np.array([r["fam"] == t for r in R])
        print("  ", t, "rise", a("rise")[s].mean().round(3), "fall", a("fall")[s].mean().round(3), "fixed-set frac pooled", (a("bF_back")[s & sp].sum() / a("bY_back")[s & sp].sum()).round(2))
