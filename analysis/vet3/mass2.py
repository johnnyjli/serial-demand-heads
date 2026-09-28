"""Layer-level checks of the mass/floor alternative.
(a) per layer z of top-25% share slope vs z of log-mass slope (does falling mass predict spreading, front vs back?)
(b) is the candidate floor (per-head min) spread more evenly than the above-floor signal?  (premise of the floor story)
(c) sink heads as a mechanistic floor: share of layer mass carried by heads with mean pos0 attention > .9, their stability
Variant via argv[3]: eager | sdpa | nosink (eager, heads with pos0_attn > .9 zeroed per item)."""
import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); from e2lib import *
sys.path.insert(0, sys.argv[1] + "/vet3"); from mass import topq, perm_idx, raw_z, adj_z, TASKS
from concurrent.futures import ProcessPoolExecutor
def job(args):
    model, variant = args; attn = "sdpa" if variant == "sdpa" else "eager"; idx = index(attn); out = []
    for c in CELLS:
        fam = c.split("/")[0]
        if fam not in TASKS or (model, c) not in idx: continue
        d = load(model, c, attn); S = d["contribution"].astype(float); k = d["k"].astype(float); n, L, H = S.shape
        pos0 = d.get("pos0_attn")
        if variant == "nosink": S = np.where(pos0 > .9, 0.0, S)
        body = np.array([int(t["id"].split("/")[-1]) for t in d["items"]])
        P = perm_idx(k, body, fam, np.random.default_rng(0)); x = np.arange(L) / (L - 1)
        Y = topq(S); M = S.sum(-1); lM = np.log(np.clip(M, 1e-12, None))
        bY, zY = raw_z(Y, k, P); bM, zM = raw_z(lM, k, P)
        ba, za = adj_z(Y, k, P, lambda l: lM[:, l:l+1])
        Sbar = S.mean(0); mn = S.min(0); sig = Sbar - mn
        r = {"model": model, "cell": c, "fam": fam, "variant": variant, "x": x.tolist(), "zY": zY.tolist(), "zM": zM.tolist(), "zadj": za.tolist(),
             "q25_mean": topq(Sbar).tolist(), "q25_floor": topq(mn).tolist(), "q25_signal": topq(np.clip(sig, 0, None)).tolist(),
             "floorfrac": (mn.sum(-1) / Sbar.sum(-1)).tolist()}
        if pos0 is not None and variant != "nosink":
            sink = pos0.mean(0) > .9                                   # [L,H] heads that are sinks on average
            r["sink_mass_frac"] = ((Sbar * sink).sum(-1) / Sbar.sum(-1)).tolist()
            r["sink_n"] = sink.sum(-1).tolist()
            stab = mn / np.clip(Sbar, 1e-12, None)
            r["stab_sink"] = float(np.nanmean(np.where(sink, stab, np.nan))) if sink.any() else None
            r["stab_nonsink"] = float(np.nanmean(np.where(~sink, stab, np.nan)))
            # slope of log mass carried by sink vs non-sink heads (back half)
            Ms = (S * sink[None]).sum(-1); Mn = (S * ~sink[None]).sum(-1); back = x >= .5
            with np.errstate(divide="ignore"):
                r["dlog_sink_back"] = float(np.nanmean(np.where(Ms.min(0) > 0, raw_z(np.log(np.clip(Ms, 1e-12, None)), k, P[:1])[0], np.nan)[back]))
            r["dlog_nonsink_back"] = float(raw_z(np.log(Mn), k, P[:1])[0][back].mean())
        out.append(r)
    return out
if __name__ == "__main__":
    vs = sys.argv[3].split(",")
    jobs = [(m, v) for v in vs for m in ORDER]
    with ProcessPoolExecutor(10) as ex: R = [r for rr in ex.map(job, jobs) for r in rr]
    json.dump(R, open(sys.argv[2], "w")); print(len(R))
