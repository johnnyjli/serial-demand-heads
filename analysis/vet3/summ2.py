import json, sys, numpy as np
R = json.load(open(sys.argv[1])); T = ["pointer", "arith", "binding", "relational"]
for v in ["eager", "sdpa", "nosink"]:
    rr = [r for r in R if r["variant"] == v]
    fz = lambda key, sel: np.array([np.mean(np.array(r[key])[sel(np.array(r["x"]))]) for r in rr])
    b = fz("zY", lambda x: x >= .5); f = fz("zY", lambda x: x < .5); ba = fz("zadj", lambda x: x >= .5); fa = fz("zadj", lambda x: x < .5)
    bm = fz("zM", lambda x: x >= .5); fm = fz("zM", lambda x: x < .5)
    print(f"{v:7s} n={len(rr)} raw back<0 {np.mean(b<0):.2f} back<front {np.mean(b<f):.2f} | mass-adjusted back<0 {np.mean(ba<0):.2f} back<front {np.mean(ba<fa):.2f} (backz {b.mean():+.2f} -> {ba.mean():+.2f}) | logM back z<0 {np.mean(bm<0):.2f} front z<0 {np.mean(fm<0):.2f} (mean {bm.mean():+.2f}/{fm.mean():+.2f})")
    for t in T:
        s = [i for i, r in enumerate(rr) if r["fam"] == t]
        print(f"      {t:10s} raw {np.mean(b[s]<0):.2f}/{np.mean(b[s]<f[s]):.2f}  adj {np.mean(ba[s]<0):.2f}/{np.mean(ba[s]<fa[s]):.2f}")
    # layer-level: all layers pooled
    X = np.concatenate([r["x"] for r in rr]); ZY = np.concatenate([r["zY"] for r in rr]); ZM = np.concatenate([r["zM"] for r in rr])
    for name, sel in [("front", X < .5), ("back", X >= .5)]:
        mf = sel & (ZM < -2); mr = sel & (ZM > 2); flat = sel & (np.abs(ZM) <= 2)
        print(f"      {name}: layers {sel.sum()}, mass falls (zM<-2) {mf.mean()/sel.mean():.2f} of layers; P(share z<-2 | mass falls) {np.mean(ZY[mf]<-2):.2f}, P(z>2|mass falls) {np.mean(ZY[mf]>2):.2f};"
              f" P(share z<-2 | mass flat) {np.mean(ZY[flat]<-2):.2f}; P(share z<-2 | mass rises) {np.mean(ZY[mr]<-2) if mr.any() else float('nan'):.2f} (n={mr.sum()}); corr(zY,zM) {np.corrcoef(ZY[sel], ZM[sel])[0,1]:+.2f}")
    if v == "eager":
        for name, sel in [("front", lambda x: x < .5), ("back", lambda x: x >= .5)]:
            g = lambda key: np.array([np.mean(np.array(r[key])[sel(np.array(r["x"]))]) for r in rr])
            qm, qf, qs, ff = g("q25_mean"), g("q25_floor"), g("q25_signal"), g("floorfrac")
            print(f"      {name}: top-25% share of mean norms {np.median(qm):.3f}, of per-head minima (floor) {np.median(qf):.3f}, of above-min signal {np.median(qs):.3f};"
                  f" floor more even than signal in {np.mean(qf<qs):.2f} of cells; min/mean mass {np.median(ff):.2f}")
        sm = np.array([np.mean(np.array(r["sink_mass_frac"])[np.array(r["x"]) >= .5]) for r in rr])
        print(f"      sink heads (mean pos0 attn>.9): back-half mass share median {np.median(sm):.3f} (IQR {np.percentile(sm,25):.3f}-{np.percentile(sm,75):.3f}); "
              f"stability min/mean sink {np.nanmedian([r['stab_sink'] for r in rr if r['stab_sink'] is not None]):.2f} vs nonsink {np.median([r['stab_nonsink'] for r in rr]):.2f}")
        ds = np.array([r["dlog_sink_back"] for r in rr], float); dn = np.array([r["dlog_nonsink_back"] for r in rr])
        print(f"      back-half dlog mass/k: sink heads median {np.nanmedian(ds)*100:+.2f}%  non-sink {np.median(dn)*100:+.2f}%")
