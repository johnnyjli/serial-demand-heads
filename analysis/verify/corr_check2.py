import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np
from scipy.stats import binomtest, wilcoxon
SP=str(paths.ANALYSIS)
R=json.load(open(f"{SP}/decomp_q25.json"))
for name,sub in [("all170",R),("serial136",[r for r in R if not r["cell"].startswith("fixed")])]:
    sig=[r for r in sub if r["p_all"]<.05]
    w=[r for r in sig if r["p_wrong"] is not None and r["p_wrong"]<.05 and np.sign(r["slope_wrong"])==np.sign(r["slope_all"])]
    k2=[r for r in sig if r["p_k2plus"]<.05 and np.sign(r["slope_k2plus"])==np.sign(r["slope_all"])]
    ctrl=[r for r in sig if np.sign(r["slope_ctrl"])==np.sign(r["slope_all"])]
    ctrls=[r for r in ctrl if abs(r["t_ctrl"])>2]
    cw=np.array([r["cw_diff"] for r in sub if r["cw_diff"] is not None])
    tc=[r for r in sub if abs(r["t_corr"])>2]
    # relative: cw_diff/mean *100 and slope rel
    rel_cw=np.array([r["cw_diff"]/r["mean"]*100 for r in sub if r["cw_diff"] is not None])
    rel_sl=np.array([r["slope_all"]/r["mean"]*100 for r in sub])
    print(name, "sig",len(sig),"ctrl same sign",len(ctrl),"ctrl |t|>2",len(ctrls),"wrong sig",len(w),"k2 sig",len(k2))
    print("  cw>0 %d / %d, binom p=%.4f, wilcoxon p=%.4f; |t_corr|>2: %d (pos %d, neg %d; ~%.1f expected by chance)"%((cw>0).sum(),len(cw),binomtest(int((cw>0).sum()),len(cw)).pvalue,wilcoxon(cw).pvalue,len(tc),sum(r['t_corr']>0 for r in tc),sum(r['t_corr']<0 for r in tc),0.0455*len(sub)))
    print("  median cw diff %.3f%% of mean; median |slope| %.3f%%/step"%(np.median(rel_cw),np.median(np.abs(rel_sl))))
