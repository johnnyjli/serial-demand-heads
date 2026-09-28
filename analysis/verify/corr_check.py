import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import json, numpy as np, sys
SP=str(paths.ANALYSIS)
for fn in ["decomp.json","decomp_clean.json","decomp_q25.json"]:
    R=json.load(open(f"{SP}/{fn}"))
    sig=[r for r in R if r["p_all"]<.05]
    keep=[r for r in sig if np.sign(r["slope_ctrl"])==np.sign(r["slope_all"])]
    keep_sig=[r for r in keep if abs(r["t_ctrl"])>2]
    wrong=[r for r in sig if r["p_wrong"] is not None and r["p_wrong"]<.05 and np.sign(r["slope_wrong"])==np.sign(r["slope_all"])]
    wrong_sign=[r for r in sig if r["slope_wrong"] is not None and np.sign(r["slope_wrong"])==np.sign(r["slope_all"])]
    k2=[r for r in sig if r["p_k2plus"]<.05 and np.sign(r["slope_k2plus"])==np.sign(r["slope_all"])]
    k2_sign=[r for r in sig if np.sign(r["slope_k2plus"])==np.sign(r["slope_all"])]
    cw=[r["cw_diff"] for r in R if r["cw_diff"] is not None]
    tc=[r for r in R if abs(r["t_corr"])>2]
    print(fn, "n",len(R),"sig",len(sig),"| ctrl same sign",len(keep),"ctrl sig",len(keep_sig),
          "| wrong-only same-sign sig",len(wrong), "same sign",len(wrong_sign),
          "| k>=2 same-sign sig",len(k2),"same sign",len(k2_sign))
    print("   cw_diff mean %.5f median %.5f pos %d neg %d; |t_corr|>2 in %d (pos %d neg %d); median |slope_all| %.5f"%(np.mean(cw),np.median(cw),sum(x>0 for x in cw),sum(x<0 for x in cw),len(tc),sum(r['t_corr']>0 for r in tc),sum(r['t_corr']<0 for r in tc), np.median([abs(r['slope_all']) for r in R])))
    viol=[(r['model'],r['cell'],r['slope_all'],r['slope_ctrl']) for r in sig if np.sign(r["slope_ctrl"])!=np.sign(r["slope_all"])]
    print("   sign flips:",viol[:5])
