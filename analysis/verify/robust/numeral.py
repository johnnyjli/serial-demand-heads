import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, os
SP=str(paths.ANALYSIS)
sys.path.insert(0, SP); sys.argv=[None, SP]
import backhalf as bh
from e2lib import ORDER
from concurrent.futures import ProcessPoolExecutor
if __name__=="__main__":
    jobs=[(m,f"fixed_digit/{w}",a,"all") for m in ORDER for w in ("w0","w1") for a in ("eager","sdpa")]
    jobs+=[(m,f"fixed_digit/{w}","eager","nosink") for m in ORDER for w in ("w0","w1")]
    with ProcessPoolExecutor(10) as ex: R=[r for r in ex.map(bh.one, jobs) if r]
    json.dump(R, open("numeral_bh.json","w")); print(len(R))
    QI="Qwen2.5-7B-Instruct"
    def s(rs,tag):
        n=len(rs); a=sum(r["back"]<0 for r in rs); b=sum(r["back"]<r["front"] for r in rs)
        print(tag,n,a,b,round(100*a/n,1),round(100*b/n,1))
    s([r for r in R if r["variant"]=="all" and ((r["attn"]=="eager" and r["model"]!=QI) or (r["attn"]=="sdpa" and r["model"]==QI))],"primary-config")
    s([r for r in R if r["variant"]=="all" and r["attn"]=="eager"],"eager-all17")
    s([r for r in R if r["variant"]=="all" and r["attn"]=="sdpa"],"sdpa")
    s([r for r in R if r["variant"]=="nosink"],"nosink")
