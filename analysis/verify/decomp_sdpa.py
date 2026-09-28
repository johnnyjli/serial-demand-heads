import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import os, sys, json, numpy as np, warnings
os.environ["E2_STAT"]="q25"; os.environ["E2_CLEAN"]="1"
warnings.filterwarnings("ignore")
SP=str(paths.ANALYSIS)
sys.path.insert(0, SP)
from e2lib import *
src=open(f"{SP}/decomp.py").read().split("def ols")[1].split('if __name__')[0]
exec("def ols"+src.replace('d = load(m, c)','d = load(m, c, ATT)'))
from concurrent.futures import ProcessPoolExecutor
ATT="sdpa"
if __name__=="__main__":
    keys=[(m,c) for m in ORDER for c in CELLS if m=="Qwen2.5-7B-Instruct"]
    with ProcessPoolExecutor(10) as ex: R=list(ex.map(one,keys))
    json.dump(R,open("decomp_q25_q7i_sdpa.json","w"))
