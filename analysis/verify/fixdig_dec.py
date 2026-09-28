import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(next(_p for _p in _pl.Path(__file__).resolve().parents if (_p / "paths.py").is_file()))); import paths  # data locations: see paths.py
import sys, json, numpy as np, os
os.environ["E2_CLEAN"] = "1"
SP = str(paths.ANALYSIS)
sys.path.insert(0, SP)
from e2lib import ORDER
sys.argv = [sys.argv[0], SP]; import consist
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
B = 2000
def one(m):
    attn = consist.ATTN.get(m, "eager"); rng = np.random.default_rng(0)
    out = {}
    for f in ["fixed_digit", "pointer"]:
        Y, k, body = consist.family_data(m, f, "q25", attn)
        obs = consist.slopes(k[None], Y)[0]; null = consist.slopes(consist.perms(k, body, f, rng), Y)
        out[f] = (obs / null.std(0)).tolist()
    return m, out
if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex: R = dict(ex.map(one, ORDER))
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixdig_z.json"), "w"))
    tq = stats.t.ppf(.975, 16)
    for f in ["fixed_digit", "pointer"]:
        prof = []
        for m in ORDER:
            z = np.array(R[m][f]); L = len(z); x = np.arange(L) / (L - 1); dec = np.minimum((x * 10).astype(int), 9)
            prof.append([z[dec == d].mean() for d in range(10)])
        prof = np.array(prof); mu = prof.mean(0); se = prof.std(0, ddof=1) / np.sqrt(17)
        print(f)
        for d in range(10): print(f"  {d*10}-{d*10+10}%  {mu[d]:+.2f} [{mu[d]-tq*se[d]:+.2f},{mu[d]+tq*se[d]:+.2f}] neg {np.sum(prof[:,d]<0)}/17")
        x = None
