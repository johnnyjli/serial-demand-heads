import numpy as np
from ilib import *
exec(open("s8_confound.py").read().split("res = {}")[0])   # reuse my own demean/ols_t/cellfit definitions
def cellx(m, f, w, attn):
    d = raw(m, f, w, attn); kp = d["keep"]; k = d["k"][kp].astype(float); body = d["body"][kp]; e = d["ent"][kp]
    wi = within(f); kd = demean(k[:, None], body, wi)[:, 0]; ed = demean(e[:, None], body, wi)[:, 0]
    return np.corrcoef(kd, ed)[0, 1]
for m in sorted(set(x for p in PAIRS for x in p)):
    r = [cellx(m, f, w, prim_attn(m)) for f in TASKS for w in ("w0", "w1")]
    R = [cellfit(m, f, w, prim_attn(m)) for f in TASKS for w in ("w0", "w1")]
    kt = np.mean([x["k"] for x in R]); ket = np.mean([x["k+ent"] for x in R])
    print(f"{m:26s} mean r(k,ent) {np.mean(r):+.2f}  back t k {kt:+.2f} -> k+ent {ket:+.2f}  ratio {ket/kt if abs(kt)>.05 else np.nan:+.2f}  expected ratio from VIF alone ~{np.mean([np.sqrt(1-x**2) for x in r]):.2f}")
