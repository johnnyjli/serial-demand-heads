from concurrent.futures import ProcessPoolExecutor
from clib import *
def go(a): raw(*a); return a
if __name__ == "__main__":
    jobs = [(m, f, w, attn_of(m)) for m in ORDER for f in FAMS for w in ("w0", "w1")] + [("Qwen2.5-7B-Instruct", f, w, "eager") for f in FAMS for w in ("w0", "w1")]
    with ProcessPoolExecutor(8) as ex: print(len(list(ex.map(go, jobs))))
