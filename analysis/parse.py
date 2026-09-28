import re
def parse(cell, text):
    q = text.split("\n\n")[-1]
    fam, w = cell.split("/")
    if fam == "binding":
        pat = r"(\w) = (\w)\." if w == "w0" else r"Let (\w) be (\w)\."
        st = re.findall(pat, q)
        qv = re.search(r"Therefore (\w) =$", q) or re.search(r"The value of (\w) is$", q)
    else:
        pat = r"The boss of (\w) is (\w)\." if w == "w0" else r"(\w) reports to (\w)\."
        st = re.findall(pat, q)
        qv = re.search(r"above (\w) is$", q) or re.search(r"(\w) ultimately reports to$", q)
    qv = qv.group(1)
    pos = {a: i for i, (a, b) in enumerate(st)}; nxt = dict(st)
    chain = [qv]
    while not chain[-1].isdigit(): chain.append(nxt[chain[-1]])
    idx = [pos[v] for v in chain[:-1]]
    return {"n_st": len(st), "q_pos": pos[qv], "root_pos": idx[-1], "hops": len(chain) - 1,
            "span": max(idx) - min(idx), "q_from_end": len(st) - 1 - pos[qv],
            "forward": sum(1 for a, b in zip(idx, idx[1:]) if b > a), "query": qv}
