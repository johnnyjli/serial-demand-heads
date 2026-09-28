"""Independent solvers from prompt text (no serial_tasks code)."""
import re
def blocks(text): return text.split("\n\n")
def solve_block(fam, w, b):
    """Return (answer, shown) for one block; shown = the digit written after the query, if any."""
    if fam in ("pointer", "fixed_digit", "ladder"):
        tbl = dict((int(a), int(c)) for a, c in re.findall(r"(\d) (?:goes|leads) to (\d)\.", b))
        s = int(re.search(r"(?:Start at|From) (\d)", b).group(1)); k = int(re.search(r"(?:Take|make) (\d+) (?:steps|moves)", b).group(1))
        for _ in range(k): s = tbl[s]
        m = re.search(r"(?:end at|finish at) (\d)\.$", b.strip()); return s, (int(m.group(1)) if m else None)
    if fam == "arith":
        if w == "w0":
            d, mod, s, k = map(int, re.search(r"add (\d) and keep the remainder after dividing by (\d+)\. Start at (\d)\. Apply the rule (\d+) times", b).groups())
            m = re.search(r"end at (\d)\.$", b.strip())
        else:
            s, k, d, mod = map(int, re.search(r"x = (\d)\. Repeat (\d+) times: x = \(x \+ (\d)\) mod (\d+)", b).groups())
            m = re.search(r"Now x = (\d)\.$", b.strip())
        return (s + k * d) % mod, (int(m.group(1)) if m else None)
    if fam in ("binding", "relational"):
        if fam == "binding":
            st = re.findall(r"(\w) = (\w)\." if w == "w0" else r"Let (\w) be (\w)\.", b)
            q = re.search(r"(?:Therefore|The value of) (\w) (?:=|is)(?: (\d)\.)?$", b.strip())
        else:
            st = re.findall(r"The boss of (\w) is (\w)\." if w == "w0" else r"(\w) reports to (\w)\.", b)
            q = re.search(r"(?:above (\w) is|(\w) ultimately reports to)(?: (\d)\.)?$", b.strip())
        nxt = dict(st); v = next(g for g in q.groups()[:-1] if g) if fam == "relational" else q.group(1)
        seen = 0
        while not v.isdigit():
            v = nxt[v]; seen += 1
            if seen > 50: raise ValueError("cycle")
        return int(v), (int(q.groups()[-1]) if q.groups()[-1] else None)
    raise ValueError(fam)
