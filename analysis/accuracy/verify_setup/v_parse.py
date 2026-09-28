"""Independent sentence-level parser for the E2 prompt blocks (written without reference to solve.py
or setup.py).  A block is split into sentences on '. '; each sentence is interpreted by its form.
parse(fam, w, block) returns a dict with the structure of the block and 'shown' (the digit written
after the question, or None for the query block)."""
import re

HEADERS = ("Rules:", "Map:", "Rule:", "Assignments:", "Facts:", "Chart:")


def sentences(block):
    s = block.strip()
    for h in HEADERS:
        if s.startswith(h):
            s = s[len(h):].strip(); break
    if s.endswith("."):
        s = s[:-1]
    return [x.strip() for x in s.split(". ")]


def parse(fam, w, block):
    S = sentences(block)
    out = {"fam": fam, "w": w, "n_sent": len(S)}
    if fam in ("pointer", "fixed_digit"):
        edges, start, steps, shown, qseen = {}, None, None, None, False
        for x in S:
            t = x.split()
            if len(t) == 4 and t[1] in ("goes", "leads") and t[2] == "to":
                a, b = int(t[0]), int(t[3]); assert a not in edges; edges[a] = b
            elif x.startswith("Start at "):
                start = int(x[len("Start at "):])
            elif x.startswith("Take ") and x.endswith(" steps"):
                steps = int(x.split()[1])
            elif x.startswith("From "):             # "From s, make k moves"
                a, b = x[len("From "):].split(", make ")
                start = int(a); steps = int(b.split()[0])
            elif x.startswith("You end at") or x.startswith("You finish at"):
                qseen = True; tail = x.split(" at", 1)[1].strip()
                shown = int(tail) if tail else None
            else:
                raise ValueError(f"unparsed sentence {x!r}")
        assert qseen and start is not None and steps is not None
        out.update(edges=edges, start=start, steps=steps, shown=shown)
        s = start
        for _ in range(steps):
            s = edges[s]
        out["answer"] = s
        # cycle through the start
        cyc, x = [start], edges[start]
        while x != start:
            cyc.append(x); x = edges[x]
        out["cycle"] = cyc
        return out
    if fam == "arith":
        if w == "w0":
            d = mod = s = kk = None; shown = None; qseen = False
            for x in S:
                if x.startswith("add "):
                    t = x.split(); d = int(t[1]); mod = int(t[-1])
                elif x.startswith("Start at "):
                    s = int(x.split()[-1])
                elif x.startswith("Apply the rule "):
                    kk = int(x.split()[3])
                elif x.startswith("You end at"):
                    qseen = True; tail = x[len("You end at"):].strip(); shown = int(tail) if tail else None
                else:
                    raise ValueError(x)
        else:
            d = mod = s = kk = None; shown = None; qseen = False
            for x in S:
                if x.startswith("Repeat "):
                    m = re.fullmatch(r"Repeat (\d+) times: x = \(x \+ (\d+)\) mod (\d+)", x)
                    kk, d, mod = map(int, m.groups())
                elif x.startswith("Now x ="):
                    qseen = True; tail = x[len("Now x ="):].strip(); shown = int(tail) if tail else None
                elif x.startswith("x = "):
                    s = int(x[len("x = "):])
                else:
                    raise ValueError(x)
        assert qseen and None not in (d, mod, s, kk)
        v = s
        for _ in range(kk):
            v = (v + d) % mod
        out.update(start=s, inc=d, mod=mod, steps=kk, shown=shown, answer=v)
        return out
    # binding / relational: statements, then the query sentence
    stm, query = [], None
    for x in S:
        if fam == "binding" and w == "w0":
            if x.startswith("Therefore "):
                query = x[len("Therefore "):]; continue
            a, b = x.split(" = "); stm.append((a, b))
        elif fam == "binding":
            if x.startswith("The value of "):
                query = x[len("The value of "):]; continue
            assert x.startswith("Let ") and " be " in x
            a, b = x[4:].split(" be "); stm.append((a, b))
        elif w == "w0":
            if x.startswith("The top boss above "):
                query = x[len("The top boss above "):]; continue
            assert x.startswith("The boss of ")
            a, b = x[len("The boss of "):].split(" is "); stm.append((a, b))
        else:
            if " ultimately reports to" in x:
                query = x; continue
            a, b = x.split(" reports to "); stm.append((a, b))
    assert query is not None
    if fam == "binding" and w == "w0":
        v, rest = query.split(" =", 1)
    elif fam == "binding":
        v, rest = query.split(" is", 1)
    elif w == "w0":
        v, rest = query.split(" is", 1)
    else:
        v, rest = query.split(" ultimately reports to", 1)
    v = v.strip(); rest = rest.strip()
    shown = int(rest) if rest else None
    lhs = [a for a, _ in stm]
    assert len(lhs) == len(set(lhs))
    nxt = dict(stm); x, hops = v, 0
    while not x.isdigit():
        x = nxt[x]; hops += 1
        assert hops < 50
    roots = sorted({int(b) for _, b in stm if b.isdigit()})
    out.update(stm=stm, qvar=v, shown=shown, answer=int(x), hops=hops, roots=roots)
    return out
