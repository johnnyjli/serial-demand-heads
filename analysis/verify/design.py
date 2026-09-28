import sys, json, re, collections, numpy as np
sys.path.insert(0, "..")
import e2lib, solve
idx = e2lib.index("eager")
def items(model, cell):
    z = np.load(idx[(model, cell)], allow_pickle=False); m = json.loads(str(z["meta"])); return m
M = "Llama-3.2-1B"
for fam in ["pointer","arith","binding","relational","fixed_digit"]:
    m0 = items(M, fam+"/w0"); m1 = items(M, fam+"/w1")
    print("==", fam, "setting:", m0["setting"], "chance", m0["chance"])
    it0 = m0["items"]; it1 = m1["items"]
    # instance sharing between wordings: same gold, k; same digits sequence
    dig = lambda t: re.findall(r"\d", t.split("\n\n")[-1])
    same_gold = all(a["gold"]==b["gold"] and a["k"]==b["k"] for a,b in zip(it0,it1))
    same_dig = sum(dig(a["text"])==dig(b["text"]) for a,b in zip(it0,it1))
    print(" w0/w1 same gold&k:", same_gold, " same digit sequence in final block:", same_dig, "/", len(it0))
    # gold balance per k
    for k in sorted({it["k"] for it in it0}):
        g = collections.Counter(it["gold"] for it in it0 if it["k"]==k)
        print("  k",k,"gold counts", dict(sorted(g.items())), "echo", {it["echo"] for it in it0 if it["k"]==k}, "answer_set sizes", {len(it["answer_set"]) for it in it0 if it["k"]==k})
    # worked examples: depth set, answers vs item gold
    eq_gold = 0; depths = collections.Counter(); ex_eq_own_k = 0; ex_eq_echo = 0; query_eq = 0; n = 0
    for it in it0:
        bl = solve.blocks(it["text"])
        ex_ans = []
        for b in bl[:-1]:
            a, s = solve.solve_block(fam, "w0", b); ex_ans.append(a)
            kk = re.search(r"Take (\d+) steps|Apply the rule (\d+) times", b)
            if kk:
                kd = int([x for x in kk.groups() if x][0]); depths[kd]+=1
                if a == kd: ex_eq_own_k += 1
            if it["echo"] is not None and str(a)==it["echo"]: ex_eq_echo += 1
            if re.sub(r" \d\.$","",b) == bl[-1]: query_eq += 1
        if it["gold"] in [str(a) for a in ex_ans]: eq_gold += 1
        n += 1
    print("  items with a worked example whose answer == item gold:", eq_gold, "/", n, " ex answer==own k:", ex_eq_own_k, " ex answer==echo:", ex_eq_echo, " example==query:", query_eq, " ex depths (written):", dict(depths))
