import sys, json, re, collections, numpy as np
sys.path.insert(0, "..")
import e2lib, solve
idx = e2lib.index("eager")
def items(model, cell):
    z = np.load(idx[(model, cell)], allow_pickle=False); return json.loads(str(z["meta"]))["items"]
M="Llama-3.2-1B"
# binding/relational statements
for fam, pat in [("binding", r"(\w) = (\w)\."), ("relational", r"The boss of (\w) is (\w)\.")]:
    its = items(M, fam+"/w0")
    cnt = collections.Counter(); chains = collections.Counter(); bodies=collections.defaultdict(set); inorder=0
    for it in its:
        fin = it["text"].split("\n\n")[-1]
        st = re.findall(pat, fin); cnt[len(st)] += 1
        roots = [y for x,y in st if y.isdigit()]; chains[len(roots)] += 1
        body = fin.rsplit(".",1)[0]
        bodies[body].add(it["k"])
    print(fam, "n statements", dict(cnt), "n roots", dict(chains), "n bodies", len(bodies), "ks per body", collections.Counter(len(v) for v in bodies.values()))
    print(" example final:", its[0]["text"].split("\n\n")[-1][:300])
# arith instance sharing
a0 = items(M,"arith/w0"); a1 = items(M,"arith/w1")
def par0(t): return tuple(map(int, re.search(r"add (\d) and keep the remainder after dividing by (\d+)\. Start at (\d)\. Apply the rule (\d+) times", t.split("\n\n")[-1]).groups()))
def par1(t):
    s,k,d,m = map(int, re.search(r"x = (\d)\. Repeat (\d+) times: x = \(x \+ (\d)\) mod (\d+)", t.split("\n\n")[-1]).groups()); return (d,m,s,k)
print("arith shared instances:", sum(par0(x["text"])==par1(y["text"]) for x,y in zip(a0,a1)), "/", len(a0))
print("arith increments used:", sorted({par0(x["text"])[0] for x in a0}), "moduli", {par0(x["text"])[1] for x in a0})
# fixed_digit: cycle through start length k+1, others fixed
f0 = items(M,"fixed_digit/w0"); ok=0; wk=collections.Counter(); ns=collections.Counter()
for it in f0:
    fin = it["text"].split("\n\n")[-1]
    tbl = dict((int(a),int(b)) for a,b in re.findall(r"(\d) goes to (\d)\.", fin)); ns[len(tbl)]+=1
    s = int(re.search(r"Start at (\d)", fin).group(1)); K=int(re.search(r"Take (\d+) steps", fin).group(1)); wk[K]+=1
    cyc=[s]; x=tbl[s]
    while x!=s: cyc.append(x); x=tbl[x]
    fixed = all(tbl[q]==q for q in tbl if q not in cyc)
    ok += (len(cyc)==it["k"]+1 and fixed)
print("fixed_digit: cycle len == k+1 & others fixed:", ok, "/", len(f0), "written K", dict(wk), "n states", dict(ns))
# worked examples in fixed_digit: states count
ex_ns = collections.Counter()
for it in f0:
    for b in it["text"].split("\n\n")[:-1]:
        ex_ns[len(re.findall(r"(\d) goes to (\d)\.", b))]+=1
print("fixed_digit worked-example table sizes", dict(ex_ns))
# pointer: single 10-cycle
p0 = items(M,"pointer/w0"); ok=0
for it in p0:
    fin = it["text"].split("\n\n")[-1]
    tbl = dict((int(a),int(b)) for a,b in re.findall(r"(\d) goes to (\d)\.", fin))
    s=0; n=0
    while True:
        s=tbl[s]; n+=1
        if s==0: break
    ok += (n==10)
print("pointer single 10-cycle:", ok, "/", len(p0))
# base vs instruct token counts (chat template?)
for a,b in [("Llama-3.1-8B","Llama-3.1-8B-Instruct"),("Llama-3.2-3B","Llama-3.2-3B-Instruct"),("Qwen2.5-7B","Qwen2.5-7B-Instruct"),("OLMo-2-1124-7B","OLMo-2-1124-7B-Instruct"),("OLMo-2-1124-7B","OLMo-2-1124-7B-SFT"),("OLMo-2-1124-7B","OLMo-2-1124-7B-DPO")]:
    za=np.load(idx[(a,"pointer/w0")]); zb=np.load(idx[(b,"pointer/w0")])
    ma=json.loads(str(za["meta"])); mb=json.loads(str(zb["meta"]))
    print(a,b, "ntok", set(za["n_tokens"].tolist()), set(zb["n_tokens"].tolist()), ma["pos0_tokens"], mb["pos0_tokens"], ma["appended_forms"], mb["appended_forms"])
