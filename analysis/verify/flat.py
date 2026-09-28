import json,sys
J=sys.argv[1]; out=open(sys.argv[2],'w')
n=0
for line in open(J):
    n+=1
    try: r=json.loads(line)
    except: continue
    msg=r.get('message') or {}
    role=msg.get('role') or r.get('type')
    c=msg.get('content')
    ts=r.get('timestamp','')
    if isinstance(c,str):
        out.write(f"\n===== L{n} {role} {ts}\n{c}\n"); continue
    if not isinstance(c,list): continue
    for b in c:
        t=b.get('type')
        if t=='text': out.write(f"\n===== L{n} {role}-text {ts}\n{b['text']}\n")
        elif t=='tool_use': out.write(f"\n===== L{n} {role}-tool_use {ts}\n{json.dumps(b.get('input'))[:20000]}\n")
        elif t=='tool_result':
            cc=b.get('content')
            if isinstance(cc,list): cc='\n'.join(x.get('text','') for x in cc if isinstance(x,dict))
            out.write(f"\n===== L{n} tool_result {ts}\n{cc}\n")
