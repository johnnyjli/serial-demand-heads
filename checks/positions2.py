import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
SP8=sys.argv[1]; OUT=sys.argv[2]; sys.path.insert(0,SP8)
from e2lib import load, ORDER
from parse import parse
def q10(S):
    P=S/S.sum(-1,keepdims=True); m=max(1,int(np.ceil(.1*S.shape[-1]))); return -np.sort(-P,-1)[...,:m].sum(-1)
rows=[]
for m in ORDER:
    attn="sdpa" if m=="Qwen2.5-7B-Instruct" else "eager"
    for fam in ("binding","relational"):
        for w in ("w0","w1"):
            d=load(m,f"{fam}/{w}",attn); S=d["contribution"].astype(float); L=d["L"]
            keep=~((S.max(-1)/np.maximum(np.median(S,-1),1e-9))>50).any(1)
            items=[t for t,kp in zip(d["items"],keep) if kp]; S=S[keep]
            x=np.arange(L)/(L-1); y=q10(S)[:,x>=.5].mean(1)
            k=np.array([t["k"] for t in items],float); body=np.array([int(t["id"].split("/")[-1]) for t in items])
            P=[parse(f"{fam}/{w}",t["text"]) for t in items]
            fw=np.array([p["forward"] for p in P],float); sp=np.array([p["span"] for p in P],float)
            def dm(a):
                a=np.asarray(a,float).copy()
                for g in np.unique(body): a[body==g]-=a[body==g].mean(0)
                return a
            def fit(cols):
                X=dm(np.column_stack(cols)); yd=dm(y[:,None])[:,0]
                beta=np.linalg.lstsq(X,yd,rcond=None)[0]; r=yd-X@beta; dof=len(yd)-X.shape[1]-len(np.unique(body))
                se=np.sqrt((r@r)/dof*np.diag(np.linalg.inv(X.T@X))); return (beta/se).tolist()
            # within a depth: does span / forward predict the share? (items with k>=2, depth fixed effects via demeaning by k too)
            mask=k>=2; tw=[]
            for var in (sp,fw):
                yy=y[mask].copy(); vv=var[mask].copy(); kk=k[mask]
                for g in np.unique(kk): yy[kk==g]-=yy[kk==g].mean(); vv[kk==g]-=vv[kk==g].mean()
                b=(vv@yy)/(vv@vv); r=yy-b*vv; se=np.sqrt((r@r)/(len(yy)-2-len(np.unique(kk)))/(vv@vv)); tw.append(b/se)
            rows.append({"model":m,"cell":f"{fam}/{w}","k_span":fit([k,sp]),"k_fwd":fit([k,fw]),"k_span_fwd":fit([k,sp,fw]),"within_k_span_t":tw[0],"within_k_fwd_t":tw[1]})
def summ(key,i):
    t=np.array([r[key][i] for r in rows]); return {"mean_t":float(t.mean()),"n_neg_sig":int((t<-1.96).sum()),"n_pos_sig":int((t>1.96).sum())}
S={"k|k+span":summ("k_span",0),"span|k+span":summ("k_span",1),"k|k+fwd":summ("k_fwd",0),"fwd|k+fwd":summ("k_fwd",1),
   "k|k+span+fwd":summ("k_span_fwd",0),"span|all":summ("k_span_fwd",1),"fwd|all":summ("k_span_fwd",2)}
for key in ("within_k_span_t","within_k_fwd_t"):
    t=np.array([r[key] for r in rows]); S[key]={"mean_t":float(t.mean()),"n_neg_sig":int((t<-1.96).sum()),"n_pos_sig":int((t>1.96).sum())}
S["n_cells"]=len(rows); print(json.dumps(S,indent=1)); json.dump({"rows":rows,"summary":S},open(f"{OUT}/positions2.json","w"),indent=1)
