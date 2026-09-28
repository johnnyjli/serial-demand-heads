import sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
SP8=sys.argv[1]; OUT=sys.argv[2]; sys.path.insert(0,SP8)
from e2lib import load, ORDER
from parse import parse
from scipy import stats
res={"positions":{}, "adjust":[]}
def q10(S):
    P=S/S.sum(-1,keepdims=True); m=max(1,int(np.ceil(.1*S.shape[-1]))); return -np.sort(-P,-1)[...,:m].sum(-1)
# 1. positions vs k (prompts are identical across models; use Llama-3.1-8B)
for fam in ("binding","relational"):
    for w in ("w0","w1"):
        d=load("Llama-3.1-8B",f"{fam}/{w}"); F=[]
        for t in d["items"]:
            p=parse(f"{fam}/{w}",t["text"]); idx_min=p["q_pos"]; 
            F.append((t["k"],p["q_pos"],p["q_from_end"],p["root_pos"],p["span"],p["n_st"],p["hops"],p["forward"]))
        F=np.array(F,float); k=F[:,0]
        assert (F[:,6]==k).all()
        row={}
        for j,name in [(1,"q_pos"),(2,"q_from_end"),(3,"root_pos"),(4,"span"),(7,"forward")]:
            r,pv=stats.spearmanr(k,F[:,j]); row[name]={"rho":float(r),"p":float(pv),"mean_by_k":[float(F[k==kk,j].mean()) for kk in range(1,7)]}
        row["n_statements"]=sorted(set(F[:,5].astype(int).tolist()))
        res["positions"][f"{fam}/{w}"]=row
        print(fam,w,{n:(round(v["rho"],2),[round(x,1) for x in v["mean_by_k"]]) for n,v in row.items() if n!="n_statements"}, "n_st",row["n_statements"])
# 2. does the back-half k-slope survive position covariates?
for m in ORDER:
    attn="sdpa" if m=="Qwen2.5-7B-Instruct" else "eager"
    for fam in ("binding","relational"):
        for w in ("w0","w1"):
            d=load(m,f"{fam}/{w}",attn); S=d["contribution"].astype(float); L=d["L"]
            keep=~((S.max(-1)/np.maximum(np.median(S,-1),1e-9))>50).any(1)
            items=[t for t,kp in zip(d["items"],keep) if kp]; S=S[keep]
            x=np.arange(L)/(L-1); y=q10(S)[:,x>=.5].mean(1)
            k=np.array([t["k"] for t in items],float)
            P=[parse(f"{fam}/{w}",t["text"]) for t in items]
            body=np.array([int(t["id"].split("/")[-1]) for t in items])
            qe=np.array([p["q_from_end"] for p in P],float); fw=np.array([p["forward"] for p in P],float); sp=np.array([p["span"] for p in P],float)
            # body fixed effects (as in the within-body design): demean within body
            def dm(a):
                a=a.copy().astype(float)
                for g in np.unique(body): a[body==g]-=a[body==g].mean(0)
                return a
            def tk(X):
                Xd=dm(X); yd=dm(y[:,None])[:,0]
                beta,res_,rk,sv=np.linalg.lstsq(Xd,yd,rcond=None); r=yd-Xd@beta
                dof=len(yd)-Xd.shape[1]-len(np.unique(body)); s2=(r@r)/dof
                se=np.sqrt(s2*np.linalg.inv(Xd.T@Xd)[0,0]); return float(beta[0]/se), float(beta[0])
            t0,b0=tk(k[:,None]); t1,b1=tk(np.column_stack([k,qe])); t2,b2=tk(np.column_stack([k,qe,fw,sp]))
            res["adjust"].append({"model":m,"cell":f"{fam}/{w}","t_k":t0,"t_k_adj":t1,"b_k":b0,"b_k_adj":b1,"t_k_adj2":t2,"b_k_adj2":b2})
A=res["adjust"]; t0=np.array([a["t_k"] for a in A]); t1=np.array([a["t_k_adj"] for a in A]); b0=np.array([a["b_k"] for a in A]); b1=np.array([a["b_k_adj"] for a in A])
summ={"n_cells":len(A),"r_t":float(np.corrcoef(t0,t1)[0,1]),"sign_agree":float(np.mean(np.sign(t0)==np.sign(t1))),
      "median_ratio_b":float(np.median(b1/b0)),"n_sig_neg_before":int((t0<-1.96).sum()),"n_sig_neg_after":int((t1<-1.96).sum()),
      "n_sig_pos_before":int((t0>1.96).sum()),"n_sig_pos_after":int((t1>1.96).sum()),"mean_t_before":float(t0.mean()),"mean_t_after":float(t1.mean())}
t2=np.array([a["t_k_adj2"] for a in A]); b2=np.array([a["b_k_adj2"] for a in A])
summ.update({"r_t2":float(np.corrcoef(t0,t2)[0,1]),"sign_agree2":float(np.mean(np.sign(t0)==np.sign(t2))),"median_ratio_b2":float(np.median(b2/b0)),
  "n_sig_neg_after2":int((t2<-1.96).sum()),"n_sig_pos_after2":int((t2>1.96).sum()),"mean_t_after2":float(t2.mean())})
res["adjust_summary"]=summ; print(json.dumps(summ,indent=1))
json.dump(res,open(f"{OUT}/positions.json","w"),indent=1)
