import sys, glob, os, numpy as np, pandas as pd, torch
sys.path.insert(0, "/workspace/libero_RL/RLinf")
from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenTransformer
F="/workspace/libero_RL/tmp/rlt_stage1/feats"; dev="cuda"
done=sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{F}/*.done")); VALS=[t for t in done if t.startswith("eval_")]; tr=[t for t in done if not t.startswith("eval_")]
ck=torch.load(f"{F}/../rlt_stage1_ckpt.pt"); SC=ck["scale"]
net=RLTTokenTransformer(input_dim=960,embed_dim=960,prefix_seq_len=128,num_layers=2,num_heads=8).to(dev); net.load_state_dict(ck["net"]); net.eval()
X={t:np.load(f"{F}/{t}.npy",mmap_mode="r") for t in done}; Mt={t:pd.read_parquet(f"{F}/{t}_meta.parquet") for t in done}
def enc(sets,stride):
    Z=[];P=[];M=[]
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        for t in sets:
            ii=np.arange(0,len(X[t]),stride)
            for k in range(0,len(ii),128):
                z=torch.from_numpy(X[t][ii[k:k+128]].astype(np.float32)/SC).to(dev)
                Z.append(net.encode_flat(z).float().cpu()); P.append(z.float().mean(1).cpu())
            M.append(Mt[t].iloc[ii])
    return torch.cat(Z),torch.cat(P),pd.concat(M)
FR=F+"_rest"
for t in done:
    if os.path.exists(f"{FR}/{t}.done"): X[t+"+rest"]=np.load(f"{FR}/{t}.npy",mmap_mode="r"); Mt[t+"+rest"]=pd.read_parquet(f"{FR}/{t}_meta.parquet")
tr=tr+[t+"+rest" for t in tr if t+"+rest" in X]; va=VALS+[t+"+rest" for t in VALS if t+"+rest" in X]
Ztr,Ptr,Mtr=enc(tr,3); Zva,Pva,Mva=enc(va,1)
inwin=((Mva.t>=40)&(Mva.t<=100)).values
def std(A,B): mu,sd=A.mean(0),A.std(0)+1e-6; return (A-mu)/sd,(B-mu)/sd
def ridge(a,b,ya,yb,lam=100.):
    a1=torch.cat([a,torch.ones(len(a),1)],1); b1=torch.cat([b,torch.ones(len(b),1)],1)
    W=torch.linalg.solve(a1.T@a1+lam*torch.eye(a1.shape[1]),a1.T@ya); p=b1@W
    return float(1-((p-yb)**2).sum(0).sum()/((yb-yb.mean(0))**2).sum(0).sum())
def logit(a,b,ya,yb,steps=300):
    w=torch.zeros(a.shape[1],requires_grad=True); bb=torch.zeros(1,requires_grad=True); o=torch.optim.LBFGS([w,bb],max_iter=steps)
    def cl():
        o.zero_grad(); l=torch.nn.functional.binary_cross_entropy_with_logits(a@w+bb,ya)+1e-2*(w**2).sum(); l.backward(); return l
    o.step(cl); return float((((b@w+bb)>0).float()==yb).float().mean())
T=lambda M,c: torch.tensor(M[c].values,dtype=torch.float32)
for name,A,B in (("z_rl (1 token, 960d)",Ztr,Zva),("mean-pooled raw tokens (960d)",Ptr,Pva)):
    a,b=std(A,B)
    for sub,mk in (("all",np.ones(len(b),bool)),("t=40..100",inwin),("outside",~inwin)):
        mk_t=torch.from_numpy(mk); bb=b[mk_t]; Mv=Mva[mk]
        r_gap=ridge(a,bb,T(Mtr,["gap"]),T(Mv,["gap"])); r_pos=ridge(a,bb,T(Mtr,["s0","s1","s2"]),T(Mv,["s0","s1","s2"]))
        acc=logit(a,bb,T(Mtr,"succ"),T(Mv,"succ"))
        print(f"[{name}] eval_v2 {sub:9s} n={mk.sum():5d} | R2 finger gap {r_gap:.3f} | R2 eef xyz {r_pos:.3f} | acc episode-success {acc:.3f} (majority {max(Mv.succ.mean(),1-Mv.succ.mean()):.3f})")
