"""Stage 1: train the RL-token encoder-decoder (VLA frozen) on cached image-token features; probe z_rl."""
import sys, glob, os, time, json, numpy as np, pandas as pd, torch
sys.path.insert(0, "/workspace/libero_RL/RLinf")
from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenTransformer
F="/workspace/libero_RL/tmp/rlt_stage1/feats"; STEPS=int(os.environ.get("STEPS",3000)); BS=64; DEMO_FRAC=float(os.environ.get("DEMO_FRAC",0.1))
done=sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{F}/*.done"))
tr=[t for t in done if not t.startswith("eval_")]; VALS=[t for t in done if t.startswith("eval_")]; assert tr and VALS, done
print("train sets:",tr,"| val (eval_v2):",VALS,flush=True)
REST=os.environ.get("REST","1")=="1"; FR=F+"_rest"
X={t:np.load(f"{F}/{t}.npy",mmap_mode="r") for t in done}; Mt={t:pd.read_parquet(f"{F}/{t}_meta.parquet") for t in done}
if REST:   # frames outside t=40..100 (every 6th); keys "<tag>+rest"
    for t in done:
        if os.path.exists(f"{FR}/{t}.done"): X[t+"+rest"]=np.load(f"{FR}/{t}.npy",mmap_mode="r"); Mt[t+"+rest"]=pd.read_parquet(f"{FR}/{t}_meta.parquet")
    print("rest sets available:",sorted(k for k in X if k.endswith("+rest")),flush=True)
tr=tr+[t+"+rest" for t in tr if t+"+rest" in X]
idx=[(t,i) for t in tr for i in range(len(X[t]))]; rng=np.random.default_rng(0)
sample=np.stack([X[t][i] for t,i in (idx[j] for j in rng.choice(len(idx),256,replace=False))]).astype(np.float32)
SCALE=float(sample.std()); print("global std of z: %.3f (inputs divided by it)"%SCALE,flush=True)
DX=np.concatenate([np.load(f"{F}/../feats_demo/sft_target3.npy")]+([np.load(f"{F}/../feats_demo_rest/sft_target3.npy")] if os.path.exists(f"{F}/../feats_demo_rest/sft_target3.npy") else [])); print("demo frames:",len(DX),"mix fraction per batch:",DEMO_FRAC,flush=True)
dev="cuda"; torch.manual_seed(0)
net=RLTTokenTransformer(input_dim=960,embed_dim=960,prefix_seq_len=128,num_layers=2,num_heads=8).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=0.01)
print("params(M): %.1f"%(sum(p.numel() for p in net.parameters())/1e6),flush=True)
def batch(items): return torch.from_numpy(np.stack([X[t][i] for t,i in items]).astype(np.float32)/SCALE).to(dev)
vi=[(k,i) for t in VALS for k in (t,t+"+rest") if k in X for i in range(0,len(X[k]),2)]; print("val frames:",len(vi),flush=True)
def demo_loss():
    net.eval()
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        l,_=net(torch.from_numpy(np.asarray(DX).astype(np.float32)/SCALE).to(dev))
    net.train(); return l.item()
def val_loss():
    net.eval(); tot=0;n=0
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        for k in range(0,len(vi),128):
            z=batch(vi[k:k+128]); l,_=net(z); tot+=l.item()*len(z); n+=len(z)
    net.train(); return tot/n
t0=time.time()
for s in range(1,STEPS+1):
    nd=int(round(BS*DEMO_FRAC)); z=batch([idx[j] for j in rng.integers(0,len(idx),BS-nd)])
    if nd: z=torch.cat([z,torch.from_numpy(np.stack([DX[j] for j in rng.integers(0,len(DX),nd)]).astype(np.float32)/SCALE).to(dev)])
    for g in opt.param_groups: g["lr"]=1e-4*min(1,s/200)
    with torch.autocast("cuda",dtype=torch.bfloat16): loss,_=net(z)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
    if s%100==0: print(f"step {s} train mse {loss.item():.4f} ({time.time()-t0:.0f}s)",flush=True)
    if s%500==0 or s==STEPS: print(f"  demo mse (in training) {demo_loss():.4f} |",end=""); print(f"  val mse {val_loss():.4f}   (input var≈1.0 after scaling; mean-pred baseline ≈ {float((batch(vi[:128]).var(dim=(0,)).mean())):.3f})",flush=True)
torch.save({"net":net.state_dict(),"scale":SCALE},f"{F}/../rlt_stage1_ckpt.pt")
# (probes: see probe_stage1.py)
