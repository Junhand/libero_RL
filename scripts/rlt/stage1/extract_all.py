"""Extract frozen-SmolVLA image-token hidden states for frames t in [40,100] step 2, per dataset (resumable)."""
import sys, os, glob, time, numpy as np, pandas as pd, torch, av
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/workspace/libero_RL/RLinf")
from omegaconf import OmegaConf
from rlinf.models.embodiment.smolvla import get_model
from rlinf.models.embodiment.pi0_fast.lerobot_io import build_lerobot_batch_from_env_obs
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
R="/workspace/libero_RL"; OUT=f"{R}/tmp/rlt_stage1/feats"; os.makedirs(OUT,exist_ok=True)
T0,T1,STEP=40,100,2; TS=list(range(T0,T1+1,STEP)); NIMG=128
TASK="put both the alphabet soup and the tomato sauce in the basket"
cfg=OmegaConf.create(dict(model_path=f"{R}/models/smolvla_libero", num_action_chunks=10, action_dim=7,
  smolvla=dict(vlm_model_name=f"{R}/models/SmolVLM2-500M-Video-Instruct", noise_seed=0, chunk_size=50, advantage_prompt=False,
  rename_map={"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"})))
m=get_model(cfg).cuda().eval(); pol=m.policy; mod=pol.model
def dec(path):
    out=[]
    with av.open(path) as c:
        for fr in c.decode(video=0):
            out.append(fr.to_ndarray(format="rgb24"))
            if len(out)>T1: break
    return np.stack(out)
def load(D,ep):
    return dec(f"{D}/videos/chunk-000/image/episode_{ep:06d}.mp4"), dec(f"{D}/videos/chunk-000/wrist_image/episode_{ep:06d}.mp4"), \
           pd.read_parquet(f"{D}/data/chunk-000/episode_{ep:06d}.parquet", columns=["state","actions","is_success"])
def feats(a,w,S):
    obs=dict(main_images=torch.from_numpy(a[TS]),wrist_images=torch.from_numpy(w[TS]),states=torch.from_numpy(S[TS].astype(np.float32)),task_descriptions=[TASK]*len(TS))
    with torch.no_grad():
        b=m._prepare_lerobot_batch(build_lerobot_batch_from_env_obs(obs,image_size=m.image_size))
        imgs,im=pol.prepare_images(b); st=pol.prepare_state(b)
        embs,pad,att=mod.embed_prefix(imgs,im,b["observation.language.tokens"],b["observation.language.attention_mask"],state=st)
        out,_=mod.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad,att),position_ids=torch.cumsum(pad,1)-1,past_key_values=None,inputs_embeds=[embs,None],use_cache=True)
        z=out[0][:,:NIMG].float()
    assert torch.isfinite(z).all() and z.abs().max()<6e4
    return z.half().cpu().numpy()
order=["r4_m2"]+[p for p in sorted(x.split("train_v2_")[1] for x in glob.glob(f"{R}/data/smolvla_recap/libero10_task0_train_v2_r*_m*")) if p!="r4_m2"]
for tag in order:
    D=f"{R}/data/smolvla_recap/libero10_task0_train_v2_{tag}"
    if os.path.exists(f"{OUT}/{tag}.done"): continue
    n=sum(1 for _ in open(f"{D}/meta/episodes.jsonl")); X=np.lib.format.open_memmap(f"{OUT}/{tag}.npy",mode="w+",dtype=np.float16,shape=(n*len(TS),NIMG,960))
    meta=[]; t0=time.time()
    with ThreadPoolExecutor(4) as ex:
        futs=[ex.submit(load,D,e) for e in range(min(8,n))]
        for e in range(n):
            a,w,d=futs[e].result(); futs[e]=None
            if e+8<n: futs.append(ex.submit(load,D,e+8))
            else: futs.append(None)
            S=np.stack(d.state.values); A=np.stack(d.actions.values)
            X[e*len(TS):(e+1)*len(TS)]=feats(a,w,S)
            succ=bool(d.is_success.any()); gap=S[:,6]-S[:,7]
            for k,t in enumerate(TS):
                meta.append((e,t,succ,*S[t].tolist(),float(gap[t]),float(A[t,6])))
            if e%50==0: print(tag,e,n,"%.0fs"%(time.time()-t0),flush=True)
    X.flush(); del X
    pd.DataFrame(meta,columns=["ep","t","succ"]+[f"s{i}" for i in range(8)]+["gap","cmd"]).to_parquet(f"{OUT}/{tag}_meta.parquet")
    open(f"{OUT}/{tag}.done","w").write("ok"); print("DONE",tag,flush=True)
