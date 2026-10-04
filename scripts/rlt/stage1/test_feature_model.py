import sys, numpy as np, pandas as pd, torch, av
sys.path.insert(0, "/workspace/libero_RL/tmp/rlinf_dev")
from omegaconf import OmegaConf
from rlinf.models.embodiment.smolvla import get_model
R="/workspace/libero_RL"
cfg=OmegaConf.create(dict(model_path=f"{R}/models/smolvla_libero", num_action_chunks=10, ref_num_action_chunks=10, action_dim=7, precision=None, load_to_device=True, is_lora=False, model_type="smolvla",
  smolvla=dict(vlm_model_name=f"{R}/models/SmolVLM2-500M-Video-Instruct", noise_seed=0, chunk_size=50, advantage_prompt=False,
  rlt_token_ckpt=f"{R}/tmp/rlt_stage1/rlt_stage1_ckpt.pt",
  rename_map={"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"})))
m=get_model(cfg).cuda().eval(); print(type(m).__name__, "z_dim", m.z_dim)
D=f"{R}/data/smolvla_recap/libero10_task0_eval_v2_m0"; ep=0
def fr(cam,n):
    out=[]
    with av.open(f"{D}/videos/chunk-000/{cam}/episode_{ep:06d}.mp4") as c:
        for f in c.decode(video=0):
            out.append(f.to_ndarray(format="rgb24"))
            if len(out)>=n: break
    return np.stack(out)
TS=[40,50,60,70,80,90,100]
a=fr("image",101); w=fr("wrist_image",101)
d=pd.read_parquet(f"{D}/data/chunk-000/episode_{ep:06d}.parquet"); S=np.stack(d.state.values).astype(np.float32)
obs=dict(main_images=torch.from_numpy(a[TS]),wrist_images=torch.from_numpy(w[TS]),states=torch.from_numpy(S[TS]),task_descriptions=["put both the alphabet soup and the tomato sauce in the basket"]*len(TS))
o=m.extract_rlt_obs(obs); print({k:(tuple(v.shape),str(v.dtype),str(v.device)) for k,v in o.items()})
# compare with the cached features of the same frames (window file: t=40..100 step 2 -> index (t-40)/2)
F=f"{R}/tmp/rlt_stage1/feats"; X=np.load(f"{F}/eval_m0.npy",mmap_mode="r")
import torch as T
ck=T.load(f"{R}/tmp/rlt_stage1/rlt_stage1_ckpt.pt"); sc=ck["scale"]
z=T.from_numpy(np.stack([X[(t-40)//2] for t in TS]).astype(np.float32)/sc).cuda()
with T.no_grad(): zc=m.rlt_module.encode_flat(z).float()
print("max|z_rl(live) - z_rl(cached)|: %.4f   (|z_rl| mean %.3f)"%((o["z_rl"]-zc).abs().max().item(), zc.abs().mean().item()))
print("cosine per frame:", torch.nn.functional.cosine_similarity(o["z_rl"],zc).cpu().numpy().round(4))
rec=np.load(f"{D}/data/chunk-000/episode_{ep:06d}.parquet".replace(".parquet",".parquet"),allow_pickle=True) if False else None
A=np.stack(d.actions.values)
ref=o["ref_chunk"].cpu().numpy(); print("ref_chunk vs recorded actions L1 per-dim:", np.abs(ref-np.stack([A[t:t+10] for t in TS])).mean((0,1)).round(3))
