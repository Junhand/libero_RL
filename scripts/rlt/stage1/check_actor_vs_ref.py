import sys, numpy as np, pandas as pd, torch, av
sys.path.insert(0, "/workspace/libero_RL/tmp/rlinf_dev")
from omegaconf import OmegaConf
from rlinf.models.embodiment.smolvla import get_model
from rlinf.models.embodiment.mlp_policy import get_model as get_mlp
R="/workspace/libero_RL"
fcfg=OmegaConf.create(dict(model_path=f"{R}/models/smolvla_libero", num_action_chunks=10, ref_num_action_chunks=10, action_dim=7, precision=None, load_to_device=True, is_lora=False, model_type="smolvla",
  smolvla=dict(vlm_model_name=f"{R}/models/SmolVLM2-500M-Video-Instruct", noise_seed=0, chunk_size=50, advantage_prompt=False, rlt_token_ckpt=f"{R}/tmp/rlt_stage1/rlt_stage1_ckpt.pt",
  rename_map={"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"})))
fm=get_model(fcfg).cuda().eval()
mcfg=OmegaConf.create(dict(model_type="rlt_mlp_policy",z_dim=960,proprio_dim=8,action_dim=7,num_action_chunks=10,ref_num_action_chunks=10,add_q_head=True,q_head_type="default",fixed_std=0.02))
mlp=get_mlp(mcfg,torch.float32)
sd=torch.load(f"{R}/tmp/rlt_stage2/logs/smoke_223023/rlt2stage-std0.02-v2-s2-task/checkpoints/global_step_400/actor/model_state_dict/full_weights.pt",map_location="cpu")
print("load_state_dict:",mlp.load_state_dict(sd)); mlp=mlp.cuda().eval()
D=f"{R}/data/smolvla_recap/libero10_task0_eval_v2_m0"; ep=0; TS=[0,20,40,50,60,80]
def fr(cam,n):
    out=[]
    with av.open(f"{D}/videos/chunk-000/{cam}/episode_{ep:06d}.mp4") as c:
        for f in c.decode(video=0):
            out.append(f.to_ndarray(format="rgb24"))
            if len(out)>=n: break
    return np.stack(out)
a=fr("image",81); w=fr("wrist_image",81); d=pd.read_parquet(f"{D}/data/chunk-000/episode_{ep:06d}.parquet"); S=np.stack(d.state.values).astype(np.float32)
obs=dict(main_images=torch.from_numpy(a[TS]),wrist_images=torch.from_numpy(w[TS]),states=torch.from_numpy(S[TS]),task_descriptions=["put both the alphabet soup and the tomato sauce in the basket"]*len(TS))
rl=fm.extract_rlt_obs(obs)
acts,_=mlp.predict_action_batch(env_obs=rl,mode="eval")
ref=rl["ref_chunk"].cpu()[:, :10]; acts=acts.cpu()
np.set_printoptions(precision=3,suppress=True)
print("actor (deterministic) vs VLA reference chunk, RMS difference per frame:",((acts-ref)**2).mean((1,2)).sqrt().numpy())
print("per-dim RMS (x,y,z,rx,ry,rz,grip):",((acts-ref)**2).mean((0,1)).sqrt().numpy())
print("reference per-dim RMS magnitude:   ",(ref**2).mean((0,1)).sqrt().numpy())
