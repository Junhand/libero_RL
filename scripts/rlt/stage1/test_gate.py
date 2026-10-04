import sys, glob, types, numpy as np, pandas as pd, torch
sys.path.insert(0, "/workspace/libero_RL/tmp/rlinf_dev")
from rlinf.envs.sim.libero.libero_env import LiberoEnv
R="/workspace/libero_RL/data/smolvla_recap"
def run(eps):
    f=types.SimpleNamespace(_rlt_hold_gap_min=0.04,_rlt_hold_gap_max=0.071,_rlt_hold_min_steps=8,
        _rlt_hold_count=np.zeros(1,np.int32),_rlt_hold_done=np.zeros(1,bool))
    res=[]
    for path in eps:
        d=pd.read_parquet(path); d["is_success"]=d["is_success"] if "is_success" in d else True; S=np.stack(d.state.values); A=np.stack(d.actions.values)
        f._rlt_hold_count[:]=0; f._rlt_hold_done[:]=False; t_conf=-1
        for t in range(len(d)):
            raw=[{"robot0_gripper_qpos":S[t,6:8]}]
            if LiberoEnv._update_rlt_hold(f,raw,A[t:t+1])[0]: t_conf=t; break
        res.append((bool(d.is_success.any()),t_conf,len(d)))
    return pd.DataFrame(res,columns=["succ","t_conf","T"])
eps=sorted(glob.glob(f"{R}/libero10_task0_eval_v2_m*/data/chunk-000/episode_*.parquet"))
r=run(eps)
for s,g in r.groupby("succ"):
    c=g[g.t_conf>=0]
    print(f"success={s}: n={len(g)}  gate closed (grasp confirmed): {len(c)} ({len(c)/len(g):.0%}) | t_conf median {c.t_conf.median():.0f} p10 {c.t_conf.quantile(.1):.0f} p90 {c.t_conf.quantile(.9):.0f}")
print("P(success | confirmed) = %.2f  P(success | never confirmed) = %.2f"%(r[r.t_conf>=0].succ.mean(), r[r.t_conf<0].succ.mean()))
# demos
d=run(sorted(glob.glob(f"{R}/libero10_task0_sft_target3/data/chunk-000/episode_*.parquet"))); print("demos t_conf:",d.t_conf.tolist())
