"""Offline check of actor_phase=1 / pre_actor flags on recorded episodes: column 0 (pre-actor) must be on until the first
confirmed grasp, column 1 (learned actor) from the release of the first object until the second grasp, both off in between."""
import sys, numpy as np, types, torch
sys.path.insert(0, "/workspace/libero_RL/tmp/rlinf_dev"); sys.path.insert(0, "/workspace/libero_RL/tmp/rlt_stage2")
from rlinf.envs.sim.libero.libero_env import LiberoEnv
from test_gate2 import gate, episodes_collected
ok = 0; tot = 0
for S, A, succ in episodes_collected("/workspace/libero_RL/tmp/rlt_stage2/diag/G_p1" if len(sys.argv) < 2 else sys.argv[1]):
    f = gate(2); f._rlt_actor_phase = 1; f._rlt_pre_actor = True
    ev = []; hold = []
    for t in range(len(S)):
        new = LiberoEnv._update_rlt_hold(f, [{"robot0_gripper_qpos": S[t, 6:8]}], A[t:t + 1])
        if new[0]: hold.append(t)
        ev.append(LiberoEnv._rlt_switch_flags(f).numpy()[0])
    ev = np.array(ev); assert ev.shape == (len(S), 2) and ev.dtype == bool
    assert not (ev[:, 0] & ev[:, 1]).any()
    pre_end = hold[0] if hold else len(S)
    tot += 1
    # pre-actor on exactly until the first confirmed grasp
    c1 = ev[:pre_end, 0].all() and not ev[pre_end:, 0].any()
    # learned actor off before the first grasp, on after the re-arm until the second grasp
    c2 = not ev[:pre_end, 1].any() and (ev[pre_end:, 1].any() == (len(hold) >= 1 and ev[pre_end:, 1].any()))
    on2 = np.where(ev[:, 1])[0]
    c3 = (len(on2) == 0) or (on2.min() > pre_end and (len(hold) < 2 or on2.max() < hold[1] + 1))
    ok += bool(c1 and c2 and c3)
print(f"{ok}/{tot} episodes consistent")
