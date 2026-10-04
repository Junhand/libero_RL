"""Offline check of the miss counter (actor_phase=1) on recorded episodes against an independent computation:
a miss = a run of >= 3 steps of (closing and gap <= 0.03) that starts while the gate is armed after the first grasp."""
import sys, numpy as np
sys.path.insert(0, "/workspace/libero_RL/tmp/rlinf_dev"); sys.path.insert(0, "/workspace/libero_RL/tmp/rlt_stage2")
from rlinf.envs.sim.libero.libero_env import LiberoEnv
from test_gate2 import gate, episodes_collected
from diag_analyze import runs
root = sys.argv[1]; agree = 0; n = 0; tot_env = []; tot_all = []; per_phase = []
for S, A, succ in episodes_collected(root):
    f = gate(2); f._rlt_actor_phase = 1; f._rlt_pre_actor = True
    f._rlt_miss_gap_max = 0.03; f._rlt_miss_min_steps = 3
    f._rlt_miss_count = np.zeros(1, np.int32); f._rlt_newly_missed = np.zeros(1, bool)
    armed2 = []; miss_t = []
    for t in range(len(S)):
        in2 = bool(f._rlt_armed[0] and f._rlt_holds_done[0] >= 1)
        LiberoEnv._update_rlt_hold(f, [{"robot0_gripper_qpos": S[t, 6:8]}], A[t:t + 1])
        armed2.append(in2)
        if f._rlt_newly_missed[0]: miss_t.append(t)
    gap = S[:, 6] - S[:, 7]; close = A[:, 6] > 0
    ref = [a for a, b in runs(close & (gap <= 0.03), 3)]            # all misses in the episode
    ref2 = [a + 2 for a in ref if armed2[min(a + 2, len(S) - 1)]]     # those counted in the second phase (3rd step of the run)
    n += 1; agree += (len(miss_t) == len(ref2)); tot_env.append(len(miss_t)); tot_all.append(len(ref))
print(f"{agree}/{n} episodes: counter == independent count | misses per episode: phase-2 counter mean {np.mean(tot_env):.2f}, all misses in episode (diag definition) mean {np.mean(tot_all):.2f}")
