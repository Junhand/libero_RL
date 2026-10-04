#!/bin/bash
# 4-epoch smoke run of the dedicated second-phase actor with the miss penalty (checks start-up, saving and the reward values).
#   PRE_ACTOR=<phase-1 full_weights.pt> ./smoke_miss.sh
cd /workspace/libero_RL/tmp/rlt_stage2_b
PRE_ACTOR=${PRE_ACTOR:?phase-1 full_weights.pt}
EPOCHS=4 ./run_smoke.sh runner.save_interval=4 env.train.total_num_envs=2 env.eval.total_num_envs=2 runner.logger.experiment_name=p2only-miss-smoke \
  env.train.rlt_policy_switch.mode=grasp_subtask env.train.rlt_policy_switch.num_phases=2 env.train.rlt_policy_switch.actor_phase=1 \
  env.train.rlt_policy_switch.pre_actor=True env.train.rlt_policy_switch.miss_penalty=0.2 env.train.max_episode_steps=400 \
  env.train.max_steps_per_rollout_epoch=400 algorithm.rlt_phase_return=False algorithm.rlt_segment_terminal=True actor.model.fixed_std=0.05 \
  rollout.rlt_pre_actor.ckpt_path=$PRE_ACTOR
