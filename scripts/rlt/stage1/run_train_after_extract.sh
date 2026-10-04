#!/bin/bash
cd /workspace/libero_RL/tmp/rlt_stage1
until [ $(ls feats_rest/*.done 2>/dev/null | wc -l) -ge 14 ]; do sleep 20; done   # 11 train_v2 + 3 eval_v2
export HF_HUB_OFFLINE=1 STEPS=6000
../../RLinf/.venv-smolvla/bin/python train_stage1.py > train_stage1.log 2>&1
../../RLinf/.venv-smolvla/bin/python probe_stage1.py > probe_stage1.log 2>&1
