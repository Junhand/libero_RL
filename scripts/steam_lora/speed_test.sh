#!/usr/bin/env bash
# 30-step speed / memory test of each variant (the first steps include start-up; compare the median step time).
cd /workspace/libero_RL/tmp/steam_lora
run() { n=$1; shift; ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> mem_$n.txt; sleep 5; done ) & mp=$!
  ./run_variant.sh speed_$n 30 "$@" > speed_$n.log 2>&1; kill $mp; echo "[$(date +%T)] $n done: $(cat speed_$n.exit)  peak GPU MiB $(sort -n mem_$n.txt | tail -1)"; }
run freezevis actor.model.freeze_vision_encoder=true actor.model.freeze_language_model=false
run freezeboth actor.model.freeze_vision_encoder=true actor.model.freeze_language_model=true
run lora +actor.model.lora_enable=true +actor.model.lora_r=16 +actor.model.lora_alpha=32 +actor.model.lora_apply_to=vision,language actor.model.freeze_vision_encoder=false actor.model.freeze_language_model=false
run full actor.model.freeze_vision_encoder=false actor.model.freeze_language_model=false
echo chain-done
