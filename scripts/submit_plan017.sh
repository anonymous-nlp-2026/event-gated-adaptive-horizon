#!/bin/bash
# plan_017: subsample OFF + trained dt_emb + dt=2 imag + H=15
#
# Purpose: Test whether trained dt_embedding alone (without subsample dt>1
#          training data) causes CIVO. Answers D038/D040.
#
# Comparison: vs 008a (same config but untrained dt_emb / critic_dt_emb_detach=True)
#   - Only difference: critic_dt_emb_detach defaults to False here (trained)
#   - subsample OFF: gate_subsample_max_dt=1 (max_dt=1 → all dt=1, no subsampling)
#   - dt=2 imagination: gate_fixed_dt=2
#   - H=15: imag_horizon=15
#   - dt_embed_dim=32 (default, same as 008a)
#
# Expected outcomes:
#   stable  → training data dt>1 transitions are CIVO trigger (supports D040)
#   collapse → trained dt_embedding itself sufficient for CIVO

source <PROJECT_ROOT> && conda activate base && \
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 && \
export MUJOCO_GL=osmesa && export PYOPENGL_PLATFORM=osmesa && \
cd <PROJECT_ROOT> && \
nohup python -u dreamer.py --configs defaults dmc_proprio \
  --task dmc_walker_walk \
  --logdir logdir/exp_017_subsample_off_trained_dt_emb \
  --steps 105000 --seed 1 --compile False --device cuda:1 \
  --gate_enabled True --gate_fixed_dt 2 \
  --gate_dt_embed_dim 32 \
  --gate_subsample_max_dt 1 \
  --imag_horizon 15 \
  > logdir/exp_017_subsample_off_trained_dt_emb.log 2>&1 &

echo "PID=$!"
echo "Log: logdir/exp_017_subsample_off_trained_dt_emb.log"
