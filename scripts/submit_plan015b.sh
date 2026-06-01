#!/bin/bash
# plan_015B: frozen dt_emb + subsample ON + dt=2 imagination + H=8
# 2x2 ablation: isolate training vs existence of dt_emb
# cuda:1, after 009e completes

source <PROJECT_ROOT> && conda activate base
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd <PROJECT_ROOT>

python -u dreamer.py --configs defaults dmc_proprio \
  --task dmc_walker_walk \
  --logdir logdir/exp_015b_frozen_dt_emb \
  --steps 105000 --seed 1 --compile False --device cuda:1 \
  --gate_enabled True --gate_fixed_dt 2 \
  --gate_subsample_max_dt 2 \
  --imag_horizon 8 \
  --dt_emb_freeze True
