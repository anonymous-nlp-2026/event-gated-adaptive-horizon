#!/bin/bash
# exp_021_full_sgs_cheetah_s1: Full-SGS Cheetah Run
# Hypothesis: Full-SGS (blocking obs+reward+cont decoder gradients to dt_emb)
# prevents CIVO in Cheetah, validating cross-environment SGS effectiveness.
# Compare with exp_010b (Cheetah CIVO trigger, NVE 8.63×, collapse).

set -euo pipefail
CUDA_ID=${CUDA_ID:-0}

cd <PROJECT_ROOT>

unset CUDA_VISIBLE_DEVICES
source <PROJECT_ROOT>
conda activate base

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=${CUDA_ID}

python -u dreamer.py \
  --configs defaults dmc_proprio \
  --task dmc_cheetah_run \
  --gate_enabled True \
  --gate_fixed_dt 2 \
  --gate_subsample_max_dt 2 \
  --imag_horizon 8 \
  --full_sgs True \
  --seed 1 \
  --steps 105000 \
  --checkpoint_every 10000 \
  --compile False \
  --device cuda:${CUDA_ID} \
  --logdir ./logdir/exp_021_full_sgs_cheetah_s1
