#!/bin/bash
# Cheetah Run experiments for event-gated-adaptive-horizon
# Usage: source this file or copy individual commands
# Replace cuda:X with actual device (e.g., cuda:0)

CD="<PROJECT_ROOT>"
COMMON="source <PROJECT_ROOT> && conda activate base && export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 && export MUJOCO_GL=osmesa && export PYOPENGL_PLATFORM=osmesa && cd $CD"

# ============================================================
# Baseline: vanilla DreamerV3 on Cheetah Run (512K steps)
# ============================================================

# exp_010a_cheetah_baseline_s1
# eval "$COMMON" && python dreamer.py --configs defaults dmc_proprio --task dmc_cheetah_run --logdir logdir/exp_010a_cheetah_baseline_s1 --steps 512000 --seed 1 --compile False --device cuda:X

# exp_010a_cheetah_baseline_s2
# eval "$COMMON" && python dreamer.py --configs defaults dmc_proprio --task dmc_cheetah_run --logdir logdir/exp_010a_cheetah_baseline_s2 --steps 512000 --seed 2 --compile False --device cuda:X

# exp_010a_cheetah_baseline_s3
# eval "$COMMON" && python dreamer.py --configs defaults dmc_proprio --task dmc_cheetah_run --logdir logdir/exp_010a_cheetah_baseline_s3 --steps 512000 --seed 3 --compile False --device cuda:X

# ============================================================
# 008d-style: trained dt_emb, subsample ON, H=8 (105K steps)
# ============================================================

# exp_010b_cheetah_trained_dt2_s1
# eval "$COMMON" && python dreamer.py --configs defaults dmc_proprio --task dmc_cheetah_run --gate_enabled True --gate_fixed_dt 2 --gate_subsample_max_dt 2 --imag_horizon 8 --seed 1 --logdir logdir/exp_010b_cheetah_trained_dt2_s1 --steps 105000 --compile False --device cuda:X

# exp_010b_cheetah_trained_dt2_s2
# eval "$COMMON" && python dreamer.py --configs defaults dmc_proprio --task dmc_cheetah_run --gate_enabled True --gate_fixed_dt 2 --gate_subsample_max_dt 2 --imag_horizon 8 --seed 2 --logdir logdir/exp_010b_cheetah_trained_dt2_s2 --steps 105000 --compile False --device cuda:X

# exp_010b_cheetah_trained_dt2_s3
# eval "$COMMON" && python dreamer.py --configs defaults dmc_proprio --task dmc_cheetah_run --gate_enabled True --gate_fixed_dt 2 --gate_subsample_max_dt 2 --imag_horizon 8 --seed 3 --logdir logdir/exp_010b_cheetah_trained_dt2_s3 --steps 105000 --compile False --device cuda:X

# ============================================================
# 008f-style: dt_embed_dim=0, subsample ON, H=8 (105K steps)
# ============================================================

# exp_010c_cheetah_no_embed_dt2_s1
# eval "$COMMON" && python dreamer.py --configs defaults dmc_proprio --task dmc_cheetah_run --gate_enabled True --gate_fixed_dt 2 --gate_subsample_max_dt 2 --imag_horizon 8 --gate_dt_embed_dim 0 --seed 1 --logdir logdir/exp_010c_cheetah_no_embed_dt2_s1 --steps 105000 --compile False --device cuda:X

# exp_010c_cheetah_no_embed_dt2_s2
# eval "$COMMON" && python dreamer.py --configs defaults dmc_proprio --task dmc_cheetah_run --gate_enabled True --gate_fixed_dt 2 --gate_subsample_max_dt 2 --imag_horizon 8 --gate_dt_embed_dim 0 --seed 2 --logdir logdir/exp_010c_cheetah_no_embed_dt2_s2 --steps 105000 --compile False --device cuda:X

# exp_010c_cheetah_no_embed_dt2_s3
# eval "$COMMON" && python dreamer.py --configs defaults dmc_proprio --task dmc_cheetah_run --gate_enabled True --gate_fixed_dt 2 --gate_subsample_max_dt 2 --imag_horizon 8 --gate_dt_embed_dim 0 --seed 3 --logdir logdir/exp_010c_cheetah_no_embed_dt2_s3 --steps 105000 --compile False --device cuda:X
