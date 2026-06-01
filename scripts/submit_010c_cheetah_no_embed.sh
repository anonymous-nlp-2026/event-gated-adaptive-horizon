#!/bin/bash
# exp_010c: Cheetah Run no dt_emb (dim=0) + subsample ON + dt=2 + H=8
# 对应 Walker Walk exp_008f 配置，验证因果控制跨环境泛化
# 不要直接执行此脚本——由 Agent 通过 submit_training_job 提交

CD="<PROJECT_ROOT>"
ENV_SETUP="source <PROJECT_ROOT> && conda activate base && export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 && export MUJOCO_GL=osmesa && export PYOPENGL_PLATFORM=osmesa && cd $CD"

# === Seed 1 ===
# tmux new-session -d -s exp_010c_s1 "eval \"$ENV_SETUP\" && python -u dreamer.py --configs defaults dmc_proprio --task dmc_cheetah_run --logdir logdir/exp_010c_cheetah_no_embed_s1 --steps 105000 --seed 1 --compile False --device cuda:0 --gate_enabled True --gate_fixed_dt 2 --gate_dt_embed_dim 0 --gate_subsample_max_dt 2 --imag_horizon 8 2>&1 | tee logdir/exp_010c_cheetah_no_embed_s1.log"

echo "exp_010c: Cheetah Run no dt_emb + subsample + dt=2 + H=8"
echo "Steps: 105000, Seed: 1"
echo "Logdir: logdir/exp_010c_cheetah_no_embed_s1"
