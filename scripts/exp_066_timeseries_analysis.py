"""
exp_066: Timeseries RSSM Fidelity Analysis
Analyze dt=1 vs dt=2 reward prediction MAE across 12 dense checkpoints.
"""
import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import networks
import models

LOGDIR = Path("<PROJECT_ROOT>/logdir/exp_066_timeseries_rssm_fidelity_s0")
CHECKPOINT_STEPS = [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000]
NUM_EPISODES = 10
SEQ_LEN = 50  # how many steps per episode to analyze


def make_config(device="cuda:0"):
    """Construct minimal config matching exp_066 settings."""
    config = SimpleNamespace(
        # Model
        dyn_stoch=32, dyn_deter=512, dyn_hidden=512,
        dyn_discrete=32, dyn_rec_depth=1,
        dyn_mean_act="none", dyn_std_act="sigmoid2", dyn_min_std=0.1,
        units=512, act="SiLU", norm=True,
        unimix_ratio=0.01, initial="learned",
        num_actions=6, device=device, precision=32,
        # Encoder/Decoder (MLP-only for dmc_proprio)
        encoder={"mlp_keys": ".*", "cnn_keys": "$^", "act": "SiLU", "norm": True,
                 "cnn_depth": 32, "kernel_size": 4, "minres": 4,
                 "mlp_layers": 5, "mlp_units": 1024, "symlog_inputs": True},
        decoder={"mlp_keys": ".*", "cnn_keys": "$^", "act": "SiLU", "norm": True,
                 "cnn_depth": 32, "kernel_size": 4, "minres": 4,
                 "mlp_layers": 5, "mlp_units": 1024,
                 "cnn_sigmoid": False, "image_dist": "mse",
                 "vector_dist": "symlog_mse", "outscale": 1.0},
        # Heads
        reward_head={"layers": 2, "dist": "symlog_disc", "loss_scale": 1.0, "outscale": 0.0},
        cont_head={"layers": 2, "loss_scale": 1.0, "outscale": 1.0},
        grad_heads=["decoder", "reward", "cont"],
        # Gate config (exp_066: gate_enabled=True, gate_fixed_dt=2, gate_subsample_max_dt=2)
        gate_enabled=True, gate_fixed_dt=2, gate_subsample_max_dt=2,
        gate_max_dt=8, gate_dt_embed_dim=32,
        gate_hidden_units=256, gate_hidden_layers=2,
        gate_tau_init=5.0, gate_tau_final=0.5, gate_tau_anneal_steps=200000,
        gate_loss_weight=1.0, gate_entropy_bonus_init=0.1,
        gate_entropy_bonus_final=0.01,
        gate_curriculum_phase1=50000, gate_curriculum_phase2=150000,
        gate_type="gumbel", gate_subsample_min_dt=1,
        dt_emb_freeze=False, reward_dt_emb_detach=False,
        # Training
        model_lr=1e-4, opt_eps=1e-8, grad_clip=1000, weight_decay=0.0, opt="adam",
        # Behavior
        discount=0.997, imag_horizon=8,
        kl_free=1.0, dyn_scale=0.5, rep_scale=0.1,
        reward_EMA=True,
    )
    return config


def make_obs_space():
    """Construct obs_space matching Walker Walk proprio."""
    from gymnasium import spaces
    return spaces.Dict({
        "height": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        "orientations": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32),
        "velocity": spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
    })


def make_act_space():
    from gymnasium import spaces
    return spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)


def load_episodes(logdir, num_episodes=10):
    """Load eval episodes from .npz files."""
    eps_dir = logdir / "eval_eps"
    files = sorted(eps_dir.glob("*.npz"))
    episodes = []
    for f in files[:num_episodes]:
        ep = dict(np.load(f))
        episodes.append(ep)
    return episodes


def load_world_model(checkpoint_path, config, obs_space, act_space, device):
    """Load WorldModel from a Dreamer checkpoint."""
    wm = models.WorldModel(obs_space, act_space, 0, config).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent_sd = checkpoint["agent_state_dict"]
    
    # Extract _wm.* keys
    wm_sd = {}
    prefix = "_wm."
    for k, v in agent_sd.items():
        if k.startswith(prefix):
            wm_sd[k[len(prefix):]] = v
    
    # Handle compiled model keys (remove _orig_mod. prefix if present)
    wm_sd_clean = {}
    for k, v in wm_sd.items():
        k_clean = k.replace("_orig_mod.", "")
        wm_sd_clean[k_clean] = v
    
    wm.load_state_dict(wm_sd_clean, strict=False)
    wm.eval()
    return wm


@torch.no_grad()
def analyze_checkpoint(wm, episodes, device, seq_len=50):
    """
    For a given world model checkpoint, compute reward prediction MAE
    at dt=1 and dt=2.
    
    Method:
    - Encode episode observations and run RSSM observe to get posterior states
    - From each posterior state at time t:
      - img_step with dt=1 → predict reward → compare with actual reward[t+1]
      - img_step with dt=2 → predict reward → compare with actual reward[t+2] (sum of t+1 and t+2)
    """
    mae_dt1_all = []
    mae_dt2_all = []
    
    for ep in episodes:
        T = min(len(ep["reward"]), seq_len + 10)
        
        # Prepare obs dict
        obs = {
            "height": torch.tensor(ep["height"][:T], device=device, dtype=torch.float32).unsqueeze(0),
            "orientations": torch.tensor(ep["orientations"][:T], device=device, dtype=torch.float32).unsqueeze(0),
            "velocity": torch.tensor(ep["velocity"][:T], device=device, dtype=torch.float32).unsqueeze(0),
            "is_first": torch.tensor(ep["is_first"][:T], device=device, dtype=torch.float32).unsqueeze(0),
            "is_terminal": torch.tensor(ep["is_terminal"][:T], device=device, dtype=torch.float32).unsqueeze(0),
            "reward": torch.tensor(ep["reward"][:T], device=device, dtype=torch.float32).unsqueeze(0),
            "action": torch.tensor(ep["action"][:T], device=device, dtype=torch.float32).unsqueeze(0),
            "discount": torch.tensor(ep["discount"][:T], device=device, dtype=torch.float32).unsqueeze(0),
        }
        
        # Preprocess
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        
        # Encode
        embed = wm.encoder(obs)  # (1, T, embed_dim)
        
        # Get actions
        actions = obs["action"]  # (1, T, 6)
        is_first = obs["is_first"]  # (1, T)
        
        # Run RSSM observe to get posterior states
        post, prior = wm.dynamics.observe(embed, actions, is_first)
        # post: dict of (1, T, ...) tensors
        
        # Get actual rewards
        rewards = ep["reward"][:T]
        
        # For each timestep t (from 5 to T-3), predict reward at t+1 (dt=1) and t+2 (dt=2)
        errors_dt1 = []
        errors_dt2 = []
        
        start_t = 5  # skip initial steps
        end_t = min(T - 3, seq_len)
        
        for t in range(start_t, end_t):
            # Get posterior state at time t
            state_t = {k: v[:, t] for k, v in post.items()}  # (1, ...)
            action_t = actions[:, t]  # (1, 6)
            
            # dt=1 prediction: one step forward
            dt1 = torch.ones(1, dtype=torch.long, device=device)
            prior_dt1 = wm.dynamics.img_step(state_t, action_t, dt=dt1, sample=False)
            feat_dt1 = wm.dynamics.get_feat(prior_dt1)
            pred_reward_dt1 = wm.heads["reward"](feat_dt1).mode().item()
            actual_reward_dt1 = rewards[t + 1]
            errors_dt1.append(abs(pred_reward_dt1 - actual_reward_dt1))
            
            # dt=2 prediction: one step forward with dt=2 embedding
            dt2 = torch.full((1,), 2, dtype=torch.long, device=device)
            prior_dt2 = wm.dynamics.img_step(state_t, action_t, dt=dt2, sample=False)
            feat_dt2 = wm.dynamics.get_feat(prior_dt2)
            pred_reward_dt2 = wm.heads["reward"](feat_dt2).mode().item()
            # For dt=2, the model should predict the reward at t+2
            actual_reward_dt2 = rewards[t + 2]
            errors_dt2.append(abs(pred_reward_dt2 - actual_reward_dt2))
        
        if errors_dt1:
            mae_dt1_all.append(np.mean(errors_dt1))
            mae_dt2_all.append(np.mean(errors_dt2))
    
    mae_dt1 = float(np.mean(mae_dt1_all)) if mae_dt1_all else 0.0
    mae_dt2 = float(np.mean(mae_dt2_all)) if mae_dt2_all else 0.0
    return mae_dt1, mae_dt2


def extract_ve_ratio_from_metrics(logdir):
    """Extract civo/ve_ratio timeseries from metrics.jsonl."""
    metrics_file = logdir / "metrics.jsonl"
    ve_data = {}
    with open(metrics_file) as f:
        for line in f:
            entry = json.loads(line)
            if "civo/ve_ratio" in entry and "step" in entry:
                ve_data[entry["step"]] = entry["civo/ve_ratio"]
    return ve_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    args = parser.parse_args()
    
    device = args.device
    config = make_config(device)
    obs_space = make_obs_space()
    act_space = make_act_space()
    
    print(f"Loading episodes from {LOGDIR / 'eval_eps'}...")
    episodes = load_episodes(LOGDIR, args.num_episodes)
    print(f"Loaded {len(episodes)} episodes")
    
    # Extract ve_ratio from metrics
    print("Extracting ve_ratio from metrics.jsonl...")
    ve_data = extract_ve_ratio_from_metrics(LOGDIR)
    
    results = {
        "steps": [],
        "reward_mae_dt1": [],
        "reward_mae_dt2": [],
        "reward_mae_gap": [],
        "reward_mae_ratio": [],
        "ve_ratio": [],
    }
    
    for step in CHECKPOINT_STEPS:
        ckpt_path = LOGDIR / f"checkpoint_{step}.pt"
        if not ckpt_path.exists():
            print(f"  Checkpoint {ckpt_path} not found, skipping")
            continue
        
        print(f"\n--- Analyzing checkpoint at step {step} ---")
        wm = load_world_model(ckpt_path, config, obs_space, act_space, device)
        
        mae_dt1, mae_dt2 = analyze_checkpoint(wm, episodes, device, args.seq_len)
        gap = mae_dt2 - mae_dt1
        ratio = mae_dt2 / mae_dt1 if mae_dt1 > 1e-8 else float("inf")
        
        # Find closest ve_ratio
        closest_ve_step = min(ve_data.keys(), key=lambda s: abs(s - step)) if ve_data else None
        ve_ratio = ve_data.get(closest_ve_step, None) if closest_ve_step and abs(closest_ve_step - step) <= 5000 else None
        
        results["steps"].append(step)
        results["reward_mae_dt1"].append(round(mae_dt1, 6))
        results["reward_mae_dt2"].append(round(mae_dt2, 6))
        results["reward_mae_gap"].append(round(gap, 6))
        results["reward_mae_ratio"].append(round(ratio, 4))
        results["ve_ratio"].append(round(ve_ratio, 6) if ve_ratio is not None else None)
        
        print(f"  MAE dt=1: {mae_dt1:.6f}")
        print(f"  MAE dt=2: {mae_dt2:.6f}")
        print(f"  Gap (dt2-dt1): {gap:.6f}")
        print(f"  Ratio (dt2/dt1): {ratio:.4f}x")
        if ve_ratio is not None:
            print(f"  VE ratio: {ve_ratio:.6f}")
        
        # Free memory
        del wm
        torch.cuda.empty_cache()
    
    # Save results
    output_path = LOGDIR / "timeseries_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Step':>8} | {'MAE dt1':>10} | {'MAE dt2':>10} | {'Gap':>10} | {'Ratio':>8} | {'VE ratio':>10}")
    print("-"*70)
    for i, step in enumerate(results["steps"]):
        ve_str = f"{results['ve_ratio'][i]:.4f}" if results['ve_ratio'][i] is not None else "N/A"
        print(f"{step:>8} | {results['reward_mae_dt1'][i]:>10.6f} | {results['reward_mae_dt2'][i]:>10.6f} | {results['reward_mae_gap'][i]:>10.6f} | {results['reward_mae_ratio'][i]:>7.3f}x | {ve_str:>10}")


if __name__ == "__main__":
    main()
