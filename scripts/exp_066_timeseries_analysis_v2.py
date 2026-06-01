"""
exp_066: Timeseries RSSM Fidelity Analysis v2

Three fidelity metrics:
1. Single-step reward MAE: dt=1 vs dt=2 (each vs matched ground truth)
2. Open-loop multi-step: H-step imagination vs actual trajectory
   - dt=1 mode: H img_steps with dt=1, covering H physical steps
   - dt=2 mode: H img_steps with dt=2, covering 2H physical steps
   Both compared against the same physical time span (2H steps).
   dt=1 does 2H img_steps; dt=2 does H img_steps.
3. State divergence: how quickly imagined states diverge from posterior states
"""
import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import networks
import models

LOGDIR = Path("<PROJECT_ROOT>/logdir/exp_066_timeseries_rssm_fidelity_s0")
CHECKPOINT_STEPS = [10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000]
IMAG_HORIZON = 8


def make_config(device="cuda:0"):
    config = SimpleNamespace(
        dyn_stoch=32, dyn_deter=512, dyn_hidden=512,
        dyn_discrete=32, dyn_rec_depth=1,
        dyn_mean_act="none", dyn_std_act="sigmoid2", dyn_min_std=0.1,
        units=512, act="SiLU", norm=True,
        unimix_ratio=0.01, initial="learned",
        num_actions=6, device=device, precision=32,
        encoder={"mlp_keys": ".*", "cnn_keys": "$^", "act": "SiLU", "norm": True,
                 "cnn_depth": 32, "kernel_size": 4, "minres": 4,
                 "mlp_layers": 5, "mlp_units": 1024, "symlog_inputs": True},
        decoder={"mlp_keys": ".*", "cnn_keys": "$^", "act": "SiLU", "norm": True,
                 "cnn_depth": 32, "kernel_size": 4, "minres": 4,
                 "mlp_layers": 5, "mlp_units": 1024,
                 "cnn_sigmoid": False, "image_dist": "mse",
                 "vector_dist": "symlog_mse", "outscale": 1.0},
        reward_head={"layers": 2, "dist": "symlog_disc", "loss_scale": 1.0, "outscale": 0.0},
        cont_head={"layers": 2, "loss_scale": 1.0, "outscale": 1.0},
        grad_heads=["decoder", "reward", "cont"],
        gate_enabled=True, gate_fixed_dt=2, gate_subsample_max_dt=2,
        gate_max_dt=8, gate_dt_embed_dim=32,
        gate_hidden_units=256, gate_hidden_layers=2,
        gate_tau_init=5.0, gate_tau_final=0.5, gate_tau_anneal_steps=200000,
        gate_loss_weight=1.0, gate_entropy_bonus_init=0.1,
        gate_entropy_bonus_final=0.01,
        gate_curriculum_phase1=50000, gate_curriculum_phase2=150000,
        gate_type="gumbel", gate_subsample_min_dt=1,
        dt_emb_freeze=False, reward_dt_emb_detach=False,
        model_lr=1e-4, opt_eps=1e-8, grad_clip=1000, weight_decay=0.0, opt="adam",
        discount=0.997, imag_horizon=8,
        kl_free=1.0, dyn_scale=0.5, rep_scale=0.1,
        reward_EMA=True,
    )
    return config


def make_obs_space():
    from gymnasium import spaces
    return spaces.Dict({
        "height": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
        "orientations": spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32),
        "velocity": spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
    })


def make_act_space():
    from gymnasium import spaces
    return spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)


def load_episodes(logdir, num_episodes=10, use_latest=True):
    """Load eval episodes. If use_latest, take the LAST num_episodes files."""
    eps_dir = logdir / "eval_eps"
    files = sorted(eps_dir.glob("*.npz"))
    if use_latest:
        files = files[-num_episodes:]
    else:
        files = files[:num_episodes]
    episodes = []
    for f in files:
        ep = dict(np.load(f))
        episodes.append(ep)
    return episodes


def load_world_model(checkpoint_path, config, obs_space, act_space, device):
    wm = models.WorldModel(obs_space, act_space, 0, config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent_sd = checkpoint["agent_state_dict"]
    wm_sd = {}
    for k, v in agent_sd.items():
        if k.startswith("_wm."):
            k_clean = k[4:].replace("_orig_mod.", "")
            wm_sd[k_clean] = v
    wm.load_state_dict(wm_sd, strict=False)
    wm.eval()
    return wm


@torch.no_grad()
def analyze_checkpoint(wm, episodes, device, H=IMAG_HORIZON):
    """
    Metrics:
    1. single_step: dt=1 pred vs actual[t+1], dt=2 pred vs actual[t+2]
    2. open_loop_reward_mae: multi-step imagination, compare cumulative rewards
       - dt=1 mode: 2H img_steps with dt=1, predict reward at each step
       - dt=2 mode: H img_steps with dt=2, predict reward at each step (covers 2 phys steps each)
       Both cover 2H physical steps. Compare total predicted reward vs actual total.
    3. open_loop_state_div: cosine distance between imagined feat and posterior feat after H steps
    """
    ss_mae_dt1 = []
    ss_mae_dt2 = []
    ol_mae_dt1 = []
    ol_mae_dt2 = []
    ol_cumrew_err_dt1 = []
    ol_cumrew_err_dt2 = []
    state_div_dt1 = []
    state_div_dt2 = []

    for ep in episodes:
        T = len(ep["reward"])
        need_len = 2 * H + 20
        if T < need_len:
            continue

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
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        
        embed = wm.encoder(obs)
        actions = obs["action"]
        is_first = obs["is_first"]
        post, prior = wm.dynamics.observe(embed, actions, is_first)
        
        rewards = ep["reward"][:T]
        
        # Analyze from multiple starting points
        for t in range(10, min(T - 2 * H - 2, 200), 5):
            state_t = {k: v[:, t] for k, v in post.items()}
            action_t = actions[:, t]
            
            # === Single-step ===
            dt1 = torch.ones(1, dtype=torch.long, device=device)
            dt2 = torch.full((1,), 2, dtype=torch.long, device=device)
            
            prior_dt1 = wm.dynamics.img_step(state_t, action_t, dt=dt1, sample=False)
            feat_dt1 = wm.dynamics.get_feat(prior_dt1)
            pred_r_dt1 = wm.heads["reward"](feat_dt1).mode().item()
            ss_mae_dt1.append(abs(pred_r_dt1 - rewards[t + 1]))
            
            prior_dt2 = wm.dynamics.img_step(state_t, action_t, dt=dt2, sample=False)
            feat_dt2 = wm.dynamics.get_feat(prior_dt2)
            pred_r_dt2 = wm.heads["reward"](feat_dt2).mode().item()
            ss_mae_dt2.append(abs(pred_r_dt2 - rewards[t + 2]))
            
            # === Open-loop multi-step: dt=1 (2H steps covering 2H physical steps) ===
            state_imag = {k: v.clone() for k, v in state_t.items()}
            pred_rewards_dt1 = []
            for h in range(2 * H):
                if t + h >= T:
                    break
                act_h = actions[:, t + h]
                state_imag = wm.dynamics.img_step(state_imag, act_h, dt=dt1, sample=False)
                feat_h = wm.dynamics.get_feat(state_imag)
                pred_r = wm.heads["reward"](feat_h).mode().item()
                pred_rewards_dt1.append(pred_r)
            
            actual_rewards_dt1 = rewards[t + 1: t + 1 + len(pred_rewards_dt1)]
            if len(pred_rewards_dt1) > 0:
                step_errors = [abs(p - a) for p, a in zip(pred_rewards_dt1, actual_rewards_dt1)]
                ol_mae_dt1.append(np.mean(step_errors))
                ol_cumrew_err_dt1.append(abs(sum(pred_rewards_dt1) - sum(actual_rewards_dt1)))
                
                # State divergence at the end
                post_feat = wm.dynamics.get_feat({k: v[:, t + len(pred_rewards_dt1)] for k, v in post.items()})
                imag_feat = wm.dynamics.get_feat(state_imag)
                cos_sim = torch.nn.functional.cosine_similarity(post_feat, imag_feat, dim=-1).item()
                state_div_dt1.append(1.0 - cos_sim)
            
            # === Open-loop multi-step: dt=2 (H steps covering 2H physical steps) ===
            state_imag = {k: v.clone() for k, v in state_t.items()}
            pred_rewards_dt2 = []
            for h in range(H):
                phys_step = t + 2 * h
                if phys_step >= T:
                    break
                act_h = actions[:, phys_step]
                state_imag = wm.dynamics.img_step(state_imag, act_h, dt=dt2, sample=False)
                feat_h = wm.dynamics.get_feat(state_imag)
                pred_r = wm.heads["reward"](feat_h).mode().item()
                pred_rewards_dt2.append(pred_r)
            
            # dt=2 predictions correspond to rewards at t+2, t+4, ..., t+2H
            actual_rewards_dt2_targets = [rewards[t + 2 * (h + 1)] for h in range(len(pred_rewards_dt2)) 
                                           if t + 2 * (h + 1) < T]
            actual_rewards_dt2_targets = actual_rewards_dt2_targets[:len(pred_rewards_dt2)]
            
            if len(pred_rewards_dt2) > 0 and len(actual_rewards_dt2_targets) == len(pred_rewards_dt2):
                step_errors = [abs(p - a) for p, a in zip(pred_rewards_dt2, actual_rewards_dt2_targets)]
                ol_mae_dt2.append(np.mean(step_errors))
                # Cumulative: sum over 2H physical steps
                # dt=2 predicts at even steps; fill in by doubling each (approximate)
                cum_pred_dt2 = sum(pred_rewards_dt2) * 2  # each covers 2 physical steps
                cum_actual = sum(rewards[t + 1: t + 1 + 2 * H])
                ol_cumrew_err_dt2.append(abs(cum_pred_dt2 - cum_actual))
                
                # State divergence
                end_phys = t + 2 * len(pred_rewards_dt2)
                if end_phys < T:
                    post_feat = wm.dynamics.get_feat({k: v[:, end_phys] for k, v in post.items()})
                    imag_feat = wm.dynamics.get_feat(state_imag)
                    cos_sim = torch.nn.functional.cosine_similarity(post_feat, imag_feat, dim=-1).item()
                    state_div_dt2.append(1.0 - cos_sim)

    return {
        "ss_mae_dt1": float(np.mean(ss_mae_dt1)) if ss_mae_dt1 else 0.0,
        "ss_mae_dt2": float(np.mean(ss_mae_dt2)) if ss_mae_dt2 else 0.0,
        "ol_mae_dt1": float(np.mean(ol_mae_dt1)) if ol_mae_dt1 else 0.0,
        "ol_mae_dt2": float(np.mean(ol_mae_dt2)) if ol_mae_dt2 else 0.0,
        "ol_cumrew_err_dt1": float(np.mean(ol_cumrew_err_dt1)) if ol_cumrew_err_dt1 else 0.0,
        "ol_cumrew_err_dt2": float(np.mean(ol_cumrew_err_dt2)) if ol_cumrew_err_dt2 else 0.0,
        "state_div_dt1": float(np.mean(state_div_dt1)) if state_div_dt1 else 0.0,
        "state_div_dt2": float(np.mean(state_div_dt2)) if state_div_dt2 else 0.0,
        "n_samples": len(ss_mae_dt1),
    }


def extract_ve_ratio_from_metrics(logdir):
    metrics_file = logdir / "metrics.jsonl"
    ve_data = {}
    value_data = {}
    target_data = {}
    train_return_data = {}
    with open(metrics_file) as f:
        for line in f:
            entry = json.loads(line)
            step = entry.get("step", None)
            if step is None:
                continue
            if "civo/ve_ratio" in entry:
                ve_data[step] = entry["civo/ve_ratio"]
            if "value_mean" in entry:
                value_data[step] = entry["value_mean"]
            if "target_mean" in entry:
                target_data[step] = entry["target_mean"]
            if "train_return" in entry:
                train_return_data[step] = entry["train_return"]
    return ve_data, value_data, target_data, train_return_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=IMAG_HORIZON)
    args = parser.parse_args()
    
    device = args.device
    config = make_config(device)
    obs_space = make_obs_space()
    act_space = make_act_space()
    
    # Use LATEST episodes (from final eval, richest reward signal)
    print(f"Loading latest {args.num_episodes} episodes...")
    episodes = load_episodes(LOGDIR, args.num_episodes, use_latest=True)
    print(f"Loaded {len(episodes)} episodes, mean reward: {np.mean([ep['reward'].sum() for ep in episodes]):.1f}")
    
    ve_data, value_data, target_data, train_return_data = extract_ve_ratio_from_metrics(LOGDIR)
    
    results = {
        "steps": [],
        "ss_mae_dt1": [], "ss_mae_dt2": [], "ss_gap": [], "ss_ratio": [],
        "ol_mae_dt1": [], "ol_mae_dt2": [], "ol_gap": [], "ol_ratio": [],
        "ol_cumrew_err_dt1": [], "ol_cumrew_err_dt2": [],
        "state_div_dt1": [], "state_div_dt2": [],
        "ve_ratio": [], "value_mean": [], "target_mean": [], "train_return": [],
    }
    
    for step in CHECKPOINT_STEPS:
        ckpt_path = LOGDIR / f"checkpoint_{step}.pt"
        if not ckpt_path.exists():
            print(f"  Checkpoint {ckpt_path} not found, skipping")
            continue
        
        print(f"\n--- Step {step} ---")
        wm = load_world_model(ckpt_path, config, obs_space, act_space, device)
        metrics = analyze_checkpoint(wm, episodes, device, args.horizon)
        
        ss_gap = metrics["ss_mae_dt2"] - metrics["ss_mae_dt1"]
        ss_ratio = metrics["ss_mae_dt2"] / metrics["ss_mae_dt1"] if metrics["ss_mae_dt1"] > 1e-8 else float("inf")
        ol_gap = metrics["ol_mae_dt2"] - metrics["ol_mae_dt1"]
        ol_ratio = metrics["ol_mae_dt2"] / metrics["ol_mae_dt1"] if metrics["ol_mae_dt1"] > 1e-8 else float("inf")
        
        closest = lambda data, s: data.get(min(data.keys(), key=lambda k: abs(k - s)), None) if data else None
        ve = closest(ve_data, step)
        vm = closest(value_data, step)
        tm = closest(target_data, step)
        tr = closest(train_return_data, step)
        
        results["steps"].append(step)
        results["ss_mae_dt1"].append(round(metrics["ss_mae_dt1"], 6))
        results["ss_mae_dt2"].append(round(metrics["ss_mae_dt2"], 6))
        results["ss_gap"].append(round(ss_gap, 6))
        results["ss_ratio"].append(round(ss_ratio, 4))
        results["ol_mae_dt1"].append(round(metrics["ol_mae_dt1"], 6))
        results["ol_mae_dt2"].append(round(metrics["ol_mae_dt2"], 6))
        results["ol_gap"].append(round(ol_gap, 6))
        results["ol_ratio"].append(round(ol_ratio, 4))
        results["ol_cumrew_err_dt1"].append(round(metrics["ol_cumrew_err_dt1"], 4))
        results["ol_cumrew_err_dt2"].append(round(metrics["ol_cumrew_err_dt2"], 4))
        results["state_div_dt1"].append(round(metrics["state_div_dt1"], 6))
        results["state_div_dt2"].append(round(metrics["state_div_dt2"], 6))
        results["ve_ratio"].append(round(ve, 6) if ve else None)
        results["value_mean"].append(round(vm, 2) if vm else None)
        results["target_mean"].append(round(tm, 2) if tm else None)
        results["train_return"].append(round(tr, 2) if tr else None)
        
        print(f"  Single-step: dt1={metrics['ss_mae_dt1']:.5f} dt2={metrics['ss_mae_dt2']:.5f} ratio={ss_ratio:.3f}x")
        print(f"  Open-loop:   dt1={metrics['ol_mae_dt1']:.5f} dt2={metrics['ol_mae_dt2']:.5f} ratio={ol_ratio:.3f}x")
        print(f"  CumRew err:  dt1={metrics['ol_cumrew_err_dt1']:.4f} dt2={metrics['ol_cumrew_err_dt2']:.4f}")
        print(f"  State div:   dt1={metrics['state_div_dt1']:.5f} dt2={metrics['state_div_dt2']:.5f}")
        if ve: print(f"  VE ratio:    {ve:.4f}")
        
        del wm
        torch.cuda.empty_cache()
    
    output_path = LOGDIR / "timeseries_analysis_v2.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Summary table
    print("\n" + "="*120)
    print("SUMMARY")
    print("="*120)
    hdr = f"{'Step':>6} | {'SS dt1':>8} {'SS dt2':>8} {'SS ratio':>9} | {'OL dt1':>8} {'OL dt2':>8} {'OL ratio':>9} | {'CumErr1':>8} {'CumErr2':>8} | {'VE ratio':>9} | {'Return':>7}"
    print(hdr)
    print("-" * 120)
    for i in range(len(results["steps"])):
        s = results["steps"][i]
        ve_str = f"{results['ve_ratio'][i]:.4f}" if results['ve_ratio'][i] else "N/A"
        tr_str = f"{results['train_return'][i]:.1f}" if results['train_return'][i] else "N/A"
        print(f"{s:>6} | {results['ss_mae_dt1'][i]:>8.5f} {results['ss_mae_dt2'][i]:>8.5f} {results['ss_ratio'][i]:>8.3f}x | "
              f"{results['ol_mae_dt1'][i]:>8.5f} {results['ol_mae_dt2'][i]:>8.5f} {results['ol_ratio'][i]:>8.3f}x | "
              f"{results['ol_cumrew_err_dt1'][i]:>8.4f} {results['ol_cumrew_err_dt2'][i]:>8.4f} | "
              f"{ve_str:>9} | {tr_str:>7}")


if __name__ == "__main__":
    main()
