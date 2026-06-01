#!/usr/bin/env python3
"""Gradient evolution analysis across 018v2 checkpoints (with/without SGS)."""

import sys
import os
import json
import copy
import numpy as np
import torch
from types import SimpleNamespace
from pathlib import Path

PROJECT_DIR = "<PROJECT_ROOT>"
sys.path.insert(0, PROJECT_DIR)

import models
import networks
import tools
import yaml


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def convert_str_numbers(cfg):
    if isinstance(cfg, dict):
        return {k: convert_str_numbers(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [convert_str_numbers(v) for v in cfg]
    if isinstance(cfg, str):
        try:
            val = float(cfg)
            if val == int(val) and 'e' not in cfg.lower() and '.' not in cfg:
                return int(val)
            return val
        except ValueError:
            return cfg
    return cfg


def build_config(device="cpu"):
    with open(os.path.join(PROJECT_DIR, "configs.yaml")) as f:
        cfgs = yaml.safe_load(f)
    cfg = copy.deepcopy(cfgs["defaults"])
    cfg = deep_merge(cfg, cfgs["dmc_proprio"])
    cfg = convert_str_numbers(cfg)

    cfg["gate_enabled"] = True
    cfg["gate_max_dt"] = 8
    cfg["gate_subsample_max_dt"] = 8
    cfg["gate_subsample_min_dt"] = 1
    cfg["gate_fixed_dt"] = 0
    cfg["gate_type"] = "gumbel"
    cfg["dt_emb_freeze"] = False
    cfg["reward_dt_emb_detach"] = False
    cfg["full_sgs"] = False
    cfg["device"] = device
    cfg["precision"] = 32
    cfg["compile"] = False
    cfg["num_actions"] = 6
    return SimpleNamespace(**cfg)


def build_spaces():
    class Space:
        def __init__(self, shape):
            self.shape = shape
    obs_space = SimpleNamespace(spaces={
        "height": Space((1,)),
        "orientations": Space((14,)),
        "velocity": Space((9,)),
    })
    act_space = SimpleNamespace(shape=(6,))
    return obs_space, act_space


def load_batch(episode_dir, batch_size=16, batch_length=64, rng=None):
    eps_files = sorted(Path(episode_dir).glob("*.npz"))
    if rng is None:
        rng = np.random.RandomState(42)
    keys = ["height", "orientations", "velocity", "action", "reward",
            "is_first", "is_terminal", "discount"]
    batch = {k: [] for k in keys}
    indices = rng.choice(len(eps_files), size=batch_size, replace=True)
    for i in indices:
        ep = np.load(str(eps_files[i]))
        T = ep["action"].shape[0]
        start = rng.randint(0, max(1, T - batch_length))
        end = start + batch_length
        for key in keys:
            batch[key].append(ep[key][start:end])
    for key in keys:
        batch[key] = np.stack(batch[key], axis=0)
    batch["is_first"][:, 0] = True
    return batch


def compute_grads(checkpoint_path, episode_dir, device="cpu", n_batches=10):
    """Compute gradient norms for each loss component w.r.t. dt_embedding."""
    config = build_config(device=device)
    obs_space, act_space = build_spaces()

    wm = models.WorldModel(obs_space, act_space, step=0, config=config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    wm_state = {k[4:]: v for k, v in ckpt["agent_state_dict"].items()
                if k.startswith("_wm.")}
    wm.load_state_dict(wm_state, strict=True)
    wm.to(device)
    wm.eval()

    dt_param = wm.dynamics._dt_embedding.weight

    rng = np.random.RandomState(42)
    results = {"obs": [], "reward": [], "kl": [], "cont": []}

    for bi in range(n_batches):
        batch = load_batch(episode_dir, batch_size=16, batch_length=64, rng=rng)
        data = wm.preprocess(batch)
        data, dt_labels = tools.subsample_variable_dt(
            data, config.gate_max_dt, config.discount,
            effective_max_dt=float(config.gate_subsample_max_dt),
            min_dt=config.gate_subsample_min_dt,
        )

        for p in wm.parameters():
            p.requires_grad_(True)

        embed = wm.encoder(data)
        post, prior = wm.dynamics.observe(
            embed, data["action"], data["is_first"], dt_labels=dt_labels
        )

        kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
            post, prior, config.kl_free, config.dyn_scale, config.rep_scale
        )
        mean_kl = torch.mean(kl_loss)

        feat = wm.dynamics.get_feat(post)

        decoder_preds = wm.heads["decoder"](feat)
        obs_loss_sum = torch.zeros(1, device=device)
        for name, pred in decoder_preds.items():
            obs_loss_sum = obs_loss_sum + torch.mean(-pred.log_prob(data[name]))

        reward_pred = wm.heads["reward"](feat)
        mean_reward = torch.mean(-reward_pred.log_prob(data["reward"]))

        cont_pred = wm.heads["cont"](feat)
        mean_cont = torch.mean(-cont_pred.log_prob(data["cont"]))

        grad_obs = torch.autograd.grad(
            obs_loss_sum, dt_param, retain_graph=True, allow_unused=True)[0]
        grad_reward = torch.autograd.grad(
            mean_reward, dt_param, retain_graph=True, allow_unused=True)[0]
        grad_kl = torch.autograd.grad(
            mean_kl, dt_param, retain_graph=True, allow_unused=True)[0]
        grad_cont = torch.autograd.grad(
            mean_cont, dt_param, retain_graph=True, allow_unused=True)[0]

        for name, g in [("obs", grad_obs), ("reward", grad_reward),
                        ("kl", grad_kl), ("cont", grad_cont)]:
            results[name].append(g.norm().item() if g is not None else 0.0)

        for p in wm.parameters():
            p.requires_grad_(False)

    avg = {k: float(np.mean(v)) for k, v in results.items()}
    std = {k: float(np.std(v)) for k, v in results.items()}
    total = sum(avg.values())
    pcts = {k: avg[k] / total * 100 if total > 0 else 0 for k in avg}

    return {"avg_norms": avg, "std_norms": std, "percentages": pcts, "total_norm": total}


def main():
    device = "cpu"
    logdir = f"{PROJECT_DIR}/logdir/exp_018v2_full_sgs_walker_s1"
    episode_dir = f"{logdir}/train_eps"

    checkpoints = [
        ("checkpoint_25000.pt", 25000),
        ("checkpoint_75000.pt", 75000),
        ("latest.pt", 112000),
    ]

    output = {
        "exp_id": "exp_018v2_full_sgs_walker_s1",
        "checkpoints_found": [c[0] for c in checkpoints],
        "with_sgs": {
            "description": "Actual training gradient (Full-SGS active, obs+reward+cont pathway blocked from dt_emb)",
            "steps": []
        },
        "without_sgs": {
            "description": "Counterfactual gradient (SGS disabled, shows underlying gradient structure)",
            "steps": []
        },
        "comparison_with_008d": {
            "note": "008d at 55K: obs=25.8%, reward=23.5%, KL=50.7%. At 105K: obs=93%, reward=0%, KL=7%.",
            "finding": ""
        }
    }

    for ckpt_name, step in checkpoints:
        ckpt_path = f"{logdir}/{ckpt_name}"
        print(f"\n{'='*60}")
        print(f"Analyzing {ckpt_name} (step={step})")
        print(f"{'='*60}")

        print("  Computing counterfactual gradients (without SGS)...")
        result = compute_grads(ckpt_path, episode_dir, device=device, n_batches=10)

        without_sgs_entry = {
            "step": step,
            "obs_norm": result["avg_norms"]["obs"],
            "reward_norm": result["avg_norms"]["reward"],
            "kl_norm": result["avg_norms"]["kl"],
            "cont_norm": result["avg_norms"]["cont"],
            "obs_pct": round(result["percentages"]["obs"], 1),
            "reward_pct": round(result["percentages"]["reward"], 1),
            "kl_pct": round(result["percentages"]["kl"], 1),
            "cont_pct": round(result["percentages"]["cont"], 1),
        }
        output["without_sgs"]["steps"].append(without_sgs_entry)

        # With SGS: effective gradient on dt_emb = only KL component
        # (SGS subtracts obs+reward+cont gradients via hook)
        kl_norm = result["avg_norms"]["kl"]
        with_sgs_entry = {
            "step": step,
            "obs_norm": 0.0,
            "reward_norm": 0.0,
            "kl_norm": kl_norm,
            "cont_norm": 0.0,
            "obs_pct": 0.0,
            "reward_pct": 0.0,
            "kl_pct": 100.0,
            "cont_pct": 0.0,
        }
        output["with_sgs"]["steps"].append(with_sgs_entry)

        total = result["total_norm"]
        print(f"  Without SGS: obs={result['percentages']['obs']:.1f}% "
              f"reward={result['percentages']['reward']:.1f}% "
              f"kl={result['percentages']['kl']:.1f}% "
              f"cont={result['percentages']['cont']:.1f}%")
        print(f"  With SGS: kl=100% (all others blocked)")

    # Generate finding
    steps_data = output["without_sgs"]["steps"]
    if len(steps_data) >= 2:
        early = steps_data[0]
        late = steps_data[-1]
        finding = (
            f"018v2 counterfactual: obs goes from {early['obs_pct']}% (step {early['step']}) "
            f"to {late['obs_pct']}% (step {late['step']}). "
            f"KL goes from {early['kl_pct']}% to {late['kl_pct']}%. "
            f"With Full-SGS active, dt_emb only sees KL gradient (100%) throughout training."
        )
        output["comparison_with_008d"]["finding"] = finding

    out_path = f"{PROJECT_DIR}/artifacts/gradient_evolution_018v2.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
