#!/usr/bin/env python3
"""Independent gradient decomposition verification.
Uses torch.autograd.grad() for clean per-loss gradient computation.
Correct 008d config: gate_fixed_dt=2, gate_subsample_max_dt=2.
"""

import sys, os, json, copy
import numpy as np
import torch
from types import SimpleNamespace
from pathlib import Path

PROJECT_DIR = "<PROJECT_ROOT>"
sys.path.insert(0, PROJECT_DIR)

import models, networks, tools, yaml

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

def build_config(device="cpu", gate_fixed_dt=2, gate_subsample_max_dt=2):
    with open(os.path.join(PROJECT_DIR, "configs.yaml")) as f:
        cfgs = yaml.safe_load(f)
    cfg = copy.deepcopy(cfgs["defaults"])
    cfg = deep_merge(cfg, cfgs["dmc_proprio"])
    cfg = convert_str_numbers(cfg)
    cfg["gate_enabled"] = True
    cfg["gate_max_dt"] = 8
    cfg["gate_subsample_max_dt"] = gate_subsample_max_dt
    cfg["gate_subsample_min_dt"] = 1
    cfg["gate_fixed_dt"] = gate_fixed_dt
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

def verify(checkpoint_path, episode_dir, device="cpu", n_batches=20,
           gate_fixed_dt=2, gate_subsample_max_dt=2):
    config = build_config(device=device, gate_fixed_dt=gate_fixed_dt,
                          gate_subsample_max_dt=gate_subsample_max_dt)
    obs_space, act_space = build_spaces()
    wm = models.WorldModel(obs_space, act_space, step=0, config=config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    wm_state = {k[4:]: v for k, v in ckpt["agent_state_dict"].items()
                if k.startswith("_wm.")}
    wm.load_state_dict(wm_state, strict=True)
    wm.to(device)
    wm.eval()

    dt_param = wm.dynamics._dt_embedding.weight
    print(f"dt_param shape: {dt_param.shape}")
    print(f"dt_param name in model: dynamics._dt_embedding.weight")
    print(f"dt_param values (first 3 rows):\n{dt_param.data[:3]}")

    kl_free = getattr(config, 'kl_free', 1.0)
    dyn_scale = getattr(config, 'dyn_scale', 0.5)
    rep_scale = getattr(config, 'rep_scale', 0.1)

    bst_max_dt = 2 if config.gate_type == 'bernoulli_st' else config.gate_max_dt
    subsample_eff_dt = float(config.gate_subsample_max_dt)

    print(f"\nConfig: gate_fixed_dt={gate_fixed_dt}, gate_subsample_max_dt={gate_subsample_max_dt}")
    print(f"bst_max_dt={bst_max_dt}, subsample_eff_dt={subsample_eff_dt}")
    print(f"kl_free={kl_free}, dyn_scale={dyn_scale}, rep_scale={rep_scale}")

    rng = np.random.RandomState(42)
    all_norms = {"obs": [], "reward": [], "kl": [], "cont": []}

    for bi in range(n_batches):
        batch = load_batch(episode_dir, batch_size=16, batch_length=64, rng=rng)
        data = wm.preprocess(batch)
        data, dt_labels = tools.subsample_variable_dt(
            data, bst_max_dt, config.discount,
            effective_max_dt=subsample_eff_dt,
            min_dt=1)

        for p in wm.parameters():
            p.requires_grad_(True)

        embed = wm.encoder(data)
        post, prior = wm.dynamics.observe(
            embed, data["action"], data["is_first"], dt_labels=dt_labels)

        kl_loss, kl_value, _, _ = wm.dynamics.kl_loss(
            post, prior, kl_free, dyn_scale, rep_scale)
        kl_loss_mean = kl_loss.mean()

        feat = wm.dynamics.get_feat(post)

        decoder_preds = wm.heads["decoder"](feat)
        obs_loss = torch.zeros(1, device=device)
        for obs_key in ["height", "orientations", "velocity"]:
            if obs_key in decoder_preds and obs_key in data:
                obs_loss = obs_loss + (-decoder_preds[obs_key].log_prob(data[obs_key]).mean())

        reward_pred = wm.heads["reward"](feat)
        reward_loss = -reward_pred.log_prob(data["reward"]).mean()

        cont_pred = wm.heads["cont"](feat)
        cont_loss = -cont_pred.log_prob(data["cont"]).mean()

        norms_batch = {}
        for loss_name, loss_val in [("obs", obs_loss), ("reward", reward_loss),
                                     ("kl", kl_loss_mean), ("cont", cont_loss)]:
            g = torch.autograd.grad(loss_val, dt_param, retain_graph=True, allow_unused=True)[0]
            if g is not None:
                norms_batch[loss_name] = g.norm().item()
            else:
                norms_batch[loss_name] = 0.0
            all_norms[loss_name].append(norms_batch[loss_name])

        for p in wm.parameters():
            p.requires_grad_(False)

        total_b = sum(norms_batch.values())
        if bi % 5 == 0 and total_b > 0:
            print(f"  batch {bi}: obs={norms_batch['obs']/total_b*100:.1f}% "
                  f"reward={norms_batch['reward']/total_b*100:.1f}% "
                  f"kl={norms_batch['kl']/total_b*100:.1f}% "
                  f"cont={norms_batch['cont']/total_b*100:.1f}%")

    avg = {k: float(np.mean(v)) for k, v in all_norms.items()}
    std = {k: float(np.std(v)) for k, v in all_norms.items()}
    total = sum(avg.values())
    pcts = {k: round(avg[k] / total * 100, 2) if total > 0 else 0.0 for k in avg}

    print(f"\n{'='*60}")
    print(f"INDEPENDENT VERIFICATION RESULT")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: gate_fixed_dt={gate_fixed_dt}, gate_subsample_max_dt={gate_subsample_max_dt}")
    print(f"Batches: {n_batches}, seed=42")
    print(f"Method: torch.autograd.grad()")
    print(f"{'='*60}")
    print(f"{'Component':<12} {'L2 norm (mean)':<16} {'std':<16} {'%':>8}")
    print(f"{'-'*52}")
    for name in ["obs", "reward", "kl", "cont"]:
        print(f"{name:<12} {avg[name]:.8f}    {std[name]:.8f}    {pcts[name]:>7.2f}%")
    print(f"{'total':<12} {total:.8f}")
    print(f"{'='*60}")

    return {
        "avg_norms": avg,
        "std_norms": std,
        "percentages": pcts,
        "total_norm": total,
        "per_batch_norms": {k: [float(x) for x in v] for k, v in all_norms.items()},
    }

if __name__ == "__main__":
    device = "cuda:0"
    ckpt = "<PROJECT_ROOT>/logdir/exp_008d_subsample_on_h8/latest.pt"
    eps_dir = "<PROJECT_ROOT>/logdir/exp_008d_subsample_on_h8/train_eps"

    print("="*60)
    print("RUN 1: CORRECT 008d config (fixed_dt=2, subsample_max=2)")
    print("="*60)
    result_correct = verify(ckpt, eps_dir, device=device, n_batches=20,
                            gate_fixed_dt=2, gate_subsample_max_dt=2)

    print("\n\n")
    print("="*60)
    print("RUN 2: WRONG config (fixed_dt=0, subsample_max=8)")
    print("="*60)
    result_wrong = verify(ckpt, eps_dir, device=device, n_batches=20,
                          gate_fixed_dt=0, gate_subsample_max_dt=8)

    output = {
        "verification_id": "independent_verification_20260517",
        "script": "scripts/verify_grad_decomp_independent.py",
        "method": "torch.autograd.grad() per loss component",
        "checkpoint": ckpt,
        "n_batches": 20,
        "seed": 42,
        "correct_config": {
            "gate_fixed_dt": 2,
            "gate_subsample_max_dt": 2,
            "results": result_correct,
        },
        "wrong_config": {
            "gate_fixed_dt": 0,
            "gate_subsample_max_dt": 8,
            "results": result_wrong,
        },
        "comparison": {
            "paper_table8_55K": {"obs_pct": 16.3, "reward_pct": 49.0, "kl_pct": 34.7},
            "D061": {"obs_pct": 93, "reward_pct": 0, "kl_pct": 7},
        }
    }

    out_path = f"{PROJECT_DIR}/artifacts/analysis/gradient_decomposition_independent_verification.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")
