#!/usr/bin/env python3
"""Signed Reward Error Analysis — optimized for CPU.

Uses pre-allocated buffers, processes episodes sequentially (B=1) to keep
memory low and avoid the O(T^2) torch.cat in static_scan.
Uses 8 episodes per experiment for tractable CPU runtime.
"""
import sys
import os
import glob
import json
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import torch

PROJECT_DIR = '<PROJECT_ROOT>'
sys.path.insert(0, PROJECT_DIR)

import networks
import tools

DEVICE = 'cpu'
DISCOUNT = 0.997
N_EPISODES = 8
SEED = 42

EXPERIMENTS = {
    'baseline_s1': {
        'logdir': f'{PROJECT_DIR}/logdir/baseline_s1',
        'max_dt': 1,
        'dt_embed_dim': 0,
        'test_dts': [1],
    },
    'exp_008d': {
        'logdir': f'{PROJECT_DIR}/logdir/exp_008d_subsample_on_h8',
        'max_dt': 8,
        'dt_embed_dim': 32,
        'test_dts': [1, 2],
    },
    'exp_008f': {
        'logdir': f'{PROJECT_DIR}/logdir/exp_008f_no_dt_embed_subsample_on_h8',
        'max_dt': 8,
        'dt_embed_dim': 0,
        'test_dts': [1, 2],
    },
}

STOCH = 32
DETER = 512
DISCRETE = 32
FEAT_SIZE = STOCH * DISCRETE + DETER
EMBED_SIZE = 1024
NUM_ACTIONS = 6
OBS_KEYS = ['height', 'orientations', 'velocity']


def load_episodes(logdir, n=N_EPISODES):
    eps_dir = os.path.join(logdir, 'train_eps')
    files = sorted(glob.glob(os.path.join(eps_dir, '*.npz')))[-n:]
    return [dict(np.load(f)) for f in files]


def build_model(max_dt, dt_embed_dim):
    encoder = networks.MLP(
        inp_dim=24, shape=None, layers=5, units=1024,
        act='SiLU', norm=True, symlog_inputs=True,
        device=DEVICE, name='Encoder',
    )
    rssm = networks.RSSM(
        stoch=STOCH, deter=DETER, hidden=512, rec_depth=1,
        discrete=DISCRETE, act='SiLU', norm=True,
        mean_act='none', std_act='sigmoid2', min_std=0.1,
        unimix_ratio=0.01, initial='learned',
        num_actions=NUM_ACTIONS, embed=EMBED_SIZE, device=DEVICE,
        max_dt=max_dt, dt_embed_dim=dt_embed_dim,
    )
    reward_head = networks.MLP(
        inp_dim=FEAT_SIZE, shape=(255,), layers=2, units=512,
        act='SiLU', norm=True, dist='symlog_disc',
        outscale=0.0, device=DEVICE, name='Reward',
    )
    return encoder, rssm, reward_head


def load_weights(encoder, rssm, reward_head, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    sd = ckpt['agent_state_dict']
    for prefix, module in [
        ('_wm.encoder._mlp.', encoder),
        ('_wm.dynamics.', rssm),
        ('_wm.heads.reward.', reward_head),
    ]:
        mod_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        module.load_state_dict(mod_sd, strict=True)


def observe_fast(rssm, embed, action, is_first, dt_labels=None):
    """Fast RSSM observe: B=1, pre-allocated buffers."""
    T = embed.shape[1]
    post_deter = torch.zeros(1, T, rssm._deter)
    post_stoch = torch.zeros(1, T, rssm._stoch, rssm._discrete)
    post_logit = torch.zeros(1, T, rssm._stoch, rssm._discrete)

    state = None
    for t in range(T):
        dt = dt_labels[:, t] if dt_labels is not None else None
        post_t, _ = rssm.obs_step(
            state, action[:, t], embed[:, t], is_first[:, t], dt=dt
        )
        state = post_t
        post_deter[:, t] = post_t['deter']
        post_stoch[:, t] = post_t['stoch']
        post_logit[:, t] = post_t['logit']

    return {'deter': post_deter, 'stoch': post_stoch, 'logit': post_logit}


def subsample_episode(episode, dt, discount=DISCOUNT):
    T = len(episode['reward'])
    indices = list(range(0, T, dt))
    cum_rewards = np.zeros(len(indices))
    for i, t_start in enumerate(indices[:-1]):
        t_end = min(t_start + dt, T)
        span = t_end - t_start
        gamma_powers = discount ** np.arange(span)
        cum_rewards[i] = np.sum(gamma_powers * episode['reward'][t_start:t_end])
    if len(indices) > 0:
        cum_rewards[-1] = episode['reward'][indices[-1]]
    return indices, cum_rewards


@torch.no_grad()
def analyze_experiment(exp_name, exp_config):
    print(f"\n{'='*60}")
    print(f"Analyzing: {exp_name}")
    print(f"{'='*60}")

    encoder, rssm, reward_head = build_model(
        exp_config['max_dt'], exp_config['dt_embed_dim']
    )
    load_weights(encoder, rssm, reward_head,
                 os.path.join(exp_config['logdir'], 'latest.pt'))
    encoder.eval()
    rssm.eval()
    reward_head.eval()

    episodes = load_episodes(exp_config['logdir'])
    print(f"Loaded {len(episodes)} episodes")

    results = {}
    for test_dt in exp_config['test_dts']:
        t0 = time.time()
        all_actual = []
        all_predicted = []

        for ep_idx, ep in enumerate(episodes):
            T = len(ep['reward'])

            if test_dt == 1:
                obs_cat = torch.cat([
                    torch.tensor(ep[k], dtype=torch.float32) for k in OBS_KEYS
                ], -1).unsqueeze(0)
                action = torch.tensor(ep['action'], dtype=torch.float32).unsqueeze(0)
                is_first = torch.tensor(ep['is_first'], dtype=torch.float32).unsqueeze(0)
                actual_r = ep['reward'].astype(np.float64)
                dt_labels = torch.ones(1, T, dtype=torch.long) if exp_config['max_dt'] > 1 else None
            else:
                indices, cum_r = subsample_episode(ep, test_dt)
                obs_cat = torch.cat([
                    torch.tensor(ep[k][indices], dtype=torch.float32) for k in OBS_KEYS
                ], -1).unsqueeze(0)
                action = torch.tensor(ep['action'][indices], dtype=torch.float32).unsqueeze(0)
                is_first = torch.tensor(ep['is_first'][indices], dtype=torch.float32).unsqueeze(0)
                actual_r = cum_r
                dt_labels = torch.full((1, len(indices)), test_dt, dtype=torch.long)

            embed = encoder(obs_cat)
            post = observe_fast(rssm, embed, action, is_first, dt_labels)
            feat = rssm.get_feat(post)
            reward_dist = reward_head(feat)
            pred_r = reward_dist.mean().squeeze(-1).squeeze(0).numpy()

            start = 1
            end = len(actual_r) - (1 if test_dt > 1 else 0)

            all_actual.extend(actual_r[start:end].tolist())
            all_predicted.extend(pred_r[start:end].tolist())

            elapsed = time.time() - t0
            print(f"    Ep {ep_idx+1}/{len(episodes)} (dt={test_dt}) {elapsed:.1f}s")

        all_actual = np.array(all_actual)
        all_predicted = np.array(all_predicted)
        signed_errors = all_predicted - all_actual

        results[f'dt={test_dt}'] = {
            'mean_signed_error': float(np.mean(signed_errors)),
            'std_signed_error': float(np.std(signed_errors)),
            'median_signed_error': float(np.median(signed_errors)),
            'mean_actual_reward': float(np.mean(all_actual)),
            'mean_predicted_reward': float(np.mean(all_predicted)),
            'n_samples': len(signed_errors),
            'pct_positive': float(np.mean(signed_errors > 0) * 100),
            'p25_signed_error': float(np.percentile(signed_errors, 25)),
            'p75_signed_error': float(np.percentile(signed_errors, 75)),
        }

        r = results[f'dt={test_dt}']
        total_t = time.time() - t0
        print(f"\n  dt={test_dt}: n={r['n_samples']} ({total_t:.1f}s)")
        print(f"    Mean signed error:   {r['mean_signed_error']:.6f}")
        print(f"    Std signed error:    {r['std_signed_error']:.6f}")
        print(f"    Median signed error: {r['median_signed_error']:.6f}")
        print(f"    P25/P75:             {r['p25_signed_error']:.6f} / {r['p75_signed_error']:.6f}")
        print(f"    Mean actual reward:  {r['mean_actual_reward']:.6f}")
        print(f"    Mean predicted:      {r['mean_predicted_reward']:.6f}")
        print(f"    % overprediction:    {r['pct_positive']:.1f}%")

    return results


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    all_results = {}
    for exp_name, exp_config in EXPERIMENTS.items():
        all_results[exp_name] = analyze_experiment(exp_name, exp_config)

    out_path = os.path.join(PROJECT_DIR, 'analysis', 'signed_reward_error_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'='*90}")
    print("SUMMARY TABLE: Signed Reward Prediction Error (positive = overprediction)")
    print(f"{'='*90}")
    print(f"{'Experiment':<15} {'dt':>4} {'Mean SE':>12} {'Std SE':>12} {'Median SE':>12} "
          f"{'%Over':>8} {'N':>8}")
    print(f"{'-'*90}")
    for exp_name, results in all_results.items():
        for dt_key, r in results.items():
            print(f"{exp_name:<15} {dt_key:>4} {r['mean_signed_error']:>12.6f} "
                  f"{r['std_signed_error']:>12.6f} {r['median_signed_error']:>12.6f} "
                  f"{r['pct_positive']:>7.1f}% {r['n_samples']:>8}")


if __name__ == '__main__':
    main()
