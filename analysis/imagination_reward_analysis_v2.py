#!/usr/bin/env python3
"""Imagination Reward Analysis v2: Random Policy Control + Extended Sample CI."""
import sys
import os
import glob
import json
import time

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import torch

PROJECT_DIR = '<PROJECT_ROOT>'
sys.path.insert(0, PROJECT_DIR)

import networks
import tools

DEVICE = 'cpu'
N_EPISODES = 20
N_INIT_STATES_PER_EP = 5
HORIZON = 15
SEED = 42
ENCODE_UP_TO = 250
N_BOOTSTRAP = 10000

STOCH = 32
DETER = 512
DISCRETE = 32
FEAT_SIZE = STOCH * DISCRETE + DETER
EMBED_SIZE = 1024
NUM_ACTIONS = 6
OBS_KEYS = ['height', 'orientations', 'velocity']

EXPERIMENTS = {
    'exp_008d': {
        'logdir': f'{PROJECT_DIR}/logdir/exp_008d_subsample_on_h8',
        'max_dt': 8,
        'dt_embed_dim': 32,
        'gate_enabled': True,
        'test_dts': [1, 2],
    },
    'exp_008f': {
        'logdir': f'{PROJECT_DIR}/logdir/exp_008f_no_dt_embed_subsample_on_h8',
        'max_dt': 8,
        'dt_embed_dim': 0,
        'gate_enabled': True,
        'test_dts': [1, 2],
    },
}


def bootstrap_ci(data, n_boot=N_BOOTSTRAP, ci=0.95):
    data = np.array(data)
    rng = np.random.RandomState(SEED)
    boot_means = np.array([np.mean(rng.choice(data, len(data), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def load_episodes(logdir, n=N_EPISODES):
    eps_dir = os.path.join(logdir, 'train_eps')
    files = sorted(glob.glob(os.path.join(eps_dir, '*.npz')))[-n:]
    print(f"  Loading {len(files)} episodes (requested {n}, available {len(sorted(glob.glob(os.path.join(eps_dir, '*.npz'))))})")
    return [dict(np.load(f)) for f in files]


def build_components(max_dt, dt_embed_dim, gate_enabled, gate_max_dt=8):
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
    actor = networks.MLP(
        inp_dim=FEAT_SIZE, shape=(NUM_ACTIONS,), layers=2, units=512,
        act='SiLU', norm=True, dist='normal', std='learned',
        min_std=0.1, max_std=1.0, absmax=1.0, temp=0.1,
        unimix_ratio=0.01, outscale=1.0, name='Actor',
    )
    value = networks.MLP(
        inp_dim=FEAT_SIZE, shape=(255,), layers=2, units=512,
        act='SiLU', norm=True, dist='symlog_disc',
        outscale=0.0, device=DEVICE, name='Value',
    )
    event_gate = None
    if gate_enabled:
        event_gate = networks.EventGate(
            feat_size=FEAT_SIZE, max_dt=gate_max_dt,
            hidden_units=256, hidden_layers=2,
            act='SiLU', norm=True,
            tau_init=5.0, tau_final=0.5, tau_anneal_steps=200000,
        )
    return encoder, rssm, reward_head, actor, value, event_gate


def load_weights(encoder, rssm, reward_head, actor, value, event_gate, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt['agent_state_dict']
    modules = [
        ('_wm.encoder._mlp.', encoder),
        ('_wm.dynamics.', rssm),
        ('_wm.heads.reward.', reward_head),
        ('_task_behavior.actor.', actor),
        ('_task_behavior.value.', value),
    ]
    if event_gate is not None:
        modules.append(('_wm.event_gate.', event_gate))
    for prefix, module in modules:
        mod_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if mod_sd:
            module.load_state_dict(mod_sd, strict=True)


@torch.no_grad()
def encode_initial_states(rssm, encoder, episodes, n_per_ep=N_INIT_STATES_PER_EP):
    all_states = []
    for ep_idx, ep in enumerate(episodes):
        T = min(len(ep['reward']), ENCODE_UP_TO)
        obs_cat = torch.cat([
            torch.tensor(ep[k][:T], dtype=torch.float32) for k in OBS_KEYS
        ], -1).unsqueeze(0).to(DEVICE)
        action = torch.tensor(ep['action'][:T], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        is_first = torch.tensor(ep['is_first'][:T], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        embed = encoder(obs_cat)
        post, _ = rssm.observe(embed, action, is_first)
        indices = np.linspace(20, T - 10, n_per_ep, dtype=int)
        for t in indices:
            state = {k: v[:, t].detach().clone() for k, v in post.items()}
            all_states.append(state)
        if (ep_idx + 1) % 5 == 0:
            print(f"    Encoded {ep_idx+1}/{len(episodes)} episodes")
    return all_states


@torch.no_grad()
def imagination_rollout(rssm, actor, reward_head, init_state, dt_value, horizon=HORIZON, random_policy=False):
    state = {k: v.clone() for k, v in init_state.items()}
    reward_preds = []
    dt_tensor = torch.tensor([dt_value], dtype=torch.long, device=DEVICE)

    for h in range(horizon):
        feat = rssm.get_feat(state)
        reward_preds.append(reward_head(feat).mean().item())
        if random_policy:
            action = torch.rand(1, NUM_ACTIONS, device=DEVICE) * 2 - 1
        else:
            action = actor(feat.detach()).sample()
        state = rssm.img_step(state, action, dt=dt_tensor)

    return np.array(reward_preds)


def run_rollouts(rssm, actor, reward_head, init_states, dt_value, random_policy=False):
    all_rewards = np.zeros((len(init_states), HORIZON))
    for si, init_state in enumerate(init_states):
        all_rewards[si] = imagination_rollout(
            rssm, actor, reward_head, init_state, dt_value, random_policy=random_policy,
        )
    per_state_mean = all_rewards.mean(axis=1)  # mean over horizon per state
    overall_mean = float(np.mean(all_rewards))
    overall_std = float(np.std(per_state_mean))
    ci_lo, ci_hi = bootstrap_ci(per_state_mean)
    cum_reward = float(np.sum([np.mean(all_rewards[:, h]) for h in range(HORIZON)]))

    per_step = []
    for h in range(HORIZON):
        per_step.append({
            'step': h,
            'mean_reward_pred': float(np.mean(all_rewards[:, h])),
            'std_reward_pred': float(np.std(all_rewards[:, h])),
        })

    return {
        'overall_mean_reward': overall_mean,
        'overall_std': overall_std,
        'ci_95_lo': ci_lo,
        'ci_95_hi': ci_hi,
        'cumulative_reward_at_H': cum_reward,
        'n_states': len(init_states),
        'per_step': per_step,
        'per_state_means': per_state_mean.tolist(),
    }


def analyze_experiment(exp_name, exp_config):
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*70}")

    t0 = time.time()
    encoder, rssm, reward_head, actor, value, event_gate = build_components(
        exp_config['max_dt'], exp_config['dt_embed_dim'], exp_config['gate_enabled'],
    )
    ckpt_path = os.path.join(exp_config['logdir'], 'latest.pt')
    load_weights(encoder, rssm, reward_head, actor, value, event_gate, ckpt_path)
    for m in [encoder, rssm, reward_head, actor, value]:
        m.to(DEVICE).eval()
    if event_gate is not None:
        event_gate.to(DEVICE).eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    episodes = load_episodes(exp_config['logdir'])

    t1 = time.time()
    init_states = encode_initial_states(rssm, encoder, episodes)
    print(f"  Encoded {len(init_states)} initial states in {time.time()-t1:.1f}s")

    results = {'trained_actor': {}, 'random_policy': {}}

    for test_dt in exp_config['test_dts']:
        for policy_name, use_random in [('trained_actor', False), ('random_policy', True)]:
            t2 = time.time()
            # Fix random seed per config for reproducibility
            torch.manual_seed(SEED + test_dt * 100 + (1 if use_random else 0))
            r = run_rollouts(rssm, actor, reward_head, init_states, test_dt, random_policy=use_random)
            elapsed = time.time() - t2
            results[policy_name][f'dt={test_dt}'] = r
            print(f"  {policy_name} dt={test_dt}: mean={r['overall_mean_reward']:.4f} ± {r['overall_std']:.4f} "
                  f"CI=[{r['ci_95_lo']:.4f}, {r['ci_95_hi']:.4f}] N={r['n_states']} ({elapsed:.1f}s)")

    del encoder, rssm, reward_head, actor, value, event_gate
    return results


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    all_results = {}
    for exp_name, exp_config in EXPERIMENTS.items():
        all_results[exp_name] = analyze_experiment(exp_name, exp_config)

    # Summary tables
    print(f"\n{'='*100}")
    print("TABLE 1: Trained Actor (Extended Sample)")
    print(f"{'='*100}")
    print(f"{'Model':<12} {'dt':>4} {'Mean':>10} {'Std':>8} {'95% CI':>22} {'N':>5} {'vs dt=1':>10}")
    print(f"{'-'*100}")
    for exp_name, res in all_results.items():
        dt1 = res['trained_actor'].get('dt=1', {}).get('overall_mean_reward', None)
        for dt_key, r in res['trained_actor'].items():
            ratio = f"{r['overall_mean_reward']/dt1:.2f}x" if dt1 and dt1 != 0 else "—"
            print(f"{exp_name:<12} {dt_key:>4} {r['overall_mean_reward']:>10.4f} {r['overall_std']:>8.4f} "
                  f"[{r['ci_95_lo']:.4f}, {r['ci_95_hi']:.4f}] {r['n_states']:>5} {ratio:>10}")

    print(f"\n{'='*100}")
    print("TABLE 2: Random Policy Control")
    print(f"{'='*100}")
    print(f"{'Model':<12} {'dt':>4} {'Mean':>10} {'Std':>8} {'95% CI':>22} {'N':>5} {'vs dt=1':>10}")
    print(f"{'-'*100}")
    for exp_name, res in all_results.items():
        dt1 = res['random_policy'].get('dt=1', {}).get('overall_mean_reward', None)
        for dt_key, r in res['random_policy'].items():
            ratio = f"{r['overall_mean_reward']/dt1:.2f}x" if dt1 and dt1 != 0 else "—"
            print(f"{exp_name:<12} {dt_key:>4} {r['overall_mean_reward']:>10.4f} {r['overall_std']:>8.4f} "
                  f"[{r['ci_95_lo']:.4f}, {r['ci_95_hi']:.4f}] {r['n_states']:>5} {ratio:>10}")

    # Conclusion table
    print(f"\n{'='*100}")
    print("CONCLUSION TABLE: Disentangling Dynamics Bias vs Actor Exploitation")
    print(f"{'='*100}")
    print(f"{'Model':<12} {'Trained dt2/dt1':>16} {'Random dt2/dt1':>16} {'Actor Exploit %':>18}")
    print(f"{'-'*100}")
    for exp_name, res in all_results.items():
        ta1 = res['trained_actor'].get('dt=1', {}).get('overall_mean_reward', 0)
        ta2 = res['trained_actor'].get('dt=2', {}).get('overall_mean_reward', 0)
        rp1 = res['random_policy'].get('dt=1', {}).get('overall_mean_reward', 0)
        rp2 = res['random_policy'].get('dt=2', {}).get('overall_mean_reward', 0)
        ta_ratio = ta2 / ta1 if ta1 != 0 else float('nan')
        rp_ratio = rp2 / rp1 if rp1 != 0 else float('nan')
        if ta_ratio > 1 and ta_ratio != float('nan'):
            actor_pct = (ta_ratio - rp_ratio) / (ta_ratio - 1) * 100 if ta_ratio != 1 else 0
            print(f"{exp_name:<12} {ta_ratio:>14.2f}x {rp_ratio:>14.2f}x {actor_pct:>16.1f}%")
        else:
            print(f"{exp_name:<12} {ta_ratio:>14.2f}x {rp_ratio:>14.2f}x {'—':>18}")

    # Strip per_state_means from JSON output (too large)
    json_results = {}
    for exp_name, res in all_results.items():
        json_results[exp_name] = {}
        for policy, dt_results in res.items():
            json_results[exp_name][policy] = {}
            for dt_key, r in dt_results.items():
                r_copy = {k: v for k, v in r.items() if k != 'per_state_means'}
                json_results[exp_name][policy][dt_key] = r_copy

    out_path = os.path.join(PROJECT_DIR, 'analysis', 'imagination_reward_results_v2.json')
    with open(out_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
