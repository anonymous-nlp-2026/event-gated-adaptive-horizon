#!/usr/bin/env python3
"""Prior-based Imagination Reward Analysis.

Encode initial states from episodes using posterior, then do H-step
imagination rollouts (prior only) with different dt values.
Uses cuda:1 for fast inference (minimal memory footprint).
"""
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

DEVICE = 'cuda:1'
N_EPISODES = 4
N_INIT_STATES_PER_EP = 5
HORIZON = 15
SEED = 42
ENCODE_UP_TO = 250

STOCH = 32
DETER = 512
DISCRETE = 32
FEAT_SIZE = STOCH * DISCRETE + DETER
EMBED_SIZE = 1024
NUM_ACTIONS = 6
OBS_KEYS = ['height', 'orientations', 'velocity']

EXPERIMENTS = {
    'baseline_s1': {
        'logdir': f'{PROJECT_DIR}/logdir/baseline_s1',
        'max_dt': 1,
        'dt_embed_dim': 0,
        'gate_enabled': False,
        'test_dts': [1],
    },
    'exp_008d': {
        'logdir': f'{PROJECT_DIR}/logdir/exp_008d_subsample_on_h8',
        'max_dt': 8,
        'dt_embed_dim': 32,
        'gate_enabled': True,
        'test_dts': [1, 2, 3, 4],
    },
    'exp_008f': {
        'logdir': f'{PROJECT_DIR}/logdir/exp_008f_no_dt_embed_subsample_on_h8',
        'max_dt': 8,
        'dt_embed_dim': 0,
        'gate_enabled': True,
        'test_dts': [1, 2],
    },
}


def load_episodes(logdir, n=N_EPISODES):
    eps_dir = os.path.join(logdir, 'train_eps')
    files = sorted(glob.glob(os.path.join(eps_dir, '*.npz')))[-n:]
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

        # Use observe for batch processing
        post, _ = rssm.observe(embed, action, is_first)

        indices = np.linspace(20, T - 10, n_per_ep, dtype=int)
        for t in indices:
            state = {k: v[:, t].detach().clone() for k, v in post.items()}
            all_states.append(state)
        print(f"    Encoded ep {ep_idx+1}/{len(episodes)}, T={T}")
    return all_states


@torch.no_grad()
def imagination_rollout(rssm, actor, reward_head, value_head, init_state, dt_value, horizon=HORIZON):
    state = {k: v.clone() for k, v in init_state.items()}
    reward_preds = []
    value_ests = []
    dt_tensor = torch.tensor([dt_value], dtype=torch.long, device=DEVICE)

    for h in range(horizon):
        feat = rssm.get_feat(state)
        reward_preds.append(reward_head(feat).mean().item())
        value_ests.append(value_head(feat).mode().item())
        action = actor(feat.detach()).sample()
        state = rssm.img_step(state, action, dt=dt_tensor)

    return np.array(reward_preds), np.array(value_ests)


@torch.no_grad()
def imagination_rollout_gate(rssm, actor, reward_head, value_head, event_gate, init_state, horizon=HORIZON, step=55000):
    state = {k: v.clone() for k, v in init_state.items()}
    reward_preds, value_ests, dts_chosen = [], [], []
    current_dt = torch.ones(1, dtype=torch.long, device=DEVICE)

    for h in range(horizon):
        feat = rssm.get_feat(state)
        reward_preds.append(reward_head(feat).mean().item())
        value_ests.append(value_head(feat).mode().item())
        action = actor(feat.detach()).sample()
        state = rssm.img_step(state, action, dt=current_dt)
        succ_feat = rssm.get_feat(state)
        _, next_dt, _ = event_gate(succ_feat.detach(), step, hard=True)
        next_dt = torch.clamp(next_dt, 1, 8)
        dts_chosen.append(next_dt.item())
        current_dt = next_dt

    return np.array(reward_preds), np.array(value_ests), np.array(dts_chosen)


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
    print(f"  Loaded {len(episodes)} episodes")

    t1 = time.time()
    init_states = encode_initial_states(rssm, encoder, episodes)
    print(f"  Encoded {len(init_states)} initial states in {time.time()-t1:.1f}s")

    results = {}

    for test_dt in exp_config['test_dts']:
        t2 = time.time()
        all_rewards = np.zeros((len(init_states), HORIZON))
        all_values = np.zeros((len(init_states), HORIZON))

        for si, init_state in enumerate(init_states):
            r, v = imagination_rollout(rssm, actor, reward_head, value, init_state, dt_value=test_dt)
            all_rewards[si] = r
            all_values[si] = v

        per_step = []
        for h in range(HORIZON):
            per_step.append({
                'step': h,
                'mean_reward_pred': float(np.mean(all_rewards[:, h])),
                'std_reward_pred': float(np.std(all_rewards[:, h])),
                'mean_value_est': float(np.mean(all_values[:, h])),
                'std_value_est': float(np.std(all_values[:, h])),
            })

        cum_reward = np.cumsum([s['mean_reward_pred'] for s in per_step])

        results[f'dt={test_dt}'] = {
            'per_step': per_step,
            'overall_mean_reward': float(np.mean(all_rewards)),
            'overall_std_reward': float(np.std(all_rewards)),
            'cumulative_reward_at_H': float(cum_reward[-1]),
            'initial_value': float(per_step[0]['mean_value_est']),
            'final_value': float(per_step[-1]['mean_value_est']),
        }

        elapsed = time.time() - t2
        r = results[f'dt={test_dt}']
        print(f"\n  dt={test_dt}: ({elapsed:.1f}s)")
        print(f"    Overall mean reward pred: {r['overall_mean_reward']:.6f}")
        print(f"    Cumulative @H={HORIZON}: {r['cumulative_reward_at_H']:.4f}")
        print(f"    Value init/final: {r['initial_value']:.4f} / {r['final_value']:.4f}")
        print(f"    Per-step: " + " ".join([f"{s['mean_reward_pred']:.4f}" for s in per_step]))

    if exp_config['gate_enabled'] and event_gate is not None:
        t3 = time.time()
        all_rewards_g = np.zeros((len(init_states), HORIZON))
        all_values_g = np.zeros((len(init_states), HORIZON))
        all_dts_g = np.zeros((len(init_states), HORIZON))

        for si, init_state in enumerate(init_states):
            r, v, dts = imagination_rollout_gate(rssm, actor, reward_head, value, event_gate, init_state)
            all_rewards_g[si] = r
            all_values_g[si] = v
            all_dts_g[si] = dts

        per_step_gate = []
        for h in range(HORIZON):
            per_step_gate.append({
                'step': h,
                'mean_reward_pred': float(np.mean(all_rewards_g[:, h])),
                'mean_value_est': float(np.mean(all_values_g[:, h])),
                'mean_dt_chosen': float(np.mean(all_dts_g[:, h])),
            })

        results['dt=gate'] = {
            'per_step': per_step_gate,
            'overall_mean_reward': float(np.mean(all_rewards_g)),
            'mean_dt_overall': float(np.mean(all_dts_g)),
            'dt_distribution': {str(i): float(np.mean(all_dts_g == i)) for i in range(1, 9)},
        }

        elapsed = time.time() - t3
        r = results['dt=gate']
        print(f"\n  dt=gate: ({elapsed:.1f}s)")
        print(f"    Overall mean reward: {r['overall_mean_reward']:.6f}")
        print(f"    Mean dt: {r['mean_dt_overall']:.2f}")
        print(f"    dt dist: {r['dt_distribution']}")
        print(f"    Per-step: " + " ".join([f"{s['mean_reward_pred']:.4f}" for s in per_step_gate]))
        print(f"    Per-step dts: " + " ".join([f"{s['mean_dt_chosen']:.1f}" for s in per_step_gate]))

    # Free GPU memory for next experiment
    del encoder, rssm, reward_head, actor, value, event_gate
    torch.cuda.empty_cache()

    return results


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    all_results = {}
    for exp_name, exp_config in EXPERIMENTS.items():
        all_results[exp_name] = analyze_experiment(exp_name, exp_config)

    out_path = os.path.join(PROJECT_DIR, 'analysis', 'imagination_reward_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY: Imagination Reward Predictions by dt")
    print(f"{'='*90}")
    print(f"{'Model':<15} {'dt':>6} {'MeanRewPred':>14} {'CumRew@H':>12} {'InitValue':>12} {'FinalValue':>12}")
    print(f"{'-'*90}")
    for exp_name, results in all_results.items():
        for dt_key, r in results.items():
            if dt_key == 'dt=gate':
                print(f"{exp_name:<15} {'gate':>6} {r['overall_mean_reward']:>14.6f} {'N/A':>12} {'N/A':>12} {'N/A':>12}  (mean_dt={r['mean_dt_overall']:.2f})")
            else:
                print(f"{exp_name:<15} {dt_key:>6} {r['overall_mean_reward']:>14.6f} {r['cumulative_reward_at_H']:>12.4f} {r['initial_value']:>12.4f} {r['final_value']:>12.4f}")

    # Key comparison
    if 'exp_008d' in all_results and 'dt=1' in all_results['exp_008d'] and 'dt=2' in all_results['exp_008d']:
        d1 = all_results['exp_008d']['dt=1']
        d2 = all_results['exp_008d']['dt=2']
        print(f"\n{'='*70}")
        print("KEY COMPARISON: exp_008d dt=2 vs dt=1 imagination")
        print(f"{'='*70}")
        diff = d2['overall_mean_reward'] - d1['overall_mean_reward']
        print(f"  Mean reward pred diff (dt=2 - dt=1): {diff:+.6f}")
        print(f"  Cumulative reward diff: {d2['cumulative_reward_at_H'] - d1['cumulative_reward_at_H']:+.4f}")
        print(f"  Value diff (initial): {d2['initial_value'] - d1['initial_value']:+.4f}")
        print(f"  Value diff (final):   {d2['final_value'] - d1['final_value']:+.4f}")

        print(f"\n  Per-step reward comparison:")
        print(f"  {'Step':>4} {'dt=1':>12} {'dt=2':>12} {'diff':>12}")
        for h in range(HORIZON):
            r1 = d1['per_step'][h]['mean_reward_pred']
            r2 = d2['per_step'][h]['mean_reward_pred']
            print(f"  {h:>4} {r1:>12.6f} {r2:>12.6f} {r2-r1:>+12.6f}")

        for dt_val in [3, 4]:
            dk = f'dt={dt_val}'
            if dk in all_results['exp_008d']:
                dd = all_results['exp_008d'][dk]
                diff_d = dd['overall_mean_reward'] - d1['overall_mean_reward']
                print(f"\n  dt={dt_val} vs dt=1 mean diff: {diff_d:+.6f}")

        if diff > 0:
            print("\n  >>> CONCLUSION: dt=2 imagination produces HIGHER reward predictions")
            print("  >>> CIVO mechanism in imagination level: CONFIRMED")
        else:
            print("\n  >>> CONCLUSION: dt=2 imagination does NOT produce higher reward predictions")
            print("  >>> CIVO needs revision: compound error through bootstrapping")

    # Also compare 008f dt=1 vs dt=2 as sanity check
    if 'exp_008f' in all_results and 'dt=1' in all_results['exp_008f'] and 'dt=2' in all_results['exp_008f']:
        d1f = all_results['exp_008f']['dt=1']
        d2f = all_results['exp_008f']['dt=2']
        print(f"\n  SANITY CHECK: exp_008f (no dt_emb) dt=2 vs dt=1")
        diff_f = d2f['overall_mean_reward'] - d1f['overall_mean_reward']
        print(f"  Mean reward pred diff: {diff_f:+.6f} (should be ~0)")


if __name__ == '__main__':
    main()
