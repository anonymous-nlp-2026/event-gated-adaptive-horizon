# Why Temporal Abstraction Can Fail in Imagination-Based RL: A Diagnostic Analysis of Variable-Rate DreamerV3

Source code and analysis scripts for reproducing the experiments in the paper.

## Directory Structure

```
.
├── dreamer.py              # Main training entry point (DreamerV3 + event gate)
├── models.py               # World model, actor-critic, CIVO detector, gate training
├── networks.py             # RSSM, encoder/decoder, EventGate network
├── tools.py                # Replay buffer, variable-dt lambda returns, utilities
├── configs.yaml            # All hyperparameters (gate_* for event gate configs)
├── exploration.py          # Exploration strategies
├── parallel.py             # Parallel environment wrapper
├── envs/                   # Environment wrappers (DMC, ManiSkill, Atari, Crafter, etc.)
├── scripts/                # Experiment launch scripts and analysis tools
├── analysis/               # Bootstrap CIs, reward error analysis
├── artifacts/              # Pre-computed analysis data and paper figures
│   ├── analysis/           # Gradient decomposition, entropy dynamics data
│   └── figures/            # Figure generation scripts
├── figures/                # Paper figures (PDF)
├── requirements.txt        # Python dependencies
└── Dockerfile              # Container build instructions
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- dm_control 1.0.9+
- MuJoCo 2.3+

```bash
pip install -r requirements.txt
```

For headless rendering (e.g., on a server without display):
```bash
bash xvfb_run.sh python dreamer.py ...
```

## Reproduction

### DreamerV3 Baseline (dt=1)
```bash
python dreamer.py --configs defaults dmc_proprio \
    --task dmc_walker_walk \
    --logdir ./logdir/baseline_s0 \
    --seed 0
```

### dt-embedding (CIVO trigger condition)
```bash
python dreamer.py --configs defaults dmc_proprio \
    --task dmc_walker_walk \
    --gate_enabled True \
    --gate_fixed_dt 2 \
    --gate_subsample_max_dt 2 \
    --gate_dt_embed_dim 32 \
    --imag_horizon 8 \
    --logdir ./logdir/dt_embed_s0 \
    --seed 0
```

### Full Event-Gated Adaptive Horizon
```bash
python dreamer.py --configs defaults dmc_proprio \
    --task dmc_walker_walk \
    --gate_enabled True \
    --gate_type gumbel \
    --gate_max_dt 8 \
    --gate_dt_embed_dim 32 \
    --imag_horizon 15 \
    --logdir ./logdir/full_gate_s0 \
    --seed 0
```

### Cheetah Run Experiments
```bash
bash scripts/cheetah_run_experiments.sh
```

### Monitoring
```bash
tensorboard --logdir ./logdir
```

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gate_enabled` | false | Enable event-gated imagination |
| `gate_max_dt` | 8 | Maximum temporal skip |
| `gate_dt_embed_dim` | 32 | dt embedding dimension |
| `gate_type` | gumbel | Gate sampling (gumbel / fixed) |
| `gate_fixed_dt` | 0 | Fixed dt (0 = use learned gate) |
| `civo_detector` | true | Enable CIVO detection metrics |
| `civo_ve_threshold` | 0.49 | Value error threshold for CIVO |
| `imag_horizon` | 15 | Imagination rollout horizon |

## Analysis Scripts

- `analysis/compute_bootstrap_cis.py` — Bootstrap confidence intervals for main results
- `scripts/compute_gradient_decomposition.py` — Gradient decomposition analysis (Section 4)
- `scripts/civo_detector.py` — CIVO detection and metrics
- `figures/gen_figures.py` — Regenerate paper figures from cached data

## License

MIT
