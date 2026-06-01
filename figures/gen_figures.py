import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})

steps = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105]

exps = {
    '008d': {
        'nve': [None, 0.846, 1.773, 1.705, 2.093, 4.129, 1.235, 1.482, 1.153, 1.726, 1.573],
        'label': '008d: Trained dt-emb (CIVO)',
        'color': '#d62728', 'ls': '-', 'lw': 2.2, 'marker': 'o', 'ms': 4,
    },
    '008f': {
        'nve': [None, 0.633, 1.02, 0.728, 0.832, 0.97, 0.367, 0.422, 0.649, 0.535, 0.297],
        'label': '008f: No dt-emb (control)',
        'color': '#2ca02c', 'ls': '-', 'lw': 2.0, 'marker': 's', 'ms': 4,
    },
    '018v2': {
        'nve': [None, 1.254, 3.998, 1.602, 1.38, 1.268, 1.044, 0.829, 1.079, 1.661, 1.341],
        'label': '018v2: Full SGS',
        'color': '#1f77b4', 'ls': '-', 'lw': 2.2, 'marker': 'D', 'ms': 4,
    },
    '009b': {
        'nve': [None, 1.194, 2.042, 1.238, 1.619, 1.413, 0.942, 0.658, 0.506, 0.43, 0.382],
        'label': '009b: Transient CIVO (self-corrected)',
        'color': '#ff7f0e', 'ls': '--', 'lw': 1.8, 'marker': '^', 'ms': 4,
    },
    '010b': {
        'nve': [None, 0.184, 2.891, 0.933, 0.879, 2.481, 8.652, 0.999, 0.984, 1.576, 1.175],
        'label': '010b: Cheetah CIVO',
        'color': '#d62728', 'ls': '--', 'lw': 1.8, 'marker': 'v', 'ms': 4,
    },
}

# ============ Figure 1: NVE Trajectories ============
fig, ax = plt.subplots(figsize=(8, 5))

for key in ['008f', '009b', '018v2', '008d', '010b']:
    e = exps[key]
    valid_idx = [i for i, v in enumerate(e['nve']) if v is not None]
    xs = [steps[i] for i in valid_idx]
    ys = [e['nve'][i] for i in valid_idx]
    ax.plot(xs, ys, color=e['color'], ls=e['ls'], lw=e['lw'],
            marker=e['marker'], ms=e['ms'], markeredgecolor='white',
            markeredgewidth=0.5, label=e['label'], zorder=3)

ax.axhline(y=1.5, color='#888888', ls=':', lw=1.2, zorder=1)
ax.text(107, 1.55, 'NVE = 1.5', fontsize=8.5, color='#666666', va='bottom')

ax.axvline(x=35, color='#cccccc', ls='--', lw=1.0, zorder=1)
ax.text(36, 9.0, 'Diagnostic\nwindow', fontsize=8, color='#aaaaaa', va='top')

ax.axhline(y=1.0, color='#cccccc', ls='-', lw=0.8, zorder=0)

ax.annotate('008d collapse\nonset (55K)',
            xy=(55, 4.129), xytext=(70, 6.5),
            fontsize=8.5, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2),
            ha='center')

ax.annotate('010b spike\n(65K)',
            xy=(65, 8.652), xytext=(82, 8.5),
            fontsize=8.5, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2),
            ha='center')

ax.set_xlabel('Training Steps (×1K)')
ax.set_ylabel('Normalized Value Excess (NVE)')
ax.set_xlim(10, 110)
ax.set_ylim(0, 9.5)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='#cccccc',
          fancybox=False)
ax.grid(True, alpha=0.3, ls='-', lw=0.5)

fig.savefig('<PROJECT_ROOT>/figures/fig_nve_trajectories.pdf')
fig.savefig('<PROJECT_ROOT>/figures/fig_nve_trajectories.png')
plt.close(fig)
print('Figure 1 saved.')

# ============ Figure 2: Gradient Decomposition ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
fig.subplots_adjust(bottom=0.18, top=0.88, wspace=0.08)

bar_width = 0.55
colors = {'obs': '#1f77b4', 'reward': '#ff7f0e', 'kl': '#2ca02c'}

# --- Left: 008d (No SGS) ---
steps_008d = [55, 105]
obs_008d   = [25.8, 93.0]
rew_008d   = [23.5, 0.0]
kl_008d    = [50.7, 7.0]

x1 = np.arange(len(steps_008d))
ax1.bar(x1, obs_008d, bar_width, label='Obs decoder', color=colors['obs'], edgecolor='white', lw=0.5)
ax1.bar(x1, rew_008d, bar_width, bottom=obs_008d, label='Reward', color=colors['reward'], edgecolor='white', lw=0.5)
ax1.bar(x1, kl_008d, bar_width, bottom=[o+r for o,r in zip(obs_008d, rew_008d)],
        label='KL', color=colors['kl'], edgecolor='white', lw=0.5)

for i, (o, r, k) in enumerate(zip(obs_008d, rew_008d, kl_008d)):
    if o > 8: ax1.text(i, o/2, f'{o:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    if r > 8: ax1.text(i, o + r/2, f'{r:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    if k > 8: ax1.text(i, o + r + k/2, f'{k:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

ax1.set_xticks(x1)
ax1.set_xticklabels([f'{s}K' for s in steps_008d])
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Gradient Contribution (%)')
ax1.set_title('008d — No SGS', fontweight='bold')
ax1.set_ylim(0, 105)
ax1.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='#cccccc', fontsize=9)

ax1.annotate('', xy=(1, 93), xytext=(0, 25.8),
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=2, connectionstyle='arc3,rad=0.3'))
ax1.text(0.5, 65, 'obs\nexplodes', fontsize=8.5, ha='center', color='#d62728', style='italic')

# --- Right: 018v2 counterfactual ---
steps_018 = [25, 75, 112]
obs_018   = [11.6, 20.8, 24.2]
rew_018   = [28.0, 22.1, 22.8]
kl_018    = [60.3, 57.1, 53.0]

x2 = np.arange(len(steps_018))
ax2.bar(x2, obs_018, bar_width, color=colors['obs'], edgecolor='white', lw=0.5)
ax2.bar(x2, rew_018, bar_width, bottom=obs_018, color=colors['reward'], edgecolor='white', lw=0.5)
ax2.bar(x2, kl_018, bar_width, bottom=[o+r for o,r in zip(obs_018, rew_018)],
        color=colors['kl'], edgecolor='white', lw=0.5)

for i, (o, r, k) in enumerate(zip(obs_018, rew_018, kl_018)):
    if o > 8: ax2.text(i, o/2, f'{o:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    if r > 8: ax2.text(i, o + r/2, f'{r:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    if k > 8: ax2.text(i, o + r + k/2, f'{k:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

ax2.set_xticks(x2)
ax2.set_xticklabels([f'{s}K' for s in steps_018])
ax2.set_xlabel('Training Steps')
ax2.set_title('018v2 — Full SGS (counterfactual)', fontweight='bold')

fig.text(0.5, 0.04, 'Actual 018v2 training: 100% KL gradient at all steps (obs/reward blocked by SGS)',
         fontsize=9, ha='center', style='italic', color='#555555')

fig.suptitle('Gradient Decomposition on dt-embedding', fontsize=13, fontweight='bold')

fig.savefig('<PROJECT_ROOT>/figures/fig_gradient_decomposition.pdf')
fig.savefig('<PROJECT_ROOT>/figures/fig_gradient_decomposition.png')
plt.close(fig)
print('Figure 2 saved.')

print('Done.')
