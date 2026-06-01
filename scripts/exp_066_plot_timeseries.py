"""
exp_066: Plot timeseries evolution of RSSM fidelity gap and VE ratio.
Generates a multi-panel figure for temporal causal chain analysis.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

LOGDIR = Path("<PROJECT_ROOT>/logdir/exp_066_timeseries_rssm_fidelity_s0")
OUTDIR = LOGDIR

with open(LOGDIR / "timeseries_analysis_v2.json") as f:
    d = json.load(f)

steps = np.array(d["steps"]) / 1000  # in K

ss_gap = np.array(d["ss_gap"])
ss_ratio = np.array(d["ss_ratio"])
ol_gap = np.array(d["ol_gap"])
ol_ratio = np.array(d["ol_ratio"])
ve_ratio = np.array(d["ve_ratio"])
ss_mae_dt1 = np.array(d["ss_mae_dt1"])
ss_mae_dt2 = np.array(d["ss_mae_dt2"])
ol_mae_dt1 = np.array(d["ol_mae_dt1"])
ol_mae_dt2 = np.array(d["ol_mae_dt2"])
state_div_dt1 = np.array(d["state_div_dt1"])
state_div_dt2 = np.array(d["state_div_dt2"])
train_return = np.array(d["train_return"])

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle("exp_066: RSSM Fidelity Gap → Value Overestimation Temporal Causal Chain\n(Walker Walk, fixed dt=2, trained dt_embedding, 65K steps)",
             fontsize=13, fontweight='bold')

# Panel 1: State-space MAE dt=1 vs dt=2
ax = axes[0, 0]
ax.plot(steps, ss_mae_dt1, 'o-', color='#2196F3', label='dt=1', markersize=5)
ax.plot(steps, ss_mae_dt2, 's-', color='#F44336', label='dt=2', markersize=5)
ax.fill_between(steps, ss_mae_dt1, ss_mae_dt2, alpha=0.15, color='#F44336')
ax.set_ylabel('State-space MAE')
ax.set_title('(a) Single-step State Prediction MAE')
ax.legend(loc='upper right')
ax.set_xlabel('Training Step (K)')
ax.grid(True, alpha=0.3)

# Panel 2: State-space fidelity ratio + VE ratio (dual axis)
ax = axes[0, 1]
color1 = '#E91E63'
color2 = '#FF9800'
ln1 = ax.plot(steps, ss_ratio, 'o-', color=color1, label='SS Fidelity Ratio (dt2/dt1)', markersize=5, linewidth=2)
ax.set_ylabel('Fidelity Ratio (dt=2 / dt=1)', color=color1)
ax.tick_params(axis='y', labelcolor=color1)
ax.set_ylim(0.9, 1.9)
ax2 = ax.twinx()
ln2 = ax2.plot(steps, ve_ratio, 'D--', color=color2, label='VE Ratio', markersize=5, linewidth=2, alpha=0.8)
ax2.set_ylabel('Value Overestimation Ratio', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper left', fontsize=9)
ax.set_title('(b) Fidelity Ratio vs VE Ratio — Temporal Alignment')
ax.set_xlabel('Training Step (K)')
ax.grid(True, alpha=0.3)

# Panel 3: Open-loop MAE dt=1 vs dt=2
ax = axes[1, 0]
ax.plot(steps, ol_mae_dt1, 'o-', color='#2196F3', label='dt=1', markersize=5)
ax.plot(steps, ol_mae_dt2, 's-', color='#F44336', label='dt=2', markersize=5)
ax.fill_between(steps, ol_mae_dt1, ol_mae_dt2, alpha=0.15, color='#F44336')
ax.set_ylabel('Open-loop MAE')
ax.set_title('(c) Multi-step Open-loop Prediction MAE')
ax.legend(loc='upper right')
ax.set_xlabel('Training Step (K)')
ax.grid(True, alpha=0.3)

# Panel 4: Open-loop fidelity ratio + VE ratio
ax = axes[1, 1]
ln1 = ax.plot(steps, ol_ratio, 'o-', color='#9C27B0', label='OL Fidelity Ratio (dt2/dt1)', markersize=5, linewidth=2)
ax.set_ylabel('OL Fidelity Ratio', color='#9C27B0')
ax.tick_params(axis='y', labelcolor='#9C27B0')
ax.set_ylim(0.8, 1.8)
ax2 = ax.twinx()
ln2 = ax2.plot(steps, ve_ratio, 'D--', color=color2, label='VE Ratio', markersize=5, linewidth=2, alpha=0.8)
ax2.set_ylabel('VE Ratio', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper left', fontsize=9)
ax.set_title('(d) Open-loop Fidelity Ratio vs VE Ratio')
ax.set_xlabel('Training Step (K)')
ax.grid(True, alpha=0.3)

# Panel 5: State divergence
ax = axes[2, 0]
ax.plot(steps, state_div_dt1, 'o-', color='#2196F3', label='dt=1', markersize=5)
ax.plot(steps, state_div_dt2, 's-', color='#F44336', label='dt=2', markersize=5)
ax.set_ylabel('State Divergence (KL)')
ax.set_title('(e) Imagined State Divergence')
ax.legend(loc='upper right')
ax.set_xlabel('Training Step (K)')
ax.grid(True, alpha=0.3)

# Panel 6: Fidelity gap + train return
ax = axes[2, 1]
ln1 = ax.plot(steps, ss_gap, 'o-', color=color1, label='SS Fidelity Gap', markersize=5, linewidth=2)
ln3 = ax.plot(steps, ol_gap, 's-', color='#9C27B0', label='OL Fidelity Gap', markersize=5, linewidth=2)
ax.set_ylabel('Fidelity Gap (MAE_dt2 - MAE_dt1)')
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax2 = ax.twinx()
ln2 = ax2.plot(steps, train_return, 'x--', color='#4CAF50', label='Train Return', markersize=6, linewidth=1.5, alpha=0.7)
ax2.set_ylabel('Train Return', color='#4CAF50')
ax2.tick_params(axis='y', labelcolor='#4CAF50')
lns = ln1 + ln3 + ln2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper left', fontsize=9)
ax.set_title('(f) Fidelity Gaps vs Training Return')
ax.set_xlabel('Training Step (K)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUTDIR / "exp_066_timeseries_fidelity_causal.png", dpi=150, bbox_inches='tight')
fig.savefig(OUTDIR / "exp_066_timeseries_fidelity_causal.pdf", bbox_inches='tight')
print(f"Saved to {OUTDIR / 'exp_066_timeseries_fidelity_causal.png'}")

# ---- Correlation analysis ----
from scipy import stats
# Pearson correlation between fidelity metrics and VE ratio
corr_ss, p_ss = stats.pearsonr(ss_ratio, ve_ratio)
corr_ol, p_ol = stats.pearsonr(ol_ratio, ve_ratio)

# Cross-correlation to find lag
def normalized_cross_corr(x, y, max_lag=5):
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    n = len(x)
    cc = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            cc[lag] = np.mean(x[:n-lag] * y[lag:]) if lag < n else 0
        else:
            cc[lag] = np.mean(x[-lag:] * y[:n+lag]) if -lag < n else 0
    return cc

cc_ss = normalized_cross_corr(ss_ratio, ve_ratio, max_lag=4)
cc_ol = normalized_cross_corr(ol_ratio, ve_ratio, max_lag=4)

peak_lag_ss = max(cc_ss, key=cc_ss.get)
peak_lag_ol = max(cc_ol, key=cc_ol.get)

print("\n===== TEMPORAL CAUSAL CHAIN ANALYSIS =====")
print(f"\nPearson corr(SS_fidelity_ratio, VE_ratio) = {corr_ss:.4f} (p={p_ss:.4f})")
print(f"Pearson corr(OL_fidelity_ratio, VE_ratio) = {corr_ol:.4f} (p={p_ol:.4f})")
print(f"\nCross-correlation peak lag (SS → VE): {peak_lag_ss} steps (positive = SS leads)")
print(f"Cross-correlation peak lag (OL → VE): {peak_lag_ol} steps (positive = OL leads)")
print(f"\nSS cross-corr by lag: {dict((k, round(v, 3)) for k, v in cc_ss.items())}")
print(f"OL cross-corr by lag: {dict((k, round(v, 3)) for k, v in cc_ol.items())}")

print("\n===== PER-CHECKPOINT DATA =====")
print(f"{'Step':>6s} | {'SS_ratio':>8s} | {'OL_ratio':>8s} | {'VE_ratio':>8s} | {'SS_gap':>8s} | {'OL_gap':>8s} | {'Return':>8s}")
print("-" * 72)
for i in range(len(d["steps"])):
    print(f"{d['steps'][i]:>6d} | {ss_ratio[i]:>8.4f} | {ol_ratio[i]:>8.4f} | {ve_ratio[i]:>8.4f} | {ss_gap[i]:>8.4f} | {ol_gap[i]:>8.4f} | {train_return[i]:>8.2f}")

# Monotonicity test
ss_diffs = np.diff(ss_ratio)
ol_diffs = np.diff(ol_ratio)
ve_diffs = np.diff(ve_ratio)
ss_mono = np.sum(ss_diffs > 0) / len(ss_diffs)
ol_mono = np.sum(ol_diffs > 0) / len(ol_diffs)
ve_mono = np.sum(ve_diffs > 0) / len(ve_diffs)

print(f"\nMonotonicity (fraction of positive increments):")
print(f"  SS_ratio: {ss_mono:.2f} ({np.sum(ss_diffs > 0)}/{len(ss_diffs)} positive)")
print(f"  OL_ratio: {ol_mono:.2f} ({np.sum(ol_diffs > 0)}/{len(ol_diffs)} positive)")
print(f"  VE_ratio: {ve_mono:.2f} ({np.sum(ve_diffs > 0)}/{len(ve_diffs)} positive)")

# Onset detection: when does each metric first exceed a threshold
ss_onset = next((d["steps"][i] for i in range(len(ss_ratio)) if ss_ratio[i] > 1.1), None)
ol_onset = next((d["steps"][i] for i in range(len(ol_ratio)) if ol_ratio[i] > 1.1), None)
ve_onset = next((d["steps"][i] for i in range(len(ve_ratio)) if ve_ratio[i] > 0.2), None)

print(f"\nOnset detection (first exceeds threshold):")
print(f"  SS_ratio > 1.1: step {ss_onset}")
print(f"  OL_ratio > 1.1: step {ol_onset}")
print(f"  VE_ratio > 0.2: step {ve_onset}")

# Final summary
early_fidelity = ss_ratio[2]  # 20K
early_ve = ve_ratio[2]
late_fidelity = ss_ratio[-1]  # 65K
late_ve = ve_ratio[-1]

print(f"\n===== SUMMARY =====")
print(f"Early training (20K): SS_ratio={early_fidelity:.3f}, VE_ratio={early_ve:.4f}")
print(f"Late training (65K):  SS_ratio={late_fidelity:.3f}, VE_ratio={late_ve:.4f}")
print(f"SS fidelity ratio growth: {late_fidelity/early_fidelity:.2f}x")
print(f"VE ratio growth: {late_ve/(early_ve+1e-8):.2f}x")

if ss_onset and ve_onset and ss_onset <= ve_onset:
    print(f"\nCONCLUSION: SUCCESS — Fidelity gap onset ({ss_onset}) <= VE onset ({ve_onset})")
    print("  RSSM fidelity degradation precedes or is concurrent with value overestimation.")
    print("  Supports temporal causal chain: RSSM fidelity gap → value overestimation.")
elif ss_onset and ve_onset:
    print(f"\nCONCLUSION: NEGATIVE — Fidelity gap onset ({ss_onset}) > VE onset ({ve_onset})")
    print("  Value overestimation appears before fidelity gap, no causal support.")
else:
    print(f"\nCONCLUSION: INCONCLUSIVE — Could not determine onset timing.")
