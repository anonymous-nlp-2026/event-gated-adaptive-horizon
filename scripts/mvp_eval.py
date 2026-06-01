#!/usr/bin/env python3
"""MVP evaluation script for event-gated-adaptive-horizon.

Reads metrics.jsonl from a training logdir and evaluates 4 MVP criteria:
  1. Return >= 80% of baseline
  2. Gate entropy > 0.3 nats
  3. Mean H_eff > 1.2x H_nominal (dt_mean > 1.2)
  4. Temporal structure: dt_autocorr > 0.15 OR dt_std > 0.5

Usage:
  python scripts/mvp_eval.py --logdir logdir/event_gated_v5_s1 --baseline-return 953.4 --mode interim
  python scripts/mvp_eval.py --logdir logdir/event_gated_v5_s1 --baseline-return 953.4 --mode final
"""

import argparse
import json
import os
import sys
import numpy as np


def load_metrics(logdir):
    path = os.path.join(logdir, "metrics.jsonl")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        sys.exit(1)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_eval_records(records):
    return [r for r in records if "eval_return" in r]


def get_model_records(records):
    """Records containing gate metrics (logged with model loss)."""
    return [r for r in records if "gate_dt_mean" in r]


def compute_return_trend(eval_records, window_steps=10000):
    """Linear regression slope of eval_return over the last window_steps env steps."""
    if len(eval_records) < 2:
        return 0.0, "INSUFFICIENT_DATA"
    max_step = eval_records[-1]["step"]
    recent = [r for r in eval_records if r["step"] >= max_step - window_steps]
    if len(recent) < 2:
        recent = eval_records[-2:]
    steps = np.array([r["step"] for r in recent], dtype=float)
    returns = np.array([r["eval_return"] for r in recent], dtype=float)
    if steps[-1] == steps[0]:
        return 0.0, "FLAT"
    # slope per 10K steps
    coeffs = np.polyfit(steps, returns, 1)
    slope_per_10k = coeffs[0] * 10000
    if slope_per_10k > 5:
        label = "TRENDING_UP"
    elif slope_per_10k < -5:
        label = "TRENDING_DOWN"
    else:
        label = "FLAT"
    return slope_per_10k, label


def fmt(val, precision=3):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def main():
    parser = argparse.ArgumentParser(description="MVP Evaluation")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--baseline-return", type=float, required=True)
    parser.add_argument("--mode", choices=["interim", "final"], default="final")
    parser.add_argument("--device", default="cuda:0", help="(unused, kept for interface consistency)")
    args = parser.parse_args()

    records = load_metrics(args.logdir)
    eval_recs = get_eval_records(records)
    model_recs = get_model_records(records)

    if not eval_recs:
        print("ERROR: no eval records found in metrics.jsonl")
        sys.exit(1)

    # --- Criterion 1: Return ---
    last_evals = eval_recs[-10:]
    eval_returns = [r["eval_return"] for r in last_evals]
    ret_mean = np.mean(eval_returns)
    ret_std = np.std(eval_returns)
    ret_ratio = ret_mean / args.baseline_return
    slope_per_10k, trend_label = compute_return_trend(eval_recs)

    # Thresholds: >=0.80 PASS, 0.70-0.79 GRAY ZONE, <0.70 FAIL
    if args.mode == "final":
        if ret_ratio >= 0.80:
            c1_status = "PASS"
        elif ret_ratio >= 0.70:
            c1_status = "GRAY ZONE"
        else:
            c1_status = "FAIL"
    else:
        c1_status = trend_label

    # --- Gate metrics (from latest model record) ---
    latest_gate = model_recs[-1] if model_recs else {}
    gate_scale = latest_gate.get("gate_scale")
    gate_loss = latest_gate.get("gate_loss")
    dt_mean = latest_gate.get("gate_dt_mean")
    dt_std = latest_gate.get("gate_dt_std")
    entropy = latest_gate.get("gate_empirical_entropy")
    entropy_ratio = latest_gate.get("gate_entropy_ratio")
    current_step = latest_gate.get("step", eval_recs[-1]["step"])

    # dt distribution
    dt_fracs = {}
    for k in range(1, 9):
        key = f"gate_dt_{k}_frac"
        if key in latest_gate:
            dt_fracs[k] = latest_gate[key]

    # --- Criterion 2: Entropy > 0.3 nats ---
    if entropy is not None:
        c2_status = "PASS" if entropy > 0.3 else ("NOT YET" if args.mode == "interim" else "FAIL")
    else:
        c2_status = "N/A (field missing)"

    # --- Criterion 3: H_eff > 1.2x H_nominal (dt_mean > 1.2) ---
    if dt_mean is not None:
        h_ratio = dt_mean  # H_eff/H_nominal = dt_mean / 1.0
        c3_status = "PASS" if h_ratio > 1.2 else ("NOT YET" if args.mode == "interim" else "FAIL")
    else:
        h_ratio = None
        c3_status = "N/A (field missing)"

    # --- Criterion 4: autocorr > 0.15 OR dt_std > 0.5 ---
    # gate_dt_autocorr not logged; check dt_std only
    dt_autocorr = latest_gate.get("gate_dt_autocorr")
    has_autocorr = dt_autocorr is not None
    has_std = dt_std is not None

    if has_autocorr or has_std:
        pass_autocorr = has_autocorr and dt_autocorr > 0.15
        pass_std = has_std and dt_std > 0.5
        if pass_autocorr or pass_std:
            c4_status = "PASS"
        else:
            c4_status = "NOT YET" if args.mode == "interim" else "FAIL"
    else:
        c4_status = "N/A (fields missing)"

    # ==================== Output ====================
    mode_label = "INTERIM" if args.mode == "interim" else "FINAL"
    bar = "═" * 50

    print(f"\n{bar}")
    print(f"  MVP Evaluation — {mode_label} ({current_step//1000}K)")
    print(f"{bar}")
    if args.mode == "interim":
        print("⚠ 中间检查——趋势判断，非 pass/fail 判定\n")
    print()

    # Criterion 1
    print(f"[Criterion 1] Return")
    print(f"  eval_return (last {len(last_evals)}): {fmt(ret_mean, 1)} ± {fmt(ret_std, 1)}")
    print(f"  baseline_return: {fmt(args.baseline_return, 1)}")
    print(f"  return_ratio: {fmt(ret_ratio, 3)}")
    print(f"  trend (last 10K): {trend_label} ({'+' if slope_per_10k >= 0 else ''}{fmt(slope_per_10k, 1)}/10K)")
    if args.mode == "interim" and gate_scale is not None:
        print(f"  gate_scale: {fmt(gate_scale, 3)}")
    print(f"  status: {c1_status}")
    print()

    # Criterion 2
    print(f"[Criterion 2] Gate Entropy")
    print(f"  empirical_entropy: {fmt(entropy, 3)}")
    if entropy_ratio is not None:
        print(f"  entropy_ratio: {fmt(entropy_ratio, 3)}")
    print(f"  threshold: > 0.3")
    print(f"  status: {c2_status}")
    print()

    # Criterion 3
    print(f"[Criterion 3] Effective Horizon")
    print(f"  dt_mean: {fmt(dt_mean, 3)}")
    print(f"  H_eff/H_nominal: {fmt(h_ratio, 3)}×")
    print(f"  threshold: > 1.2×")
    print(f"  status: {c3_status}")
    print()

    # Criterion 4
    print(f"[Criterion 4] Temporal Structure")
    print(f"  dt_autocorr: {fmt(dt_autocorr)}")
    print(f"  dt_std: {fmt(dt_std, 3)}")
    print(f"  threshold: autocorr > 0.15 OR std > 0.5")
    print(f"  status: {c4_status}")
    print()

    # Gate dt distribution
    if dt_fracs:
        print(f"[Gate Δt Distribution]")
        for k in sorted(dt_fracs.keys()):
            v = dt_fracs[k]
            bar_len = int(v * 40)
            print(f"  dt={k}: {'█' * bar_len}{'░' * (40 - bar_len)} {v*100:5.1f}%")
        print()

    # Stability
    print(f"[Stability]")
    print(f"  gate_loss: {fmt(gate_loss, 4)}")
    print(f"  return_trend: {'+' if slope_per_10k >= 0 else ''}{fmt(slope_per_10k, 1)} per 10K steps")
    if gate_scale is not None:
        print(f"  gate_scale: {fmt(gate_scale, 3)}")
    print()

    # Overall
    print(bar)
    if args.mode == "interim":
        parts = []
        if gate_scale is not None and gate_scale < 1.0:
            parts.append(f"gate ramping (scale={fmt(gate_scale, 2)})")
        parts.append(f"return trend: {trend_label}")
        print(f"  Overall: INTERIM — {', '.join(parts)}")
    else:
        statuses = [c1_status, c2_status, c3_status, c4_status]
        n_pass = sum(1 for s in statuses if s == "PASS")
        n_fail = sum(1 for s in statuses if s == "FAIL")
        if n_pass == 4:
            overall = "MVP ACHIEVED ✓"
        elif n_fail > 0:
            failed = []
            for i, s in enumerate(statuses, 1):
                if s == "FAIL":
                    failed.append(f"C{i}")
            overall = f"NOT MET — failed: {', '.join(failed)}"
        else:
            overall = f"PARTIAL — {n_pass}/4 passed"
        print(f"  Overall: {overall}")
    print(bar)
    print()


if __name__ == "__main__":
    main()
