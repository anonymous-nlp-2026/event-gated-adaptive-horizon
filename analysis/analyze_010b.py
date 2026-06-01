#!/usr/bin/env python3
"""CIVO analysis for exp_010b_cheetah_civo_trigger.
Computes Normalized Value Estimation (NVE) ratio vs baseline and determines CIVO verdict.
"""

import json
import os
import sys
import numpy as np

BASELINE_PATH = "<PROJECT_ROOT>/logdir/exp_010a_cheetah_baseline_s1/metrics.jsonl"
EXP_PATH = "<PROJECT_ROOT>/logdir/exp_010b_cheetah_civo_trigger/metrics.jsonl"
OUTPUT_PATH = "<PROJECT_ROOT>/analysis/010b_civo_analysis.json"

NVE_THRESHOLD = 1.5
SUSTAINED_STEPS = 35000
COLLAPSE_RATIO = 0.5


def load_metrics(path):
    """Load metrics.jsonl, return dict: step -> {field: value}."""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = row.get("step")
            if step is None:
                continue
            if step not in data:
                data[step] = {}
            data[step].update(row)
    return data


def extract_eval_value_pairs(metrics_dict):
    """Extract (step, eval_return, value_mean) triples where both fields exist."""
    pairs = []
    for step in sorted(metrics_dict.keys()):
        rec = metrics_dict[step]
        er = rec.get("eval_return")
        vm = rec.get("value_mean")
        if er is not None and vm is not None:
            pairs.append({"step": step, "eval_return": er, "value_mean": vm})
    return pairs


def compute_ve_ratio(pairs):
    """Compute ve_ratio = value_mean / eval_return for each entry."""
    for p in pairs:
        if abs(p["eval_return"]) > 1e-8:
            p["ve_ratio"] = p["value_mean"] / p["eval_return"]
        else:
            p["ve_ratio"] = float("nan")
    return pairs


def nearest_baseline_ve(baseline_pairs, target_step):
    """Find baseline ve_ratio at nearest step."""
    if not baseline_pairs:
        return float("nan")
    best = min(baseline_pairs, key=lambda x: abs(x["step"] - target_step))
    return best["ve_ratio"]


def analyze(dry_run=False):
    # Load baseline
    if not os.path.exists(BASELINE_PATH):
        print(f"ERROR: Baseline not found: {BASELINE_PATH}")
        sys.exit(1)

    baseline_metrics = load_metrics(BASELINE_PATH)
    baseline_pairs = extract_eval_value_pairs(baseline_metrics)
    baseline_pairs = compute_ve_ratio(baseline_pairs)

    if dry_run:
        print(f"=== DRY RUN (baseline only) ===")
        print(f"Baseline entries with both eval_return & value_mean: {len(baseline_pairs)}")
        if baseline_pairs:
            print(f"Step range: {baseline_pairs[0]['step']} - {baseline_pairs[-1]['step']}")
            print(f"Sample ve_ratios:")
            for p in baseline_pairs[:5]:
                print(f"  step={p['step']:>7d}  eval_return={p['eval_return']:>10.3f}  "
                      f"value_mean={p['value_mean']:>10.4f}  ve_ratio={p['ve_ratio']:.4f}")
            ve_ratios = [p["ve_ratio"] for p in baseline_pairs if not np.isnan(p["ve_ratio"])]
            print(f"Baseline ve_ratio stats: mean={np.mean(ve_ratios):.4f}, "
                  f"std={np.std(ve_ratios):.4f}, min={np.min(ve_ratios):.4f}, max={np.max(ve_ratios):.4f}")
        print("DRY RUN PASSED - script can process baseline data correctly.")
        return

    # Load experiment
    if not os.path.exists(EXP_PATH):
        print(f"ERROR: Experiment metrics not found: {EXP_PATH}")
        sys.exit(1)

    exp_metrics = load_metrics(EXP_PATH)
    exp_pairs = extract_eval_value_pairs(exp_metrics)
    exp_pairs = compute_ve_ratio(exp_pairs)

    if not exp_pairs:
        print("ERROR: No eval checkpoints with both eval_return and value_mean found in 010b.")
        sys.exit(1)

    # Compute NVE
    timeline = []
    for p in exp_pairs:
        bve = nearest_baseline_ve(baseline_pairs, p["step"])
        if abs(bve) > 1e-8 and not np.isnan(bve):
            nve = p["ve_ratio"] / bve
        else:
            nve = float("nan")
        timeline.append({
            "step": p["step"],
            "eval_return": p["eval_return"],
            "value_mean": p["value_mean"],
            "ve_ratio": round(p["ve_ratio"], 6),
            "baseline_ve": round(bve, 6),
            "nve": round(nve, 4) if not np.isnan(nve) else None
        })

    # CIVO detection
    valid_nve = [(t["step"], t["nve"]) for t in timeline if t["nve"] is not None]

    nve_peak = max((n for _, n in valid_nve), default=0)
    nve_peak_step = next((s for s, n in valid_nve if n == nve_peak), None)

    # Sustained detection: find longest consecutive window above threshold
    above_threshold = [(s, n) for s, n in valid_nve if n > NVE_THRESHOLD]
    sustained = False
    sustained_window = None

    if len(above_threshold) >= 2:
        # Check if there's a contiguous span >= SUSTAINED_STEPS
        start_step = above_threshold[0][0]
        end_step = above_threshold[0][0]
        best_start, best_end = start_step, end_step

        for i in range(1, len(above_threshold)):
            cur_step = above_threshold[i][0]
            prev_step = above_threshold[i - 1][0]
            # Allow gaps up to 15K steps (eval every 10K)
            if cur_step - prev_step <= 15000:
                end_step = cur_step
            else:
                if end_step - start_step > best_end - best_start:
                    best_start, best_end = start_step, end_step
                start_step = cur_step
                end_step = cur_step

        if end_step - start_step > best_end - best_start:
            best_start, best_end = start_step, end_step

        if best_end - best_start >= SUSTAINED_STEPS:
            sustained = True
            sustained_window = f"{best_start//1000}K-{best_end//1000}K"

    # Collapse detection
    eval_returns = [(t["step"], t["eval_return"]) for t in timeline]
    if eval_returns:
        peak_return = max(er for _, er in eval_returns)
        peak_return_step = next(s for s, er in eval_returns if er == peak_return)
        final_return = eval_returns[-1][1]
        collapse_ratio = (peak_return - final_return) / peak_return if peak_return > 1e-8 else 0
        collapse_detected = collapse_ratio > COLLAPSE_RATIO
    else:
        peak_return = 0
        peak_return_step = 0
        final_return = 0
        collapse_ratio = 0
        collapse_detected = False

    # CIVO verdict
    if sustained:
        verdict = "CIVO_PRESENT"
    elif nve_peak > NVE_THRESHOLD:
        verdict = "CIVO_TRANSIENT"
    else:
        verdict = "CIVO_ABSENT"

    result = {
        "exp_id": "exp_010b_cheetah_civo_trigger",
        "environment": "cheetah_run",
        "civo_verdict": verdict,
        "nve_peak": round(nve_peak, 4),
        "nve_peak_step": nve_peak_step,
        "nve_sustained_above_threshold": sustained,
        "nve_sustained_window": sustained_window,
        "eval_return_peak": round(peak_return, 3),
        "eval_return_peak_step": peak_return_step,
        "eval_return_final": round(final_return, 3),
        "collapse_detected": collapse_detected,
        "collapse_ratio": round(collapse_ratio, 4),
        "timeline": timeline
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"{'='*50}")
    print(f"CIVO Analysis: exp_010b_cheetah_civo_trigger")
    print(f"{'='*50}")
    print(f"Verdict:       {verdict}")
    print(f"NVE peak:      {nve_peak:.4f} (step {nve_peak_step})")
    print(f"Sustained:     {sustained} ({sustained_window or 'N/A'})")
    print(f"Eval peak:     {peak_return:.1f} (step {peak_return_step})")
    print(f"Eval final:    {final_return:.1f}")
    print(f"Collapse:      {collapse_detected} (ratio={collapse_ratio:.4f})")
    print(f"Checkpoints:   {len(timeline)}")
    print(f"{'='*50}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    analyze(dry_run=dry_run)
