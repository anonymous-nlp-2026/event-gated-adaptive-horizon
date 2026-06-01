#!/usr/bin/env python3
"""Supplement v3 AUROC CI with margin analysis and exact binomial CI."""

import json
import math
import os
import numpy as np

ARTIFACTS = "<PROJECT_ROOT>/artifacts"

with open(os.path.join(ARTIFACTS, "nve_threshold_v3.json")) as f:
    v3 = json.load(f)

pe = v3["per_experiment"]

def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial (Clopper-Pearson) 95% CI."""
    if k == 0:
        lo = 0.0
    else:
        try:
            from scipy.stats import beta as beta_dist
            lo = beta_dist.ppf(alpha / 2, k, n - k + 1)
        except ImportError:
            lo = (1 + (n - k + 1) / (k * _f_quantile(alpha / 2, 2 * k, 2 * (n - k + 1)))) ** -1

    if k == n:
        hi = 1.0
        lo_simple = alpha ** (1.0 / n)
        lo = lo_simple
    else:
        try:
            from scipy.stats import beta as beta_dist
            hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
        except ImportError:
            hi = 1.0

    return round(lo, 4), round(hi, 4)

def compute_auroc_manual(labels, scores):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    concordant = 0.0
    for i in range(len(labels)):
        if labels[i] == 1:
            for j in range(len(labels)):
                if labels[j] == 0:
                    if scores[i] > scores[j]:
                        concordant += 1.0
                    elif scores[i] == scores[j]:
                        concordant += 0.5
    return concordant / (n_pos * n_neg)

def bootstrap_auroc_detailed(labels, scores, n_bootstrap=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    aurocs = []
    skipped = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bl = [labels[i] for i in idx]
        bs = [scores[i] for i in idx]
        if sum(bl) == 0 or sum(bl) == len(bl):
            skipped += 1
            continue
        a = compute_auroc_manual(bl, bs)
        if not math.isnan(a):
            aurocs.append(a)
    if len(aurocs) < 10:
        return {"ci_lower": None, "ci_upper": None, "n_valid": len(aurocs),
                "n_skipped": skipped, "note": "too few valid resamples"}
    ci = np.percentile(aurocs, [2.5, 97.5])
    return {
        "ci_lower": round(float(ci[0]), 4),
        "ci_upper": round(float(ci[1]), 4),
        "mean": round(float(np.mean(aurocs)), 4),
        "std": round(float(np.std(aurocs)), 4),
        "n_valid": len(aurocs),
        "n_skipped": skipped,
        "n_unique_values": len(set(round(a, 6) for a in aurocs)),
        "min": round(float(min(aurocs)), 4),
        "max": round(float(max(aurocs)), 4),
    }

def analyze_task(task_data, task_name):
    pos_ids = task_data["positive"]
    neg_ids = task_data["negative"]
    
    pos_scores = [pe[eid]["max_nve_35K_plus"] for eid in pos_ids]
    neg_scores = [pe[eid]["max_nve_35K_plus"] for eid in neg_ids]
    
    labels = [1] * len(pos_ids) + [0] * len(neg_ids)
    scores = pos_scores + neg_scores
    
    n = len(labels)
    n_pos = len(pos_ids)
    n_neg = len(neg_ids)
    
    pos_min = min(pos_scores)
    neg_max = max(neg_scores)
    margin = pos_min - neg_max
    separated = margin > 0
    
    # AUROC
    auroc = compute_auroc_manual(labels, scores)
    
    # Bootstrap with more detail
    boot = bootstrap_auroc_detailed(labels, scores, n_bootstrap=2000, seed=42)
    
    # For perfect accuracy case (all thresholds between neg_max and pos_min work)
    # Find best threshold accuracy
    if separated:
        k_correct = n
    else:
        # find best threshold accuracy from v3 data
        best_acc = max(t["accuracy"] for t in task_data["thresholds"].values())
        k_correct = round(best_acc * n)
    
    acc_ci = clopper_pearson(k_correct, n)
    
    result = {
        "task": task_name,
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_experiments": {eid: round(pe[eid]["max_nve_35K_plus"], 4) for eid in pos_ids},
        "negative_experiments": {eid: round(pe[eid]["max_nve_35K_plus"], 4) for eid in neg_ids},
        "positive_min_nve": round(pos_min, 4),
        "negative_max_nve": round(neg_max, 4),
        "margin": round(margin, 4),
        "perfect_separation": separated,
        "auroc": round(auroc, 4),
        "auroc_bootstrap": boot,
        "accuracy_at_best_threshold": k_correct / n,
        "accuracy_clopper_pearson_95_ci": list(acc_ci),
        "accuracy_ci_note": f"Clopper-Pearson exact binomial CI for {k_correct}/{n} correct"
    }
    return result

# Analyze all three tasks
task_b1 = analyze_task(v3["task_b1"], "B1: predict CIVO collapse (009b=negative)")
task_b2 = analyze_task(v3["task_b2"], "B2: predict CIVO collapse (009b=positive)")
task_a = analyze_task(v3["task_a"], "A: detect trained dt_emb presence")

supplement = {
    "meta": {
        "source": "nve_threshold_v3.json",
        "purpose": "Supplement AUROC CI analysis for small-sample perfect separation",
        "v3_ci_method": "Bootstrap percentile, 1000 resamples, seed=42",
        "v3_ci_problem": "With perfect separation and n=8, all valid bootstrap resamples yield AUROC=1.0, so CI=[1.0,1.0] is mathematically correct but uninformative",
        "supplement_ci_method": "Clopper-Pearson exact binomial CI on classification accuracy (more informative for perfect separation)",
    },
    "task_a": task_a,
    "task_b1": task_b1,
    "task_b2": task_b2,
    "paper_recommendation": {
        "what_to_report": [
            "Sample sizes: n, n_positive, n_negative",
            "Separation margin: positive_min - negative_max",
            "Exact binomial (Clopper-Pearson) 95% CI on classification accuracy",
            "Note that AUROC CI=[1,1] is trivially correct under perfect separation",
            "Bootstrap details: n_resamples, n_skipped (single-class), all valid=1.0"
        ],
        "why": "AUROC CI methods (DeLong, bootstrap percentile, BCa) all degenerate to [1,1] under perfect separation. The Clopper-Pearson CI on accuracy is the more informative uncertainty quantification for small n."
    }
}

outpath = os.path.join(ARTIFACTS, "nve_threshold_v3_ci_supplement.json")
with open(outpath, "w") as f:
    json.dump(supplement, f, indent=2)

# Print summary
print("=" * 70)
print("AUROC CI SUPPLEMENT ANALYSIS")
print("=" * 70)

for task in [task_a, task_b1, task_b2]:
    print(f"\n--- {task['task']} ---")
    print(f"  n={task['n']} (pos={task['n_positive']}, neg={task['n_negative']})")
    print(f"  pos_min={task['positive_min_nve']}, neg_max={task['negative_max_nve']}, margin={task['margin']}")
    print(f"  Perfect separation: {task['perfect_separation']}")
    print(f"  AUROC={task['auroc']}")
    b = task['auroc_bootstrap']
    print(f"  Bootstrap AUROC: CI=[{b['ci_lower']}, {b['ci_upper']}], "
          f"mean={b['mean']}, std={b['std']}, "
          f"valid={b['n_valid']}, skipped={b['n_skipped']}, "
          f"unique_values={b['n_unique_values']}")
    print(f"  Accuracy Clopper-Pearson 95% CI: {task['accuracy_clopper_pearson_95_ci']}")
    print(f"  ({task['accuracy_ci_note']})")

print(f"\nSaved to {outpath}")
