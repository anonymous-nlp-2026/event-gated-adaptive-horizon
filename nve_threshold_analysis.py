import json
import os
import numpy as np

LOGDIR = "<PROJECT_ROOT>/logdir"
PAPER_DATA = os.path.join(LOGDIR, "paper_data_walker_walk.json")

# Ground truth labels
GROUND_TRUTH = {
    "exp_001_baseline": "healthy",
    "exp_006_fixed_dt2": "collapsed",
    "exp_007_fixed_dt4": "collapsed",
    "exp_008a_no_subsample_dt2": "healthy",
    "exp_008c_no_subsample_dt2_h8": "collapsed",
    "exp_008d_subsample_on_h8": "collapsed",
    "exp_008e_v3_walker_s1": "healthy",
    "exp_008e_v3_walker_s2": "collapsed",
    "exp_008f_no_dt_embed_subsample_on_h8": "healthy",
}

# Mapping from task exp_id to actual logdir name (for metrics.jsonl fallback)
DIR_MAP = {
    "exp_001_baseline": "baseline_s1",
    "exp_006_fixed_dt2": "fixed_dt2_v1_s1",
    "exp_007_fixed_dt4": "fixed_dt4_v2_s1",
    "exp_008a_no_subsample_dt2": "exp_008a_fixed_dt2_no_subsample",
    "exp_008c_no_subsample_dt2_h8": "exp_008c_fixed_dt2_no_subsample_h8",
    "exp_008d_subsample_on_h8": "exp_008d_subsample_on_h8",
    "exp_008e_v3_walker_s1": "exp_008e_v3_walker_s1",
    "exp_008e_v3_walker_s2": "exp_008e_v3_walker_s2",
    "exp_008f_no_dt_embed_subsample_on_h8": "exp_008f_no_dt_embed_subsample_on_h8",
}

def load_from_paper_data(paper_data, exp_id):
    """Try to load trajectory from paper_data JSON."""
    exps = paper_data.get("experiments", {})
    if exp_id in exps:
        traj = exps[exp_id].get("trajectory", [])
        return [(p["step"], p.get("ve_ratio")) for p in traj if p.get("ve_ratio") is not None]
    return None

def load_from_metrics_jsonl(logdir_name):
    """Load eval_return and value_mean from metrics.jsonl, compute ve_ratio."""
    path = os.path.join(LOGDIR, logdir_name, "metrics.jsonl")
    if not os.path.exists(path):
        return None
    
    eval_data = {}  # step -> eval_return
    value_data = {}  # step -> value_mean
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = rec.get("step")
            if step is None:
                continue
            if "eval_return" in rec:
                eval_data[step] = rec["eval_return"]
            if "value_mean" in rec:
                value_data[step] = rec["value_mean"]
    
    # Compute ve_ratio at steps where both exist
    results = []
    for step in sorted(set(eval_data.keys()) & set(value_data.keys())):
        er = eval_data[step]
        vm = value_data[step]
        if er != 0:
            ve = vm / er
        else:
            ve = float('inf') if vm > 0 else 0.0
        results.append((step, ve))
    return results

# Load paper data
with open(PAPER_DATA) as f:
    paper_data = json.load(f)

baseline_ve_ref = paper_data["metadata"]["baseline_ve_reference"]
baseline_ve = {int(k): v for k, v in baseline_ve_ref.items()}

# Collect all experiment data
exp_data = {}
missing_exps = []

for exp_id in GROUND_TRUTH:
    # Try paper data first
    traj = load_from_paper_data(paper_data, exp_id)
    if traj:
        exp_data[exp_id] = traj
        continue
    
    # Fallback to metrics.jsonl
    logdir_name = DIR_MAP.get(exp_id)
    if logdir_name:
        traj = load_from_metrics_jsonl(logdir_name)
        if traj:
            exp_data[exp_id] = traj
            continue
    
    missing_exps.append(exp_id)

print("=" * 80)
print("NVE THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 80)

if missing_exps:
    print(f"\n⚠ Missing experiments: {missing_exps}")

# Compute NVE for each experiment at each step
print("\n## 1. Baseline V/E Reference")
print(f"{'Step':>8} | {'Baseline V/E':>12}")
print("-" * 25)
for step in sorted(baseline_ve.keys()):
    print(f"{step:>8} | {baseline_ve[step]:>12.4f}")

print("\n## 2. Per-Experiment NVE Trajectories")
print(f"(NVE = experiment_ve_ratio / baseline_ve_ratio at same step)")

nve_data = {}  # exp_id -> {step: nve}
for exp_id, traj in exp_data.items():
    nve_data[exp_id] = {}
    for step, ve in traj:
        if step in baseline_ve and baseline_ve[step] > 0:
            nve = ve / baseline_ve[step]
            nve_data[exp_id][step] = nve

# Print NVE trajectories
for exp_id in sorted(nve_data.keys()):
    gt = GROUND_TRUTH[exp_id]
    print(f"\n### {exp_id} (GT: {gt})")
    steps = sorted(nve_data[exp_id].keys())
    for step in steps:
        nve = nve_data[exp_id][step]
        print(f"  step {step:>6}: NVE = {nve:.4f}")

# For classification, use multiple strategies
# Strategy 1: Max NVE across all steps
# Strategy 2: NVE at 105K (if available), else last available step
# Strategy 3: Mean NVE over last 3 checkpoints

print("\n## 3. Summary Statistics per Experiment")
print(f"{'Experiment':<45} | {'GT':>10} | {'Max NVE':>8} | {'Final NVE':>9} | {'Mean Last3':>10}")
print("-" * 95)

summary = {}
for exp_id in sorted(nve_data.keys()):
    gt = GROUND_TRUTH[exp_id]
    nve_vals = nve_data[exp_id]
    steps = sorted(nve_vals.keys())
    
    if not steps:
        continue
    
    max_nve = max(nve_vals.values())
    final_nve = nve_vals[steps[-1]]
    last3 = [nve_vals[s] for s in steps[-3:]]
    mean_last3 = np.mean(last3)
    
    summary[exp_id] = {
        "gt": gt,
        "max_nve": max_nve,
        "final_nve": final_nve,
        "mean_last3": mean_last3,
    }
    
    print(f"{exp_id:<45} | {gt:>10} | {max_nve:>8.4f} | {final_nve:>9.4f} | {mean_last3:>10.4f}")

# Threshold analysis
THRESHOLDS = [1.3, 1.5, 1.7, 2.0]
STRATEGIES = {
    "max_nve": "Max NVE (any step)",
    "final_nve": "Final Step NVE",
    "mean_last3": "Mean Last-3 NVE",
}

print("\n## 4. Classification Results per Threshold")

# Exclude baseline from classification (it's the reference)
classify_exps = {k: v for k, v in summary.items() if k != "exp_001_baseline"}

for strategy_key, strategy_name in STRATEGIES.items():
    print(f"\n### Strategy: {strategy_name}")
    
    for threshold in THRESHOLDS:
        tp, tn, fp, fn = 0, 0, 0, 0
        predictions = {}
        
        for exp_id, s in classify_exps.items():
            score = s[strategy_key]
            predicted = "collapsed" if score > threshold else "healthy"
            actual = s["gt"]
            predictions[exp_id] = (predicted, actual, score)
            
            if predicted == "collapsed" and actual == "collapsed":
                tp += 1
            elif predicted == "healthy" and actual == "healthy":
                tn += 1
            elif predicted == "collapsed" and actual == "healthy":
                fp += 1
            elif predicted == "healthy" and actual == "collapsed":
                fn += 1
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n  Threshold = {threshold}")
        print(f"  {'Experiment':<45} | {'Score':>7} | {'Predicted':>10} | {'Actual':>10} | {'Correct':>7}")
        print(f"  {'-'*90}")
        for exp_id in sorted(predictions.keys()):
            pred, act, score = predictions[exp_id]
            correct = "✓" if pred == act else "✗"
            print(f"  {exp_id:<45} | {score:>7.4f} | {pred:>10} | {act:>10} | {correct:>7}")
        
        print(f"\n  TP={tp} TN={tn} FP={fp} FN={fn}")
        print(f"  Accuracy={accuracy:.4f}  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

# AUROC computation
print("\n## 5. AUROC Analysis")
print("(Using max_nve as primary score, collapsed=positive)")

from collections import defaultdict

for strategy_key, strategy_name in STRATEGIES.items():
    scores = []
    labels = []  # 1=collapsed, 0=healthy
    
    for exp_id, s in classify_exps.items():
        scores.append(s[strategy_key])
        labels.append(1 if s["gt"] == "collapsed" else 0)
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    # Manual AUROC (since we may not have sklearn)
    # Sort by score descending
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]
    sorted_scores = scores[sorted_indices]
    
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        print(f"\n  {strategy_name}: Cannot compute AUROC (need both classes)")
        continue
    
    # Compute AUROC via trapezoidal rule
    tpr_list = [0.0]
    fpr_list = [0.0]
    tp_count = 0
    fp_count = 0
    
    prev_score = None
    for i, (score, label) in enumerate(zip(sorted_scores, sorted_labels)):
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1
        
        if i == len(sorted_scores) - 1 or sorted_scores[i+1] != score:
            tpr_list.append(tp_count / n_pos)
            fpr_list.append(fp_count / n_neg)
    
    auroc = np.trapz(tpr_list, fpr_list)
    
    print(f"\n  {strategy_name}:")
    print(f"    N={len(scores)}, Positives(collapsed)={n_pos}, Negatives(healthy)={n_neg}")
    print(f"    AUROC = {auroc:.4f}")
    
    # Show score distribution
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    print(f"    Collapsed scores: min={pos_scores.min():.4f} mean={pos_scores.mean():.4f} max={pos_scores.max():.4f}")
    print(f"    Healthy scores:   min={neg_scores.min():.4f} mean={neg_scores.mean():.4f} max={neg_scores.max():.4f}")

# Optimal threshold recommendation
print("\n## 6. Optimal Threshold Recommendation")

best_f1 = 0
best_config = None

for strategy_key, strategy_name in STRATEGIES.items():
    # Try many thresholds
    all_scores = [classify_exps[e][strategy_key] for e in classify_exps]
    test_thresholds = np.arange(0.5, 3.0, 0.05)
    
    for t in test_thresholds:
        tp, tn, fp, fn = 0, 0, 0, 0
        for exp_id, s in classify_exps.items():
            score = s[strategy_key]
            predicted = "collapsed" if score > t else "healthy"
            actual = s["gt"]
            if predicted == "collapsed" and actual == "collapsed": tp += 1
            elif predicted == "healthy" and actual == "healthy": tn += 1
            elif predicted == "collapsed" and actual == "healthy": fp += 1
            elif predicted == "healthy" and actual == "collapsed": fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        if f1 > best_f1 or (f1 == best_f1 and accuracy > (best_config[5] if best_config else 0)):
            best_f1 = f1
            best_config = (strategy_name, t, tp, tn, fp, fn, accuracy, precision, recall, f1)

if best_config:
    sn, t, tp, tn, fp, fn, acc, prec, rec, f1 = best_config
    print(f"\n  Best Configuration:")
    print(f"    Strategy: {sn}")
    print(f"    Threshold: {t:.2f}")
    print(f"    TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"    Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

# Also check the requested thresholds specifically
print("\n  Requested Thresholds Summary (Max NVE strategy):")
print(f"  {'Threshold':>10} | {'Acc':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6}")
print(f"  {'-'*48}")
for t in THRESHOLDS:
    tp, tn, fp, fn = 0, 0, 0, 0
    for exp_id, s in classify_exps.items():
        score = s["max_nve"]
        predicted = "collapsed" if score > t else "healthy"
        actual = s["gt"]
        if predicted == "collapsed" and actual == "collapsed": tp += 1
        elif predicted == "healthy" and actual == "healthy": tn += 1
        elif predicted == "collapsed" and actual == "healthy": fp += 1
        elif predicted == "healthy" and actual == "collapsed": fn += 1
    
    total = tp + tn + fp + fn
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"  {t:>10.1f} | {acc:>6.4f} | {prec:>6.4f} | {rec:>6.4f} | {f1:>6.4f}")

