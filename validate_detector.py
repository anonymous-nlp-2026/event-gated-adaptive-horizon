import json
import os

def load_metrics(logdir):
    """Load eval_return and value_mean timelines from metrics.jsonl"""
    path = os.path.join(logdir, "metrics.jsonl")
    if not os.path.exists(path):
        return None, None
    
    eval_data = {}  # step -> eval_return
    value_data = {}  # step -> value_mean
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except:
                continue
            step = d.get("step")
            if step is None:
                continue
            if "eval_return" in d:
                eval_data[step] = d["eval_return"]
            if "value_mean" in d:
                value_data[step] = d["value_mean"]
    
    # Get steps where both eval_return and value_mean exist
    common_steps = sorted(set(eval_data.keys()) & set(value_data.keys()))
    
    eval_timeline = [(s, eval_data[s]) for s in common_steps]
    value_timeline = [(s, value_data[s]) for s in common_steps]
    
    return eval_timeline, value_timeline

def run_detector(eval_timeline, value_timeline):
    """
    Run improved CIVO detector:
    - ve_ratio = value_mean / eval_return
    - condition 1: ve_ratio >= 0.49
    - condition 2: delta_ve > 0.1
    - condition 3: prev_delta_ve > 0.1 AND delta_ve > prev_delta_ve (acceleration)
    - condition 4: last 3 evals, >=2 declined
    """
    checkpoints = []
    prev_ve_ratio = None
    prev_delta_ve = None
    eval_history = []
    trigger_step = None
    
    for i in range(len(eval_timeline)):
        step = eval_timeline[i][0]
        eval_ret = eval_timeline[i][1]
        value_mean = value_timeline[i][1]
        
        if eval_ret <= 0:
            checkpoints.append({
                "step": step,
                "eval_return": eval_ret,
                "value_mean": value_mean,
                "ve_ratio": None,
                "delta_ve": None,
                "prev_delta_ve": None,
                "acceleration": False,
                "eval_decline_count": 0,
                "all_met": False,
                "note": "eval_return <= 0, skipped"
            })
            continue
        
        ve_ratio = value_mean / eval_ret
        eval_history.append(eval_ret)
        
        if prev_ve_ratio is None:
            checkpoints.append({
                "step": step,
                "eval_return": eval_ret,
                "value_mean": value_mean,
                "ve_ratio": ve_ratio,
                "delta_ve": None,
                "prev_delta_ve": None,
                "acceleration": False,
                "eval_decline_count": 0,
                "all_met": False,
                "note": "first checkpoint"
            })
            prev_ve_ratio = ve_ratio
            continue
        
        delta_ve = ve_ratio - prev_ve_ratio
        
        # Eval decline check (2/3 rule)
        eval_decline_count = 0
        if len(eval_history) >= 3:
            for j in range(-1, -3, -1):
                if eval_history[j] < eval_history[j-1]:
                    eval_decline_count += 1
            eval_decline = (eval_decline_count >= 2)
        elif len(eval_history) >= 2:
            if eval_history[-1] < eval_history[-2]:
                eval_decline_count = 1
            eval_decline = (eval_history[-1] < eval_history[-2])
        else:
            eval_decline = False
        
        # Check all conditions
        cond1 = ve_ratio >= 0.49
        cond2 = delta_ve > 0.1
        cond3 = (prev_delta_ve is not None and prev_delta_ve > 0.1 and delta_ve > prev_delta_ve)
        cond4 = eval_decline
        
        all_met = cond1 and cond2 and cond3 and cond4
        
        if all_met and trigger_step is None:
            trigger_step = step
        
        checkpoints.append({
            "step": step,
            "eval_return": round(eval_ret, 4),
            "value_mean": round(value_mean, 6),
            "ve_ratio": round(ve_ratio, 6),
            "delta_ve": round(delta_ve, 6),
            "prev_delta_ve": round(prev_delta_ve, 6) if prev_delta_ve is not None else None,
            "acceleration": cond3,
            "eval_decline_count": eval_decline_count,
            "cond1_ve_thresh": cond1,
            "cond2_delta_ve": cond2,
            "cond3_accel": cond3,
            "cond4_eval_decline": cond4,
            "all_met": all_met
        })
        
        prev_delta_ve = delta_ve
        prev_ve_ratio = ve_ratio
    
    return trigger_step, checkpoints

# Define experiments
LOGBASE = "<PROJECT_ROOT>/logdir"
experiments = [
    {"exp_id": "exp_008d_s0", "logdir": f"{LOGBASE}/exp_008d_subsample_on_h8", "expected": "TP"},
    {"exp_id": "exp_010b_cheetah", "logdir": f"{LOGBASE}/exp_010b_cheetah_civo_trigger", "expected": "TP"},
    {"exp_id": "exp_008d_s3", "logdir": f"{LOGBASE}/exp_008d_civo_trigger_s3", "expected": "TP"},
    {"exp_id": "exp_008f_s3", "logdir": f"{LOGBASE}/exp_008f_s3_no_dt_embed_walker", "expected": "TN"},
    {"exp_id": "exp_009c_008f_s2", "logdir": f"{LOGBASE}/exp_009c_008f_walker_s2", "expected": "TN"},
    {"exp_id": "exp_009h_008f_s0", "logdir": f"{LOGBASE}/exp_009h_008f_walker_s0", "expected": "TN"},
]

results = {}
for exp in experiments:
    exp_id = exp["exp_id"]
    logdir = exp["logdir"]
    expected = exp["expected"]
    
    eval_tl, value_tl = load_metrics(logdir)
    
    if eval_tl is None or len(eval_tl) == 0:
        results[exp_id] = {
            "logdir": logdir,
            "trigger_step": None,
            "expected": expected,
            "verdict": "data_unavailable",
            "checkpoints": []
        }
        continue
    
    trigger_step, checkpoints = run_detector(eval_tl, value_tl)
    
    if expected == "TP":
        verdict = "TP" if trigger_step is not None else "FN"
    else:  # expected TN
        verdict = "TN" if trigger_step is None else "FP"
    
    results[exp_id] = {
        "logdir": logdir,
        "trigger_step": trigger_step,
        "expected": expected,
        "verdict": verdict,
        "num_checkpoints": len(checkpoints),
        "checkpoints": checkpoints
    }
    
    print(f"{exp_id}: trigger_step={trigger_step}, expected={expected}, verdict={verdict}, n_ckpt={len(checkpoints)}")

# Summary
tp = sum(1 for r in results.values() if r["verdict"] == "TP")
fn = sum(1 for r in results.values() if r["verdict"] == "FN")
tn = sum(1 for r in results.values() if r["verdict"] == "TN")
fp = sum(1 for r in results.values() if r["verdict"] == "FP")
unavail = sum(1 for r in results.values() if r["verdict"] == "data_unavailable")

positive_total = tp + fn
negative_total = tn + fp

# Calculate mean lead (for TP cases, how early before performance peaks does it trigger)
lead_steps = []
for exp_id, r in results.items():
    if r["verdict"] == "TP" and r["checkpoints"]:
        # Find peak eval step
        peak_step = max(r["checkpoints"], key=lambda c: c.get("eval_return", 0) if c.get("eval_return") else 0)["step"]
        lead = peak_step - r["trigger_step"]
        lead_steps.append(lead)

mean_lead = sum(lead_steps) / len(lead_steps) if lead_steps else None

summary = {
    "TPR": f"{tp}/{positive_total}" if positive_total > 0 else "N/A",
    "FPR": f"{fp}/{negative_total}" if negative_total > 0 else "N/A",
    "mean_lead_K": round(mean_lead / 1000, 1) if mean_lead is not None else None,
    "details": f"TP={tp}, FN={fn}, TN={tn}, FP={fp}, unavailable={unavail}"
}

output = {
    "version": 3,
    "date": "2026-05-17",
    "code_status": {
        "ve_threshold": 0.49,
        "eval_decline_rule": "2/3",
        "confirmed_at": "2026-05-17"
    },
    "results": results,
    "summary": summary
}

# Save to file
outpath = "<PROJECT_ROOT>/detector_validation_v3.json"
with open(outpath, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n=== SUMMARY ===")
print(f"TPR: {summary['TPR']}")
print(f"FPR: {summary['FPR']}")
print(f"Mean lead: {summary['mean_lead_K']}K steps")
print(f"Details: {summary['details']}")
print(f"\nSaved to {outpath}")
