import json

with open('<PROJECT_ROOT>/detector_validation_v3.json') as f:
    data = json.load(f)

for exp_id, r in data['results'].items():
    print(f'=== {exp_id} === trigger={r["trigger_step"]} verdict={r["verdict"]}')
    for c in r['checkpoints']:
        step = c['step']
        ve = c.get('ve_ratio')
        dve = c.get('delta_ve')
        pdve = c.get('prev_delta_ve')
        accel = c.get('cond3_accel', c.get('acceleration', '-'))
        edec = c.get('eval_decline_count', '-')
        met = c.get('all_met', '-')
        ve_s = f"{ve:.4f}" if ve is not None else "None"
        dve_s = f"{dve:.4f}" if dve is not None else "None"
        pdve_s = f"{pdve:.4f}" if pdve is not None else "None"
        print(f'  step={step:>6} ve={ve_s:>8} dve={dve_s:>8} pdve={pdve_s:>8} accel={str(accel):>5} edec={edec} met={met}')
