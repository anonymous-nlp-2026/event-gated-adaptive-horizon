import json

with open('<PROJECT_ROOT>/detector_validation_v3.json') as f:
    data = json.load(f)

for exp_id, r in data['results'].items():
    if r['verdict'] == 'data_unavailable':
        print(f'{exp_id}: data_unavailable')
        continue
    print(f'=== {exp_id} === trigger={r["trigger_step"]} verdict={r["verdict"]}')
    for c in r['checkpoints']:
        er = c.get('eval_return', 'N/A')
        vm = c.get('value_mean', 'N/A')
        er_s = f"{er:.2f}" if isinstance(er, (int, float)) else str(er)
        vm_s = f"{vm:.4f}" if isinstance(vm, (int, float)) else str(vm)
        met = c.get('all_met', False)
        print(f"  step={c['step']:>6}  eval_ret={er_s:>10}  val_mean={vm_s:>12}  met={met}")
    # Find peak eval
    eval_rets = [(c['step'], c['eval_return']) for c in r['checkpoints'] if isinstance(c.get('eval_return'), (int, float))]
    if eval_rets:
        peak = max(eval_rets, key=lambda x: x[1])
        print(f"  PEAK eval: step={peak[0]} eval_return={peak[1]:.2f}")
        if r['trigger_step']:
            print(f"  Lead = peak({peak[0]}) - trigger({r['trigger_step']}) = {peak[0] - r['trigger_step']}")
    print()
