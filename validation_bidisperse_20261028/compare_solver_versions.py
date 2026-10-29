#!/usr/bin/env python3
"""Compare identical fixtures/settings across the bounded line-search change."""
from pathlib import Path
import csv
import json
import statistics

root=Path(__file__).resolve().parent
results=[]
for old,new,label in [
    ('ar5_n1000_k30_pilot','ar5_n1000_k30_pilot_v2','AR5_N1000_kn30_completed_200_steps'),
    ('ar5_n1000_k300_trace','ar5_n1000_k300_trace_v2','AR5_N1000_kn300_common_accepted_prefix')]:
    a=list(csv.DictReader((root/old/'history.csv').open()))
    b=list(csv.DictReader((root/new/'history.csv').open()))
    am=json.loads((root/old/'run_metadata.json').read_text());bm=json.loads((root/new/'run_metadata.json').read_text())
    for k in ['particles_sha256','kn_N_per_m','kt_kr_N_per_m','dt_s','threads']:
        assert am[k]==bm[k],k
    count=min(len(a),len(b));a=a[:count];b=b[:count]
    assert all(x['step']==y['step'] for x,y in zip(a,b))
    r={'case':label,'v1':old,'v2':new,'common_last_step':int(a[-1]['step']),
       'identical_fixture_and_numerical_parameters':True,
       'timing_caveat':'Single-thread processes ran concurrently with other validation/build work; wall timing is indicative, not an isolated speed benchmark.'}
    for key in ['profile_total_s','newton_iterations','linear_iterations','geometry_iterations']:
        x=statistics.mean(float(row[key]) for row in a[1:]);y=statistics.mean(float(row[key]) for row in b[1:])
        r[key]={'v1_mean':x,'v2_mean':y,'v2_over_v1':y/x if x else None}
    for key in ['eta_relative','max_overlap_fraction_min_semiaxis','solver_relative_residual']:
        r[key+'_maximum_absolute_difference']=max(abs(float(x[key])-float(y[key])) for x,y in zip(a,b))
    results.append(r)
output={'comparisons':results,'physical_equations_changed':False,
        'newton_iteration_budget_changed':False,'timestep_changed':False,'tolerances_relaxed':False}
(root/'SOLVER_COST_COMPARISON.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
