#!/usr/bin/env python3
"""Bounded production replay; immutable binary and exact fixture provenance."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(out):
    with (out/'history.csv').open() as f:
        rows=list(csv.DictReader(f))
    report={'recorded_rows':len(rows),'last_step':int(rows[-1]['step']) if rows else None}
    if not rows:
        return report
    for key in rows[0]:
        if key=='time_integrator':
            report[key]=sorted(set(r[key] for r in rows))
            continue
        values=[float(r[key]) for r in rows]
        finite=[x for x in values if math.isfinite(x)]
        report[key]={'min':min(finite) if finite else None,
                     'max':max(finite) if finite else None,
                     'mean':statistics.mean(finite) if finite else None,
                     'last':values[-1] if math.isfinite(values[-1]) else None,
                     'nonfinite':len(values)-len(finite)}
    completed=[r for r in rows if int(r['step'])>0]
    report['mean_completed_step_s']=statistics.mean(float(r['profile_total_s']) for r in completed) if completed else None
    report['max_recorded_overlap_fraction']=max(float(r[k]) for r in rows for k in (
        'max_overlap_fraction_min_semiaxis','actual_endpoint_max_overlap_fraction_min_semiaxis'))
    report['overlap_below_five_percent']=report['max_recorded_overlap_fraction']<.05
    report['max_relative_power_balance_error']=max(abs(float(r['power_balance_error_W']))/
        max(abs(float(r['hydro_dissipation_W'])),abs(float(r['contact_power_W'])),abs(float(r['stress_power_W'])),1e-300) for r in rows)
    report['steady_state_certified']=False
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--plugin',type=Path,required=True)
    p.add_argument('--lmp',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--dt',type=float,default=2.5e-6)
    p.add_argument('--kn',type=float,required=True)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--trace',action='store_true')
    a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    fixture=a.fixture.resolve();plugin=a.plugin.resolve()
    shutil.copyfile(fixture/'particles.data',out/'particles.data')
    for filename in ['parameters.json','packing_clearance.json']:
        if (fixture/filename).exists():shutil.copyfile(fixture/filename,out/('fixture_'+filename))
    text=(fixture/'in.bridge').read_text()
    text=re.sub(r'^plugin load .*$', 'plugin load ${slurry_plugin}',text,flags=re.M)
    text=re.sub(r'^run \d+$',f'run {a.steps}',text,flags=re.M)
    text=re.sub(r'^timestep .*$',f'timestep {a.dt:.17g}',text,flags=re.M)
    text=re.sub(r'history history.csv \d+ ','history history.csv 1 ',text)
    text=re.sub(r'^thermo \d+$','thermo 40',text,flags=re.M)
    text=re.sub(r'dump trajectory all custom \d+', 'dump trajectory all custom 400',text)
    lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('fix hydro '):
            tokens=line.split();b=tokens.index('slurry/bridge')
            tokens[b+5]=str(a.kn)
            tokens[tokens.index('contact')+2]=str(.8*a.kn)
            tokens[tokens.index('rolling')+2]=str(.8*a.kn)
            lines[i]=' '.join(tokens)
    (out/'in.bridge').write_text('\n'.join(lines)+'\n')
    metadata={'fixture':str(fixture),'plugin':str(plugin),'plugin_sha256':sha(plugin),
        'particles_sha256':sha(out/'particles.data'),'input_sha256':sha(out/'in.bridge'),
        'steps':a.steps,'dt_s':a.dt,'kn_N_per_m':a.kn,'kt_kr_N_per_m':.8*a.kn,'threads':a.threads,
        'history_interval_steps':1,'backend':'nearfield','contact_scheme':'imex',
        'steady_state_certified':False,'attraction_tested':False}
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS=str(a.threads),OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS=str(a.threads))
    env['SLURRY_BRIDGE_TRACE']='1' if a.trace else '0'
    start=time.monotonic()
    with (out/'console.log').open('w') as log:
        proc=subprocess.run([str(a.lmp.resolve()),'-var','slurry_plugin','"'+str(plugin)+'"','-in','in.bridge'],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
    report=summarize(out) if (out/'history.csv').exists() else {}
    report.update(metadata);report.update(returncode=proc.returncode,wall_seconds=time.monotonic()-start)
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ['returncode','last_step','wall_seconds','mean_completed_step_s','max_recorded_overlap_fraction']}),flush=True)
    return proc.returncode

if __name__=='__main__':
    raise SystemExit(main())
