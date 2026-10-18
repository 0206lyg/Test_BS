#!/usr/bin/env python3
"""Actual serial LAMMPS integration smoke; no physics or viscosity gate claim."""
import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--lmp', type=Path, required=True)
p.add_argument('--plugin', type=Path, required=True)
p.add_argument('--table', type=Path, required=True)
p.add_argument('--project', type=Path, default=Path(__file__).resolve().parent.parent/'lammps_slurry')
p.add_argument('--output', type=Path, required=True, help='Directory for generated run logs and trajectories')
p.add_argument('--summary', type=Path, default=None, help='Optional summary path; defaults inside --output')
a = p.parse_args()
for key in ('lmp','plugin','table','project'): setattr(a,key,getattr(a,key).resolve())
sys.path.insert(0,str(a.project/'tests'))
import validate_implicit as base
import validate_rheology as rheology
fixtures = Path(__file__).resolve().parent/'runtime_fixtures'
root = a.output.resolve(); root.mkdir(parents=True,exist_ok=True)
summary_path = a.summary.resolve() if a.summary else root/'bulk_smoke_summary.json'
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
summary = {'schema_version':1,'scope':'Serial actual-plugin integration/stability smoke only','aspect_ratio':10,'source_and_interpolation_physics_ready':False,'cases':[], 'limitations':['Coarse smoke table has no quantitative viscosity validity.','No equilibrium or bulk viscosity convergence established.','No MPI or scaling test.'], 'inputs':{str(f):sha(f) for f in (a.lmp,a.plugin,a.table)}}
exitcode = 0
try:
    provenance_path = Path(str(a.table)+'.json')
    provenance = json.loads(provenance_path.read_text())
    if provenance.get('profile') != 'smoke' or provenance.get('physics_ready') is not False:
        raise ValueError('This harness requires an explicitly unvalidated smoke-profile table.')
    summary['table_provenance'] = provenance
    summary['inputs'][str(provenance_path)] = sha(provenance_path)
    with a.table.open('rb') as f:
        if f.readline().decode().strip() != 'SLURRY_SPD_LOGFOURIER_V2': raise ValueError('Smoke requires the explicit-join SPD table V2 format')
        values=f.readline().decode().split()
        summary['table_grid']={'ar':float(values[0]),'theta_nodes':int(values[3]),'phi_nodes':int(values[4]),'gap_nodes':int(values[5])}
        joins=f.readline().decode().split()
        if len(joins)!=int(joins[0])+1: raise ValueError('Malformed matching-join header')
        summary['table_grid']['matching_joins']=[float(x) for x in joins[1:]]
        summary['table_grid']['format']='SLURRY_SPD_LOGFOURIER_V2'
    for name in ('two_asymmetric_3nm','eight_connected_stack'):
        folder = root/name;folder.mkdir(parents=True,exist_ok=True)
        for filename in ('fixture.json','particles.data'):
            shutil.copyfile(fixtures/name/filename,folder/filename)
        config = json.loads((folder/'fixture.json').read_text())
        record = {'name':name,'status':'running','steps':config['steps'],'dt_s':config['timestep_s'],'shear_rate_per_s':config['shear_rate_per_s'],'particle_count':len(config['particles']),'fixture_sha256':sha(folder/'particles.data')}
        summary['cases'].append(record)
        lines = [f'plugin load "{a.plugin}"', 'units si','atom_style ellipsoid','boundary p p p','atom_modify map array','processors 1 1 1','newton on','read_data particles.data','comm_modify vel yes','pair_style zero 3.75e-6','pair_coeff * *','neighbor 3e-8 bin','neigh_modify every 1 delay 0 check yes','fix shear all deform 0 xy erate 100 remap v units box',
          f'fix integrate all slurry/implicit 0.01 100 7.5e-8 1.5e-10 1.0 lubrication finite_gap_fast finite_gap_fast_table "{a.table}" finite_gap_range 0.5 contact_gap 3e-9 friction 0.1 80 solver_preconditioner ilu solver_layout root solver_forcing fixed solver_cache on history history.csv 1 0',
          'timestep 1e-6','compute orientation all property/atom quatw quati quatj quatk','dump trajectory all custom 1 trajectory.dump id xu yu zu vx vy vz ix iy iz c_orientation[*]','dump_modify trajectory sort id format float %.17g','thermo 1','thermo_style custom step time atoms f_integrate[4]','thermo_modify lost error flush yes format float %.17g','run 20']
        (folder/'in.smoke').write_text('\n'.join(lines)+'\n',encoding='ascii')
        record['input_sha256']=sha(folder/'in.smoke')
        env=os.environ.copy();env['OMP_NUM_THREADS']='1'
        begin=time.perf_counter()
        with (folder/'stdout.txt').open('w') as stream:
            result=subprocess.run([str(a.lmp),'-log','log.lammps','-in','in.smoke'],cwd=folder,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        record['formal_exit_code']=result.returncode;record['elapsed_seconds']=time.perf_counter()-begin
        if result.returncode: raise RuntimeError(name+': LAMMPS failed; see stdout.txt')
        frames=base.parse_dump(folder/'trajectory.dump')
        history=rheology.parse_history(folder/'history.csv')
        expected=set(range(config['steps']+1))
        if set(frames)!=expected or set(history)!=expected:raise AssertionError(name+': missing trajectory/history steps')
        for frame in frames.values():
            if len(frame['atoms'])!=record['particle_count']:raise AssertionError(name+': lost particles')
            for particle in frame['atoms'].values():
                if not all(math.isfinite(v) for v in particle.values()):raise AssertionError(name+': nonfinite trajectory')
                q=[particle['c_orientation[%d]'%k] for k in range(1,5)]
                if abs(sum(v*v for v in q)-1)>2e-10:raise AssertionError(name+': nonunit quaternion')
        for step,row in history.items():
            rheology.check_force_residual(row,name+' step'+str(step))
            for key in ('sigma_xy_Pa','eta_relative','sigma_finite_gap_Pa','finite_gap_correction_power_W','finite_gap_strain_bound_min'):
                if not math.isfinite(row[key]):raise AssertionError(name+': nonfinite '+key)
        pairs=[row['lubrication_pairs'] for row in history.values()]
        if min(pairs)<1:raise AssertionError(name+': fast pair path was inactive')
        if not any(abs(row['sigma_finite_gap_Pa'])>0 for row in history.values()):raise AssertionError(name+': missing finite-gap stress contribution')
        record['energy']=rheology.check_energy(history,name)
        record['min_active_pairs']=min(pairs);record['max_active_pairs']=max(pairs)
        record['max_force_residual_N']=max(row['force_residual_N'] for row in history.values())
        record['max_force_residual_over_tolerance']=max(row['force_residual_N']/row['force_tolerance_N'] for row in history.values())
        record['minimum_strain_bound']=min(row['finite_gap_strain_bound_min'] for row in history.values())
        record['final_eta_relative_smoke_only']=history[config['steps']]['eta_relative']
        record['final_sigma_finite_gap_Pa_smoke_only']=history[config['steps']]['sigma_finite_gap_Pa']
        record['status']='pass'
        summary_path.write_text(json.dumps(summary,indent=2)+'\n')
except Exception as e:
    exitcode=2;summary['error']=str(e)
    if summary['cases'] and summary['cases'][-1]['status']=='running':summary['cases'][-1]['status']='fail'
summary['formal_exit_code']=exitcode
summary['not_run']=[name for name in ('two_asymmetric_3nm','eight_connected_stack') if name not in [c['name'] for c in summary['cases']]]
summary['status']='pass' if exitcode==0 else 'stopped_at_failure'
summary_path.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
sys.exit(exitcode)
