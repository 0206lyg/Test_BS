"""Compare FMM action on a saved exact-solve density; never solve a new system."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('GFORTRAN_UNBUFFERED_ALL','1')
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from backend import Backend
from geometry import build_mesh
from matrix_free import MatrixFreeContext
from memory_bounded_dense import _check_cgroup_headroom
from run_reference import logged_fortran, array_sha256, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh diagnostic output')
    source=json.loads(args.reference.read_text())
    if (source.get('status')!='complete' or source.get('node_count')!=300 or
        source.get('order')!=4 or source.get('level')!=0 or source.get('h_nm')!=150. or
        source.get('aspect_ratio')!=20. or source.get('isolated_body') is not None):
        raise ValueError('Only the recorded N300 p4/l0 pair reference at150nm is allowed')
    if sha(args.library)!=source['library_sha256']:
        raise ValueError('Selected library differs from saved exact reference')
    density_path=Path(source['solution_file'])
    if sha(density_path)!=source['solution_sha256']:
        raise ValueError('Saved checkpoint SHA mismatch')
    for name in ('backend.py','geometry.py'):
        if sha(Path(__file__).with_name(name))!=source['source_hashes'][name]:
            raise ValueError('Saved geometry/backend implementation mismatch')
    record={'schema':'saved_exact_density_fmm_action_comparison_v1',
            'physical_accuracy_qualified':False,'contains_new_stokes_solve':False,
            'scope':'One saved N300 AR20 pair at150nm; collocation operator action only, not independent off-grid accuracy',
            'reference':{'path':str(args.reference.resolve()),'sha256':sha(args.reference)},
            'library_sha256':sha(args.library),'density_sha256':sha(density_path),
            'far_order_increment':48,'quadrature_tolerance':source['quadrature_tolerance'],
            'per_rhs':[],'source_sha256':{name:sha(Path(__file__).with_name(name)) for name in
                ('probe_saved_fmm_action.py','matrix_free.py','backend.py','geometry.py')},
            'status':'preflight'}
    started=time.monotonic()
    try:
        record['cgroup_admission']=_check_cgroup_headroom(512*1024**2,250_000_000)
        write_json(args.output,record)
        backend=Backend(args.library)
        mesh=backend.coefficients(build_mesh(150.,4,0,backend.discretization(4)[0],
            patch_descriptors=source['patch_descriptors']))
        with np.load(density_path,allow_pickle=False) as saved:
            for key in ('srcvals','weights','body_ids','norders','ixyzs','iptype'):
                if not np.array_equal(saved[key],mesh[key]):
                    raise ValueError('Saved/rebuilt mesh differs: '+key)
            x=saved['density'].reshape(900,17).copy()
            rhs=saved['rhs'].copy()
        if array_sha256(rhs)!=source['rhs_sha256']:
            raise ValueError('Saved RHS SHA mismatch')
        t=time.monotonic()
        exact=logged_fortran(args.output.with_suffix('.exact.log'),lambda:
            backend.matrix(mesh,source['quadrature_tolerance'],source['alpha'],source['beta'],max_memory_gb=.1),backend.lib)
        ax=exact@x
        record['exact_matrix_seconds']=time.monotonic()-t
        record['exact_matrix_bytes']=exact.nbytes
        record['exact_action_rhs_residual_by_rhs']=(np.linalg.norm(ax-rhs,axis=0)/np.linalg.norm(rhs,axis=0)).tolist()
        context=logged_fortran(args.output.with_suffix('.prepare.log'),lambda:
            MatrixFreeContext(backend,mesh,eps=source['quadrature_tolerance'],
                              alpha=source['alpha'],beta=source['beta'],
                              far_order_increment=48,max_memory_gb=.25),backend.lib)
        record['context']=context.metadata()
        record['status']='applying_saved_density'
        write_json(args.output,record)
        output=np.empty_like(ax)

        def apply(j):
            t=time.monotonic();output[:,j]=context.matvec(x[:,j]);seconds=time.monotonic()-t
            record['per_rhs'].append({'rhs':j,'mode':source['mode_names'][j],
                'seconds':seconds,
                'relative_operator_action_error':float(np.linalg.norm(output[:,j]-ax[:,j])/np.linalg.norm(ax[:,j])),
                'true_rhs_relative_residual':float(np.linalg.norm(output[:,j]-rhs[:,j])/np.linalg.norm(rhs[:,j])),
                'maximum_pointwise_error_over_boundary_mode_scale':float(np.max(np.linalg.norm((output[:,j]-ax[:,j]).reshape(300,3),axis=1))/np.max(np.linalg.norm(rhs[:,j].reshape(300,3),axis=1)))})
            write_json(args.output,record)
            return seconds

        first=logged_fortran(args.output.with_suffix('.first_action.log'),lambda:apply(0),backend.lib)
        record.update(first_action_seconds=first,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      completed_rhs=len(record['per_rhs']),context=context.metadata())
        write_json(args.output,record)
        print(json.dumps({'first_action_seconds':first,'peak_rss_kib':record['peak_rss_kib']}),flush=True)
        if first>15 or record['peak_rss_kib']>512*1024:
            raise RuntimeError('First application exceeded the light-probe time/RSS envelope')
        logged_fortran(args.output.with_suffix('.remaining_actions.log'),lambda:[apply(j) for j in range(1,17)],backend.lib)
        record.update(status='complete',completed_rhs=17,context=context.metadata(),
            maximum_relative_operator_action_error=max(v['relative_operator_action_error'] for v in record['per_rhs']),
            maximum_true_rhs_relative_residual=max(v['true_rhs_relative_residual'] for v in record['per_rhs']),
            full17_action_relative_frobenius_error=float(np.linalg.norm(output-ax)/np.linalg.norm(ax)),
            total_action_seconds=sum(v['seconds'] for v in record['per_rhs']),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            wall_seconds=time.monotonic()-started)
    except BaseException as exc:
        record.update(status='failed',failure_type=type(exc).__name__,failure_message=str(exc),
                      wall_seconds=time.monotonic()-started,
                      completed_rhs=len(record['per_rhs']),
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(args.output,record)
        raise
    write_json(args.output,record)
    print(json.dumps({key:record[key] for key in ['status','maximum_relative_operator_action_error',
        'maximum_true_rhs_relative_residual','total_action_seconds','wall_seconds','peak_rss_kib']}),flush=True)


if __name__=='__main__':
    main()
