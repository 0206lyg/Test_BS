#!/usr/bin/env python3
"""Read-only flake trajectory diagnostics (Python 3 + NumPy).

Usage: python3 diagnose_flake.py [run-directory]
Omitting the directory chooses the most recently modified complete/nonempty
flake trajectory under ./runs. It never starts a simulation or changes its input.
Outputs: flake_layering.csv and flake_diagnostics_summary.json in the run directory.

Nematic alignment is not evidence of layers. Density peaks use allowed periodic
reciprocal-lattice modes; their wavelength can be a harmonic of the layer spacing.
The seeded uniform-position null is a finite-N noise reference, not an equilibrium
hard-particle null, a physical p-value, or a pass/fail test. Velocity quantities are
center velocities, not face-surface slip or frictional dissipation.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys
from dataclasses import dataclass
import numpy as np


@dataclass
class Frame:
    step: int
    origin: np.ndarray
    lattice: np.ndarray  # columns are lattice vectors
    names: list
    values: np.ndarray

    def column(self, name):
        return self.values[:, self.names.index(name)]

    def columns(self, names):
        return self.values[:, [self.names.index(name) for name in names]]


def read_frames(path, warnings=None):
    """Stream standard/restricted-triclinic LAMMPS custom dump frames.

    A partial trailing frame from a running/aborted job is ignored with a warning.
    General triclinic 'abc origin' dumps are deliberately rejected, not misread.
    """
    warnings = [] if warnings is None else warnings
    with Path(path).open() as stream:
        while True:
            line = stream.readline()
            if not line:
                return
            if not line.strip():
                continue
            if line.strip() != 'ITEM: TIMESTEP':
                raise ValueError('Expected ITEM: TIMESTEP in trajectory')
            try:
                step = int(stream.readline())
                if stream.readline().strip() != 'ITEM: NUMBER OF ATOMS':
                    raise EOFError('missing atom-count header')
                n = int(stream.readline())
                header = stream.readline().split()
                if header[:3] != ['ITEM:', 'BOX', 'BOUNDS']:
                    raise EOFError('missing box header')
                if 'abc' in header:
                    raise ValueError('General triclinic abc dumps are unsupported; use the existing restricted-triclinic dump')
                tilted = all(x in header for x in ('xy', 'xz', 'yz'))
                box = []
                for _ in range(3):
                    fields = stream.readline().split()
                    if len(fields) < (3 if tilted else 2):
                        raise EOFError('partial box')
                    box.append([float(x) for x in fields])
                xy, xz, yz = (box[0][2], box[1][2], box[2][2]) if tilted else (0., 0., 0.)
                xlo = box[0][0] - min(0., xy, xz, xy+xz)
                xhi = box[0][1] - max(0., xy, xz, xy+xz)
                ylo = box[1][0] - min(0., yz)
                yhi = box[1][1] - max(0., yz)
                zlo, zhi = box[2][:2]
                origin = np.array([xlo, ylo, zlo])
                H = np.array([[xhi-xlo, xy, xz], [0., yhi-ylo, yz], [0., 0., zhi-zlo]])
                if np.linalg.det(H) <= 0:
                    raise ValueError('Nonpositive simulation-cell volume')
                atom_header = stream.readline().split()
                if atom_header[:2] != ['ITEM:', 'ATOMS']:
                    raise EOFError('missing atom header')
                names = atom_header[2:]
                rows = []
                for _ in range(n):
                    atom_line = stream.readline()
                    if not atom_line.endswith('\n'):
                        raise EOFError('unterminated atom row')
                    row = atom_line.split()
                    if len(row) != len(names):
                        raise EOFError('partial atom rows')
                    rows.append([float(x) for x in row])
                values = np.asarray(rows, dtype=float).reshape(n, len(names))
                if n < 2 or not np.isfinite(values).all():
                    raise ValueError('Require at least two atoms and finite dump data')
            except (EOFError, StopIteration) as error:
                warnings.append('Ignored incomplete trailing frame: '+str(error))
                return
            except ValueError as error:
                # A final empty timestep/count line is common while LAMMPS writes.
                if 'invalid literal for int()' in str(error):
                    warnings.append('Ignored incomplete trailing frame: '+str(error))
                    return
                raise
            yield Frame(step, origin, H, names, values)


def positions(frame):
    for keys, scaled in ((('x','y','z'),False), (('xu','yu','zu'),False), (('xs','ys','zs'),True)):
        if all(k in frame.names for k in keys):
            x = frame.columns(keys)
            return frame.origin + x @ frame.lattice.T if scaled else x
    raise ValueError('Trajectory lacks x/y/z, xu/yu/zu, or xs/ys/zs')


def axes(frame):
    keys = ['c_shape[%d]' % i for i in range(4,8)]
    if not all(k in frame.names for k in keys):
        keys = ['quatw','quati','quatj','quatk']
    if not all(k in frame.names for k in keys):
        return None
    q = frame.columns(keys)
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms < 1.e-14):
        raise ValueError('Invalid zero quaternion')
    w,x,y,z = (q / norms[:,None]).T
    return np.column_stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)))


def nematic(axis):
    if axis is None:
        return math.nan, np.full(3, np.nan), math.nan
    tensor = (3*axis.T @ axis/len(axis)-np.eye(3))/2
    eig, vectors = np.linalg.eigh(tensor)
    director = vectors[:,-1]
    if director[np.argmax(np.abs(director))] < 0:
        director = -director
    # The leading-eigenvalue gap warns when the selected director is poorly defined.
    return float(eig[-1]), director, float(eig[-1]-eig[-2])


def canonical_mode(m):
    m = tuple(int(v) for v in m)
    for v in m:
        if v:
            return tuple(-x for x in m) if v < 0 else m
    return None


def density_modes(H, direction, min_spacing, max_index=128, max_modes=512, exact_gradient=False):
    """Allowed reciprocal vectors, cycles/m = H^-T m (without 2*pi)."""
    invH = np.linalg.inv(H)
    longest = np.linalg.svd(H, compute_uv=False)[0]
    qmin, qmax = 2/longest, 1/min_spacing
    if exact_gradient:
        # For the existing restricted-triclinic xz cell, m=(0,0,k) is exactly along z.
        upper = min(max_index, int(math.floor(H[2,2]/min_spacing)))
        M = np.array([(0,0,k) for k in range(2,upper+1)], dtype=int)
        return M.reshape(-1,3), upper == max_index and H[2,2]/min_spacing > max_index
    if not np.isfinite(direction).all():
        return np.empty((0,3),dtype=int), False
    direction = direction / np.linalg.norm(direction)
    found = set()
    # One-dimensional ray plus immediate lattice neighbors samples a 15-degree cone.
    # No arbitrary continuous wavevectors: translating by a periodic image is invariant.
    count = max(64, min(4*max_index, int(math.ceil(2*longest*qmax))))
    shifts = [(0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    clipped = False
    for q in np.linspace(qmin, qmax, count):
        base = np.rint(H.T @ direction * q).astype(int)
        for shift in shifts:
            m = canonical_mode(base + shift)
            if m is None:
                continue
            if max(abs(v) for v in m) > max_index:
                clipped = True
                continue
            k = np.asarray(m) @ invH
            kn = np.linalg.norm(k)
            if kn > 0 and qmin*(1-1e-12) <= kn <= qmax*(1+1e-12) and abs(k @ direction)/kn >= math.cos(math.radians(15)):
                found.add(m)
    modes = sorted(found, key=lambda m: np.linalg.norm(np.asarray(m) @ invH))
    if len(modes)>max_modes:
        modes = [modes[k] for k in np.linspace(0,len(modes)-1,max_modes,dtype=int)]
        clipped = True
    return np.asarray(modes,dtype=int).reshape(-1,3), clipped


def peak_order(frac, modes):
    if not len(modes):
        return np.empty(0)
    return np.abs(np.exp(2j*np.pi*(frac @ modes.T)).mean(axis=0))


class UniformNull:
    def __init__(self, repeats=32, seed=1729):
        self.repeats = repeats
        self.seed = seed
        self.cache = {}

    def maxima(self, n, modes):
        key = (n, modes.tobytes())
        if key not in self.cache:
            rng = np.random.default_rng(self.seed)
            peaks = [float(peak_order(rng.random((n,3)),modes).max()) for _ in range(self.repeats)]
            if len(self.cache)>=16:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = np.asarray(peaks)
        return self.cache[key]


def density_metric(frac, H, modes, null, prefix):
    keys = ['mode_x','mode_y','mode_z','peak_order','peak_structure_factor','peak_wavelength_m',
            'wave_normal_x','wave_normal_y','wave_normal_z','uniform_max_order_p95','uniform_max_exceedance_fraction','modes_scanned']
    out = {prefix+'_'+key: math.nan for key in keys}
    out[prefix+'_modes_scanned'] = len(modes)
    if not len(modes):
        return out
    orders = peak_order(frac,modes)
    m = modes[int(np.argmax(orders))]
    k = m @ np.linalg.inv(H)
    kn = np.linalg.norm(k)
    peak = float(orders.max())
    reference = null.maxima(len(frac),modes)
    vals = [*m,peak,len(frac)*peak*peak,1/kn,*(k/kn),float(np.quantile(reference,.95)),float((1+np.count_nonzero(reference>=peak))/(1+len(reference))),len(modes)]
    out.update({prefix+'_'+key: float(value) for key,value in zip(keys,vals)})
    return out


def velocity_metrics(frame, x, director, rate, a):
    out = {key: math.nan for key in ['velocity_nonaffine_rms_m_s','velocity_nonaffine_rms_over_shear_a',
           'velocity_nonaffine_gradient_rms_m_s','velocity_nonaffine_director_normal_rms_m_s',
           'velocity_nonaffine_director_tangent_rms_m_s','velocity_actual_gradient_rms_m_s',
           'velocity_affine_gradient_rms_m_s','velocity_actual_director_normal_rms_m_s','velocity_actual_flow_rms_m_s']}
    if not all(k in frame.names for k in ('vx','vy','vz')) or not math.isfinite(rate):
        return out
    v = frame.columns(('vx','vy','vz'))
    affine = np.zeros_like(v)
    affine[:,0] = rate*x[:,2]
    nonaffine = v-affine
    nonaffine -= nonaffine.mean(axis=0)  # uniform translation is irrelevant
    actual = v-v.mean(axis=0)
    affine -= affine.mean(axis=0)
    rms = lambda q: float(np.sqrt(np.mean(q*q)))
    total = float(np.sqrt(np.mean(np.sum(nonaffine**2,axis=1))))
    out.update(velocity_nonaffine_rms_m_s=total,
        velocity_nonaffine_rms_over_shear_a=total/(abs(rate)*a) if rate and a>0 else math.nan,
        velocity_nonaffine_gradient_rms_m_s=rms(nonaffine[:,2]),
        velocity_actual_gradient_rms_m_s=rms(actual[:,2]),
        velocity_affine_gradient_rms_m_s=rms(affine[:,2]),
        velocity_actual_flow_rms_m_s=rms(actual[:,0]))
    if np.isfinite(director).all():
        normal = nonaffine @ director
        out['velocity_nonaffine_director_normal_rms_m_s'] = rms(normal)
        out['velocity_nonaffine_director_tangent_rms_m_s'] = math.sqrt(max(0.,total*total-float(np.mean(normal**2))))
        out['velocity_actual_director_normal_rms_m_s'] = rms(actual @ director)
    return out


def parallel_neighbor_motion(frame, x, axis, rate, a, c):
    """Center-motion proxy for close projected near-parallel neighbors.

    Select mutual axis angle <=10 deg, projected center separation <=2a,
    and center separation along the bisector <=4c. This is NOT a contact/film
    classification or a surface gap. Enumerate all images in that finite support.
    """
    out = {key:math.nan for key in ['parallel_neighbor_count','parallel_neighbor_normal_center_speed_rms_m_s',
         'parallel_neighbor_tangent_center_speed_rms_m_s','parallel_neighbor_affine_tangent_center_speed_rms_m_s']}
    if axis is None or not math.isfinite(rate) or not (a>0 and c>0) or not all(k in frame.names for k in ('vx','vy','vz')):
        return out
    n = len(x)
    ii,jj = np.triu_indices(n,0)
    dot = np.sum(axis[ii]*axis[jj],axis=1)
    mask = np.abs(dot)>=math.cos(math.radians(10))
    ii,jj,dot = ii[mask],jj[mask],dot[mask]
    normals = axis[ii]+np.where(dot>=0,1.,-1.)[:,None]*axis[jj]
    normals /= np.linalg.norm(normals,axis=1)[:,None]
    H, invH = frame.lattice, np.linalg.inv(frame.lattice)
    original = x[jj]-x[ii]
    base = -np.rint(original @ invH.T).astype(int)
    rc = math.hypot(2*a,4*c)
    limits = np.ceil(rc*np.linalg.norm(invH,axis=1)+.5).astype(int)
    # The supported run cell is far larger than the particle. Fail explicitly
    # rather than explode in pathological user-edited extremely small/skew cells.
    if np.prod(2*limits+1)>20000:
        raise ValueError('Cell too small/skew for neighbor diagnostic image enumeration')
    v = frame.columns(('vx','vy','vz'))
    count=0; normal2=tangent2=affine2=0.
    for off in itertools.product(*(range(-m,m+1) for m in limits)):
        image = base+np.asarray(off)
        shift = image @ H.T
        d = original+shift
        dn = np.sum(d*normals,axis=1)
        proj2 = np.sum(d*d,axis=1)-dn*dn
        # Half of each nonzero self-image pair; all i<j images are retained.
        positive = (image[:,0]>0)|((image[:,0]==0)&(image[:,1]>0))|((image[:,0]==0)&(image[:,1]==0)&(image[:,2]>0))
        selected = (np.abs(dn)<=4*c)&(proj2<=(2*a)**2)&((ii!=jj)|positive)
        if not selected.any():
            continue
        dv = v[jj[selected]]-v[ii[selected]]
        dv[:,0] += rate*shift[selected,2]  # Lees-Edwards image velocity
        nn = normals[selected]
        vn = np.sum(dv*nn,axis=1)
        va = np.zeros_like(dv); va[:,0]=rate*d[selected,2]
        van = np.sum(va*nn,axis=1)
        count += len(vn)
        normal2 += float(vn@vn)
        tangent2 += float(np.sum(dv*dv)-vn@vn)
        affine2 += float(np.sum(va*va)-van@van)
    out['parallel_neighbor_count']=count
    if count:
        out['parallel_neighbor_normal_center_speed_rms_m_s']=math.sqrt(max(0.,normal2/count))
        out['parallel_neighbor_tangent_center_speed_rms_m_s']=math.sqrt(max(0.,tangent2/count))
        out['parallel_neighbor_affine_tangent_center_speed_rms_m_s']=math.sqrt(max(0.,affine2/count))
    return out


def finite_json(value):
    if isinstance(value,dict):
        return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value,np.generic):
        value=value.item()
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


def select_run(argument=None):
    if argument:
        path=Path(argument).expanduser().resolve()
        run=path.parent if path.is_file() else path
        trajectory=path if path.is_file() else run/'trajectory.lammpstrj'
        if not trajectory.is_file():
            raise FileNotFoundError('Missing trajectory: '+str(trajectory))
        return run,trajectory
    candidates=[]
    for path in Path('runs').glob('**/trajectory.lammpstrj'):
        params=path.parent/'parameters.json'
        try:
            metadata=json.loads(params.read_text())
            shape=metadata.get('particle_shape',metadata.get('shape'))
            if shape=='flake' and path.stat().st_size:
                # Do not auto-select a newly created file containing only a header.
                first = next(read_frames(path), None)
                if first is not None:
                    candidates.append(path)
        except (OSError,ValueError):
            continue
    if not candidates:
        raise FileNotFoundError('No flake trajectory under ./runs; pass the run directory explicitly')
    for candidate in sorted(candidates,key=lambda p:p.stat().st_mtime_ns,reverse=True):
        if next(read_frames(candidate),None) is not None:
            path=candidate.resolve()
            return path.parent,path
    raise ValueError('No complete flake trajectory frame under ./runs')


def csv_tail(path):
    if not path.is_file():
        return None
    latest=None; count=0
    with path.open() as stream:
        for row in csv.DictReader(stream):
            # Exclude a partially written row; retain the latest complete row.
            if None in row or any(v is None or v=='' for v in row.values()):
                continue
            latest=row; count+=1
    if latest is None:
        return None
    converted={}
    for k,v in latest.items():
        try:
            converted[k]=float(v)
        except ValueError:
            converted[k]=v
    return {'rows':count,'last_complete_row':converted}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_directory',nargs='?')
    parser.add_argument('--output',type=Path,help='output directory (default: run directory)')
    parser.add_argument('--stride',type=int,default=1,help='analyze every kth saved frame')
    parser.add_argument('--max-frames',type=int,default=0,help='0 means all complete frames')
    parser.add_argument('--null-repeats',type=int,default=32,help='seeded uniform-reference samples')
    parser.add_argument('--max-mode-index',type=int,default=128)
    args=parser.parse_args(argv)
    if args.stride<1 or args.max_frames<0 or args.null_repeats<2 or args.max_mode_index<2:
        parser.error('Require stride>=1, max-frames>=0, null-repeats>=2, max-mode-index>=2')
    run,trajectory=select_run(args.run_directory)
    print('Selected run:',run,flush=True)
    metadata=json.loads((run/'parameters.json').read_text()) if (run/'parameters.json').is_file() else {}
    a=float(metadata.get('a_m',math.nan)); c=float(metadata.get('c_m',metadata.get('thickness_nm',math.nan)*.5e-9))
    rate=float(metadata.get('shear_rate',math.nan)); dt=float(metadata.get('timestep',math.nan))
    if not (math.isfinite(c) and c>0):
        raise ValueError('parameters.json must provide c_m or thickness_nm for the density-mode wavelength range')
    output=args.output or run;output.mkdir(parents=True,exist_ok=True)
    warnings=[]; rows=[]; null=UniformNull(args.null_repeats)
    for index,frame in enumerate(read_frames(trajectory,warnings)):
        if index%args.stride:
            continue
        x=positions(frame); axis=axes(frame)
        S,director,gap=nematic(axis)
        frac=(x-frame.origin) @ np.linalg.inv(frame.lattice).T
        frac%=1
        row={'step':frame.step,'time_s':frame.step*dt,'strain':abs(rate)*frame.step*dt,'count':len(x),
             'nematic_S':S,'director_x':director[0],'director_y':director[1],'director_z':director[2],
             'nematic_eigenvalue_gap':gap}
        for label,direction,gradient in [('gradient',np.array([0.,0.,1.]),True),('director',director,False)]:
            modes,limited=density_modes(frame.lattice,direction,2*c,args.max_mode_index,exact_gradient=gradient)
            row.update(density_metric(frac,frame.lattice,modes,null,label))
            row[label+'_mode_search_limited']=int(limited)
        row.update(velocity_metrics(frame,x,director,rate,a))
        row.update(parallel_neighbor_motion(frame,x,axis,rate,a,c))
        for label in ('gradient','director_normal'):
            speed=row['velocity_nonaffine_'+label+'_rms_m_s']
            row['nonaffine_'+label+'_thickness_per_strain']=speed/(abs(rate)*2*c) if rate else math.nan
        rows.append(row)
        if len(rows)==1 or len(rows)%20==0:
            print('Frames %d, step %d: S=%.4g, gradient density peak=%.4g (uniform p95=%.4g)'%
                  (len(rows),frame.step,S,row['gradient_peak_order'],row['gradient_uniform_max_order_p95']),flush=True)
        if args.max_frames and len(rows)>=args.max_frames:
            break
    if not rows:
        raise ValueError('No complete trajectory frames yet; simulation was not started or modified')
    with (output/'flake_layering.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if not np.isfinite(rows[-1]['nematic_S']):
        warnings.append('Quaternion columns unavailable: orientation/director diagnostics omitted')
    if not np.isfinite(rows[-1]['velocity_nonaffine_rms_m_s']):
        warnings.append('Velocity columns or shear rate unavailable: velocity diagnostics omitted')
    summary={'diagnostic_version':'20261024-read-only-layering-v1','run_directory':str(run),
             'trajectory':str(trajectory),'frames_analyzed':len(rows),'first_frame':rows[0],'last_frame':rows[-1],
             'warnings':warnings,'uniform_null_repeats':args.null_repeats,'uniform_null_seed':1729,
             'mode_search':{'maximum_integer_index':args.max_mode_index,'minimum_wavelength_m':2*c,
                            'director_cone_degrees':15,'maximum_director_modes':512},
             'history':csv_tail(run/'history.csv'),'channels':csv_tail(run/'flake_channels.csv'),
             'interpretation':[
                 'Nematic S measures axis alignment, not positional layering.',
                 'Density order is |mean(exp(2*pi*i*m.s))|, with integer m and fractional position s; it is invariant under periodic-image shifts.',
                 'Peak wavelength may be a harmonic. The finite-N uniform max-peak reference controls this diagnostic search, not excluded-volume correlations or physical equilibrium.',
                 'Layering evidence requires positional density modulation plus aligned axes; sliding evidence additionally requires small cross-layer motion with finite tangential relative motion.',
                 'Nonaffine velocities subtract gamma_dot*z in x and the mean translation; image-relative neighbor velocities include gamma_dot*image_shift_z.',
                 'Parallel-neighbor speeds are center-velocity proxies for axis angle<=10 degrees, projected separation<=2a and bisector separation<=4c; they are not certified contacts or face-surface slip.',
                 'A low viscosity and small solver residual do not establish the physical accuracy of the near-field model.',
                 'Unweighted-film power, when present in channels, uses frozen solved velocities and is not a re-solved counterfactual viscosity.',
                 'Snapshot diagnostics are not steady-state time averages.']}
    (output/'flake_diagnostics_summary.json').write_text(json.dumps(finite_json(summary),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    last=rows[-1]
    print('Saved:',output/'flake_layering.csv')
    print('Saved:',output/'flake_diagnostics_summary.json')
    print('Latest: strain=%.6g, nematic S=%.4g, gradient density order=%.4g, uniform-max p95=%.4g'%
          (last['strain'],last['nematic_S'],last['gradient_peak_order'],last['gradient_uniform_max_order_p95']))
    print('Interpretation: alignment alone is not layering; density modulation and cross-layer motion must be considered together.')
    if summary['channels']:
        print('Runtime channel diagnostics found:',run/'flake_channels.csv')
        channels=summary['channels']['last_complete_row']
        required=['eta_relative','eta_relative_face','eta_relative_rim','eta_relative_middle','eta_relative_contact']
        if all(k in channels for k in required):
            print('Channel snapshot step %g: eta_r=%.6g = 1 + face %.6g + rim %.6g + middle %.6g + contact %.6g'%
                  tuple([channels['step']]+[channels[k] for k in required]))
            face=channels.get('face_power_W',0.)
            extra=channels.get('extra_face_frozen_power_W',0.)
            if face+extra>0:
                print('Angle-taper omitted face-power fraction within evaluated domain: %.4g%% (frozen velocities, NOT predicted viscosity)'%(100*extra/(face+extra)))
            print('Angle-excluded cutoff-active pairs: %g; film loss outside the angle domain is UNKNOWN'%channels.get('angle_excluded_pairs',math.nan))
            print('Active contacts: %g; contacted particles: %g; summed normal load: %.6g N'%
                  (channels.get('active_contacts',math.nan),channels.get('contacted_particles',math.nan),channels.get('normal_load_sum_N',math.nan)))
    else:
        print('No runtime channel CSV: existing trajectories still support layering/motion diagnostics; angle-power attribution needs a diagnostic-enabled run.')
    for warning in warnings:
        print('WARNING:',warning,file=sys.stderr)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError) as error:
        print('diagnose_flake.py:',error,file=sys.stderr)
        raise SystemExit(2)
