#!/usr/bin/env python3
"""Independent whole-pair analytic-ring MFS reference, with all 17 motions.

No scattering basis, finite-film coefficient, or matched provider is imported.
The analytic-ring kernel is supplied explicitly so its hash is recorded.
An individual run is diagnostic; qualification also needs refinement agreement.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigvalsh, lstsq
from scipy.special import roots_legendre


def strain_basis():
    q = 1 / math.sqrt(2)
    t = 1 / math.sqrt(6)
    return np.array([np.diag([q, -q, 0]), np.diag([t, t, -2*t]),
                     [[0, q, 0], [q, 0, 0], [0, 0, 0]],
                     [[0, 0, q], [0, 0, 0], [q, 0, 0]],
                     [[0, 0, 0], [0, 0, q], [0, q, 0]]])


def rotation(axis):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    x = np.array([1., 0., 0.])
    if abs(x @ axis) > .9:
        x = np.array([0., 1., 0.])
    x -= axis * (axis @ x)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(axis, x), axis))


def geometry(a, h, offset):
    # Parallel spheroids, normal-frame counterpart of lateral offset offset*a.
    # At h=0 this has exactly the requested lateral offset in the body frame.
    q = offset / 2
    theta = math.atan(q / (a * math.sqrt(1-q*q)))
    axis = np.array([math.sin(theta), 0., math.cos(theta)])
    tensor = a*a*np.eye(3) + (1-a*a)*np.outer(axis, axis)
    normal = np.array([0., 0., 1.])
    support = tensor @ normal / math.sqrt(normal @ tensor @ normal)
    return [(axis.copy(), -support), (axis.copy(), support+h*normal)]


def nodes(nt, nphi, a, body, half=False, shift=0.):
    axis, center = body
    mu, _ = roots_legendre(nt)
    phi = (np.arange(nphi) + (.5 if half else 0.) + shift) * (math.pi if half else 2*math.pi) / nphi
    rho = a*np.sqrt(1-mu*mu)
    local = np.stack((rho[:, None]*np.cos(phi), rho[:, None]*np.sin(phi),
                      np.broadcast_to(mu[:, None], (nt, nphi))), axis=-1)
    return local.reshape(-1, 3) @ rotation(axis).T + center


def rhs(targets, bodies, nbody, a):
    result = np.zeros((3*len(targets), 17))
    es = strain_basis()
    for body, (_, center) in enumerate(bodies):
        points = targets[body*nbody:(body+1)*nbody] - center
        block = np.zeros((nbody, 3, 17))
        block[:, :, 6*body:6*body+3] = np.eye(3)
        for k in range(3):
            block[:, :, 6*body+3+k] = np.cross(np.eye(3)[k], points) / a
        for k in range(5):
            block[:, :, 12+k] = -points @ es[k].T / a
        result[3*body*nbody:3*(body+1)*nbody] = block.reshape(-1, 17)
    return result


def crossmat(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])


def moment_map(nlat, order, depth, a, bodies):
    nm = 2*order+1
    result = np.zeros((17, 6*nlat*nm))
    mu, _ = roots_legendre(nlat)
    rho = math.sqrt(a*a-1+depth*depth)*np.sqrt(1-mu*mu)
    z = depth*mu
    es = strain_basis()
    for b, (axis, _) in enumerate(bodies):
        q = rotation(axis)
        for latitude in range(nlat):
            start = 3*(b*nlat+latitude)*nm
            result[6*b:6*b+3, start:start+3] = np.eye(3)
            positions = [(0, z[latitude]*q[:, 2])]
            if order:
                positions += [(1, .5*rho[latitude]*q[:, 0]), (2, .5*rho[latitude]*q[:, 1])]
            for harmonic, position in positions:
                cols = slice(start+3*harmonic, start+3*harmonic+3)
                result[6*b+3:6*b+6, cols] = crossmat(position)/a
                for k in range(5):
                    result[12+k, cols] = -(es[k] @ position)/a
    return result


def midpoint_projection(bodies, h, a):
    middle = np.array([0., 0., .5*h])
    result = np.zeros((6, 12))
    result[:3, :3], result[:3, 6:9] = -np.eye(3), np.eye(3)
    result[3:, 3:6], result[3:, 9:12] = -np.eye(3), np.eye(3)
    result[:3, 3:6] = crossmat((middle-bodies[0][1])/a)
    result[:3, 9:12] = -crossmat((middle-bodies[1][1])/a)
    return result


def reduce(grand, projection):
    aa, cc, ee = grand[:12, :12], grand[:12, 12:], grand[12:, 12:]
    ac = np.linalg.solve(aa, cc)
    k = np.linalg.inv(projection @ np.linalg.solve(aa, projection.T))
    f = projection @ ac
    answer = np.block([[k, k@f], [f.T@k, ee-cc.T@ac+f.T@k@f]])
    return (answer+answer.T)/2


def free_strain(reduced):
    return reduced[6:, 6:] - reduced[6:, :6] @ np.linalg.solve(reduced[:6, :6], reduced[:6, 6:])


def power_error(a, b):
    return float(np.max(np.abs(eigvalsh(a, b)-1)))


def run(args):
    start = time.time()
    def progress(stage, **data):
        print(json.dumps(dict(stage=stage, seconds=time.time()-start, **data)), flush=True)
    kernel_path = Path(args.kernel).resolve()
    spec = importlib.util.spec_from_file_location('independent_analytic_ring', str(kernel_path))
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    bodies = geometry(args.ar, args.h, args.offset)
    nlat, order = args.nlat, args.order
    nt = int(math.ceil(nlat*args.oversample))
    nphi = int(math.ceil((2*order+1)*args.oversample/2))
    nbody = nt*nphi
    targets = np.concatenate([nodes(nt, nphi, args.ar, body, half=True) for body in bodies])
    b = rhs(targets, bodies, nbody, args.ar)
    progress('kernel_start', rows=3*len(targets), columns=6*nlat*(2*order+1), matrix_gb=3*len(targets)*6*nlat*(2*order+1)*8/1e9)
    matrix = kernel.matrix(targets, nlat, order, args.depth, bodies, a=args.ar)
    progress('kernel_complete')
    az_even = np.array([True] + [x for _ in range(order) for x in (True, False)])
    src_even = np.tile((az_even[:, None] == np.array([True, False, True])[None, :]).ravel(), 2*nlat)
    rhs_even = np.array([True, False, True, False, True, False]*2 + [True, True, False, True, False])
    coef = np.zeros((matrix.shape[1], 17))
    ranks = []
    for parity, sm, rm in (('even', src_even, rhs_even), ('odd', ~src_even, ~rhs_even)):
        # A Fortran copy avoids an extra full matrix copy inside LAPACK.
        submatrix = np.asfortranarray(matrix[:, sm])
        co, _, rank, _ = lstsq(submatrix, b[:, rm], cond=args.rcond, lapack_driver='gelsy', check_finite=False, overwrite_a=True, overwrite_b=False)
        coef[np.ix_(sm, rm)] = co
        ranks.append(int(rank))
        del submatrix
        progress('least_squares_complete', parity=parity, rank=int(rank))
    collocation = float(np.linalg.norm(matrix@coef-b)/np.linalg.norm(b))
    del matrix
    grand = moment_map(nlat, order, args.depth, args.ar, bodies) @ coef
    symmetric = (grand+grand.T)/2
    reduced = reduce(symmetric, midpoint_projection(bodies, args.h, args.ar))
    ntc, npc = nt+5, 2*nphi+4
    # Keep a shifted-azimuth equatorial ring even when the GL check grid has
    # even latitude count. Thin oblate rim errors can otherwise be missed.
    check_nbody = (ntc+1)*npc
    checks = np.concatenate([np.concatenate((nodes(ntc, npc, args.ar, body, shift=.371),
                                            nodes(1, npc, args.ar, body, shift=.173))) for body in bodies])
    check_rhs = rhs(checks, bodies, check_nbody, args.ar)
    err = np.empty_like(check_rhs)
    for first in range(0, len(checks), 128):
        last = min(first+128, len(checks))
        c = kernel.matrix(checks[first:last], nlat, order, args.depth, bodies, a=args.ar)
        err[3*first:3*last] = c@coef-check_rhs[3*first:3*last]
    percol = np.linalg.norm(err, axis=0)/np.linalg.norm(check_rhs, axis=0)
    equator_rows = np.concatenate([np.arange(3*(body*check_nbody+ntc*npc), 3*((body+1)*check_nbody)) for body in range(2)])
    equator_percol = np.linalg.norm(err[equator_rows], axis=0)/np.linalg.norm(check_rhs[equator_rows], axis=0)
    worst_boundary = float(math.sqrt(max(0., eigvalsh(err.T@err, check_rhs.T@check_rhs)[-1])))
    reciprocity = float(np.linalg.norm(grand-grand.T)/np.linalg.norm(grand))
    mineig = float(eigvalsh(symmetric)[0])
    report = dict(method='independent_whole_pair_analytic_ring_MFS_full17',
                  scope='Single parallel-offset pose. No runtime scattering basis or finite-film model is used.',
                  ar=args.ar, gap_over_c=args.h, offset_contact_a=args.offset,
                  nlat=nlat, order=order, depth=args.depth, oversample=args.oversample, rcond=args.rcond,
                  reflection_parity=True, parity_ranks=ranks, unknowns=len(coef),
                  bodies=[dict(axis=axis.tolist(), center=center.tolist()) for axis, center in bodies],
                  units='c=eta=1; input Ui,aOmegai,Uj,aOmegaj,aE5; output F,T/a,intrinsicS/a',
                  kernel_sha256=hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  collocation_relative_l2=collocation,
                  boundary_relative_l2_per_column=percol.tolist(),
                  boundary_max_column=float(percol.max()), boundary_worst_all17=worst_boundary,
                  shifted_equator_max_column=float(equator_percol.max()),
                  raw_reciprocity=reciprocity, minimum_symmetric_eigenvalue=mineig,
                  grand17_raw=grand.tolist(), reduced11=reduced.tolist(), free_strain5=free_strain(reduced).tolist(),
                  boundary_gate_passed=bool(percol.max()<=.001 and worst_boundary<=.001 and reciprocity<=.001 and mineig>0),
                  refinement_status='not_run', qualified_reference=False)
    if args.previous:
        old = json.loads(Path(args.previous).read_text())
        for key, value in [('ar', args.ar), ('gap_over_c', args.h), ('offset_contact_a', args.offset)]:
            if old[key] != value:
                raise ValueError('Refinement geometry mismatch: '+key)
        energy = power_error(reduced, np.array(old['reduced11']))
        stress = power_error(free_strain(reduced), np.array(old['free_strain5']))
        report.update(refinement_status='completed', previous_sha256=hashlib.sha256(Path(args.previous).read_bytes()).hexdigest(),
                      refinement_all11_error=energy, refinement_free_strain_error=stress,
                      previous_boundary_max_column=old['boundary_max_column'],
                      previous_boundary_worst_all17=old['boundary_worst_all17'],
                      qualified_reference=bool(report['boundary_gate_passed'] and energy<=.001 and stress<=.001))
    report['seconds'] = time.time()-start
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    progress('complete', **{key:report[key] for key in ('boundary_max_column', 'boundary_worst_all17', 'raw_reciprocity', 'qualified_reference')})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--previous')
    parser.add_argument('--ar', type=float, default=10.)
    parser.add_argument('--h', type=float, default=.75)
    parser.add_argument('--offset', type=float, default=.5)
    parser.add_argument('--nlat', type=int, default=64)
    parser.add_argument('--order', type=int, default=10)
    parser.add_argument('--depth', type=float, default=.5)
    parser.add_argument('--oversample', type=float, default=1.15)
    parser.add_argument('--rcond', type=float, default=1e-11)
    options = parser.parse_args()
    if not (options.ar>=1 and options.h>0 and 0<=options.offset<2 and options.nlat>=4 and options.order>=1 and 0<options.depth<1 and options.oversample>=1):
        parser.error('Invalid geometry or resolution')
    run(options)
