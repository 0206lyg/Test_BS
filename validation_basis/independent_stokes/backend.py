"""Thin, explicit int64/float64 interface to pinned fmm3dbie.

The reference solves the original exterior Stokes Dirichlet problem.  No
lubrication coefficients enter this module.  Its combined layer density is NOT
traction.  Physical moments are extracted from the representation's exact
monopole, rotlet and symmetric dipole moments.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import numpy as np


class Backend:
    def __init__(self, library):
        self.path = Path(library).resolve()
        self.lib = ctypes.CDLL(str(self.path), mode=ctypes.RTLD_GLOBAL)

    def call(self, name, *args):
        arrays = []
        for value in args:
            if isinstance(value, (int, np.integer)):
                value = np.asarray(value, dtype=np.int64)
            elif isinstance(value, (float, np.floating)):
                value = np.asarray(value, dtype=np.float64)
            elif not isinstance(value, np.ndarray):
                raise TypeError(f"Unsupported argument to {name}: {type(value)}")
            if value.dtype not in (np.dtype('float64'), np.dtype('int64')):
                raise TypeError(f"Wrong ABI dtype {value.dtype} for {name}")
            if not value.flags.f_contiguous:
                raise ValueError(f"Non-Fortran-contiguous argument to {name}")
            arrays.append(value)
        fn = getattr(self.lib, name + '_')
        fn.restype = None
        fn(*[ctypes.c_void_p(a.ctypes.data) for a in arrays])

    def discretization(self, order, kind=11):
        n = (order+1)**2 if kind != 1 else (order+1)*(order+2)//2
        uv = np.empty((2, n), order='F')
        tocoef = np.empty((n, n), order='F')
        toval = np.empty_like(tocoef, order='F')
        w = np.empty(n)
        self.call('get_disc_exps', order, n, kind, uv, tocoef, toval, w)
        return uv, tocoef, toval, w

    def coefficients(self, mesh):
        n = mesh['srcvals'].shape[1]
        out = np.empty((9, n), order='F')
        vals = np.asfortranarray(mesh['srcvals'][:9])
        self.call('surf_vals_to_coefs', 9, len(mesh['norders']),
                  mesh['norders'], mesh['ixyzs'], mesh['iptype'], n, vals, out)
        mesh['srccoefs'] = out
        w = np.empty(n)
        self.call('get_qwts', len(mesh['norders']), mesh['norders'],
                  mesh['ixyzs'], mesh['iptype'], n, mesh['srcvals'], w)
        mesh['weights'] = w
        return mesh

    def matrix(self, mesh, eps, alpha=1., beta=1., max_memory_gb=6.):
        n = mesh['srcvals'].shape[1]
        # Includes original matrix, its LU copy, quadrature workspace and slack.
        estimate = 224*n*n + 256*n*17
        if estimate > max_memory_gb*1e9:
            raise MemoryError(f"Dense reference needs about {estimate/1e9:.2f} GB "
                              f"for {n} nodes; budget {max_memory_gb:g} GB")
        out = np.empty((3*n, 3*n), order='F')
        self.call('stok_comb_vel_matgen', len(mesh['norders']), mesh['norders'],
                  mesh['ixyzs'], mesh['iptype'], n, mesh['srccoefs'],
                  mesh['srcvals'], eps, np.array([alpha, beta]), 1, out)
        return out

    def interpolate(self, mesh, values, patch_ids, uv):
        """Interpolate nodal fields at independent points of the SAME patches."""
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError('values must be (nfields,npts)')
        result = np.empty((values.shape[0], len(patch_ids)))
        cache = {}
        for j, p in enumerate(patch_ids):
            p = int(p)
            order, kind = int(mesh['norders'][p]), int(mesh['iptype'][p])
            if (order, kind) not in cache:
                cache[order, kind] = self.discretization(order, kind)[1]
            begin, end = mesh['ixyzs'][p:p+2] - 1
            pols = np.empty(end-begin)
            self.call('get_basis_pols', np.asarray(uv[:, j]).copy(), order,
                      end-begin, kind, pols)
            weights = pols @ cache[order, kind]
            result[:, j] = values[:, begin:end] @ weights
        return result

    def boundary_values(self, mesh, density, patch_ids, uv, eps,
                        alpha=1., beta=1., chunk=32):
        """Off-collocation surface trace with independently generated quadrature.

        Uses the polynomial panel's coordinates, records its deviation from the
        exact spheroid separately in the driver.  The +beta/2 exterior jump is
        added explicitly.  These checks are sampled, not a rigorous L-infinity
        bound on the entire surface.
        """
        npatch = len(mesh['norders'])
        n = mesh['srcvals'].shape[1]
        points = self.interpolate(mesh, mesh['srcvals'][:3], patch_ids, uv)
        qflat = density.transpose(1, 2, 0).reshape(3*density.shape[2], n)
        qtrace = self.interpolate(mesh, qflat, patch_ids, uv)
        qtrace = qtrace.reshape(3, density.shape[2], -1).transpose(2, 0, 1)
        result = np.empty_like(qtrace)
        for begin in range(0, len(patch_ids), chunk):
            end = min(begin+chunk, len(patch_ids))
            nt = end-begin
            nnz = nt*npatch
            row = np.arange(nt+1, dtype=np.int64)*npatch+1
            col = np.tile(np.arange(1, npatch+1, dtype=np.int64), nt)
            iq = np.r_[1, 1+np.cumsum(np.diff(mesh['ixyzs'])[col-1])].astype(np.int64)
            nquad = int(iq[-1]-1)
            near = np.empty((6, nquad), order='F')
            self.call('getnearquad_stok_comb_vel_eval', npatch, mesh['norders'],
                      mesh['ixyzs'], mesh['iptype'], n, mesh['srccoefs'],
                      mesh['srcvals'], 3, nt,
                      np.asfortranarray(points[:, begin:end]),
                      np.asarray(patch_ids[begin:end]+1, dtype=np.int64),
                      np.asfortranarray(uv[:, begin:end]), eps,
                      np.array([alpha, beta]), 1, nnz, row, col, iq, 1.25,
                      nquad, near)
            v = beta*.5*qtrace[begin:end].copy()
            for k, (i, j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
                kernel = near[k].reshape(nt, n)
                v[:, i] += kernel @ density[:, j, :]
                if i != j:
                    v[:, j] += kernel @ density[:, i, :]
            result[begin:end] = v
        return points.T, result


def representation_moments(points, normals, weights, density, body_ids,
                           centers, strain_basis, alpha=1., beta=1., a=20.):
    """Return positive-resistance rows [−Fhydro,−Thydro/a,Sphysical/a].

    D here is +3 r_i r_j (r.n)/(4*pi*r^5), as in fmm3dbie.  Its symmetric
    force dipole is +2 beta STF sym(n tensor q).  With exterior rigid total
    boundary velocity the intrinsic particle stresslet is minus this dipole
    including the single-layer first moment.  No extra ambient 2*V*E term
    is to be added to this combined-density expression.
    """
    out = np.zeros((17, density.shape[2]))
    flux = []
    for body in range(2):
        mask = body_ids == body
        if not np.any(mask):
            flux.append(np.zeros(density.shape[2]).tolist())
            continue
        q = density[mask]
        w = weights[mask]
        r = points[mask] - centers[body]
        normal = normals[mask]
        out[6*body:6*body+3] = alpha*np.einsum('n,nij->ij', w, q)
        torque = np.cross(r[:, :, None], q, axisa=1, axisb=1, axisc=1)
        out[6*body+3:6*body+6] = alpha*np.einsum('n,nij->ij', w, torque)/a
        effective = alpha*r+2*beta*normal
        for k, e in enumerate(strain_basis):
            # e is symmetric and traceless: explicit STF formation unnecessary.
            out[12+k] -= np.einsum('n,ni,ij,njr->r', w, effective, e, q)/a
        flux.append(np.einsum('n,ni,nir->r', w, normal, q).tolist())
    return out, flux
