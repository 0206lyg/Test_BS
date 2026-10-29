"""Reusable exterior Stokes operator for the pinned fmm3dbie backend.

Preparation mirrors ``stok_comb_vel_solver`` in upstream stok_comb_vel.f90.
The near weights and oversampled geometry are built once and reused for every
GMRES application and right-hand side. The FMM's internal tree/work arrays are
still constructed by upstream on each application. No lubrication model,
symmetrization, eigenvalue clipping, or change of boundary conditions is used.

Ordering matches Backend.matrix: [qx(node0), qy(node0), qz(node0), ...].
Only the exterior operator is exposed: its jump is +beta/2 and the interior
normal-flux rank correction is exactly zero. Quadrature/linear convergence
does not establish physical density or geometry resolution.
"""
from __future__ import annotations

import time
import warnings
import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.sparse import bsr_matrix, eye
from scipy.sparse.linalg import LinearOperator, gmres


class GMRESConvergenceError(RuntimeError):
    """Retains partial results so a failed experiment can be recorded."""
    def __init__(self, solution, diagnostics):
        self.solution = solution
        self.diagnostics = diagnostics
        failed = [v['rhs'] for v in diagnostics['right_hand_sides'] if not v['converged']]
        super().__init__(f'Matrix-free GMRES did not converge for RHS {failed}')


class MatrixFreeContext:
    """Prepare the original exterior combined-layer operator once.

    ``backend`` is an existing Backend instance; its shared library is never
    rebuilt. ``mesh`` must already have srccoefs and weights. All native input
    arrays are copied, so later caller edits cannot alter this context.
    ``max_memory_gb`` limits retained arrays before their large allocations;
    native FMM/quadrature temporary memory and GMRES workspace are additional.
    ``far_order_increment`` adds a caller-controlled amount to upstream's
    estimated far quadrature orders. Flattened, coarse patches can require
    this refinement even when eps is small; eps alone is not an error bound.
    ``near_radius_factor`` expands the near correction region. These two
    controls change quadrature only, never the original integral equation.
    ``all_patches_near=True`` instead stores the same six kernel entries as
    dense assembly, using 48*npts**2 bytes plus indexing. It skips FMM/far
    quadrature and applies these weights directly without a dense matrix/LU.
    This option removes far-oversampling uncertainty but is still quadratic
    storage and work; near-quadrature and density refinement remain necessary.
    """
    def __init__(self, backend, mesh, eps=1e-7, alpha=1., beta=1.,
                 max_memory_gb=4., far_order_increment=0,
                 near_radius_factor=1., all_patches_near=False,
                 near_apply_library=None):
        start = time.monotonic()
        if not (0 < eps < 1 and np.isfinite(alpha) and np.isfinite(beta)
                and alpha > 0 and beta > 0 and np.isfinite(max_memory_gb)
                and max_memory_gb > 0):
            raise ValueError('Positive finite combined strengths, tolerance and memory budget required')
        if int(far_order_increment) != far_order_increment or far_order_increment < 0:
            raise ValueError('far_order_increment must be a nonnegative integer')
        if not np.isfinite(near_radius_factor) or near_radius_factor < 1:
            raise ValueError('near_radius_factor must be finite and at least one')
        self.backend = backend
        self.eps = float(eps)
        self.alpha, self.beta = float(alpha), float(beta)
        self.far_order_increment = int(far_order_increment)
        self.near_radius_factor = float(near_radius_factor)
        self.all_patches_near = bool(all_patches_near)
        self._near_apply = None
        self.near_apply_metadata = None
        if near_apply_library is not None:
            if not self.all_patches_near:
                raise ValueError('near_apply_library is only used with all_patches_near')
            try:
                from .near_apply import NearApply
            except ImportError:
                from near_apply import NearApply
            self._near_apply = NearApply(near_apply_library)
            self.near_apply_metadata = self._near_apply.metadata()
        self.budget = int(max_memory_gb * 1e9)
        self.mesh = {name: np.array(mesh[name], order='F', copy=True)
                     for name in ['norders', 'ixyzs', 'iptype', 'srcvals',
                                  'srccoefs', 'weights']}
        if 'body_ids' in mesh:
            self.mesh['body_ids'] = np.array(mesh['body_ids'], dtype=np.int64, copy=True)
        self.n = self.mesh['srcvals'].shape[1]
        self.p = len(self.mesh['norders'])
        if self.n < 1 or self.p < 1 or self.mesh['ixyzs'][0] != 1 or self.mesh['ixyzs'][-1] != self.n+1:
            raise ValueError('Invalid one-based patch offsets')
        if np.any(np.diff(self.mesh['ixyzs']) <= 0):
            raise ValueError('Empty or reversed patch')
        if len(np.unique(self.mesh['iptype'])) != 1:
            raise ValueError('Mixed patch types are unsupported by upstream average-type near-radius selection')
        self.dpars = np.array([alpha, beta, 0.], dtype=np.float64)
        # The native complex wave number is exactly 0+0j. Backend.call accepts
        # float64 storage; this view has the identical 16-byte complex ABI.
        self.zero_complex = np.zeros(1, dtype=np.complex128).view(np.float64)
        self.zero_integer = np.zeros(1, dtype=np.int64)
        self.matvec_calls = 0
        self.matvec_seconds = 0.
        self._patch_factors = None
        self._body_factors = None
        self._prepare()
        self.preparation_seconds = time.monotonic()-start
        self.operator = LinearOperator((3*self.n, 3*self.n), matvec=self.matvec,
                                       matmat=self.matmat, dtype=np.float64)

    def _base(self):
        m = self.mesh
        return (self.p, m['norders'], m['ixyzs'], m['iptype'], self.n,
                m['srccoefs'], m['srcvals'])

    def retained_bytes(self):
        seen = set()
        total = 0
        for value in list(self.mesh.values()) + list(vars(self).values()):
            if isinstance(value, np.ndarray) and id(value) not in seen:
                total += value.nbytes
                seen.add(id(value))
        if self._patch_factors is not None:
            total += sum(lu.nbytes+piv.nbytes for _, _, lu, piv in self._patch_factors)
        if self._body_factors is not None:
            total += sum(dofs.nbytes+lu.nbytes+piv.nbytes for dofs, lu, piv in self._body_factors)
        return total

    def _check_memory(self, additional):
        predicted = self.retained_bytes()+int(additional)
        if predicted > self.budget:
            raise MemoryError(f'Matrix-free retained arrays need {predicted/1e9:.3f} GB, '
                              f'budget {self.budget/1e9:g} GB; native temporary arrays are additional')

    def _prepare(self):
        call, m = self.backend.call, self.mesh
        self.targets = np.asfortranarray(m['srcvals'][:3])
        rfac = np.zeros((), dtype=np.float64)
        rfac0 = np.zeros((), dtype=np.float64)
        call('get_rfacs', int(np.floor(np.mean(m['norders']))),
             int(m['iptype'][0]), rfac, rfac0)
        self.rfac = float(rfac)*self.near_radius_factor
        self.rfac0 = float(rfac0)
        self.centroids = np.empty((3, self.p), order='F')
        self.radii = np.empty(self.p)
        call('get_centroid_rads', self.p, m['norders'], m['ixyzs'], m['iptype'],
             self.n, m['srccoefs'], self.centroids, self.radii)
        near_radii = np.asfortranarray(self.radii*self.rfac)
        if self.all_patches_near:
            self.nnz = self.n*self.p
        else:
            nnz = np.zeros((), dtype=np.int64)
            call('findnearmem', self.centroids, self.p, near_radii, 3,
                 self.targets, self.n, nnz)
            self.nnz = int(nnz)
        self._check_memory(8*(self.n+1+2*self.nnz+1))
        if self.all_patches_near:
            self.row_ptr = np.arange(self.n+1, dtype=np.int64)*self.p+1
            self.col_ind = np.tile(np.arange(1, self.p+1, dtype=np.int64), self.n)
        else:
            self.row_ptr = np.empty(self.n+1, dtype=np.int64)
            self.col_ind = np.empty(self.nnz, dtype=np.int64)
            call('findnear', self.centroids, self.p, near_radii, 3, self.targets,
                 self.n, self.row_ptr, self.col_ind)
        self.iquad = np.empty(self.nnz+1, dtype=np.int64)
        call('get_iquad_rsc', self.p, m['ixyzs'], self.n, self.nnz,
             self.row_ptr, self.col_ind, self.iquad)
        self.nquad = int(self.iquad[-1]-1)
        if (self.row_ptr[0] != 1 or self.row_ptr[-1] != self.nnz+1 or
                self.iquad[0] != 1 or np.any(np.diff(self.row_ptr) < 1)):
            raise RuntimeError('Invalid native near-interaction structure')

        if self.all_patches_near:
            self.novers = self.estimated_oversampling_orders = np.empty(0, dtype=np.int64)
            self.ixyzso = np.ones(1, dtype=np.int64)
            self.noversampled = 0
            self.srcover = np.empty((12, 0), order='F')
            self.wover = np.empty(0)
            self._check_memory(48*self.nquad)
        else:
            self.novers = np.empty(self.p, dtype=np.int64)
            self.ixyzso = np.empty(self.p+1, dtype=np.int64)
            call('get_far_order', self.eps, self.p, m['norders'], m['ixyzs'],
                 m['iptype'], self.centroids, self.radii, self.n, m['srccoefs'],
                 3, self.n, self.targets, 0, self.zero_complex, self.nnz,
                 self.row_ptr, self.col_ind, self.rfac, self.novers, self.ixyzso)
            self.estimated_oversampling_orders = self.novers.copy()
            if self.far_order_increment:
                self.novers += self.far_order_increment
                kind = int(m['iptype'][0])
                widths = ((self.novers+1)*(self.novers+2)//2 if kind == 1
                          else (self.novers+1)**2)
                self.ixyzso[:] = np.r_[1, 1+np.cumsum(widths)]
            self.noversampled = int(self.ixyzso[-1]-1)
            self._check_memory(8*(13*self.noversampled+6*self.nquad))
            self.srcover = np.empty((12, self.noversampled), order='F')
            self.wover = np.empty(self.noversampled)
            call('oversample_geom', *self._base(), self.novers, self.ixyzso,
                 self.noversampled, self.srcover)
            call('get_qwts', self.p, self.novers, self.ixyzso, m['iptype'],
                 self.noversampled, self.srcover, self.wover)
        self.wnear = np.zeros((6, self.nquad), order='F')
        call('getnearquad_stok_comb_vel', *self._base(), self.eps,
             self.dpars[:2], 1, self.nnz, self.row_ptr, self.col_ind,
             self.iquad, self.rfac0, self.nquad, self.wnear)
        # Do not allocate a full-size boolean array for quadratic all-near
        # storage (hundreds of MB at the larger reference meshes).
        flat_near = self.wnear.ravel(order='F')
        finite_near = all(np.isfinite(flat_near[start:start+(1 << 20)]).all()
                          for start in range(0, flat_near.size, 1 << 20))
        if not (finite_near and np.isfinite(self.srcover).all()
                and np.isfinite(self.wover).all()):
            raise FloatingPointError('Non-finite prepared quadrature')

    def matvec(self, vector):
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vector.size != 3*self.n or not np.isfinite(vector).all():
            raise ValueError('matvec needs a finite point-interleaved vector of length 3*npts')
        sigma = np.asfortranarray(vector.reshape(self.n, 3).T)
        potential = np.empty((3, self.n), order='F')
        start = time.monotonic()
        if self.all_patches_near:
            if self._near_apply is not None:
                raw = self._near_apply.matvec(self.wnear, vector)
                potential[:] = np.asarray(raw).reshape(self.n, 3).T
            else:
                potential.fill(0.)
                # Each view has source node as its last index; einsum
                # deliberately avoids BLAS packing an entire strided N-by-N
                # kernel copy.
                kernels = self.wnear.reshape(6, self.n, self.n)
                for k, (i, j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
                    potential[i] += np.einsum('ts,s->t', kernels[k], sigma[j], optimize=False)
                    if i != j:
                        potential[j] += np.einsum('ts,s->t', kernels[k], sigma[i], optimize=False)
        else:
            self.backend.call('lpcomp_stok_comb_vel_addsub', *self._base(),
                              self.eps, 3, self.dpars, 0, self.zero_complex,
                              0, self.zero_integer, self.nnz, self.row_ptr,
                              self.col_ind, self.iquad, self.nquad, 6, self.wnear,
                              self.novers, self.noversampled, self.ixyzso,
                              self.srcover, self.wover, self.n, self.mesh['weights'],
                              3, sigma, potential)
        self.matvec_calls += 1
        self.matvec_seconds += time.monotonic()-start
        result = potential.ravel(order='F') + (self.beta/2)*vector
        if not np.isfinite(result).all():
            raise FloatingPointError('Non-finite matrix-free potential')
        return result

    def matmat(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] != 3*self.n:
            raise ValueError('matmat expects (3*npts,nrhs)')
        if self.all_patches_near and self._near_apply is not None:
            start = time.monotonic()
            result = self._near_apply.matmat(self.wnear, vectors) + (self.beta/2)*vectors
            self.matvec_calls += vectors.shape[1]
            self.matvec_seconds += time.monotonic()-start
            return result
        return np.column_stack([self.matvec(vectors[:, j]) for j in range(vectors.shape[1])])

    def near_matrix(self, include_jump=True, max_extra_memory_gb=1.):
        """Optional sparse near operator, not the complete Stokes operator.

        Exposes exact prepared target/source-node 3x3 blocks for experiments
        with sparse preconditioners. The FMM far contribution is omitted.
        Native one-based CSR arrays and six weights are also public members.
        """
        # Adding the diagonal can temporarily retain both sparse matrices.
        estimate = (2 if include_jump else 1)*(80*self.nquad + 8*(self.n+1))
        if estimate > max_extra_memory_gb*1e9:
            raise MemoryError(f'Expanded near matrix needs at least {estimate/1e9:.3f} GB')
        columns = np.empty(self.nquad, dtype=np.int64)
        pointers = self.iquad[self.row_ptr-1]-1
        for interaction, source_patch in enumerate(self.col_ind-1):
            begin, end = self.iquad[interaction:interaction+2]-1
            first, last = self.mesh['ixyzs'][source_patch:source_patch+2]-1
            columns[begin:end] = np.arange(first, last)
        blocks = np.empty((self.nquad, 3, 3))
        for k, (i, j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
            blocks[:, i, j] = self.wnear[k]
            blocks[:, j, i] = self.wnear[k]
        result = bsr_matrix((blocks, columns, pointers), shape=(3*self.n, 3*self.n))
        if include_jump:
            result = result + (self.beta/2)*eye(3*self.n, format='bsr')
        return result

    def patch_preconditioner(self):
        """Patch-self block Jacobi; the physical operator remains unchanged."""
        if self._patch_factors is None:
            widths = np.diff(self.mesh['ixyzs'])
            self._check_memory(8*np.sum((3*widths)**2)+4*3*self.n)
            factors = []
            for patch in range(self.p):
                first, last = self.mesh['ixyzs'][patch:patch+2]-1
                width = int(last-first)
                block = np.zeros((3*width, 3*width), order='F')
                for local_row, target in enumerate(range(first, last)):
                    begin, end = self.row_ptr[target:target+2]-1
                    matches = np.flatnonzero(self.col_ind[begin:end] == patch+1)
                    if len(matches) != 1:
                        raise RuntimeError('Self patch missing or duplicated in near structure')
                    interaction = int(begin+matches[0])
                    qb, qe = self.iquad[interaction:interaction+2]-1
                    for k, (i, j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
                        block[3*local_row+i, j::3] = self.wnear[k, qb:qe]
                        block[3*local_row+j, i::3] = self.wnear[k, qb:qe]
                block.flat[::len(block)+1] += self.beta/2
                with warnings.catch_warnings():
                    warnings.simplefilter('error', LinAlgWarning)
                    lu, piv = lu_factor(block, overwrite_a=True, check_finite=True)
                factors.append((3*int(first), 3*int(last), lu, piv))
            self._patch_factors = factors
        def apply(vector):
            out = np.empty(3*self.n)
            for first, last, lu, piv in self._patch_factors:
                out[first:last] = lu_solve((lu, piv), np.asarray(vector)[first:last],
                                           check_finite=False)
            return out
        return LinearOperator(self.operator.shape, matvec=apply, dtype=np.float64)

    def body_preconditioner(self):
        """Exact individual-body LU blocks, for the all-patches-near mode.

        This is an optional stronger preconditioner. It adds approximately
        72*sum(N_body**2) bytes to retained memory; the pair operator still
        includes every cross-body interaction without alteration.
        """
        if not self.all_patches_near or 'body_ids' not in self.mesh:
            raise ValueError('Body LU requires all_patches_near and mesh body_ids')
        if self._body_factors is None:
            owners = self.mesh['body_ids']
            if owners.shape != (self.n,):
                raise ValueError('body_ids must have one entry per node')
            groups = [np.flatnonzero(owners == body) for body in np.unique(owners)]
            self._check_memory(sum(72*len(nodes)**2+36*len(nodes) for nodes in groups))
            kernels = self.wnear.reshape(6, self.n, self.n)
            factors = []
            for nodes in groups:
                width = len(nodes)
                block = np.empty((3*width, 3*width), order='F')
                contiguous = bool(np.all(np.diff(nodes) == 1))
                for k, (i, j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
                    if contiguous:
                        entry = kernels[k, nodes[0]:nodes[-1]+1, nodes[0]:nodes[-1]+1]
                        block[i::3, j::3] = entry
                        block[j::3, i::3] = entry
                    else:
                        # Bound advanced-indexing temporaries for reordered
                        # meshes instead of copying an entire scalar kernel.
                        for start in range(0, width, 64):
                            end = min(start+64, width)
                            entry = kernels[k][np.ix_(nodes[start:end], nodes)]
                            block[3*start+i:3*end+i:3, j::3] = entry
                            block[3*start+j:3*end+j:3, i::3] = entry
                block.flat[::len(block)+1] += self.beta/2
                with warnings.catch_warnings():
                    warnings.simplefilter('error', LinAlgWarning)
                    lu, piv = lu_factor(block, overwrite_a=True, check_finite=True)
                dofs = (3*nodes[:, None]+np.arange(3)).ravel()
                factors.append((dofs, lu, piv))
            self._body_factors = factors
        def apply(vector):
            out = np.empty(3*self.n)
            for dofs, lu, piv in self._body_factors:
                out[dofs] = lu_solve((lu, piv), np.asarray(vector)[dofs], check_finite=False)
            return out
        return LinearOperator(self.operator.shape, matvec=apply, dtype=np.float64)

    def solve(self, rhs, rtol=1e-9, atol=0., restart=80, maxiter=100,
              preconditioner='patch', initial=None, fail_on_nonconvergence=True,
              progress=None):
        """Solve one or many RHS; maxiter counts restart cycles, not matvecs.

        Returns (solution, diagnostics). Each RHS has an independently evaluated
        unpreconditioned residual. A failed RHS raises GMRESConvergenceError by
        default; the exception retains the partial solution and diagnostics.
        With this default, later RHS are not attempted after a failed RHS.
        Optional ``progress(index, solved_prefix_copy, record_copy)`` is called
        after each attempted RHS, permitting a caller to checkpoint results.
        The convergence record applies only to this discretized operator.
        """
        rhs = np.asarray(rhs, dtype=np.float64)
        was_vector = rhs.ndim == 1
        if was_vector:
            rhs = rhs[:, None]
        if rhs.ndim != 2 or rhs.shape[0] != 3*self.n or not np.isfinite(rhs).all():
            raise ValueError('rhs must be finite and have shape (3*npts,nrhs)')
        if preconditioner == 'patch':
            precond = self.patch_preconditioner()
        elif preconditioner == 'body':
            precond = self.body_preconditioner()
        elif preconditioner is None or preconditioner == 'none':
            precond = None
        elif isinstance(preconditioner, LinearOperator):
            precond = preconditioner
        else:
            raise ValueError('preconditioner must be patch, body, none or a LinearOperator')
        if initial is not None:
            initial = np.asarray(initial, dtype=np.float64).reshape(rhs.shape)
        solution = np.zeros_like(rhs)
        records = []
        start, before = time.monotonic(), self.matvec_calls
        for j in range(rhs.shape[1]):
            history = []
            norm = np.linalg.norm(rhs[:, j])
            if norm == 0:
                info, absolute, relative = 0, 0., 0.
            else:
                solution[:, j], info = gmres(
                    self.operator, rhs[:, j],
                    x0=None if initial is None else initial[:, j],
                    rtol=rtol, atol=atol, restart=restart, maxiter=maxiter,
                    M=precond, callback=lambda r: history.append(float(r)),
                    callback_type='pr_norm')
                absolute = float(np.linalg.norm(self.matvec(solution[:, j])-rhs[:, j]))
                relative = absolute/norm
            record = {'rhs': j, 'info': int(info), 'iterations': len(history),
                      'relative_residual': relative, 'absolute_residual': absolute,
                      'converged': bool(info == 0 and absolute <= max(atol, rtol*norm)),
                      'preconditioned_relative_residual_history': history}
            records.append(record)
            if progress is not None:
                progress_record = dict(record)
                progress_record['preconditioned_relative_residual_history'] = history.copy()
                progress(j, solution[:, :j+1].copy(), progress_record)
            if fail_on_nonconvergence and not record['converged']:
                break
        diagnostics = self.metadata()
        diagnostics.update(right_hand_sides=records,
                           all_converged=all(r['converged'] for r in records),
                           solve_seconds=time.monotonic()-start,
                           solve_matvec_calls=self.matvec_calls-before,
                           requested_rhs_count=rhs.shape[1], attempted_rhs_count=len(records),
                           rtol=rtol, atol=atol, restart=restart,
                           max_restart_cycles=maxiter,
                           preconditioner=preconditioner if isinstance(preconditioner, str) else
                           ('none' if precond is None else 'supplied_LinearOperator'))
        if was_vector:
            solution = solution[:, 0]
        if fail_on_nonconvergence and not diagnostics['all_converged']:
            raise GMRESConvergenceError(solution, diagnostics)
        return solution, diagnostics

    def metadata(self):
        return {'operator': ('fmm3dbie_exterior_combined_all_near_v1' if self.all_patches_near
                             else 'fmm3dbie_exterior_combined_matrix_free_v1'),
                'node_count': self.n, 'patch_count': self.p,
                'near_patch_interactions': self.nnz, 'near_node_interactions': self.nquad,
                'oversampled_nodes': self.noversampled,
                'oversampling_orders': self.novers.tolist(),
                'estimated_oversampling_orders': self.estimated_oversampling_orders.tolist(),
                'far_order_increment': self.far_order_increment,
                'near_radius_factor': self.near_radius_factor,
                'all_patches_near': self.all_patches_near,
                'near_apply_library': self.near_apply_metadata,
                'quadrature_tolerance': self.eps, 'alpha': self.alpha, 'beta': self.beta,
                'exterior_jump': self.beta/2, 'normal_rank_correction': 0.,
                'preparation_seconds': self.preparation_seconds,
                'retained_array_bytes': self.retained_bytes(),
                'matvec_calls': self.matvec_calls, 'matvec_seconds': self.matvec_seconds,
                'physical_accuracy_qualified': False}
