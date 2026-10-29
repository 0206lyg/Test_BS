"""Independent exact oblate-spheroid geometry, with recorded AR20 defaults.

No production geometry, scattering basis, lubrication coefficient, or Stokes
solver is imported. Public coordinates use c=1 and a=aspect_ratio. The default
``recorded_pose(gap_nm)`` retains a=20 and one length unit of 150 nm. Generic
``spheroid_pose`` accepts AR, c in nm, and three orientation angles in radians.
The outward normal
points OUT OF THE SOLID and into the exterior fluid.

The 17 input coordinates are [U_i, a*Omega_i, U_j, a*Omega_j, a*E_0,...,a*E_4].
U and Omega are relative to the ambient velocity/vorticity at each centre.
Thus a strain column is -E_k @ (x-centre)/a, NOT -E_k @ x/a. The conjugate
outputs are [F_i,T_i/a,F_j,T_j/a,sum(S_intrinsic)/a], with force on the fluid.
This module supplies velocities, not a prescription that BIE density is force.

For quadrangular fmm3dbie patches, call ``build_mesh`` with the backend's own
GL reference nodes. Optional anisotropic rim and local contact refinement keep
the analytic surface exact. For a triangular backend, call ``unit_sphere_triangles`` and
``evaluate_panels`` at the backend's own reference-triangle nodes. The latter
returns analytic positions, first derivatives and normals; ``srcvals`` has
the usual (12,npoints) layout x,x_u,x_v,n. It does not construct polynomial
coefficients, close-evaluation rules, or a Stokes discretization.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from typing import ClassVar

import numpy as np
from numpy.polynomial.legendre import leggauss

THETA_I = 0.0049245303592113559
THETA_J = 0.041010932014513654
PHI = 1.7016960206944713
A_NM = 3000.0
C_NM = 150.0
MODE_NAMES = (
    "Ui_x", "Ui_y", "Ui_z", "aOmega_i_x", "aOmega_i_y", "aOmega_i_z",
    "Uj_x", "Uj_y", "Uj_z", "aOmega_j_x", "aOmega_j_y", "aOmega_j_z",
    "aE_xx_minus_yy", "aE_xx_yy_minus_2zz", "aE_xy", "aE_xz", "aE_yz",
)


def strain_basis() -> np.ndarray:
    """Five orthonormal symmetric trace-free matrices, in full17 order."""
    s, t = 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)
    return np.array([
        np.diag([s, -s, 0.0]), np.diag([t, t, -2.0*t]),
        [[0, s, 0], [s, 0, 0], [0, 0, 0]],
        [[0, 0, s], [0, 0, 0], [s, 0, 0]],
        [[0, 0, 0], [0, 0, s], [0, s, 0]],
    ], dtype=float)


def _rotation(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x = np.array([1.0, 0.0, 0.0])
    x -= axis * np.dot(axis, x)
    if np.linalg.norm(x) < 1e-12:
        x = np.array([0.0, 1.0, 0.0])
        x -= axis * np.dot(axis, x)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(axis, x), axis))


@dataclass(frozen=True)
class SpheroidPose:
    """Two equal oblate spheroids with supplied orientations; gap is the exact minimum surface distance.

    The two closest points are (0,0,0) and (0,0,gap_over_c). Parallel support
    planes and convexity certify the global minimum; this is not a centre
    separation or an approximate centre-line gap. Zero gap is valid geometry
    but is not a disjoint exterior Stokes domain. Negative gaps are rejected.
    """
    gap_nm: float
    aspect_ratio: float = 20.0
    length_unit_nm: float = C_NM
    theta_i: float = THETA_I
    theta_j: float = THETA_J
    phi: float = PHI
    axes: np.ndarray = field(init=False, repr=False)
    rotations: np.ndarray = field(init=False, repr=False)
    centers: np.ndarray = field(init=False, repr=False)
    shape_tensors: np.ndarray = field(init=False, repr=False)
    inverse_shape_tensors: np.ndarray = field(init=False, repr=False)
    c: ClassVar[float] = 1.0

    @property
    def a(self) -> float:
        return self.aspect_ratio

    @property
    def is_recorded(self) -> bool:
        return (self.aspect_ratio == 20.0 and self.length_unit_nm == C_NM and
                self.theta_i == THETA_I and self.theta_j == THETA_J and self.phi == PHI)

    def __post_init__(self) -> None:
        if not math.isfinite(self.gap_nm) or self.gap_nm < 0.0:
            raise ValueError("gap_nm must be finite and nonnegative")
        if not math.isfinite(self.aspect_ratio) or self.aspect_ratio < 1.0:
            raise ValueError("aspect_ratio must be finite and at least 1 (oblate or sphere)")
        if not math.isfinite(self.length_unit_nm) or self.length_unit_nm <= 0.0:
            raise ValueError("c_nm must be finite and positive")
        if not all(math.isfinite(t) and 0.0 <= t <= math.pi for t in (self.theta_i, self.theta_j)):
            raise ValueError("theta_i and theta_j must be finite radians in [0,pi]")
        if not math.isfinite(self.phi):
            raise ValueError("phi must be finite radians")
        axes = np.array([
            [math.sin(self.theta_i), 0.0, math.cos(self.theta_i)],
            [math.sin(self.theta_j)*math.cos(self.phi), math.sin(self.theta_j)*math.sin(self.phi),
             math.cos(self.theta_j)],
        ])
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        rotations = np.stack([_rotation(axis) for axis in axes])
        tensors = np.stack([self.a**2*np.eye(3) + (1-self.a**2)*np.outer(q, q)
                            for q in axes])
        inverse = np.stack([np.eye(3)/self.a**2 + (1-1/self.a**2)*np.outer(q, q)
                            for q in axes])
        support = tensors[:, :, 2] / np.sqrt(tensors[:, 2, 2])[:, None]
        centers = support * np.array([-1.0, 1.0])[:, None]
        centers[1, 2] += self.gap_over_c
        for name, value in (("axes", axes), ("rotations", rotations),
                            ("centers", centers), ("shape_tensors", tensors),
                            ("inverse_shape_tensors", inverse)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def gap_over_c(self) -> float:
        return self.gap_nm / self.length_unit_nm

    @property
    def contact_points(self) -> np.ndarray:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, self.gap_over_c]])

    @property
    def exact_area(self) -> float:
        """Area of ONE spheroid in c^2 units."""
        if self.a == self.c:
            return 4.0*math.pi*self.a**2
        e = math.sqrt(1.0 - (self.c/self.a)**2)
        return 2.0*math.pi*self.a**2 * (
            1.0 + self.c**2/(self.a**2*e)*math.atanh(e))

    @property
    def exact_volume(self) -> float:
        """Volume of ONE spheroid in c^3 units."""
        return 4.0*math.pi*self.a**2*self.c/3.0

    def contact_direction(self, body: int) -> np.ndarray:
        """Preimage of the closest point on the local unit parameter sphere."""
        _body(body)
        local = self.rotations[body].T @ (self.contact_points[body]-self.centers[body])
        local = local / np.array([self.a, self.a, self.c])
        return local / np.linalg.norm(local)

    def implicit_residual(self, points: np.ndarray, body: int) -> np.ndarray:
        _body(body)
        r = np.asarray(points, dtype=float) - self.centers[body]
        return np.einsum("...i,ij,...j->...", r, self.inverse_shape_tensors[body], r)-1.0

    def outward_normals(self, points: np.ndarray, body: int) -> np.ndarray:
        _body(body)
        r = np.asarray(points, dtype=float) - self.centers[body]
        gradient = r @ self.inverse_shape_tensors[body]
        size = np.linalg.norm(gradient, axis=-1)
        if np.any(size == 0.0):
            raise ValueError("Normal is undefined at the centre")
        return gradient / size[..., None]

    def metadata(self) -> dict:
        return {
            "geometry": ("exact_recorded_AR20_oblate_pair_v1" if self.is_recorded
                         else "exact_generic_oblate_pair_v1"),
            "aspect_ratio": self.aspect_ratio, "a_nm": self.a*self.length_unit_nm,
            "c_nm": self.length_unit_nm,
            "length_unit_nm": self.length_unit_nm, "coordinate_units": "c=1",
            "h_nm": self.gap_nm, "h_over_c": self.gap_over_c,
            "theta_i": self.theta_i, "theta_j": self.theta_j, "phi": self.phi,
            "axes": self.axes.tolist(), "centers": self.centers.tolist(),
            "rotations": self.rotations.tolist(),
            "contact_points": self.contact_points.tolist(),
            "normal_orientation": "out of solid, into exterior fluid",
            "mode_names": list(MODE_NAMES),
        }


class RecordedPose(SpheroidPose):
    """Compatibility constructor for the unchanged recorded AR20 geometry."""

    def __init__(self, gap_nm: float) -> None:
        super().__init__(gap_nm=gap_nm)


def spheroid_pose(gap_nm: float, *, aspect_ratio: float = 20.0, c_nm: float = C_NM,
                  theta_i: float = THETA_I, theta_j: float = THETA_J,
                  phi: float = PHI) -> SpheroidPose:
    """General equal-oblate-pair API; all input lengths are nm and angles radians."""
    return SpheroidPose(float(gap_nm), float(aspect_ratio), float(c_nm),
                        float(theta_i), float(theta_j), float(phi))


def recorded_pose(gap_nm: float) -> RecordedPose:
    return RecordedPose(float(gap_nm))


def canonical_geometry(gap_over_c: float, ar: float = 20.0) -> RecordedPose:
    """Convenience API for a dimensionless gap; returns a RecordedPose object."""
    if ar != 20.0:
        raise ValueError("This module is restricted to the recorded AR20 pose")
    return recorded_pose(float(gap_over_c)*C_NM)


def _body(body: int) -> None:
    if body not in (0, 1):
        raise ValueError("body must be 0 or 1")


@dataclass(frozen=True)
class SurfaceEvaluation:
    points: np.ndarray
    du: np.ndarray
    dv: np.ndarray
    normals: np.ndarray
    jacobian: np.ndarray
    # mu,phi coordinates have negative orientation; triangle u,v is positive.
    parameter_orientation: int

    @property
    def srcvals(self) -> np.ndarray:
        """Fortran-contiguous (12,N) x,x_u,x_v,outward-normal samples."""
        return np.asfortranarray(np.concatenate(
            [x.reshape(-1, 3) for x in (self.points, self.du, self.dv, self.normals)],
            axis=1).T)


def _map_sphere(pose: SpheroidPose, body: int, sphere: np.ndarray,
                du: np.ndarray, dv: np.ndarray, focus: float,
                orientation: int) -> SurfaceEvaluation:
    """Optional smooth Mobius reparameterization concentrates at the contact.

    0<=focus<1 is a dimensionless sphere-map parameter, not a geometric cutoff.
    It changes quadrature/panel density but does not change the spheroid. For
    focus>0 the local linear compression at contact is sqrt((1-focus)/(1+focus)).
    The far side is correspondingly coarser; it requires its own resolution test.
    """
    _body(body)
    if not math.isfinite(focus) or not 0.0 <= focus < 1.0:
        raise ValueError("focus must be finite and in [0,1)")
    if focus:
        k = pose.contact_direction(body)
        t = sphere @ k
        s = math.sqrt(1.0-focus*focus)
        denominator = 1.0 + focus*t
        mapped = (s*sphere + ((1.0-s)*t+focus)[..., None]*k) / denominator[..., None]
        def derivative(d: np.ndarray) -> np.ndarray:
            dt = d @ k
            return (s*d+(1.0-s)*dt[..., None]*k-
                    focus*dt[..., None]*mapped) / denominator[..., None]
        du, dv = derivative(du), derivative(dv)
        sphere = mapped
    # Row-vector convention: L maps a local unit-sphere vector into world space.
    L = pose.rotations[body] @ np.diag([pose.a, pose.a, pose.c])
    points = sphere @ L.T + pose.centers[body]
    xu, xv = du @ L.T, dv @ L.T
    area_vector = orientation*np.cross(xu, xv)
    jacobian = np.linalg.norm(area_vector, axis=-1)
    if np.any(jacobian <= 0.0):
        raise ValueError("Degenerate surface parameterization")
    normals = area_vector / jacobian[..., None]
    return SurfaceEvaluation(points, xu, xv, normals, jacobian, orientation)


def evaluate_mu_phi(pose: SpheroidPose, body: int, mu: np.ndarray,
                    phi: np.ndarray, focus: float = 0.0) -> SurfaceEvaluation:
    """Analytic tensor chart, -1<mu<1 and periodic phi; derivatives are d/dmu,d/dphi.

    The poles are chart singularities; triangular panels include them without a
    chart singularity. Gauss-Legendre tensor nodes never use the endpoints.
    """
    mu, phi = np.broadcast_arrays(np.asarray(mu, dtype=float), np.asarray(phi, dtype=float))
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(phi))) or np.any(np.abs(mu) >= 1.0):
        raise ValueError("Tensor chart requires finite phi and -1<mu<1")
    r = np.sqrt(1.0-mu*mu)
    cosine, sine = np.cos(phi), np.sin(phi)
    sphere = np.stack((r*cosine, r*sine, mu), axis=-1)
    du = np.stack((-mu*cosine/r, -mu*sine/r, np.ones_like(mu)), axis=-1)
    dv = np.stack((-r*sine, r*cosine, np.zeros_like(mu)), axis=-1)
    return _map_sphere(pose, body, sphere, du, dv, focus, -1)


@dataclass(frozen=True)
class SurfaceNodes:
    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    body_ids: np.ndarray
    du: np.ndarray
    dv: np.ndarray
    reference_weights: np.ndarray
    panel_ids: np.ndarray | None = None

    @property
    def srcvals(self) -> np.ndarray:
        return np.asfortranarray(np.concatenate(
            (self.points, self.du, self.dv, self.normals), axis=1).T)


def tensor_nodes(pose: SpheroidPose, n_mu: int, n_phi: int, *, phase: float = 0.0,
                 focus: float = 0.0) -> SurfaceNodes:
    """Independent Gauss(mu) x periodic(phi) integration on both bodies."""
    if n_mu < 2 or n_phi < 3 or not math.isfinite(phase):
        raise ValueError("n_mu>=2, n_phi>=3 and finite phase required")
    mu, w = leggauss(n_mu)
    phi = (np.arange(n_phi)+phase)*2.0*math.pi/n_phi
    values = [evaluate_mu_phi(pose, i, mu[:, None], phi[None, :], focus) for i in (0, 1)]
    rw = np.broadcast_to(w[:, None]*2.0*math.pi/n_phi, (n_mu, n_phi)).reshape(-1)
    flatten = lambda name: np.concatenate([getattr(v, name).reshape(-1, 3) for v in values])
    return SurfaceNodes(flatten("points"), flatten("normals"),
                        np.concatenate([v.jacobian.ravel()*rw for v in values]),
                        np.repeat(np.arange(2), n_mu*n_phi), flatten("du"), flatten("dv"),
                        np.tile(rw, 2))


def unit_sphere_triangles(subdivisions: int = 0) -> np.ndarray:
    """Conforming outward triangular sphere patches, shape (20*4**level,3,3).

    Edges are exact normalized chords (great-circle arcs); each subdivision
    normalizes shared midpoints. No planar triangle approximates the physical
    surface: evaluate_panels supplies its exact curved geometry.
    """
    if not isinstance(subdivisions, (int, np.integer)) or not 0 <= subdivisions <= 8:
        raise ValueError("subdivisions must be an integer in [0,8]")
    t = (1.0+math.sqrt(5.0))/2.0
    vertices = np.array([[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],
                         [0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],
                         [t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]], dtype=float)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.array([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
                      [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
                      [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
                      [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]])
    triangles = vertices[faces]
    for _ in range(subdivisions):
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        ab, bc, ca = a+b, b+c, c+a
        ab /= np.linalg.norm(ab, axis=1)[:, None]
        bc /= np.linalg.norm(bc, axis=1)[:, None]
        ca /= np.linalg.norm(ca, axis=1)[:, None]
        triangles = np.concatenate([np.stack(v, axis=1) for v in
                                    ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))])
    direction = np.einsum("ij,ij->i", np.cross(triangles[:,1]-triangles[:,0],
                                              triangles[:,2]-triangles[:,0]),
                          triangles.sum(axis=1))
    if np.any(direction <= 0.0):
        raise RuntimeError("Internal triangle orientation error")
    return triangles


def evaluate_panels(pose: SpheroidPose, body: int, triangles: np.ndarray,
                    uv: np.ndarray, focus: float = 0.0) -> SurfaceEvaluation:
    """Exact smooth map for backend-supplied triangular quadrature nodes.

    triangles: (npatch,3,3) unit-sphere corners, outward orientation.
    uv: (nquad,2), u>=0,v>=0,u+v<=1. Returns arrays (npatch,nquad,3).
    x_u cross x_v points outward. This makes positions/derivatives/normals usable
    by a high-order triangular BIE builder without tensor-chart pole singularities.
    """
    triangles, uv = np.asarray(triangles, dtype=float), np.asarray(uv, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3,3) or uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("Expected triangles (P,3,3) and uv (Q,2)")
    if not (np.all(np.isfinite(triangles)) and np.all(np.isfinite(uv))):
        raise ValueError("Nonfinite panel geometry")
    if np.any(uv < -1e-14) or np.any(uv.sum(axis=1) > 1.0+1e-14):
        raise ValueError("uv lies outside the reference triangle")
    if np.max(np.abs(np.linalg.norm(triangles, axis=-1)-1.0)) > 1e-12:
        raise ValueError("Panel corners must lie on the unit sphere")
    barycentric = np.column_stack((1.0-uv.sum(axis=1), uv))
    chord = np.einsum("qk,pkj->pqj", barycentric, triangles)
    size = np.linalg.norm(chord, axis=-1)
    if np.any(size <= 1e-14):
        raise ValueError("A panel crosses the parameter-sphere centre")
    sphere = chord/size[..., None]
    def derivative(d: np.ndarray) -> np.ndarray:
        d = d[:, None, :]
        return (d-sphere*np.sum(sphere*d, axis=-1)[..., None])/size[..., None]
    du = derivative(triangles[:,1]-triangles[:,0])
    dv = derivative(triangles[:,2]-triangles[:,0])
    sign = np.sum(np.cross(du, dv)*sphere, axis=-1)
    if np.any(sign <= 0.0):
        raise ValueError("Panel corner orientation must be outward")
    return _map_sphere(pose, body, sphere, du, dv, focus, 1)


def cube_sphere_patches(subdivisions: int = 0) -> np.ndarray:
    """Outward cube-face quadrilaterals, shape (6*4**level,4,3).

    Corners lie on the CUBE, not the sphere, in order 00,10,11,01. Their
    normalized bilinear map is an exact sphere patch. The local patch domain
    is [-1,1]^2, matching fmm3dbie iptype=11 quadrangular GL patches.
    """
    if not isinstance(subdivisions, (int, np.integer)) or not 0 <= subdivisions <= 8:
        raise ValueError("subdivisions must be an integer in [0,8]")
    frames = [([1,0,0],[0,1,0],[0,0,1]), ([-1,0,0],[0,-1,0],[0,0,1]),
              ([0,1,0],[-1,0,0],[0,0,1]), ([0,-1,0],[1,0,0],[0,0,1]),
              ([0,0,1],[1,0,0],[0,1,0]), ([0,0,-1],[1,0,0],[0,-1,0])]
    patches = []
    for n, u, v in frames:
        n, u, v = np.array(n), np.array(u), np.array(v)
        patches.append(np.stack((n-u-v,n+u-v,n+u+v,n-u+v)))
    result = np.asarray(patches, dtype=float)
    for _ in range(subdivisions):
        result = refine_quad_patches(result, np.arange(len(result)))
    return result


def refine_quad_patches(patches: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Split selected parametric quad patches into four exact subpatches.

    This supports geometric local refinement without changing either body.
    Neighbouring patches need not have matching polynomial nodes; a backend
    must support independent surface patches before using selective refinement.
    """
    patches = np.asarray(patches, dtype=float)
    indices = np.asarray(indices, dtype=int)
    if patches.ndim != 3 or patches.shape[1:] != (4,3):
        raise ValueError("Expected cube patches (P,4,3)")
    if np.any(indices < 0) or np.any(indices >= len(patches)):
        raise ValueError("Refinement index out of range")
    chosen = np.zeros(len(patches), dtype=bool);chosen[indices] = True
    p = patches[chosen]
    a,b,c,d = p[:,0],p[:,1],p[:,2],p[:,3]
    ab,bc,cd,da,mid = (a+b)/2,(b+c)/2,(c+d)/2,(d+a)/2,(a+b+c+d)/4
    children = [np.stack(v,axis=1) for v in
                ((a,ab,mid,da),(ab,b,bc,mid),(mid,bc,c,cd),(da,mid,cd,d))]
    return np.concatenate([patches[~chosen], *children])


def refine_rim_patches(patches: np.ndarray, levels: int,
                       half_width: float = 0.05) -> np.ndarray:
    """Binary z subdivision near the oblate rim, without azimuthal subdivision.

    Side cube faces have z varying only in their second parameter. Panels
    intersecting |cube z|<=half_width are split in that direction; cap faces
    remain unchanged. The band is a parameter-space resolution choice, not a
    physical gap cutoff. This construction assumes the unfocused sphere map.
    """
    if not isinstance(levels, (int,np.integer)) or not 0 <= levels <= 12:
        raise ValueError("rim_levels must be an integer in [0,12]")
    if not math.isfinite(half_width) or not 0.0 <= half_width < 1.0:
        raise ValueError("rim_width must be finite and in [0,1)")
    patches = np.asarray(patches,dtype=float).copy()
    if patches.ndim != 3 or patches.shape[1:] != (4,3):
        raise ValueError("Expected cube patches (P,4,3)")
    for _ in range(levels):
        zmin,zmax = patches[:,:,2].min(axis=1),patches[:,:,2].max(axis=1)
        chosen = ((zmin <= half_width) & (zmax >= -half_width) &
                  (zmax-zmin > 1e-15))
        p = patches[chosen]
        if not len(p):
            break
        a,b,c,d = p[:,0],p[:,1],p[:,2],p[:,3]
        # cube_sphere_patches and all supported refinements preserve this
        # alignment, so these two halves split z and preserve azimuth width.
        if np.max(np.abs(a[:,2]-b[:,2])) > 1e-13 or np.max(np.abs(c[:,2]-d[:,2])) > 1e-13:
            raise ValueError("Rim refinement requires z-aligned side-face patches")
        bc,da = (b+c)/2,(d+a)/2
        patches = np.concatenate((patches[~chosen],np.stack((a,b,bc,da),axis=1),
                                  np.stack((da,bc,c,d),axis=1)))
    return patches


def refine_contact_patches(patches: np.ndarray, direction: np.ndarray,
                           levels: int, radius: float = 0.25) -> np.ndarray:
    """Refine a conservative angular neighbourhood of a contact preimage.

    radius is in radians on the local unit parameter sphere. A patch is kept
    whenever its enclosing spherical cap intersects this neighbourhood, so the
    contact cannot be missed merely because no corner lies close to it. The
    optional Mobius map fixes the contact direction and preserves its inclusion.
    """
    if not isinstance(levels,(int,np.integer)) or not 0 <= levels <= 8:
        raise ValueError("contact_levels must be an integer in [0,8]")
    if not math.isfinite(radius) or not 0.0 <= radius < math.pi:
        raise ValueError("contact_radius must be finite and in [0,pi)")
    patches = np.asarray(patches,dtype=float).copy()
    direction = np.asarray(direction,dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) == 0.0:
        raise ValueError("Contact direction must be a finite nonzero 3-vector")
    direction = direction/np.linalg.norm(direction)
    for _ in range(levels):
        centers = patches.mean(axis=1)
        centers /= np.linalg.norm(centers,axis=1)[:,None]
        corners = patches/np.linalg.norm(patches,axis=2)[:,:,None]
        bound = np.arccos(np.clip(np.einsum("pki,pi->pk",corners,centers),-1,1)).max(axis=1)
        distance = np.arccos(np.clip(centers@direction,-1,1))
        patches = refine_quad_patches(patches,np.flatnonzero(distance <= radius+bound+1e-14))
    return patches


def refine_descriptors(descriptors: list, indices: np.ndarray,
                       axis: int | None = None) -> list:
    """Refine selected tagged patches, preserving parents' order and body IDs.

    axis=None makes four children; axis=0 or 1 makes two children by splitting
    the corresponding reference coordinate. The exact normalized cube map is
    unchanged. Child descriptors can be passed directly to build_mesh.
    """
    if axis not in (None,0,1):
        raise ValueError("axis must be None, 0, or 1")
    indices = np.asarray(indices,dtype=int)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(descriptors)):
        raise ValueError("Refinement indices must be in descriptor range")
    chosen = set(indices.tolist())
    out = []
    for index,(body,vertices) in enumerate(descriptors):
        _body(body)
        patch = np.asarray(vertices,dtype=float)
        if patch.shape != (4,3) or not np.all(np.isfinite(patch)):
            raise ValueError("Expected finite descriptor corners (4,3)")
        if index not in chosen:
            out.append((int(body),patch.copy()))
            continue
        a,b,c,d = patch
        ab,bc,cd,da,mid = (a+b)/2,(b+c)/2,(c+d)/2,(d+a)/2,(a+b+c+d)/4
        if axis == 0:
            children = ((a,ab,cd,d),(ab,b,c,cd))
        elif axis == 1:
            children = ((a,b,bc,da),(da,bc,c,d))
        else:
            children = ((a,ab,mid,da),(ab,b,bc,mid),(mid,bc,c,cd),(da,mid,cd,d))
        out.extend((int(body),np.stack(child)) for child in children)
    return out


def _quad_uv(uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, dtype=float)
    # The backend API supplies (2,Q); (Q,2) is also accepted except the
    # intrinsically ambiguous 2x2 case, for which backend column layout wins.
    if uv.ndim == 2 and uv.shape[0] == 2:
        uv = uv.T
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.all(np.isfinite(uv)):
        raise ValueError("Quad uv must have shape (2,Q) or (Q,2)")
    if np.any(np.abs(uv) > 1.0+1e-14):
        raise ValueError("Quad uv lies outside [-1,1]^2")
    return uv


def evaluate_quad_panels(pose: SpheroidPose, body: int, patches: np.ndarray,
                         uv: np.ndarray, focus: float = 0.0) -> SurfaceEvaluation:
    """Exact x,x_u,x_v,n for [-1,1]^2 quadrangles, shape (P,Q,3)."""
    patches, uv = np.asarray(patches, dtype=float), _quad_uv(uv)
    if patches.ndim != 3 or patches.shape[1:] != (4,3) or not np.all(np.isfinite(patches)):
        raise ValueError("Expected finite cube patches (P,4,3)")
    u,v = uv[:,0],uv[:,1]
    w = .25*np.stack(((1-u)*(1-v),(1+u)*(1-v),(1+u)*(1+v),(1-u)*(1+v)),axis=1)
    wu = .25*np.stack((-(1-v),1-v,1+v,-(1+v)),axis=1)
    wv = .25*np.stack((-(1-u),-(1+u),1+u,1-u),axis=1)
    chord = np.einsum("qk,pkj->pqj",w,patches)
    size = np.linalg.norm(chord,axis=-1)
    if np.any(size <= 1e-14):
        raise ValueError("A cube patch crosses the sphere centre")
    sphere = chord/size[...,None]
    def derivative(weights: np.ndarray) -> np.ndarray:
        d = np.einsum("qk,pkj->pqj",weights,patches)
        return (d-sphere*np.sum(sphere*d,axis=-1)[...,None])/size[...,None]
    du,dv = derivative(wu),derivative(wv)
    if np.any(np.sum(np.cross(du,dv)*sphere,axis=-1) <= 0.0):
        raise ValueError("Quad orientation must be outward")
    return _map_sphere(pose,body,sphere,du,dv,focus,1)


def build_mesh(gap_nm: float, order: int, level: int, uv: np.ndarray,
               isolated_body: int | None = None, focus: float = 0.0,
               rim_levels: int = 0, rim_width: float = 0.05,
               contact_levels: int = 0, contact_radius: float = 0.25,
               patch_descriptors: list | None = None,
               pose: SpheroidPose | None = None) -> dict:
    """Build exact geometry arrays for a quadrangular fmm3dbie backend.

    ``uv`` is the backend's (2,(order+1)^2) GL node array. All arrays use c=1;
    ``ixyzs`` is 1-based and all integer arrays are int64. Body IDs retain 0/1
    even for an isolated-body geometry check. No solver or coefficient transform
    is called. ``exact_patch_values(patch_id,new_uv)`` accepts a zero-based patch
    ID and returns fresh exact (12,Q) values for independent geometry checks.

    ``rim_levels`` binary-refines local equatorial z only (see refine_rim_patches)
    and requires focus=0. ``contact_levels`` adds independent local 4-way patch
    refinement for each body (see refine_contact_patches). Neither changes the
    physical gap, shape, or orientation. Each is an independent convergence axis.
    ``patch_descriptors`` overrides the cube/refinement pipeline with tagged
    (body,4x3 cube vertices) patches, in exactly the supplied order. This supports
    a posteriori refinement using refine_descriptors. It must cover each selected
    body exactly once; overlap/coverage checks remain the caller's responsibility.
    """
    if not isinstance(order,(int,np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    if isolated_body is not None:
        _body(isolated_body)
    if pose is None:
        pose = recorded_pose(gap_nm)
    elif not isinstance(pose, SpheroidPose) or pose.gap_nm != float(gap_nm):
        raise ValueError("Explicit pose must have the same dimensional gap_nm")
    nodes = _quad_uv(uv)
    if len(nodes) != (order+1)**2:
        raise ValueError("iptype11 requires (order+1)^2 backend GL nodes")
    if rim_levels and focus != 0.0:
        raise ValueError("Equatorial rim refinement requires focus=0; refine the two axes separately")
    bodies = (0,1) if isolated_body is None else (isolated_body,)
    if patch_descriptors is None:
        cube = refine_rim_patches(cube_sphere_patches(level),rim_levels,rim_width)
        body_patches = [refine_contact_patches(cube,pose.contact_direction(body),
                                              contact_levels,contact_radius) for body in bodies]
        descriptors = [(body,patch.copy()) for body,patches in zip(bodies,body_patches) for patch in patches]
    else:
        descriptors = []
        for body,vertices in patch_descriptors:
            _body(body)
            if body not in bodies:
                raise ValueError("Descriptor includes a body excluded by isolated_body")
            patch = np.asarray(vertices,dtype=float)
            if patch.shape != (4,3) or not np.all(np.isfinite(patch)):
                raise ValueError("Expected finite descriptor corners (4,3)")
            if not np.allclose(patch[0]+patch[2],patch[1]+patch[3],rtol=0,atol=1e-13):
                raise ValueError("Descriptors must be affine cube-face rectangles")
            cube_face = (np.ptp(patch,axis=0) < 1e-13) & (np.abs(np.abs(patch[0])-1.0) < 1e-13)
            if not np.any(cube_face) or np.any(np.abs(patch) > 1.0+1e-13):
                raise ValueError("Descriptor is not on a unit cube face")
            descriptors.append((int(body),patch.copy()))
        if set(body for body,_ in descriptors) != set(bodies):
            raise ValueError("Descriptors must include every requested body")
    nq = len(nodes);npatches = len(descriptors)
    srcvals = np.empty((12,npatches*nq),dtype=float,order="F")
    for body in bodies:
        indices = np.array([j for j,(owner,_) in enumerate(descriptors) if owner == body],dtype=int)
        patches = np.stack([descriptors[j][1] for j in indices])
        values = evaluate_quad_panels(pose,body,patches,nodes,focus).srcvals
        slots = (indices[:,None]*nq+np.arange(nq)[None,:]).ravel()
        srcvals[:,slots] = values
    def exact_patch_values(patch_id: int, new_uv: np.ndarray) -> np.ndarray:
        if not 0 <= patch_id < npatches:
            raise ValueError("patch_id out of range")
        body,patch = descriptors[patch_id]
        return evaluate_quad_panels(pose,body,patch[None,:,:],new_uv,focus).srcvals
    def locate_direction(body: int, unitdirection: np.ndarray) -> tuple[int,np.ndarray]:
        """Locate an ACTUAL local-sphere direction; return zero-based patch, uv(2,).

        The direction is after any focus map: diag(a,a,c)*direction rotated and
        shifted is its physical point. The inverse focus map is applied here.
        Shared-edge points may belong to either adjacent patch; first wins.
        """
        _body(body)
        direction = np.asarray(unitdirection,dtype=float)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) == 0:
            raise ValueError("unitdirection must be a finite nonzero 3-vector")
        direction = direction/np.linalg.norm(direction)
        if focus:
            k = pose.contact_direction(body);t = float(direction@k)
            s = math.sqrt(1.0-focus*focus)
            direction = (s*direction+((1.0-s)*t-focus)*k)/(1.0-focus*t)
        for patch_id,(owner,patch) in enumerate(descriptors):
            if owner != body:
                continue
            a,b,_,d = patch
            normal = np.cross(b-a,d-a)
            denominator = float(normal@direction)
            if denominator <= 0.0:
                continue
            chord = direction*float(normal@a)/denominator
            basis = np.column_stack((b-a,d-a))
            coordinates = np.linalg.lstsq(basis,chord-a,rcond=None)[0]
            if (np.linalg.norm(basis@coordinates-(chord-a)) < 1e-11 and
                    np.all(coordinates >= -1e-12) and np.all(coordinates <= 1.0+1e-12)):
                return patch_id,2.0*np.clip(coordinates,0,1)-1.0
        raise ValueError("Direction is outside the requested body's mesh")
    contact_locations = []
    for body in bodies:
        direction = pose.contact_direction(body)
        patch_id,local_uv = locate_direction(body,direction)
        contact_locations.append({"body": int(body),"patch_id": patch_id,
                                  "uv": local_uv.tolist(),"direction": direction.tolist()})
    return {
        "srcvals": srcvals, "norders": np.full(npatches,order,dtype=np.int64),
        "ixyzs": np.arange(npatches+1,dtype=np.int64)*nq+1,
        "iptype": np.full(npatches,11,dtype=np.int64),
        "body_ids": np.repeat(np.array([body for body,_ in descriptors],dtype=np.int64),nq),
        "centers": pose.centers.copy(), "axes": pose.axes.copy(),
        "rotations": pose.rotations.copy(), "pose_metadata": pose.metadata(),
        "patches": descriptors, "exact_patch_values": exact_patch_values,
        "locate_direction": locate_direction,"contact_patch_id_uv": contact_locations,
        "order": int(order), "level": int(level), "focus": float(focus),
        "rim_levels": int(rim_levels), "rim_width": float(rim_width),
        "contact_levels": int(contact_levels), "contact_radius": float(contact_radius),
        "custom_patch_descriptors": patch_descriptors is not None,
        "npatches": npatches, "npts": srcvals.shape[1],
    }


def triangular_nodes(pose: SpheroidPose, subdivisions: int, order: int,
                     *, focus: float = 0.0) -> SurfaceNodes:
    """Duffy-Gauss geometry quadrature; not a backend-specific VR node rule."""
    if order < 2:
        raise ValueError("order must be at least 2")
    x, w = leggauss(order)
    u, w = (x+1.0)/2.0, w/2.0
    uu, tt = np.meshgrid(u, u, indexing="ij")
    uv = np.column_stack((uu.ravel(), ((1.0-uu)*tt).ravel()))
    rw = (w[:, None]*w[None, :]*(1.0-uu)).ravel()
    triangles = unit_sphere_triangles(subdivisions)
    values = [evaluate_panels(pose, i, triangles, uv, focus) for i in (0, 1)]
    flatten = lambda name: np.concatenate([getattr(v, name).reshape(-1,3) for v in values])
    count = len(triangles)*len(uv)
    return SurfaceNodes(flatten("points"), flatten("normals"),
                        np.concatenate([(v.jacobian*rw).ravel() for v in values]),
                        np.repeat(np.arange(2), count), flatten("du"), flatten("dv"),
                        np.tile(rw, 2*len(triangles)),
                        np.repeat(np.arange(2*len(triangles)), len(uv)))


def quadrangular_nodes(pose: SpheroidPose, subdivisions: int, order: int,
                       *, focus: float = 0.0, rim_levels: int = 0,
                       contact_levels: int = 0) -> SurfaceNodes:
    """Independent numpy GL quadrature of the exact backend-compatible quads.

    This checks geometry only. A Stokes backend must use its own reference-node
    ordering and close-quadrature corrections, not assume this numpy ordering.
    """
    x,w = leggauss(order+1)
    u,v = np.meshgrid(x,x,indexing="ij")
    uv = np.stack((u.ravel(),v.ravel()))
    mesh = build_mesh(pose.gap_nm,order,subdivisions,uv,focus=focus,
                      rim_levels=rim_levels,contact_levels=contact_levels,pose=pose)
    sv = mesh["srcvals"]
    rw = np.tile(np.outer(w,w).ravel(),mesh["npatches"])
    jac = np.linalg.norm(np.cross(sv[3:6].T,sv[6:9].T),axis=1)
    return SurfaceNodes(sv[:3].T,sv[9:12].T,rw*jac,mesh["body_ids"],
                        sv[3:6].T,sv[6:9].T,rw,
                        np.repeat(np.arange(mesh["npatches"]),(order+1)**2))


def velocity_modes(points: np.ndarray, body_ids: np.ndarray,
                   pose: SpheroidPose) -> np.ndarray:
    """All 17 boundary disturbance velocities, shape (N,3,17), in world axes."""
    points, body_ids = np.asarray(points, dtype=float), np.asarray(body_ids)
    if points.ndim != 2 or points.shape[1] != 3 or body_ids.shape != (len(points),):
        raise ValueError("Expected points (N,3) and body_ids (N,)")
    if not np.all(np.isfinite(points)) or np.any((body_ids != 0) & (body_ids != 1)):
        raise ValueError("Invalid points or body identifiers")
    out = np.zeros((len(points), 3, 17))
    es = strain_basis()
    for body in (0, 1):
        mask = body_ids == body
        r = points[mask]-pose.centers[body]
        out[mask, :, 6*body:6*body+3] = np.eye(3)
        for k in range(3):
            out[mask, :, 6*body+3+k] = np.cross(np.eye(3)[k], r)/pose.a
        for k in range(5):
            out[mask, :, 12+k] = -r@es[k].T/pose.a
    return out


def _integrals(pose: SpheroidPose, nodes: SurfaceNodes) -> dict:
    area, volume, normal_flux, ellipsoid_error, normal_error = [], [], [], [], []
    for body in (0, 1):
        mask = nodes.body_ids == body
        x, normal, weight = nodes.points[mask], nodes.normals[mask], nodes.weights[mask]
        area.append(float(weight.sum()))
        volume.append(float(np.sum(weight*np.sum((x-pose.centers[body])*normal, axis=1))/3.0))
        normal_flux.append(float(np.linalg.norm(np.sum(weight[:,None]*normal, axis=0))/weight.sum()))
        ellipsoid_error.append(float(np.max(np.abs(pose.implicit_residual(x, body)))))
        normal_error.append(float(np.max(np.linalg.norm(normal-pose.outward_normals(x, body), axis=1))))
    return {"nodes": len(nodes.points), "area": area, "volume": volume,
            "area_relative_error": (np.abs(np.array(area)/pose.exact_area-1.0)).tolist(),
            "volume_relative_error": (np.abs(np.array(volume)/pose.exact_volume-1.0)).tolist(),
            "normal_flux_over_area": normal_flux,
            "max_implicit_surface_residual": ellipsoid_error,
            "max_normal_difference_from_implicit_geometry": normal_error}


def geometry_checks(pose: SpheroidPose) -> dict:
    """Small same-pose geometry tests only; never a Stokes/reference accuracy gate."""
    contact_normals = np.stack([pose.outward_normals(pose.contact_points[i], i) for i in (0,1)])
    contact_error = [float(abs(pose.implicit_residual(pose.contact_points[i], i))) for i in (0,1)]
    tensor = [_integrals(pose, tensor_nodes(pose, n, 24, phase=0.371)) for n in (16,32,64,128)]
    triangles = [_integrals(pose, triangular_nodes(pose, level, 8)) for level in (0,1,2)]
    quads = [{"order": 6,"level": level,
              **_integrals(pose,quadrangular_nodes(pose,level,6))} for level in (0,1,2,3)]
    rim = [{"order": 6,"level": 1,"rim_levels": levels,
            **_integrals(pose,quadrangular_nodes(pose,1,6,rim_levels=levels))}
           for levels in (0,1,2,4)]
    local = _integrals(pose,quadrangular_nodes(pose,1,6,rim_levels=2,contact_levels=1))
    fine = tensor_nodes(pose, 128, 24, phase=0.371)
    rhs = velocity_modes(fine.points, fine.body_ids, pose)
    flux = np.einsum("n,nc,nck->k", fine.weights, fine.normals, rhs)
    mode_flux = float(np.max(np.abs(flux))/fine.weights.sum())
    # Independent analytic chart versus finite differences at off-grid points.
    uv = np.array([[.17,.23],[.29,.41],[.03,.19]])
    tri = unit_sphere_triangles(1)[[0,7,29]]
    derivative_errors = []
    for body in (0,1):
        base = evaluate_panels(pose, body, tri, uv, focus=.4)
        for k, derivative in ((0,base.du),(1,base.dv)):
            delta = np.zeros_like(uv);delta[:,k] = 1e-6
            difference = (evaluate_panels(pose,body,tri,uv+delta,focus=.4).points-
                          evaluate_panels(pose,body,tri,uv-delta,focus=.4).points)/(2e-6)
            derivative_errors.append(float(np.linalg.norm(difference-derivative)/np.linalg.norm(derivative)))
    focused = _integrals(pose, tensor_nodes(pose, 128, 192, phase=.371, focus=.4))
    quad_derivative_errors = []
    quad_uv = np.array([[.17,.23],[-.29,.41],[.03,-.19]])
    cube = cube_sphere_patches(1)[[0,7,19]]
    locator_errors = []
    gx,_ = leggauss(5)
    gu,gv = np.meshgrid(gx,gx,indexing="ij")
    guv = np.stack((gu.ravel(),gv.ravel()))
    rng = np.random.default_rng(58120)
    for focus in (0.0,.4):
        for body in (0,1):
            base = evaluate_quad_panels(pose,body,cube,quad_uv,focus=focus)
            for k,derivative in ((0,base.du),(1,base.dv)):
                delta = np.zeros_like(quad_uv);delta[:,k] = 1e-6
                difference = (evaluate_quad_panels(pose,body,cube,quad_uv+delta,focus=focus).points-
                              evaluate_quad_panels(pose,body,cube,quad_uv-delta,focus=focus).points)/(2e-6)
                quad_derivative_errors.append(float(np.linalg.norm(difference-derivative)/np.linalg.norm(derivative)))
        mesh = build_mesh(pose.gap_nm,4,1,guv,focus=focus,
                          rim_levels=2 if not focus else 0,contact_levels=1,pose=pose)
        for body in (0,1):
            directions = np.vstack((pose.contact_direction(body),rng.normal(size=(32,3))))
            directions /= np.linalg.norm(directions,axis=1)[:,None]
            for direction in directions:
                patch_id,coordinates = mesh["locate_direction"](body,direction)
                actual = mesh["exact_patch_values"](patch_id,coordinates[:,None])[:3,0]
                expected = pose.rotations[body]@(np.array([pose.a,pose.a,pose.c])*direction)+pose.centers[body]
                locator_errors.append(float(np.linalg.norm(actual-expected)))
    passed = (max(contact_error) < 1e-11 and
              np.max(np.abs(contact_normals-np.array([[0,0,1],[0,0,-1]]))) < 1e-11 and
              max(tensor[-1]["area_relative_error"]) < 1e-8 and
              max(tensor[-1]["volume_relative_error"]) < 1e-11 and
              max(focused["area_relative_error"]) < 1e-7 and
              max(quads[-1]["area_relative_error"]) < 1e-7 and
              max(rim[-1]["area_relative_error"]) < 1e-7 and
              max(local["area_relative_error"]) < 1e-7 and
              mode_flux < 1e-11 and max(derivative_errors) < 1e-8 and
              max(quad_derivative_errors) < 1e-8 and max(locator_errors) < 1e-11)
    return {"scope": "supplied pose; exact geometry/quadrature only, no Stokes solve",
            "geometry_passed": bool(passed), "physical_reference_qualified": False,
            "pose": pose.metadata(), "exact_single_body_area": pose.exact_area,
            "exact_single_body_volume": pose.exact_volume,
            "exact_gap_nm": float(np.linalg.norm(pose.contact_points[1]-pose.contact_points[0])*pose.length_unit_nm),
            "contact_surface_residual": contact_error, "contact_normals": contact_normals.tolist(),
            "tensor_refinement": tensor, "triangular_refinement": triangles,
            "quadrangular_refinement": quads,"anisotropic_rim_refinement": rim,
            "locally_refined_quadrangles": local,
            "focused_tensor": focused, "max_relative_panel_derivative_error": max(derivative_errors),
            "max_relative_quad_derivative_error": max(quad_derivative_errors),
            "max_direction_locator_position_error": max(locator_errors),
            "max_mode_normal_flux_over_total_area": mode_flux}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-nm", type=float, required=True)
    parser.add_argument("--check", action="store_true", help="run small geometry-only checks")
    args = parser.parse_args()
    pose = recorded_pose(args.gap_nm)
    result = geometry_checks(pose) if args.check else pose.metadata()
    print(json.dumps(result, indent=2))
    return 0 if not args.check or result["geometry_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
