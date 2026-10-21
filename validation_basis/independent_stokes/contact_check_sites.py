"""Additional boundary-check sites from the exact two-spheroid gap geometry.

This module only selects off-grid surface targets. It neither changes the BIE
operator nor establishes a boundary-error bound. The historical closest-point
and circular-ring targets are retained. Gap-Hessian ellipses supplement them;
their different physical widths resolve the face/rim distinction that a single
parameter-sphere radius cannot represent.
"""
from __future__ import annotations

import math

import numpy as np


def _gap_hessian(pose):
    contributions = []
    for body in (0, 1):
        q = np.asarray(pose.inverse_shape_tensors[body], dtype=float)
        r = pose.contact_points[body] - pose.centers[body]
        gradient = q @ r
        if (not np.isfinite(gradient).all() or abs(gradient[2]) == 0 or
                np.linalg.norm(gradient[:2]) > 1e-10 * abs(gradient[2])):
            raise ValueError('Expected the exact closest-point common xy tangent plane')
        surface_hessian = -q[:2, :2] / gradient[2]
        contributions.append(-surface_hessian if body == 0 else surface_hessian)
    total = np.sum(contributions, axis=0)
    values, vectors = np.linalg.eigh(total)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('The exact gap Hessian must be positive definite')
    return total, values, vectors


def _projection_radius(pose, direction):
    """Positive distance from the contact projection to either silhouette."""
    result = math.inf
    for body in (0, 1):
        center = pose.centers[body, :2]
        inverse = np.linalg.inv(pose.shape_tensors[body, :2, :2])
        aa = float(direction @ inverse @ direction)
        bb = float(-direction @ inverse @ center)
        cc = float(center @ inverse @ center - 1)
        if aa <= 0 or cc >= 0:
            raise ValueError('Contact must be inside both projected spheroids')
        root = math.sqrt(bb * bb - aa * cc)
        radius = -cc / (bb + root) if bb >= 0 else (-bb + root) / aa
        result = min(result, radius)
    return result


def _surface_direction(pose, body, xy):
    """Unit parameter-sphere direction for the exact upper/lower surface."""
    q = np.asarray(pose.inverse_shape_tensors[body], dtype=float)
    r0 = pose.contact_points[body] - pose.centers[body]
    gradient = q @ r0
    aa = float(q[2, 2])
    bb = float(xy @ q[:2, 2] + gradient[2])
    cc = float(xy @ q[:2, :2] @ xy + 2 * xy @ gradient[:2] + r0 @ gradient - 1)
    discriminant = bb * bb - aa * cc
    if discriminant < -1e-12 * max(bb * bb, abs(aa * cc), 1.):
        raise ValueError('Hessian boundary-check target lies outside a projection')
    root = math.sqrt(max(discriminant, 0.))
    branch = 1. if body == 0 else -1.
    denominator = bb + branch * root
    z = (-cc / denominator if branch * bb >= 0 and denominator != 0
         else (-bb + branch * root) / aa)
    point = np.array([xy[0], xy[1], pose.contact_points[body, 2] + z])
    local = pose.rotations[body].T @ (point - pose.centers[body])
    direction = local / np.array([pose.a, pose.a, pose.c])
    size = float(np.linalg.norm(direction))
    if not math.isfinite(size) or abs(size - 1.) > 1e-9:
        raise ValueError('Exact surface target failed the spheroid equation')
    return direction / size


def build_contact_check_sites(mesh, pose, *, edge_avoidance=1e-3,
                              ring_factors=(.25, 1.), azimuth_count=8,
                              projection_fraction=.8):
    """Return ``(patch_ids, uv, metadata)`` for legacy and Hessian targets.

    ``uv`` has shape (2,K), is Fortran contiguous, and avoids shared panel edges
    required by the backend's self quadrature. Thus a requested exact contact
    on an edge is sampled just inside the selected panel, as in the historical
    driver. Each Hessian ellipse has .5*x.T*H*x = factor**2*h. For large gaps,
    each ray is shortened to remain inside ``projection_fraction`` times the
    closer projected silhouette; no extrapolated surface branch is sampled.
    """
    if not 0 < edge_avoidance < .1 or not 0 < projection_fraction < 1:
        raise ValueError('Invalid panel-edge avoidance or projection fraction')
    if (isinstance(azimuth_count, bool) or int(azimuth_count) != azimuth_count or
            not 4 <= azimuth_count <= 128):
        raise ValueError('azimuth_count must be an integer in [4,128]')
    factors = tuple(float(value) for value in ring_factors)
    if not factors or not all(math.isfinite(value) and 0 < value <= 4 for value in factors):
        raise ValueError('ring_factors must be finite and in (0,4]')
    h = float(pose.gap_over_c)
    if not math.isfinite(h) or h <= 0:
        raise ValueError('Boundary checks require a positive finite surface gap')
    contacts = mesh['contact_patch_id_uv']
    bodies = [int(item['body']) for item in contacts]
    if not bodies or len(set(bodies)) != len(bodies) or any(body not in (0, 1) for body in bodies):
        raise ValueError('Expected at most one closest-point entry per body')
    patch_ids, targets = [], []
    edge_adjusted = 0

    def add(body, direction):
        nonlocal edge_adjusted
        pid, uv = mesh['locate_direction'](body, direction)
        bounded = np.clip(np.asarray(uv, dtype=float), -1 + edge_avoidance, 1 - edge_avoidance)
        edge_adjusted += int(np.any(bounded != uv))
        patch_ids.append(int(pid))
        targets.append(bounded)

    # Preserve the original 17 requested sites per selected body exactly.
    legacy_width = math.sqrt(h / pose.a)
    for item in contacts:
        body = int(item['body'])
        direction = np.asarray(item['direction'], dtype=float)
        tangent = np.cross(direction, [1., 0., 0.])
        if np.linalg.norm(tangent) < 1e-12:
            tangent = np.cross(direction, [0., 1., 0.])
        tangent /= np.linalg.norm(tangent)
        other = np.cross(direction, tangent)
        add(body, direction)
        for radius in (.25 * legacy_width, legacy_width):
            for angle in np.arange(8) * np.pi / 4 + .173:
                candidate = direction + radius * (np.cos(angle) * tangent + np.sin(angle) * other)
                add(body, candidate / np.linalg.norm(candidate))
    legacy_count = len(patch_ids)
    hessian, curvature, vectors = _gap_hessian(pose)
    semiaxes = np.sqrt(2 * h / curvature)
    rays = []
    clipped = 0
    max_used_radius = 0.
    for factor in factors:
        for angle in np.arange(azimuth_count) * (2 * np.pi / azimuth_count) + .173:
            xy = vectors @ (factor * semiaxes * np.array([np.cos(angle), np.sin(angle)]))
            radius = float(np.linalg.norm(xy))
            direction = xy / radius
            limit = projection_fraction * _projection_radius(pose, direction)
            if radius > limit:
                xy = direction * limit
                clipped += 1
            max_used_radius = max(max_used_radius, float(np.linalg.norm(xy)))
            rays.append(xy)
    for body in bodies:
        for xy in rays:
            add(body, _surface_direction(pose, body, xy))
    uv = np.asfortranarray(np.asarray(targets, dtype=float).T)
    metadata = {
        'schema': 'exact_spheroid_contact_check_sites_v1',
        'scope': 'sampled polynomial-panel trace targets; no global error bound',
        'legacy_sites_retained': True,
        'legacy_site_count': legacy_count,
        'hessian_site_count': len(patch_ids) - legacy_count,
        'site_count': len(patch_ids),
        'body_ids': bodies,
        'gap_hessian_c_inverse': hessian.tolist(),
        'gap_doubling_ellipse_semiaxes_c': semiaxes.tolist(),
        'gap_doubling_ellipse_semiaxes_nm': (semiaxes * pose.length_unit_nm).tolist(),
        'ring_factors': list(factors),
        'azimuth_count': int(azimuth_count),
        'projection_fraction': float(projection_fraction),
        'projection_clipped_ray_count': clipped,
        'maximum_used_xy_radius_c': max_used_radius,
        'edge_avoidance_uv': float(edge_avoidance),
        'edge_adjusted_site_count': edge_adjusted,
        'closest_point_on_shared_edge_is_shifted': True,
    }
    return np.asarray(patch_ids, dtype=np.int64), uv, metadata
