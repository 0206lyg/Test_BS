#!/usr/bin/env python3
"""Exact gap-shape screening at the one recorded AR20 pose; no Stokes solve.

The common contact tangent plane is the world xy plane. For the lower body
use its upper z branch, and for the upper body its lower z branch. Their
difference is the exact vertical film thickness wherever both projections
exist. A quadratic gap-doubling ellipse is a geometric length scale, not a
hydrodynamic error estimate or a validated matching interval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from geometry import C_NM,RecordedPose,recorded_pose

GAPS_NM = (150.0,75.0,30.0,10.0,3.0,1.0)


def relative_surface_z(pose: RecordedPose, body: int, xy: np.ndarray):
    """Exact selected surface z relative to its closest-point tangent plane.

    Returns z, discriminant, and the projected-silhouette membership mask.
    Outside the silhouette, z is NaN: there is no vertical film column there.
    The branch sign is fixed by the body's upper/lower face, not by a changing
    centre-line or a guessed local normal.
    """
    xy = np.asarray(xy,dtype=float)
    if body not in (0,1) or xy.shape[-1] != 2 or not np.all(np.isfinite(xy)):
        raise ValueError("Expected body0/1 and finite xy(...,2)")
    q = pose.inverse_shape_tensors[body]
    r0 = pose.contact_points[body]-pose.centers[body]
    qr0 = q@r0
    b = xy@q[:2,2]+qr0[2]
    c = np.einsum("...i,ij,...j->...",xy,q[:2,:2],xy)+2*xy@qr0[:2]+float(r0@qr0-1)
    a = float(q[2,2])
    discriminant = b*b-a*c
    inside = discriminant >= 0.0
    root = np.sqrt(np.maximum(discriminant,0.0))
    branch = 1.0 if body == 0 else -1.0
    denominator = b+branch*root
    # Rationalized expression avoids cancellation near the contact. On the
    # other side of a turning point use the direct branch expression instead.
    direct = (-b+branch*root)/a
    with np.errstate(divide="ignore",invalid="ignore"):
        stable = -c/denominator
    z = np.where(branch*b >= 0,stable,direct)
    return np.where(inside,z,np.nan),discriminant,inside


def contact_hessians(pose: RecordedPose):
    """Positive body contributions to the gap Hessian, from implicit quadrics.

    At contact grad_xy(F)=0 and F_z=2*(Q*r0)_z. Implicit differentiation gives
    z_,ab=-Q_ab/(Q*r0)_z. The lower body's contribution is -H(z_lower), and
    the upper body's contribution is +H(z_upper).
    """
    contributions = []
    tangent_gradients = []
    for body in (0,1):
        q = pose.inverse_shape_tensors[body]
        gradient = q@(pose.contact_points[body]-pose.centers[body])
        surface = -q[:2,:2]/gradient[2]
        contributions.append(-surface if body == 0 else surface)
        tangent_gradients.append(gradient[:2].tolist())
    return np.asarray(contributions),tangent_gradients


def hessian_finite_difference(pose: RecordedPose, body: int, step: float):
    def z(x,y):
        return float(relative_surface_z(pose,body,np.array([x,y]))[0])
    z0 = z(0,0)
    h = np.array([[(z(step,0)-2*z0+z(-step,0))/step**2,0],
                  [0,(z(0,step)-2*z0+z(0,-step))/step**2]])
    h[0,1] = h[1,0] = (z(step,step)-z(step,-step)-z(-step,step)+z(-step,-step))/(4*step**2)
    return -h if body == 0 else h


def nearest_projected_rim(pose: RecordedPose, body: int) -> dict:
    """Closest projected silhouette to the contact origin, from an exact ellipse."""
    center = pose.centers[body,:2]
    inverse = np.linalg.inv(pose.shape_tensors[body,:2,:2])
    constant = float(center@inverse@center-1)
    if constant >= 0:
        raise ValueError("The contact projection is not strictly inside the silhouette")
    def radius(angle):
        d = np.array([math.cos(angle),math.sin(angle)])
        aa,bb = float(d@inverse@d),float(-d@inverse@center)
        return float(-constant/(bb+math.sqrt(bb*bb-aa*constant)))
    angles = np.arange(720)*2*math.pi/720
    values = np.array([radius(t) for t in angles])
    index = int(np.argmin(values))
    left,right = angles[index]-2*math.pi/720,angles[index]+2*math.pi/720
    ratio = (math.sqrt(5)-1)/2
    c,d = right-ratio*(right-left),left+ratio*(right-left)
    fc,fd = radius(c),radius(d)
    for _ in range(70):
        if fc < fd:
            right,d,fd = d,c,fc
            c = right-ratio*(right-left);fc = radius(c)
        else:
            left,c,fc = c,d,fd
            d = left+ratio*(right-left);fd = radius(d)
    angle = .5*(left+right)
    distance = radius(angle)
    point = distance*np.array([math.cos(angle),math.sin(angle)])
    return {"body":body,"distance_nm":distance*C_NM,
            "azimuth_degrees":math.degrees(angle)%360,"point_xy_nm":(point*C_NM).tolist(),
            "projected_ellipse_equation_residual":float(abs((point-center)@inverse@(point-center)-1))}


def invalid_sectors(mask: np.ndarray, angles_degrees: np.ndarray) -> list:
    """Inclusive sampled-angle intervals; adjacent 0/360 sectors stay separate."""
    if not np.any(mask):
        return []
    starts = np.flatnonzero(mask & np.r_[True,~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:],True])
    return [[float(angles_degrees[a]),float(angles_degrees[b])] for a,b in zip(starts,ends)]


def diagnostic(angle_count: int = 720) -> dict:
    if angle_count < 32:
        raise ValueError("Use at least32 angular samples")
    pose = recorded_pose(GAPS_NM[0])
    body_hessians,tangent_gradients = contact_hessians(pose)
    total = body_hessians.sum(axis=0)
    curvature,eigenvectors = np.linalg.eigh(total)
    if np.any(curvature <= 0):
        raise ValueError("Contact gap is not locally positive definite")
    # Fix eigenvector signs solely for stable, reproducible angular labels.
    for i in range(2):
        if eigenvectors[np.argmax(np.abs(eigenvectors[:,i])),i] < 0:
            eigenvectors[:,i] *= -1
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:,1] *= -1
    differences = []
    for step in (.1,.03,.01,.003,.001):
        fd = np.asarray([hessian_finite_difference(pose,b,step) for b in (0,1)])
        differences.append({"step_c":step,"step_nm":step*C_NM,
                            "body_relative_frobenius_errors":[
                                float(np.linalg.norm(fd[b]-body_hessians[b])/np.linalg.norm(body_hessians[b]))
                                for b in (0,1)]})
    rim = [nearest_projected_rim(pose,b) for b in (0,1)]
    angles = np.arange(angle_count)*2*math.pi/angle_count
    circle = np.stack((np.cos(angles),np.sin(angles)),axis=1)
    angle_degrees = np.degrees(angles)
    results = []
    for gap_nm in GAPS_NM:
        current = recorded_pose(gap_nm)
        h = current.gap_over_c
        semiaxes = np.sqrt(2*h/curvature)
        xy = (circle*semiaxes[None,:])@eigenvectors.T
        lower,dl,inside_l = relative_surface_z(current,0,xy)
        upper,du,inside_u = relative_surface_z(current,1,xy)
        valid = inside_l & inside_u
        exact = h+upper-lower
        quadratic = h+.5*np.einsum("ni,ij,nj->n",xy,total,xy)
        error = (exact-2*h)/(2*h)
        implicit_errors = []
        silhouette_mismatches = []
        for body,z,inside in ((0,lower,inside_l),(1,upper,inside_u)):
            points = np.column_stack((xy[inside],z[inside]+current.contact_points[body,2]))
            implicit_errors.append(float(np.max(np.abs(current.implicit_residual(points,body)))))
            center = current.centers[body,:2]
            projection_inverse = np.linalg.inv(current.shape_tensors[body,:2,:2])
            projected_inside = np.einsum("ni,ij,nj->n",xy-center,projection_inverse,xy-center) <= 1
            silhouette_mismatches.append(int(np.sum(inside != projected_inside)))
        indices = np.flatnonzero(valid)
        def extremum(index):
            return {"ellipse_angle_degrees":float(angle_degrees[index]),
                    "world_azimuth_degrees":float(np.degrees(np.arctan2(xy[index,1],xy[index,0]))%360),
                    "point_xy_nm":(xy[index]*C_NM).tolist(),
                    "exact_gap_nm":float(exact[index]*C_NM),
                    "signed_error_over_quadratic_full_gap":float(error[index])}
        worst = indices[np.argmax(np.abs(error[valid]))]
        positive = indices[np.argmax(error[valid])]
        negative = indices[np.argmin(error[valid])]
        results.append({
            "h_nm":gap_nm,"h_over_c":h,"h_over_a":h/current.a,
            "quadratic_full_gap_nm":2*gap_nm,
            "ellipse_semiaxes_nm":(semiaxes*C_NM).tolist(),
            "ellipse_major_axis_over_nearest_rim_distance":float(max(semiaxes)*C_NM/min(x["distance_nm"] for x in rim)),
            "ellipse_equation_max_relative_residual":float(np.max(np.abs(quadratic/(2*h)-1))),
            "exact_surface_implicit_residual_by_body":implicit_errors,
            "silhouette_membership_mismatches_with_projected_quadric":silhouette_mismatches,
            "angle_count":angle_count,"valid_gap_columns":int(valid.sum()),
            "outside_lower_projection_count":int((~inside_l).sum()),
            "outside_upper_projection_count":int((~inside_u).sum()),
            "outside_either_projection_count":int((~valid).sum()),
            "outside_sectors_ellipse_degrees":invalid_sectors(~valid,angle_degrees),
            "valid_sample_max_abs_error_over_quadratic_full_gap":float(np.max(np.abs(error[valid]))),
            "valid_sample_rms_error_over_quadratic_full_gap":float(np.sqrt(np.mean(error[valid]**2))),
            "worst_valid_sample":extremum(worst),
            "largest_positive_error":extremum(positive),
            "smallest_signed_error":extremum(negative),
            "sample_columns":["ellipse_angle_degrees","x_nm","y_nm","exact_gap_nm",
                              "signed_error_over_quadratic_full_gap","inside_lower","inside_upper"],
            "samples":[[float(angle_degrees[i]),float(xy[i,0]*C_NM),float(xy[i,1]*C_NM),
                        float(exact[i]*C_NM) if valid[i] else None,
                        float(error[i]) if valid[i] else None,bool(inside_l[i]),bool(inside_u[i])]
                       for i in range(angle_count)]})
    geometry_checks_passed = (max(differences[-1]["body_relative_frobenius_errors"]) < 1e-7 and
                              max(r["projected_ellipse_equation_residual"] for r in rim) < 1e-12 and
                              max(max(g["exact_surface_implicit_residual_by_body"]) for g in results) < 1e-11 and
                              sum(sum(g["silhouette_membership_mismatches_with_projected_quadric"]) for g in results) == 0)
    return {
        "schema":"recorded_ar20_exact_gap_geometry_v1",
        "scope":"exact geometry at one fixed AR20 pose; no Stokes solve",
        "hydrodynamic_accuracy_validated":False,"matching_interval_selected":False,
        "geometry_checks_passed":bool(geometry_checks_passed),
        "geometry":pose.metadata(),
        "interpretation":"Signed error=(exact vertical gap-2h)/(2h) on .5*x^T*H*x=h. "
                         "Outside either projected silhouette the two-surface vertical gap is undefined. "
                         "Finite-angle maxima are sampled geometric errors, not hydrodynamic error bounds.",
        "body_gap_hessians_c_inverse":body_hessians.tolist(),
        "body_principal_curvature_radii_nm":[(C_NM/np.linalg.eigvalsh(h)).tolist() for h in body_hessians],
        "gap_hessian_c_inverse":total.tolist(),
        "gap_principal_curvature_radii_nm":(C_NM/curvature).tolist(),
        "principal_directions_xy_columns":eigenvectors.tolist(),
        "contact_implicit_half_gradient_xy":tangent_gradients,
        "hessian_finite_difference_checks":differences,
        "nearest_projected_rims":rim,
        "gaps":results,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--angles",type=int,default=720)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    result = diagnostic(args.angles)
    result["source_sha256"] = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("gap_geometry.py","geometry.py")}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output),"geometry_checks_passed":result["geometry_checks_passed"],
                      "hydrodynamic_accuracy_validated":False}))
    return 0 if result["geometry_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
