#!/usr/bin/env python3
"""Evidence-guided mesh proposals for the one recorded AR20 BIE reference.

No Stokes solve or acceptance decision occurs here. The sampled boundary
residual ranks patches; tensor Legendre density tails choose split directions.
The tails are resolution indicators, not estimates of reference uncertainty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import legvander

from geometry import THETA_I, THETA_J, PHI, refine_descriptors


def density_tails(density: np.ndarray, uv: np.ndarray, npatches: int) -> np.ndarray:
    """Return each patch's last-three-degree L2 tails in its u and v axes.

    Columns are normalized by their global density RMS to prevent one strong
    resistance mode dominating the direction selection. Legendre L2 weights
    integrate in parameter space; these are not physical force/stress errors.
    Backend node order is recovered from uv, never assumed.
    """
    density,uv = np.asarray(density,dtype=float),np.asarray(uv,dtype=float)
    if uv.ndim != 2 or uv.shape[0] != 2 or density.ndim != 3 or density.shape[1:] != (3,17):
        raise ValueError("Expected uv(2,Q), density(N,3,17)")
    x,iu = np.unique(uv[0],return_inverse=True)
    y,iv = np.unique(uv[1],return_inverse=True)
    q = uv.shape[1]
    if len(x) != len(y) or len(x)**2 != q or len(np.unique(iu*len(y)+iv)) != q:
        raise ValueError("Expected a complete tensor quadrature grid")
    if density.shape[0] != npatches*q or not np.all(np.isfinite(density)):
        raise ValueError("Density/patch count mismatch or nonfinite density")
    n = len(x)
    if n < 4:
        raise ValueError("Directional tails require order>=3")
    ax,bx = np.linalg.inv(legvander(x,n-1)),np.linalg.inv(legvander(y,n-1))
    norms = np.sqrt(np.mean(density*density,axis=(0,1)))
    active = norms > 1e-100
    norms[~active] = 1.0
    weights = 2.0/(2*np.arange(n)+1)
    w = weights[:,None]*weights[None,:]
    result = np.zeros((npatches,2))
    for p in range(npatches):
        values = np.empty((n,n,3,17))
        values[iu,iv] = density[p*q:(p+1)*q]
        coefficients = np.einsum("iu,jv,uvcm->ijcm",ax,bx,values)
        energy = (coefficients/norms[None,None,None,:])**2*w[:,:,None,None]
        energy[:,:,:,~active] = 0.0
        total = float(energy.sum())
        if total:
            result[p] = np.sqrt([energy[-3:].sum()/total,energy[:,-3:].sum()/total])
    return result


def refine_with_axes(descriptors: list, choices: dict[int,int | None]) -> list:
    """Refine tagged patches with independent binary/four-way choices."""
    if any(i < 0 or i >= len(descriptors) for i in choices):
        raise ValueError("Patch index out of range")
    result = []
    for i,descriptor in enumerate(descriptors):
        if i in choices:
            result.extend(refine_descriptors([descriptor],[0],axis=choices[i]))
        else:
            result.extend(refine_descriptors([descriptor],[]))
    return result


def propose(record: dict, tails: np.ndarray, *, max_nodes: int = 10000,
            output_order: int | None = None, target: float = .001,
            anisotropy_ratio: float = 3.0, rim_width: float = .05) -> dict:
    """Add a rim refinement then spend remaining nodes on large residuals.

    An isotropic patch split costs three added patches; a binary split costs
    one. Rim panels are split in z first. For large residuals with comparable
    u/v tails, an existing binary choice may be upgraded to four-way.
    """
    if (record.get("aspect_ratio") != 20 or
            abs(record.get("theta_i",0)-THETA_I) > 1e-14 or
            abs(record.get("theta_j",0)-THETA_J) > 1e-14 or
            abs(record.get("phi",0)-PHI) > 1e-14):
        raise ValueError("Only the exact recorded AR20 pose is supported")
    if record.get("focus",0) != 0:
        raise ValueError("Rim-aware proposals require the unfocused cube map")
    descriptors = record["patch_descriptors"]
    scores = np.asarray(record["validation"]["boundary_per_patch_max_scaled_error"],dtype=float)
    tails = np.asarray(tails,dtype=float)
    if scores.shape != (len(descriptors),) or tails.shape != (len(descriptors),2):
        raise ValueError("Score/descriptor shape mismatch")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(tails)) or np.any(scores < 0) or np.any(tails < 0):
        raise ValueError("Invalid residual/tail indicators")
    order = record["order"] if output_order is None else output_order
    if not isinstance(order,(int,np.integer)) or order < 3 or max_nodes < 1 or target <= 0 or anisotropy_ratio <= 1:
        raise ValueError("Invalid adaptation settings")
    per_patch = (order+1)**2
    capacity = max_nodes//per_patch-len(descriptors)
    if capacity < 0:
        raise ValueError("Parent mesh at requested order already exceeds node budget")
    choices,reasons = {},{}
    for i,(body,vertices) in enumerate(descriptors):
        patch = np.asarray(vertices,dtype=float)
        spans = np.ptp(patch,axis=0)
        if spans[2] <= 1e-14 or patch[:,2].min() > rim_width or patch[:,2].max() < -rim_width:
            continue
        if capacity == 0:
            break
        du,dv = patch[1]-patch[0],patch[3]-patch[0]
        axis = int(abs(dv[2]) > abs(du[2]))
        choices[i] = axis;reasons[i] = "equatorial normal/density resolution"
        capacity -= 1
    for i in np.argsort(scores)[::-1]:
        i = int(i)
        if scores[i] <= target:
            break
        small,large = float(tails[i].min()),float(tails[i].max())
        desired = int(np.argmax(tails[i])) if large > anisotropy_ratio*max(small,1e-100) else None
        if i in choices:
            if desired == choices[i]:
                continue
            if choices[i] is None:
                continue
            desired = None
            cost = 2
        else:
            cost = 3 if desired is None else 1
        if cost > capacity:
            continue
        choices[i] = desired
        reasons[i] = "sampled no-slip residual and directional density tails"
        capacity -= cost
    children = refine_with_axes(descriptors,choices)
    return {
        "schema":"independent_ar20_mesh_proposal_v1",
        "physical_accuracy_qualified":False,"contains_new_stokes_solve":False,
        "aspect_ratio":20.0,"theta_i":THETA_I,"theta_j":THETA_J,"phi":PHI,
        "h_over_c":record["h_over_c"],"h_nm":record["h_nm"],
        "focus":0.0,"isolated_body":record.get("isolated_body"),"order":int(order),
        "parent_patch_count":len(descriptors),"patch_count":len(children),
        "node_count":len(children)*per_patch,"max_nodes":int(max_nodes),
        "target_sampled_boundary_error":float(target),
        "selection_is_error_bound":False,
        "split_choices":[{"parent_patch":i,"axis":choices[i],"reason":reasons[i],
                          "boundary_score":float(scores[i]),"directional_tails":tails[i].tolist()}
                         for i in sorted(choices)],
        "patch_descriptors":[[int(body),vertices.tolist()] for body,vertices in children],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent",required=True,type=Path)
    p.add_argument("--library",required=True)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--max-nodes",type=int,default=10000)
    p.add_argument("--order",type=int)
    p.add_argument("--target",type=float,default=.001)
    args = p.parse_args()
    from backend import Backend
    record = json.loads(args.parent.read_text())
    with np.load(record["solution_file"]) as saved:
        density = saved["density"]
    uv = Backend(args.library).discretization(record["order"])[0]
    tails = density_tails(density,uv,len(record["patch_descriptors"]))
    result = propose(record,tails,max_nodes=args.max_nodes,output_order=args.order,target=args.target)
    result["parent_path"] = str(args.parent.resolve())
    result["parent_sha256"] = hashlib.sha256(args.parent.read_bytes()).hexdigest()
    result["helper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output),"nodes":result["node_count"],
                      "patches":result["patch_count"],"physical_accuracy_qualified":False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
