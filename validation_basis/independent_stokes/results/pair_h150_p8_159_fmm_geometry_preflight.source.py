import os
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['GFORTRAN_UNBUFFERED_ALL']='1'
import sys,json,time,hashlib,resource
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/scratch/581ed100c5ad/Test_BS/validation_basis/independent_stokes')
sys.path.insert(0,str(ROOT))
from backend import Backend
from geometry import build_mesh
from run_reference import logged_fortran
library=Path('/workspace/scratch/581ed100c5ad/reference_deps_quadparams_highorder_v1/lib/libfmm3dbie.so')
proposal=ROOT/'results/pair_h150_p8_hpc_refinement_proposal.json'
out=ROOT/'results/pair_h150_p8_159_fmm_geometry_preflight.json'
data=json.loads(proposal.read_text()); start=time.monotonic()
b=Backend(library); mesh=b.coefficients(build_mesh(150.,8,0,b.discretization(8)[0],patch_descriptors=data['patch_descriptors']))
n=mesh['srcvals'].shape[1]; p=len(mesh['norders']); targets=np.asfortranarray(mesh['srcvals'][:3])
rfac=np.zeros((),dtype=np.float64);rfac0=np.zeros((),dtype=np.float64)
b.call('get_rfacs',8,11,rfac,rfac0)
cms=np.empty((3,p),order='F');rads=np.empty(p)
b.call('get_centroid_rads',p,mesh['norders'],mesh['ixyzs'],mesh['iptype'],n,mesh['srccoefs'],cms,rads)
nnz=np.zeros((),dtype=np.int64);near_rads=np.asfortranarray(rads*rfac)
b.call('findnearmem',cms,p,near_rads,3,targets,n,nnz); nnz=int(nnz)
row=np.empty(n+1,dtype=np.int64);col=np.empty(nnz,dtype=np.int64);iq=np.empty(nnz+1,dtype=np.int64)
b.call('findnear',cms,p,near_rads,3,targets,n,row,col)
b.call('get_iquad_rsc',p,mesh['ixyzs'],n,nnz,row,col,iq)
novers=np.empty(p,dtype=np.int64);ixo=np.empty(p+1,dtype=np.int64)
logged_fortran(out.with_suffix('.native.log'),lambda:b.call('get_far_order',1e-7,p,mesh['norders'],mesh['ixyzs'],mesh['iptype'],cms,rads,n,mesh['srccoefs'],3,n,targets,0,np.zeros(1,dtype=np.complex128).view(np.float64),nnz,row,col,float(rfac),novers,ixo),b.lib)
cases=[]
for increment in (0,16,32,48):
 orders=novers+increment; widths=(orders+1)**2; ns=int(widths.sum())
 per_target=np.add.reduceat(widths[col-1],row[:-1]-1)
 nquad=int(iq[-1]-1)
 # native lpcomp allocates 5(3,ns) arrays and 4(3,nmax) near copies,
 # with private near copies inside the OpenMP parallel loop. FMM tree excluded.
 cases.append(dict(increment=increment,far_order_min=int(orders.min()),far_order_max=int(orders.max()),oversampled_nodes=ns,
   retained_near_bytes=48*nquad,retained_oversampled_geometry_bytes=104*ns,
   oversampled_direct_subtraction_interactions_per_matvec=int(per_target.sum()),
   max_oversampled_near_sources_per_target=int(per_target.max()),
   wrapper_explicit_temporary_estimate_bytes=120*ns+48*n+96*int(per_target.max())*9,
   warning='Temporary estimate excludes FMM tree/multipole arrays and quadrature construction; no matvec or solve performed'))
record=dict(schema='fmm_geometry_only_preflight_v1',physical_accuracy_qualified=False,contains_new_stokes_solve=False,
 node_count=n,patch_count=p,rfac=float(rfac),near_patch_interactions=nnz,near_node_interactions=int(iq[-1]-1),
 near_fraction=int(iq[-1]-1)/(n*n),patch_self_LU_bytes=72*int(np.sum(np.diff(mesh['ixyzs'])**2)),
 gmres_restart80_basis_bytes=8*3*n*81,near_cases=cases,
 library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),proposal_sha256=hashlib.sha256(proposal.read_bytes()).hexdigest(),
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),wall_seconds=time.monotonic()-start,
 peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
