#!/usr/bin/env python3
"""Check angular source quadrature on the actual solved pair's dominant RHS."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre
from ar20_boundary_check import samples
import analytic_ring_reference as ring
from full17_reference import rotation
p=Path(sys.argv[1]);d=json.loads(p.read_text());C=np.fromfile(d['coefficients_file']).reshape(2,d['source_rings'],-1,3,17)
points,ids,labels,axes,centers=samples(d,11,23,8,16)
vel=np.empty((len(points),3));cross=np.empty_like(vel)
for begin in range(0,len(points),16):
 end=min(begin+16,len(points));out=[]
 for b in range(2):out.append((ring.ring_block(points[begin:end],d['source_rings'],d['source_order'],d['source_depth'],axes[b],centers[b],a=d['aspect_ratio'])@C[b,...,2].reshape(-1)).reshape(-1,3))
 vel[begin:end]=out[0]+out[1]
 for b in range(2):cross[begin:end][ids[begin:end]==b]=out[1-b][ids[begin:end]==b]
expected=np.zeros_like(vel);expected[ids==0,2]=1.
errors=np.linalg.norm(vel-expected,axis=1)
selected=np.unique(np.r_[np.argsort(errors)[-4:],np.flatnonzero(labels=='gap_patch')[0],np.flatnonzero((labels=='gap_patch')&(ids==1))[0]])
mu=roots_legendre(d['source_rings'])[0];depth=d['source_depth'];ar=d['aspect_ratio'];rho=np.sqrt(ar*ar-1+depth*depth)*np.sqrt(1-mu*mu)
result={'solution':str(p),'rhs':2,'points':points[selected].tolist(),'body_ids':ids[selected].tolist(),'analytic_cross':cross[selected].tolist(),'quadrature':{}}
for nphi in [256,512,1024,2048]:
 phi=2*np.pi*np.arange(nphi)/nphi;H=np.ones((nphi,2*d['source_order']+1))
 for m in range(1,d['source_order']+1):H[:,2*m-1]=np.cos(m*phi);H[:,2*m]=np.sin(m*phi)
 actual=np.zeros((len(selected),3))
 for b in range(2):
  src=np.stack((rho[:,None]*np.cos(phi),rho[:,None]*np.sin(phi),np.broadcast_to(depth*mu[:,None],(len(mu),nphi))),axis=-1).reshape(-1,3)@rotation(axes[b]).T+centers[b]
  force=np.einsum('qm,rmc->rqc',H,C[b,...,2]).reshape(-1,3)/nphi
  for j,i in enumerate(selected):
   if ids[i]==b:continue
   delta=points[i]-src;r2=np.sum(delta*delta,axis=1);r=np.sqrt(r2)
   actual[j]=np.sum(force/r[:,None]+delta*(np.sum(delta*force,axis=1)/(r2*r))[:,None],axis=0)/(8*np.pi)
 result['quadrature'][str(nphi)]={'cross':actual.tolist(),'max_absolute_error':float(np.max(np.linalg.norm(actual-cross[selected],axis=1)))}
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v['max_absolute_error'] for k,v in result['quadrature'].items()}))
