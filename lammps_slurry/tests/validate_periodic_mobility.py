#!/usr/bin/env python3
"""Independent Fourier/tail checks; needs NumPy/SciPy only for validation.
Usage: python validate_periodic_mobility.py --probe ./test_periodic_mobility
No LAMMPS, source table, or former validation package is imported.
"""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn,erfc,sici

def tail_formula(b):
    z=b*b;om=-np.expm1(-z);e=np.exp(-z)
    return np.array([erfc(b)+om/(np.sqrt(np.pi)*b),
      erfc(b)+(3*om/b-2*(om-z*e)/b**3)/np.sqrt(np.pi)])
def probe(exe,shapes,positions,H,xi,kmax=0):
    words=[str(len(shapes)),str(xi),str(kmax)]+list(map(str,H.ravel()))
    for A,x in zip(shapes,positions):words.extend(map(str,x));words.extend(map(str,A.ravel()))
    t=time.perf_counter();s=subprocess.check_output([str(exe),'dump'],input=(' '.join(words)+'\n').encode()).decode()
    lines=s.splitlines();return np.loadtxt(lines[1:]),int(lines[0]),time.perf_counter()-t

def stf():
    B=np.zeros((5,3,3));B[0]=np.diag([1,-1,0])/np.sqrt(2);B[1]=np.diag([1,1,-2])/np.sqrt(6)
    for b,(i,j) in enumerate([(0,1),(0,2),(1,2)],2):B[b,i,j]=B[b,j,i]=1/np.sqrt(2)
    return B

def independent_fourier(shapes,positions,H,kmax):
    # Unsmoothed direct Fourier sum. Half lattice with exact physical spherical cutoff.
    reciprocal=2*np.pi*np.linalg.inv(H).T
    bound=np.ceil(kmax*np.linalg.norm(H,axis=0)/(2*np.pi)).astype(int)
    inds=np.array([(i,j,k) for i in range(-bound[0],bound[0]+1)
      for j in range(-bound[1],bound[1]+1) for k in range(-bound[2],bound[2]+1)
      if i>0 or (i==0 and j>0) or (i==0 and j==0 and k>0)])
    ks=inds@reciprocal.T;ks=ks[np.linalg.norm(ks,axis=1)<=kmax*(1+1e-13)]
    M=np.zeros((11*len(shapes),)*2);S=stf()
    for start in range(0,len(ks),8192):
      k=ks[start:start+8192];kn=np.linalg.norm(k,axis=1);n=k/kn[:,None];P=np.eye(3)-n[:,:,None]*n[:,None,:]
      C=np.zeros((len(k),3,3));C[:,0,1]=-n[:,2];C[:,0,2]=n[:,1];C[:,1,0]=n[:,2];C[:,1,2]=-n[:,0];C[:,2,0]=-n[:,1];C[:,2,1]=n[:,0]
      B=np.concatenate([-.5*C,np.einsum('aij,nj->nia',S,n)],axis=2)
      Psi=[]
      for A,x in zip(shapes,positions):
        a=np.sqrt(np.einsum('ni,ij,nj->n',n,A,n));q=a*kn;phase=np.exp(-1j*(k@x))
        G=np.empty((len(k),3,11),complex);G[:,:,:3]=np.sinc(q[:,None,None]/np.pi)*np.eye(3)*phase[:,None,None]
        G[:,:,3:]=-1j*(3*spherical_jn(1,q)/a*phase)[:,None,None]*B;Psi.append(G)
      G=np.concatenate(Psi,axis=2);PG=np.einsum('nij,nja->nia',P,G)
      M+=np.einsum('n,nia,nib->ab',2/(np.linalg.det(H)*kn*kn),G.conj(),PG).real
    return M,len(ks)

def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--quick',action='store_true');args=p.parse_args();exe=args.probe.resolve()
    tail=[]
    cpp_factors=np.loadtxt(subprocess.check_output([str(exe),'factors']).decode().splitlines())
    for x,gf,gd,hf,hd,si in cpp_factors:
      si_ref=sici(2*x)[0];assert abs(si-si_ref)<3e-14
      # Integrate the low-frequency hard-cutoff part; finite, positive integrand.
      f=quad(lambda q:np.sinc(q/np.pi)**2,0,x,epsabs=1e-12,limit=1000)[0]/(np.pi/2)
      d=quad(lambda q:spherical_jn(1,q)**2,0,x,epsabs=1e-12,limit=1000)[0]/(np.pi/6)
      assert np.max(np.abs([hf-(1-f),hd-(1-d)]))<1e-11
      if x>=.08:assert np.max(np.abs(tail_formula(x)-[gf,gd]))<1e-13
    for b in [.01,.08,.5,1,4,20,80]:
      # Integrate the low-pass part: exponentially convergent, no oscillatory infinite tail.
      f=quad(lambda q:np.exp(-(q/b)**2)*np.sinc(q/np.pi)**2,0,12*b,epsabs=1e-12,limit=1000)[0]/(np.pi/2)
      d=quad(lambda q:np.exp(-(q/b)**2)*spherical_jn(1,q)**2,0,12*b,epsabs=1e-12,limit=1000)[0]/(np.pi/6)
      err=float(np.max(np.abs(tail_formula(b)-[1-f,1-d])));assert err<1e-8;tail.append(dict(b=b,absolute_error=err))
    A=np.diag([4.,4.,1.]);H=np.array([[9,3,1],[0,8,2],[0,0,10.]])
    b=[A,A];x=[np.array([.3,.8,1.7]),np.array([3.,2.,2.5])]
    cpp,K,_=probe(exe,b,x,H,0,4);py,Kpy=independent_fourier(b,x,H,4)
    err=float(np.linalg.norm(cpp-py)/np.linalg.norm(py));assert K==Kpy and err<1e-12
    report=dict(scope='Same periodic Fourier model; numerical convergence and explicit local-tail approximation, not Stokes accuracy.',tail_integrals=tail,independent_fourier=dict(relative_error=err,modes=K),cases=[])
    if not args.quick:
      fixtures=json.loads((Path(__file__).parent/'data/periodic_fixed_cases.json').read_text())['cases']
      H=np.eye(3)*100.
      for fixture in fixtures:
        name=fixture['name'];shapes=[np.array(fixture['shape_i']),np.array(fixture['shape_j'])];r=np.array(fixture['r']);near=np.array(fixture['near'])

        xs=[np.zeros(3),r];refs=[]
        for kmax in [2.,3.,4.]:
          M,K,seconds=probe(exe,shapes,xs,H,0,kmax);refs.append(M);report.setdefault('reference_cost',[]).append(dict(case=name,kmax=kmax,modes=K,seconds=seconds))
        crossref=refs[-1][:11,11:];den=np.linalg.norm(crossref)
        # Resolve self convergence with the analytic positive high-k continuum
        # self tail. This hard-cutoff option is a validation reference, not the
        # production shear implementation (its mode-entry jumps are finite).
        completed=[probe(exe,shapes,xs,H,-1,k)[0] for k in [2.,3.,4.]]
        refR=np.linalg.inv(completed[-1]);fullref=refR+near
        qx=np.zeros(22);qx[0]=.5;qx[11]=-.5
        qn=np.zeros(22);qn[:3]=np.array(fixture['normal'])*.5;qn[11:14]=-qn[:3]
        qe=np.zeros(22);qe[9]=qe[20]=-1/np.sqrt(2)
        modes_input={'relative_x':qx,'relative_normal':qn,'common_strain_xz':qe}

        row=dict(name=name,cell_c=100.,pure_fourier_cross_change_2_to_3=float(np.linalg.norm(refs[1][:11,11:]-refs[0][:11,11:])/den),pure_fourier_cross_change_3_to_4=float(np.linalg.norm(refs[2][:11,11:]-refs[1][:11,11:])/den),split=[])
        row['hard_tail_reference']={'kmax':[2,3,4],
          'M23':float(np.linalg.norm(completed[1]-completed[0])/np.linalg.norm(completed[-1])),
          'M34':float(np.linalg.norm(completed[2]-completed[1])/np.linalg.norm(completed[-1])),
          'R23':float(np.linalg.norm(np.linalg.inv(completed[1])-np.linalg.inv(completed[0]))/np.linalg.norm(refR)),
          'R34':float(np.linalg.norm(np.linalg.inv(completed[2])-np.linalg.inv(completed[1]))/np.linalg.norm(refR))}

        for spectral in [2,4,8]:
          M,K,seconds=probe(exe,shapes,xs,H,spectral/20);diff=M[:11,11:]-crossref
          split=dict(xi_a=spectral,modes=K,seconds=seconds,cross_relative_difference=float(np.linalg.norm(diff)/den),cross_FF_relative_difference=float(np.linalg.norm(diff[:3,:3])/np.linalg.norm(crossref[:3,:3])),cross_moment_relative_difference=float(np.linalg.norm(diff[3:,3:])/np.linalg.norm(crossref[3:,3:])))
          R=np.linalg.inv(M);full=R+near;split['middle_R_relative_difference']=float(np.linalg.norm(R-refR)/np.linalg.norm(refR));split['combined_R_relative_difference']=float(np.linalg.norm(full-fullref)/np.linalg.norm(fullref));split['modes_response']={}
          for key,q in modes_input.items():
            exact=(fullref@q).reshape(2,11);delta=((full-fullref)@q).reshape(2,11)
            split['modes_response'][key]={typ:float(np.linalg.norm(delta[:,sl])/np.linalg.norm(exact[:,sl])) for typ,sl in [('F',slice(0,3)),('T',slice(3,6)),('S',slice(6,11))]}
            split['modes_response'][key]['power_relative_difference']=float(q@(full-fullref)@q/(q@fullref@q))
          row['split'].append(split)
        report['cases'].append(row)
    data=json.dumps(report,indent=2)+'\n';print(data)
    if args.output:args.output.write_text(data)
if __name__=='__main__':main()
