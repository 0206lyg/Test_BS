"""Analytic Fourier coefficients of interior Stokeslet rings, eta=1.
Hypergeometric angular integration; source force coefficients use world Cartesian components; harmonics use body azimuth.
No lubrication coefficients appear. Diagnostic implementation.
"""
import numpy as np
import math
from scipy.special import hyp2f1, roots_legendre, poch

def coefficients(r,s,z,order,p):
    ap=(r+s)**2+z*z; am=(r-s)**2+z*z
    root=np.sqrt(ap*am); A=r*r+s*s+z*z
    C=.5*(A+root); t=r*s/C
    # Roundoff guard only, points must not lie on source rings.
    t=np.clip(t,0.,np.nextafter(1.,0.))
    return np.stack([poch(p,m)/float(math.factorial(m)) * t**m * hyp2f1(p+m,p,m+1,t*t)/C**p for m in range(order+1)],axis=-1)

def ring_block(targets,nlat,order,depth,axis,center,a=10.,c=1.,source_mu=None):
    from math import factorial
    from scipy.special import gamma
    axis=np.asarray(axis,float);axis=axis/np.linalg.norm(axis)
    x=np.array([1.,0.,0.]);
    if abs(x@axis)>.9:x=np.array([0.,1.,0.])
    x-=axis*(x@axis);x/=np.linalg.norm(x)
    Q=np.column_stack([x,np.cross(axis,x),axis]);local=(targets-center)@Q
    rho=np.hypot(local[:,0],local[:,1]);ang=np.arctan2(local[:,1],local[:,0])
    mu=roots_legendre(nlat)[0] if source_mu is None else np.asarray(source_mu)
    if len(mu)!=nlat:raise ValueError("source_mu length must equal nlat")
    cs=c*depth;aa=np.sqrt(a*a-c*c+cs*cs)
    s=aa*np.sqrt(1-mu*mu)[None,:];r=rho[:,None];z=local[:,2,None]-cs*mu[None,:]
    ap=(r+s)**2+z*z;am=(r-s)**2+z*z;C=.5*(r*r+s*s+z*z+np.sqrt(ap*am));t=r*s/C
    t=np.clip(t,0.,np.nextafter(1.,0.));t2=t*t
    def co(p,M):
        return np.stack([poch(p,m)/factorial(m)*t**m*hyp2f1(p+m,p,m+1,t2)/C**p for m in range(M+1)],axis=-1)
    I=co(.5,order);J=co(1.5,order+2)
    cp=np.cos(ang);sp=np.sin(ang)
    P=np.zeros((len(targets),3,3));P[:,0,0]=cp;P[:,0,1]=-sp;P[:,1,0]=sp;P[:,1,1]=cp;P[:,2,2]=1
    W=np.einsum('ij,tjk->tik',Q,P)
    out=np.empty((len(targets),3,nlat,2*order+1,3))
    for m in range(order+1):
        jm=J[:,:,m];jc=.5*(J[:,:,abs(m-1)]+J[:,:,m+1]);jcc=.5*jm+.25*(J[:,:,abs(m-2)]+J[:,:,m+2]);jss=jm-jcc
        js=.5*(J[:,:,abs(m-1)]-J[:,:,m+1]);jsc=.25*(J[:,:,abs(m-2)]-J[:,:,m+2])
        E=np.zeros((len(targets),nlat,3,3));O=np.zeros_like(E)
        E[:,:,0,0]=I[:,:,m]+r*r*jm-2*r*s*jc+s*s*jcc
        E[:,:,1,1]=I[:,:,m]+s*s*jss;E[:,:,2,2]=I[:,:,m]+z*z*jm
        E[:,:,0,2]=E[:,:,2,0]=z*(r*jm-s*jc)
        O[:,:,0,1]=O[:,:,1,0]=-r*s*js+s*s*jsc
        O[:,:,1,2]=O[:,:,2,1]=-s*z*js
        E=np.einsum('tij,tsjk,tlk->tsil',W,E,W);O=np.einsum('tij,tsjk,tlk->tsil',W,O,W)
        if m==0:out[:,:,:,0,:]=E.transpose(0,2,1,3)
        else:
            ca=np.cos(m*ang)[:,None,None,None];sa=np.sin(m*ang)[:,None,None,None]
            out[:,:,:,2*m-1,:]=(ca*E-sa*O).transpose(0,2,1,3)
            out[:,:,:,2*m,:]=(sa*E+ca*O).transpose(0,2,1,3)
    return out.reshape(3*len(targets),nlat*(2*order+1)*3)/(8*np.pi)

def matrix(targets,nlat,order,depth,bodies,nquad=None,a=10.,c=1.,source_mu=None):
    out=np.empty((len(targets)*3,len(bodies)*nlat*(2*order+1)*3))
    for start in range(0,len(targets),32):
        out[3*start:3*(start+32)]=np.concatenate([ring_block(targets[start:start+32],nlat,order,depth,axis,center,a,c,source_mu) for axis,center in bodies],axis=1)
    return out

if __name__=='__main__':
    import sys,time
    sys.path.insert(0,'ar10_resistance_audit/stokes')
    import fourier_mfs as old
    targets=np.array([[9.999,0.,0.],[3.,4.,1.5],[-1.,.2,2.]])
    bodies=[(np.array([0.,0.,1.]),np.zeros(3))]
    for q in [1024,4096,16384]:
        t0=time.time();M=matrix(targets,24,5,.5,bodies);N=old.matrix(targets,24,5,.5,bodies,q)
        print(q, np.linalg.norm(M-N)/np.linalg.norm(M),np.max(np.abs(M-N)),time.time()-t0,flush=True)
