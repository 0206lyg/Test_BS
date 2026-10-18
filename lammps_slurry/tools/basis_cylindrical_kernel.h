#ifndef SLURRY_BASIS_CYLINDRICAL_KERNEL_H
#define SLURRY_BASIS_CYLINDRICAL_KERNEL_H

// Offline-only isolated axisymmetric Stokeslet operator, eta=1.
// Targets and source rings are (rho,z) pairs in the body frame. For a complex
// cylindrical force amplitude f_m, the physical source is Re[f_m exp(i*m*phi)].
// K_m maps that amplitude to the cylindrical velocity amplitude. At target
// azimuth zero the cylindrical and Cartesian observation frames coincide.
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace SlurryBasisCylindrical {
using Complex=std::complex<double>;

// Cartesian real-harmonic coefficient operator for ONE source ring.
// Rows are [mode0 xyz, cos1 xyz, sin1 xyz, cos2 xyz, sin2 xyz,...].
// Columns are complex cylindrical radial, azimuthal and axial amplitudes.
// Real physical output coefficients are Re[D_m*f_m]; do not insert another
// factor two if f_m is already an amplitude rather than a normalized FFT
// coefficient. In particular, positive-mode amplitudes equal 2*FFT_m/N.
inline Eigen::MatrixXcd cylindrical_to_cartesian(int mode,int order) {
  if(mode<0||order<mode+1)
    throw std::invalid_argument("Cartesian source order must include cylindrical mode+1");
  Eigen::MatrixXcd D=Eigen::MatrixXcd::Zero(3*(2*order+1),3);
  const Complex I(0.,1.);
  auto add=[&](int signed_order,int component,int column,Complex coefficient) {
    const int k=std::abs(signed_order);
    if(k==0)D(component,column)+=coefficient;
    else {
      D(3*(2*k-1)+component,column)+=coefficient;
      D(3*(2*k)+component,column)+=(signed_order>0?I:-I)*coefficient;
    }
  };
  // Rz(phi) multiplies cylindrical force components. Its sine/cosine factors
  // shift angular order by +/-1; axial force keeps the original order.
  add(mode+1,0,0,.5);add(mode+1,0,1,.5*I);
  add(mode-1,0,0,.5);add(mode-1,0,1,-.5*I);
  add(mode+1,1,0,-.5*I);add(mode+1,1,1,.5);
  add(mode-1,1,0,.5*I);add(mode-1,1,1,.5);
  add(mode,2,2,1.);
  return D;
}

// K_m = (1/2pi) integral G(x_t-x_s(phi))*Rz(phi)*exp(i*m*phi) dphi.
// G(d)=(I/|d|+dd^T/|d|^3)/(8pi). The source coefficients absorb latitude
// quadrature weights; no extra source-ring or surface-area weights occur here.
// Kernel quadrature nquad is independent of the retained force Fourier order.
// Users must refine it because a target close to an interior ring can require
// many more azimuth nodes than merely resolving the force's retained modes.
inline std::vector<Eigen::MatrixXcd> modal_kernel(
    const Eigen::MatrixXd& target_rhoz,const Eigen::MatrixXd& source_rhoz,
    int max_mode,int nquad) {
  if(target_rhoz.cols()!=2||source_rhoz.cols()!=2||target_rhoz.rows()<1||
     source_rhoz.rows()<1||max_mode<0||nquad<2*max_mode+4||
     !target_rhoz.allFinite()||!source_rhoz.allFinite()||
     target_rhoz.col(0).minCoeff()<0.||source_rhoz.col(0).minCoeff()<0.)
    throw std::invalid_argument("Invalid cylindrical Stokes kernel dimensions");
  const int nt=int(target_rhoz.rows()),nr=int(source_rhoz.rows());
  const double pi=std::acos(-1.),inv8pi=1./(8*pi),weight=1./nquad;
  std::vector<double> cosine(nquad),sine(nquad);
  Eigen::MatrixXcd harmonics(nquad,max_mode+1);
  for(int q=0;q<nquad;++q) {
    const double phi=2*pi*q/nquad;cosine[q]=std::cos(phi);sine[q]=std::sin(phi);
    const Complex phase(cosine[q],sine[q]);harmonics(q,0)=weight;
    for(int m=1;m<=max_mode;++m)harmonics(q,m)=harmonics(q,m-1)*phase;
  }
  std::vector<Eigen::MatrixXcd> result;
  result.reserve(max_mode+1);
  for(int m=0;m<=max_mode;++m)result.emplace_back(Eigen::MatrixXcd::Zero(3*nt,3*nr));
  Eigen::Matrix<double,9,Eigen::Dynamic> sampled(9,nquad);
  for(int t=0;t<nt;++t)for(int r=0;r<nr;++r) {
    const double rt=target_rhoz(t,0),rs=source_rhoz(r,0);
    const double dz=target_rhoz(t,1)-source_rhoz(r,1);
    for(int q=0;q<nquad;++q) {
      const double c=cosine[q],s=sine[q];
      const double d[3]={rt-rs*c,-rs*s,dz};
      const double r2=d[0]*d[0]+d[1]*d[1]+dz*dz;
      if(!(r2>1e-25)||!std::isfinite(r2))
        throw std::runtime_error("Cylindrical kernel target touches an interior source");
      const double z=inv8pi/std::sqrt(r2),w=z/r2;
      // d^T Rz(phi), avoiding a dense 3x3 multiply in the quadrature loop.
      const double rotated_d[3]={rt*c-rs,-rt*s,dz};
      const double rotation[9]={c,-s,0.,s,c,0.,0.,0.,1.};
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        sampled(3*i+j,q)=z*rotation[3*i+j]+w*d[i]*rotated_d[j];
    }
    const Eigen::MatrixXcd integrated=sampled.cast<Complex>()*harmonics;
    for(int m=0;m<=max_mode;++m)for(int i=0;i<3;++i)for(int j=0;j<3;++j)
      result[m](3*t+i,3*r+j)=integrated(3*i+j,m);
  }
  return result;
}
}
#endif
