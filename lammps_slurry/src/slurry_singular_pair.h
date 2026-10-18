#ifndef SLURRY_SINGULAR_PAIR_H
#define SLURRY_SINGULAR_PAIR_H

// Analytic subtraction basis for tabulating a finite-gap correction.
// Units and coordinates match SlurryScattering::Provider::film11:
// c=eta=1, [three midpoint slips, a*relative spin, five a*E modes].
// The full prediction is leading11 + interpolated(target11-leading11).
// This leading matrix is signed; it must not be eigenvalue-clipped or used as
// a standalone finite-gap resistance. It omits the subleading normal-normal
// logarithm, so its residual is not asserted bounded as h tends to zero.
#include "slurry_scattering_provider.h"
#include "slurry_spheroid_hydro_math.h"

namespace SlurryFastSingular {
using SlurryScattering::Vec3;
using SlurryScattering::Mat11;

inline Mat11 leading11(const std::array<Vec3,2>& axes,double h_eff,double ar) {
  if(!(h_eff>0.)||!std::isfinite(h_eff)||!(ar>=1.)||!std::isfinite(ar))
    throw std::invalid_argument("Singular basis needs finite oblate AR and positive effective gap");
  namespace gr=slurry::gr_re2::graphite;
  namespace singular=SlurrySpheroidHydro;
  const auto bodies=SlurryScattering::geometry(axes,h_eff,ar);
  gr::Body bi,bj;gr::Body* body[]={&bi,&bj};
  for(int k=0;k<2;++k) {
    body[k]->axes={ar,ar,1.};
    for(int i=0;i<3;++i) {
      body[k]->position[i]=bodies[k].center[i];
      for(int j=0;j<3;++j)body[k]->rotation[3*i+j]=bodies[k].rotation(i,j);
    }
  }
  gr::GapResult gap;gap.normal={0.,0.,1.};gap.gap=h_eff;gap.converged=true;
  for(int k=0;k<3;++k) {
    gap.leverI[k]=-bodies[0].center[k];
    gap.leverJ[k]=(k==2?h_eff:0.)-bodies[1].center[k];
  }
  // h_eff already contains the caller's physical regularization. Adding an
  // epsilon here would regularize twice. Cutoff is unused in this explicit path.
  const auto pair=singular::make_pair(bi,bj,gap,1.,0.,1.,true);
  Eigen::Matrix<double,6,17> T=Eigen::Matrix<double,6,17>::Zero();
  for(int k=0;k<17;++k) {
    singular::Vec12 q{};singular::Mat3 E{};
    if(k<12)q[k]=1.;
    else for(int j=0;j<9;++j)E[j]=SlurryGrandPair::strain_basis()[k-12][j]/ar;
    const auto v=singular::local_velocity(pair,q,{},E,ar);
    for(int j=0;j<6;++j)T(j,k)=v[j];
  }
  Eigen::Matrix<double,6,6> K;
  for(int i=0;i<6;++i)for(int j=0;j<6;++j)K(i,j)=pair.resistance[6*i+j];
  const auto B=SlurryScattering::midpoint_map(bodies,h_eff,ar);
  const Eigen::Matrix<double,6,6> BB=B*B.transpose();
  const Eigen::Matrix<double,12,6> P=B.transpose()*BB.ldlt().solve(Eigen::Matrix<double,6,6>::Identity());
  Eigen::Matrix<double,17,11> L=Eigen::Matrix<double,17,11>::Zero();
  L.topLeftCorner<12,6>()=P;L.bottomRightCorner<5,5>().setIdentity();
  const Eigen::Matrix<double,6,11> A=T*L;
  const Mat11 H=A.transpose()*K*A;
  if(!H.allFinite())throw std::runtime_error("Nonfinite analytic singular matrix");
  return .5*(H+H.transpose()).eval();
}
}
#endif
