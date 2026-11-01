#pragma once
// Finite-face correction and EF cap from the validated 2026-11-01 candidate.
#include "slurry_spheroid_hydro_math.h"
#include <algorithm>
#include <cmath>
#ifndef SLURRY_VALIDATION_TAPER_START
#define SLURRY_VALIDATION_TAPER_START 0.75
#endif
static_assert(SLURRY_VALIDATION_TAPER_START>=0 && SLURRY_VALIDATION_TAPER_START<1, "Taper start must lie in [0,1)");
namespace validation {
namespace sh=SlurrySpheroidHydro;
namespace gr=slurry::gr_re2::graphite;
using sh::Vec3; using sh::Mat3;
struct Diagnostic { double log_factor=1,normal_factor=1,D=0,Dref=0,Lproj=0,taper=1; };
inline gr::GapResult support_gap(gr::Body& a,gr::Body& b,Vec3 n,double h) {
 n=gr::normalized(n); gr::GapResult g;g.normal=n;g.gap=h;g.converged=true;
 auto support=[&](const gr::Body& p) {auto s=gr::rotatedDiagonal(p.rotation,{p.axes[0]*p.axes[0],p.axes[1]*p.axes[1],p.axes[2]*p.axes[2]});auto sn=gr::mul(s,n);return gr::scale(sn,1/std::sqrt(gr::dot(n,sn)));};
 g.leverI=support(a);g.leverJ=gr::scale(support(b),-1);a.position={};b.position=gr::add(gr::sub(g.leverI,g.leverJ),gr::scale(n,h));return g;
}
inline double taper(double g,double cutoff) { double q=std::max(0.,g)/cutoff;constexpr double start=SLURRY_VALIDATION_TAPER_START; if(q<=start)return 1;if(q>=1)return 0;double s=(q-start)/(1-start);double t=1-s;return t*t*t*(1+3*s+6*s*s); }
inline sh::Pair candidate(const gr::Body&a,const gr::Body&b,const gr::GapResult&g,double eta,double eps,double cap,Diagnostic&diag,int angles=64,int taper_policy=0) {
 auto p=sh::make_pair(a,b,g,eta,eps,cap);if(!p.active)return p;
 if(!p.exact_sphere) {
 const auto qi=sh::detail::implicit_shape(a),qj=sh::detail::implicit_shape(b);
 const auto ki=gr::nfCurvature(a,p.normal),kj=gr::nfCurvature(b,p.normal);
 const double rhoi=2/(sh::detail::contract(p.tangent1,ki,p.tangent1)+sh::detail::contract(p.tangent2,ki,p.tangent2));
 const double rhoj=2/(sh::detail::contract(p.tangent1,kj,p.tangent1)+sh::detail::contract(p.tangent2,kj,p.tangent2));
 diag.Dref=.5*(1/rhoi+1/rhoj)*std::pow(std::min(rhoi,rhoj),2);
 const Vec3 ni=p.normal,nj=gr::scale(p.normal,-1);
 auto radius=[](const Mat3&q,const Vec3&n,const Vec3&point,const Vec3&e) {double A=sh::detail::contract(n,q,n),B=sh::detail::contract(e,q,n),C=sh::detail::contract(e,q,e),s=sh::detail::contract(n,q,point);return s/(std::sqrt(A*C)-B);};
 double sw=0,sl=0,sd=0;const double pi=std::acos(-1.);
 // Symmetric quadrature: half the angles in each particle's principal frame.
 // This preserves exact pair exchange at finite quadrature order. If one
 // curvature is isotropic, use the other frame for both halves; a spherical
 // body's arbitrary tangent frame must not introduce coordinate dependence.
 double ai=sh::detail::contract(p.tangent1,ki,p.tangent1),di=sh::detail::contract(p.tangent2,ki,p.tangent2);
 double aj=sh::detail::contract(p.tangent1,kj,p.tangent1),dj=sh::detail::contract(p.tangent2,kj,p.tangent2),bj=sh::detail::contract(p.tangent1,kj,p.tangent2);
 double phase_i=0,phase_j=.5*std::atan2(2*bj,aj-dj);
 bool iso_i=std::abs(ai-di)<1e-12*(ai+di),iso_j=std::hypot(aj-dj,2*bj)<1e-12*(aj+dj);
 if(iso_i&&!iso_j)phase_i=phase_j;if(iso_j&&!iso_i)phase_j=phase_i;
 const int half=angles/2;
 for(double phase:{phase_i,phase_j})for(int k=0;k<half;++k){double theta=phase+2*pi*k/half;const auto e=gr::add(gr::scale(p.tangent1,std::cos(theta)),gr::scale(p.tangent2,std::sin(theta)));double curvature=.5*(sh::detail::contract(e,ki,e)+sh::detail::contract(e,kj,e));double r=std::min(radius(qi,ni,g.leverI,e),radius(qj,nj,g.leverJ,e));double ell=curvature*r*r,w=1/curvature;sw+=w;sl+=w*std::log1p(ell/p.regularized_gap);sd+=w*ell;}
 diag.D=sd/sw;diag.Lproj=sl/sw;
 diag.log_factor=std::min(1.,diag.Lproj/std::log1p(diag.Dref/p.regularized_gap));
 auto cn=[&](double D){return std::pow(D/(p.regularized_gap+D),2);};
 diag.normal_factor=std::min(1.,cn(diag.D)/cn(diag.Dref));
{const double cmi=*std::min_element(a.axes.begin(),a.axes.end()),cmj=*std::min_element(b.axes.begin(),b.axes.end());
 const double Lc=std::min(rhoi/2,cmi)+std::min(rhoj/2,cmj),Lk=rhoi/2+rhoj/2,hr=p.regularized_gap,lk=std::log(Lk/hr);
 if(lk>0) diag.log_factor=std::min(diag.log_factor,std::max(std::log(Lc/hr),0.)/lk);
 diag.normal_factor=std::min(diag.normal_factor,std::pow((.5*Lc/(hr+.5*Lc))/(.5*Lk/(hr+.5*Lk)),2));}
 const double K=p.normal_resistance;
 sh::Vec6 m{}; for(int k=0;k<6;++k)m[k]=diag.log_factor*p.normal_cross[k]/K;m[2]=1;
 p.squeeze={};sh::detail::outer_add(p.squeeze,m,diag.normal_factor*K);
 for(int k=0;k<6;++k)p.normal_cross[k]*=diag.log_factor*diag.normal_factor;
 p.normal_resistance*=diag.normal_factor;
 for(int k=0;k<36;++k){p.shear[k]*=diag.log_factor;p.pump[k]*=diag.log_factor;p.resistance[k]=p.squeeze[k]+p.shear[k]+p.pump[k];}
 }
 if(taper_policy==2 || (taper_policy==1&&!p.exact_sphere)) {
  diag.taper=taper(p.gap,p.cutoff);
  if(p.exact_sphere){p.sphere.squeeze_resistance*=diag.taper;p.sphere.shear_resistance*=diag.taper;p.sphere.pump_resistance*=diag.taper;}
  p.normal_resistance*=diag.taper;for(auto&x:p.normal_cross)x*=diag.taper;
  for(int k=0;k<36;++k){p.squeeze[k]*=diag.taper;p.shear[k]*=diag.taper;p.pump[k]*=diag.taper;p.resistance[k]*=diag.taper;}
 }
 return p;
}
}
