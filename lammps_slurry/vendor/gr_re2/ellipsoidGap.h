#ifndef SLURRY_GR_RE2_GRAPHITE_ELLIPSOID_GAP_H
#define SLURRY_GR_RE2_GRAPHITE_ELLIPSOID_GAP_H
#include "particleMath.h"
#include <limits>

// SLURRY SCOPE BEGIN
namespace slurry { namespace gr_re2 {

namespace graphite {
// A cached separating normal is an in-memory acceleration, never a restart file.
struct GapCache {Vec3 normal{0.,0.,1.};bool valid=false;};
struct GapResult {
  double gap=0.;
  Vec3 normal{0.,0.,1.},leverI{},leverJ{};
  bool converged=false;
  int iterations=0;
};
namespace gap_detail {
struct Eval {double h;Vec3 pI,pJ,gradient;Mat3 hessian;};
inline Eval eval(const Vec3& n,const Vec3& r,const Mat3& qi,const Mat3& qj) {
  const Vec3 vi=mul(qi,n),vj=mul(qj,n);
  const double si=std::sqrt(dot(n,vi)),sj=std::sqrt(dot(n,vj));
  Eval e{};e.pI=scale(vi,1./si);e.pJ=scale(vj,-1./sj);
  e.gradient=add(sub(r,e.pI),e.pJ);e.h=dot(n,r)-si-sj;
  for(int k=0;k<3;++k)for(int l=0;l<3;++l)e.hessian[3*k+l]=
    -qi[3*k+l]/si+vi[k]*vi[l]/(si*si*si)-qj[3*k+l]/sj+vj[k]*vj[l]/(sj*sj*sj);
  return e;
}
}
// Exact support geometry: maximize n.(xJ-xI)-sqrt(n.QI.n)-sqrt(n.QJ.n)
// on the unit sphere. At a disjoint-pair maximum the two support points are
// the true closest surface points. This is not a Gay-Berne contact estimate.
inline GapResult closestEllipsoidGap(const Body& bi,const Body& bj,GapCache* cache=nullptr) {
  double length=0.;for(double a:bi.axes){if(!(a>0.))throw std::domain_error("Nonpositive ellipsoid axis");length=std::max(length,a);}
  for(double a:bj.axes){if(!(a>0.))throw std::domain_error("Nonpositive ellipsoid axis");length=std::max(length,a);}
  const Vec3 r=scale(sub(bj.position,bi.position),1./length);
  Vec3 di{},dj{};for(int k=0;k<3;++k){di[k]=std::pow(bi.axes[k]/length,2);dj[k]=std::pow(bj.axes[k]/length,2);}
  const Mat3 qi=rotatedDiagonal(bi.rotation,di),qj=rotatedDiagonal(bj.rotation,dj);
  Vec3 n=cache&&cache->valid?normalized(cache->normal):(norm(r)>1e-20?normalized(r):Vec3{0.,0.,1.});
  if(dot(n,r)<0.)n=scale(n,-1.);
  auto e=gap_detail::eval(n,r,qi,qj);
  GapResult result;
  for(int iter=0;iter<100;++iter) {
    result.iterations=iter+1;
    const double radial=dot(n,e.gradient);
    const Vec3 gradient=sub(e.gradient,scale(n,radial));
    if(norm(gradient)<2e-13*(1.+norm(r))){result.converged=true;break;}
    const Vec3 reference=std::abs(n[0])<.8?Vec3{1.,0.,0.}:Vec3{0.,1.,0.};
    const Vec3 t1=normalized(cross(n,reference)),t2=cross(n,t1);
    const double g1=dot(t1,gradient),g2=dot(t2,gradient);
    const double h11=dot(t1,mul(e.hessian,t1))-radial;
    const double h12=dot(t1,mul(e.hessian,t2));
    const double h22=dot(t2,mul(e.hessian,t2))-radial;
    const double det=h11*h22-h12*h12;
    Vec3 step;
    if(h11<0.&&h22<0.&&det>1e-20)step=add(scale(t1,(-h22*g1+h12*g2)/det),scale(t2,(h12*g1-h11*g2)/det));
    else step=scale(gradient,.25/std::max(norm(gradient),1e-30));
    double stepNorm=norm(step);if(stepNorm>.5){step=scale(step,.5/stepNorm);stepNorm=.5;}
    bool accepted=false;
    for(int ls=0;ls<28;++ls) {
      const double factor=std::ldexp(1.,-ls),angle=factor*stepNorm;
      const Vec3 candidate=normalized(add(scale(n,std::cos(angle)),scale(step,std::sin(angle)/stepNorm)));
      const auto trial=gap_detail::eval(candidate,r,qi,qj);
      // Near the optimum an objective change can round to zero; accept an
      // improving stationarity residual instead, without loosening accuracy.
      const Vec3 trialGrad=sub(trial.gradient,scale(candidate,dot(candidate,trial.gradient)));
      if(trial.h>e.h+1e-4*factor*dot(gradient,step)||(trial.h>=e.h-4e-15&&norm(trialGrad)<norm(gradient))) {
        n=candidate;e=trial;accepted=true;break;
      }
    }
    if(!accepted)break;
  }
  result.normal=n;result.gap=e.h*length;result.leverI=scale(e.pI,length);result.leverJ=scale(e.pJ,length);
  if(!result.converged)throw std::runtime_error("Exact ellipsoid gap iteration did not converge");
  if(!std::isfinite(result.gap)||!finite(result.leverI)||!finite(result.leverJ))throw std::runtime_error("Nonfinite ellipsoid gap");
  if(cache){cache->normal=n;cache->valid=true;}
  return result;
}
} // namespace graphite

} } // SLURRY SCOPE END
#endif
