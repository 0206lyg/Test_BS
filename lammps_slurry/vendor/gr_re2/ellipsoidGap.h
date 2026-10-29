#ifndef SLURRY_GR_RE2_GRAPHITE_ELLIPSOID_GAP_H
#define SLURRY_GR_RE2_GRAPHITE_ELLIPSOID_GAP_H
#include "particleMath.h"
#include <limits>
#include <sstream>
#include <iomanip>
#include <vector>

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
  // Stationarity and global certification are distinct for deep penetration.
  // The latter is established by a support upper bound, never by multistart.
  bool global_certified=false;
  int starts=0;
  double stationarity_error=0.,global_gap_error_bound=std::numeric_limits<double>::infinity();
};
namespace gap_detail {
struct Eval {double h,si,sj;Vec3 pI,pJ,gradient;Mat3 hessian;};
inline Eval eval(const Vec3& n,const Vec3& r,const Mat3& qi,const Mat3& qj) {
  const Vec3 vi=mul(qi,n),vj=mul(qj,n);
  const double si=std::sqrt(dot(n,vi)),sj=std::sqrt(dot(n,vj));
  if(!(si>0.)||!(sj>0.)||!std::isfinite(si+sj))
    throw std::runtime_error("Nonfinite ellipsoid support geometry");
  Eval e{};e.si=si;e.sj=sj;e.pI=scale(vi,1./si);e.pJ=scale(vj,-1./sj);
  e.gradient=add(sub(r,e.pI),e.pJ);e.h=dot(n,r)-si-sj;
  for(int k=0;k<3;++k)for(int l=0;l<3;++l)e.hessian[3*k+l]=
    -qi[3*k+l]/si+vi[k]*vi[l]/(si*si*si)-qj[3*k+l]/sj+vj[k]*vj[l]/(sj*sj*sj);
  return e;
}
struct Tangent {
  Vec3 gradient,t1,t2;
  double norm=0.,h11=0.,h12=0.,h22=0.,largest=0.;
};
inline Tangent tangent(const Vec3& n,const Eval& e) {
  Tangent t;
  const double radial=dot(n,e.gradient);
  t.gradient=sub(e.gradient,scale(n,radial));t.norm=norm(t.gradient);
  const Vec3 reference=std::abs(n[0])<.8?Vec3{1.,0.,0.}:Vec3{0.,1.,0.};
  t.t1=normalized(cross(n,reference));t.t2=cross(n,t.t1);
  t.h11=dot(t.t1,mul(e.hessian,t.t1))-radial;
  t.h12=dot(t.t1,mul(e.hessian,t.t2));
  t.h22=dot(t.t2,mul(e.hessian,t.t2))-radial;
  t.largest=.5*(t.h11+t.h22)+std::hypot(.5*(t.h11-t.h22),t.h12);
  return t;
}
// Each ellipsoid contains a ball tangent at Qn/s with radius amin^2/s.
// Their Minkowski sum contains the sum of those balls. Its support gives
// f(m) <= |(f(n)+rho)n+grad_S f(n)|-rho for every unit m. This is a genuine
// global upper bound, including shallow overlap, not a local-Hessian test.
inline double upper_error(const Eval& e,const Tangent& t,double min_i,double min_j) {
  const double rho=min_i/e.si+min_j/e.sj,radial=e.h+rho;
  if(radial>0.)
    return t.norm*(t.norm/(std::hypot(radial,t.norm)+radial));
  return std::hypot(radial,t.norm)-radial;
}
struct Candidate {
  Vec3 normal{};Eval value{};Tangent tangential{};
  bool converged=false,certified=false;
  int iterations=0;
  double upper=std::numeric_limits<double>::infinity();
};
inline Candidate maximize(Vec3 n,const Vec3& r,const Mat3& qi,const Mat3& qj,
                          double min_i,double min_j,double accuracy) {
  Candidate out;out.normal=normalized(n);
  out.value=eval(out.normal,r,qi,qj);
  const double eps=std::numeric_limits<double>::epsilon();
  const double objective_roundoff=64.*eps*(1.+norm(r));
  for(int iter=0;iter<120;++iter) {
    out.iterations=iter+1;out.tangential=tangent(out.normal,out.value);
    const auto &t=out.tangential;
    const double curvature_roundoff=64.*eps*(1.+std::abs(t.h11)+std::abs(t.h22));
    // A zero gradient alone can be a saddle or a minimum for overlapping
    // ellipsoids. In particular, never accept the equator of a deep oblate
    // overlap merely because the center-line normal is stationary.
    if(t.norm<=accuracy&&t.largest<=curvature_roundoff) {
      out.converged=true;
      out.upper=upper_error(out.value,t,min_i,min_j);
      out.certified=out.upper<=accuracy;
      return out;
    }
    Vec3 step{};
    const double g1=dot(t.t1,t.gradient),g2=dot(t.t2,t.gradient);
    const double det=t.h11*t.h22-t.h12*t.h12;
    if(t.largest < -curvature_roundoff && det>0.) {
      step=add(scale(t.t1,(-t.h22*g1+t.h12*g2)/det),
               scale(t.t2,(t.h12*g1-t.h11*g2)/det));
    } else if(t.largest>curvature_roundoff&&t.norm<1e-7*(1.+norm(r))) {
      // Escape a stationary nonmaximum along its positive-curvature mode.
      double x=t.h12,y=t.largest-t.h11;
      if(std::hypot(x,y)<1e-30){x=t.largest-t.h22;y=t.h12;}
      if(std::hypot(x,y)<1e-30){x=1.;y=0.;}
      step=scale(add(scale(t.t1,x),scale(t.t2,y)),.25/std::hypot(x,y));
      const Vec3 plus=normalized(add(scale(out.normal,std::cos(.25)),scale(step,std::sin(.25)/.25)));
      const Vec3 minus=normalized(add(scale(out.normal,std::cos(.25)),scale(step,-std::sin(.25)/.25)));
      if(eval(minus,r,qi,qj).h>eval(plus,r,qi,qj).h)step=scale(step,-1.);
    } else {
      step=scale(t.gradient,.25/std::max(t.norm,1e-300));
    }
    double step_norm=norm(step);
    if(!(step_norm>0.)||!std::isfinite(step_norm))break;
    if(step_norm>.5){step=scale(step,.5/step_norm);step_norm=.5;}
    const double slope=std::max(0.,dot(t.gradient,step));
    bool accepted=false;
    for(int ls=0;ls<40;++ls) {
      const double factor=std::ldexp(1.,-ls),angle=factor*step_norm;
      const Vec3 trial_normal=normalized(add(scale(out.normal,std::cos(angle)),
                                      scale(step,std::sin(angle)/step_norm)));
      const auto trial=eval(trial_normal,r,qi,qj);
      const double trial_gradient=norm(sub(trial.gradient,
                                 scale(trial_normal,dot(trial_normal,trial.gradient))));
      // Objective changes below floating-point resolution must be judged by
      // stationarity, not by an impossible strict increase in rounded f.
      const bool residual_improved=trial_gradient<t.norm&&
                                   trial.h>=out.value.h-objective_roundoff;
      const bool escaped_nonmaximum=t.largest>curvature_roundoff&&
                                    trial.h>out.value.h+objective_roundoff;
      if(trial.h>out.value.h+1e-4*factor*slope||residual_improved||escaped_nonmaximum) {
        out.normal=trial_normal;out.value=trial;accepted=true;break;
      }
    }
    if(!accepted)break;
  }
  out.tangential=tangent(out.normal,out.value);
  out.upper=upper_error(out.value,out.tangential,min_i,min_j);
  return out;
}
}
// Exact support geometry: maximize n.(xJ-xI)-sqrt(n.QI.n)-sqrt(n.QJ.n)
// on the unit sphere. At a disjoint-pair maximum the two support points are
// the true closest surface points. This is not a Gay-Berne contact estimate.
// A contained-ball upper bound certifies the global maximum for separated
// and most shallow-overlap configurations. Deep overlaps use deterministic
// multiple starts and return the highest converged maximum; global_certified
// remains false unless the same actual bound succeeds. Preserve negative
// gaps without clipping: zero is the geometric surface-contact onset.
inline GapResult closestEllipsoidGap(const Body& bi,const Body& bj,GapCache* cache=nullptr) {
  double length=0.;for(double a:bi.axes){if(!(a>0.)||!std::isfinite(a))throw std::domain_error("Invalid ellipsoid axis");length=std::max(length,a);}
  for(double a:bj.axes){if(!(a>0.)||!std::isfinite(a))throw std::domain_error("Invalid ellipsoid axis");length=std::max(length,a);}
  if(!finite(bi.position)||!finite(bj.position))throw std::domain_error("Nonfinite ellipsoid center");
  const Vec3 r=scale(sub(bj.position,bi.position),1./length);
  const double distance=norm(r),accuracy=2e-13*(1.+distance);
  const bool valid_cache=cache&&cache->valid&&finite(cache->normal)&&norm(cache->normal)>1e-20;
  Vec3 n=valid_cache?normalized(cache->normal):(distance>1e-20?scale(r,1./distance):Vec3{0.,0.,1.});
  if(dot(n,r)<0.)n=scale(n,-1.);
  GapResult result;
  if(bi.axes[0]==bi.axes[1]&&bi.axes[1]==bi.axes[2]&&
     bj.axes[0]==bj.axes[1]&&bj.axes[1]==bj.axes[2]) {
    if(distance>0.)n=scale(r,1./distance);
    result.normal=n;result.gap=distance*length-bi.axes[0]-bj.axes[0];
    result.leverI=scale(n,bi.axes[0]);result.leverJ=scale(n,-bj.axes[0]);
    result.converged=result.global_certified=true;result.global_gap_error_bound=0.;
    if(cache){cache->normal=n;cache->valid=true;}
    return result;
  }
  Vec3 di{},dj{};for(int k=0;k<3;++k){di[k]=std::pow(bi.axes[k]/length,2);dj[k]=std::pow(bj.axes[k]/length,2);}
  const Mat3 qi=rotatedDiagonal(bi.rotation,di),qj=rotatedDiagonal(bj.rotation,dj);
  const double min_i=*std::min_element(di.begin(),di.end()),min_j=*std::min_element(dj.begin(),dj.end());
  auto best=gap_detail::maximize(n,r,qi,qj,min_i,min_j,accuracy);
  auto largest_observed=best;
  result.iterations=best.iterations;result.starts=1;
  if(!best.converged||!best.certified) {
    std::vector<Vec3> starts;
    if(distance>1e-20)starts.push_back(scale(r,1./distance));
    // Both pole signs matter: deep overlaps can have two strict local
    // maxima even when both normals point into the center-line hemisphere.
    for(const auto *body:{&bi,&bj})for(int k=0;k<3;++k) {
      const Vec3 axis{body->rotation[k],body->rotation[3+k],body->rotation[6+k]};
      starts.push_back(axis);starts.push_back(scale(axis,-1.));
    }
    constexpr int angular_starts=32;
    const double golden=std::acos(-1.)*(3.-std::sqrt(5.));
    for(int k=0;k<angular_starts;++k) {
      const double z=1.-2.*(k+.5)/angular_starts,w=std::sqrt(std::max(0.,1.-z*z)),angle=k*golden;
      starts.push_back({w*std::cos(angle),w*std::sin(angle),z});
    }
    for(const auto &start:starts) {
      const auto trial=gap_detail::maximize(start,r,qi,qj,min_i,min_j,accuracy);
      result.iterations+=trial.iterations;++result.starts;
      if(trial.value.h>largest_observed.value.h)largest_observed=trial;
      if(trial.converged&&(!best.converged||trial.value.h>best.value.h))best=trial;
      if(best.converged&&best.certified)break;
    }
  }
  // A failed start is useful evidence, not something to silently discard:
  // if it already exceeds every converged maximum, report that unresolved
  // candidate instead of returning a known inferior penetration normal.
  if(!best.certified&&largest_observed.value.h>best.value.h+accuracy) {
    best=largest_observed;best.converged=false;
  }
  result.normal=best.normal;result.gap=best.value.h*length;
  result.leverI=scale(best.value.pI,length);result.leverJ=scale(best.value.pJ,length);
  result.converged=best.converged;result.global_certified=best.certified;
  result.stationarity_error=best.tangential.norm*length;result.global_gap_error_bound=best.upper*length;
  if(!result.converged) {
    std::ostringstream message;message<<std::scientific<<std::setprecision(9)
      <<"Ellipsoid support-gap iteration did not converge; iterations="<<result.iterations
      <<", starts="<<result.starts<<", gradient="<<result.stationarity_error
      <<", gap="<<result.gap<<", tangent_hessian_max="<<best.tangential.largest*length;
    throw std::runtime_error(message.str());
  }
  if(!std::isfinite(result.gap)||!finite(result.leverI)||!finite(result.leverJ))throw std::runtime_error("Nonfinite ellipsoid gap");
  if(cache){cache->normal=result.normal;cache->valid=true;}
  return result;
}
} // namespace graphite

} } // SLURRY SCOPE END
#endif
