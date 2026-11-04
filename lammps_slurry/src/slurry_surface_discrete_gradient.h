#ifndef SLURRY_SURFACE_DISCRETE_GRADIENT_H
#define SLURRY_SURFACE_DISCRETE_GRADIENT_H

#include "slurry_surface_background.h"
#include <algorithm>
#include <map>
#include <sstream>

// Potential-consistent average-vector-field (AVF) pair wrench. The path has
// linear RELATIVE translation and laboratory exponential rotations. No force
// law, opening length, background switch, or potential is modified. Quadrature
// is local to this pair and never solves or advances the particle system.
//
// An averaged torque is objective and conserves the PATH-AVERAGED total pair
// moment. It need not balance r_end x F_average (nor r_mid x F_average). The
// integrated orbital moment and virial are returned explicitly; replacing
// either by an endpoint displacement times the averaged force is incorrect.
namespace SlurrySurfaceDiscreteGradient {
namespace gr=slurry::gr_re2::graphite;
namespace sc=SlurrySurfaceCohesion;
namespace sb=SlurrySurfaceBackground;
using gr::Vec3;using gr::Mat3;

struct Sample {
  gr::GapResult geometry;
  sc::Result cohesion;
  sb::Result background;
};
inline Sample sample(const gr::Body&i,const gr::Body&j,const sc::Parameters&c,
                     const sb::Parameters&b,gr::GapCache*cache=nullptr) {
  Sample out;out.geometry=gr::closestEllipsoidGap(i,j,cache);
  out.cohesion=sc::evaluate(i,j,out.geometry,c);
  out.background=sb::evaluate(i,j,out.geometry,b);return out;
}
struct Path {
  gr::Body initial_i,initial_j,final_i,final_j;
  // Must agree with final relative position and orientation. A common
  // translation is deliberately immaterial to the pair calculation.
  Vec3 relative_displacement{},rotation_i{},rotation_j{};
};
struct Options {
  double relative_work_tolerance=1e-7;
  double relative_force_tolerance=1e-5;
  int maximum_evaluations=129,maximum_depth=20;
};
struct KernelIntegral {
  Vec3 force_i{},force_j{},torque_i{},torque_j{};
  Vec3 orbital_moment{}; // integral (r_j-r_i) x force_i ds
  Mat3 symmetric_virial{}; // integral sym[(r_j-r_i) outer force_i] ds
  double initial_energy=0.,endpoint_energy=0.,average_energy=0.;
  double work=0.,work_defect=0.,work_scale=0.,force_error_estimate=0.;
  // Quasi-Newton approximation: integral s*dF_gap/dgap ds, retaining the
  // instantaneous curvature but lagging its derivatives and normal changes.
  // For background, only its exact near branch is included; far/switch
  // derivatives are intentionally lagged. This is NOT an exact Jacobian.
  double approximate_normal_tangent=0.;
};
struct Result {
  KernelIntegral cohesion,background;
  Sample initial,endpoint;
  int evaluations=0,panels=0,maximum_depth=0;
  bool converged=false;
};
struct Failure:std::runtime_error {using std::runtime_error::runtime_error;};

namespace detail {
using Values=std::array<double,23>;
struct Node {Sample state;Values cohesion{},background{};double gap_rate=0.;};
inline Values add(const Values&a,const Values&b) {
  Values c{};for(std::size_t k=0;k<c.size();++k)c[k]=a[k]+b[k];return c;
}
inline Values simpson(double width,const Values&a,const Values&m,const Values&b) {
  Values c{};for(std::size_t k=0;k<c.size();++k)c[k]=width*(a[k]+4.*m[k]+b[k])/6.;return c;
}
inline double norm3(const Values&a,int offset) {
  return std::hypot(std::hypot(a[offset],a[offset+1]),a[offset+2]);
}
inline double wrench_norm(const Values&a,double length) {
  return std::max({norm3(a,0),norm3(a,3)/length,norm3(a,6)/length});
}
inline double work(const Values&a,const Path&p) {
  double w=0.;for(int k=0;k<3;++k)
    w+=-a[k]*p.relative_displacement[k]+a[3+k]*p.rotation_i[k]+a[6+k]*p.rotation_j[k];
  return w;
}
inline double max_axis(const gr::Body&b){return *std::max_element(b.axes.begin(),b.axes.end());}
inline double min_axis(const gr::Body&b){return *std::min_element(b.axes.begin(),b.axes.end());}
template<class Kernel>
inline Values values(const Kernel&k,const Vec3&r,double tangent) {
  Values v{};for(int a=0;a<3;++a){v[a]=k.force_i[a];v[3+a]=k.torque_i[a];v[6+a]=k.torque_j[a];}
  const auto orbital=gr::cross(r,k.force_i);for(int a=0;a<3;++a)v[9+a]=orbital[a];
  for(int a=0;a<3;++a)for(int b=0;b<3;++b)v[12+3*a+b]=.5*(r[a]*k.force_i[b]+k.force_i[a]*r[b]);
  v[21]=tangent;v[22]=k.energy;return v;
}

class Integrator {
  const Path&p;const sc::Parameters&c;const sb::Parameters&b;const Options&o;
  std::map<double,Node> nodes;gr::GapCache cache;
  double length=0.,gap_second_bound=0.;
  double c_work_scale=0.,b_work_scale=0.,c_force_scale=0.,b_force_scale=0.;
  int evaluations=0,panels=0,deepest=0;
  Values total_c{},total_b{};double c_error=0.,b_error=0.;

  gr::Body body(double s,bool first) const {
    auto out=first?p.initial_i:p.initial_j;
    out.rotation=gr::rotateLaboratory(out.rotation,gr::scale(first?p.rotation_i:p.rotation_j,s));
    if(!first)out.position=gr::add(out.position,gr::scale(p.relative_displacement,s));
    return out;
  }
  Node make(double s,const Sample*known=nullptr) {
    if(!known&&evaluations>=o.maximum_evaluations)
      throw Failure("surface AVF pair quadrature exceeded its evaluation budget");
    const auto i=body(s,true),j=body(s,false);Node n;
    if(known)n.state=*known;
    else{n.state=sample(i,j,c,b,&cache);++evaluations;}
    const auto&r=n.state.geometry;
    if(!r.converged||!std::isfinite(r.gap)||!gr::finite(r.normal))
      throw Failure("surface AVF requires converged pair geometry");
    n.gap_rate=gr::dot(r.normal,p.relative_displacement)
      -gr::dot(p.rotation_i,gr::cross(r.leverI,r.normal))
      +gr::dot(p.rotation_j,gr::cross(r.leverJ,r.normal));
    double ct=0.,bt=0.;
    if(c.work_of_separation>0.&&r.gap>0.&&r.gap<c.opening_range)
      ct=-s*n.state.cohesion.attraction_at_contact/c.opening_range;
    const double h=b.reference_gap+r.gap;
    if(b.hamaker>0.&&r.gap>0.&&h<=b.curvature_switch) {
      // F_gap = A*R/(6h^2) - A*R*sigma^6/(180h^8).
      const double radius=n.state.background.attraction_at_contact/
        (2.*std::acos(-1.)*sb::work_of_separation(b));
      bt=s*b.hamaker*radius/(h*h*h)*(-1./3.+2.*std::pow(b.sigma/h,6)/45.);
    }
    const auto displacement=gr::sub(j.position,i.position);
    n.cohesion=values(n.state.cohesion,displacement,ct);
    n.background=values(n.state.background,displacement,bt);return n;
  }
  const Node&at(double s) {
    auto it=nodes.find(s);if(it!=nodes.end())return it->second;
    return nodes.emplace(s,make(s)).first->second;
  }
  // For any fixed separating normal, the signed support function is a lower
  // bound on the maximum support gap. Its path second derivative is >= -C:
  // |d^2 h_body/ds^2| <= (amax^2/amin+amax)*|theta|^2.
  // Thus g(s0)+g'(s0)t-C*t^2/2 lower-bounds the TRUE gap throughout a panel.
  // A concave quadratic attains its minimum at a panel endpoint. This detects
  // otherwise invisible entry into a narrow support, even if all sampled
  // force values are zero; fixed-orientation monotone departure is certified
  // without endless refinement at an exactly zero cutoff force.
  bool support_excluded(double lo,double hi,const std::array<double,5>&x,
                        const std::array<const Node*,5>&n,double cutoff) const {
    double lower=-std::numeric_limits<double>::infinity();
    for(int k=0;k<5;++k) {
      const double a=lo-x[k],z=hi-x[k];
      const double g=n[k]->state.geometry.gap,d=n[k]->gap_rate;
      const double candidate=std::min(g+d*a-.5*gap_second_bound*a*a,
                                      g+d*z-.5*gap_second_bound*z*z);
      lower=std::max(lower,candidate);
    }
    // Coordinate roundoff only, not a physical enlargement of the support.
    return lower>=cutoff-32.*std::numeric_limits<double>::epsilon()*length;
  }
  bool check_kernel(const Values&coarse,const Values&fine,double delta_energy,
                    double width,double ws,double fs,bool hidden,double&err) const {
    Values difference{};for(std::size_t k=0;k<difference.size();++k)difference[k]=(fine[k]-coarse[k])/15.;
    err=wrench_norm(difference,length);
    const double work_error=std::abs(work(fine,p)+delta_energy);
    // Subtracting two close endpoint energies eventually exposes support-gap
    // arithmetic at the particle-length scale. Do not demand a local work
    // difference below that roundoff scale. The independent final WHOLE-PATH
    // energy check below still uses the requested tolerance without this term.
    const double arithmetic=128.*std::numeric_limits<double>::epsilon()*fs*length;
    return !hidden&&err<=o.relative_force_tolerance*fs*width&&
      work_error<=o.relative_work_tolerance*ws*width+arithmetic;
  }
  void integrate(double lo,double hi,int depth) {
    deepest=std::max(deepest,depth);const double width=hi-lo,mid=.5*(lo+hi);
    const std::array<double,5>x{lo,.5*(lo+mid),mid,.5*(mid+hi),hi};
    const std::array<const Node*,5>n{&at(x[0]),&at(x[1]),&at(x[2]),&at(x[3]),&at(x[4])};
    const auto coarse_c=simpson(width,n[0]->cohesion,n[2]->cohesion,n[4]->cohesion);
    const auto fine_c=add(simpson(.5*width,n[0]->cohesion,n[1]->cohesion,n[2]->cohesion),
                          simpson(.5*width,n[2]->cohesion,n[3]->cohesion,n[4]->cohesion));
    const auto coarse_b=simpson(width,n[0]->background,n[2]->background,n[4]->background);
    const auto fine_b=add(simpson(.5*width,n[0]->background,n[1]->background,n[2]->background),
                          simpson(.5*width,n[2]->background,n[3]->background,n[4]->background));
    bool zero_c=true,zero_b=true;for(const auto node:n) {
      zero_c=zero_c&&!node->state.cohesion.active;zero_b=zero_b&&!node->state.background.active;
    }
    const bool hidden_c=c.work_of_separation>0.&&zero_c&&!support_excluded(lo,hi,x,n,c.opening_range);
    const bool hidden_b=b.hamaker>0.&&zero_b&&!support_excluded(lo,hi,x,n,b.far_cutoff-b.reference_gap);
    double ce=0.,be=0.;
    const bool okc=c.work_of_separation==0.||check_kernel(coarse_c,fine_c,
      n[4]->state.cohesion.energy-n[0]->state.cohesion.energy,width,c_work_scale,c_force_scale,hidden_c,ce);
    const bool okb=b.hamaker==0.||check_kernel(coarse_b,fine_b,
      n[4]->state.background.energy-n[0]->state.background.energy,width,b_work_scale,b_force_scale,hidden_b,be);
    if(okc&&okb){total_c=add(total_c,fine_c);total_b=add(total_b,fine_b);c_error+=ce;b_error+=be;++panels;return;}
    if(depth>=o.maximum_depth){std::ostringstream message;message<<"surface AVF pair quadrature exceeded its refinement depth"
      <<" interval="<<lo<<","<<hi<<" hidden="<<hidden_c<<","<<hidden_b
      <<" work="<<std::abs(work(fine_c,p)+n[4]->state.cohesion.energy-n[0]->state.cohesion.energy)/c_work_scale
      <<","<<std::abs(work(fine_b,p)+n[4]->state.background.energy-n[0]->state.background.energy)/b_work_scale
      <<" width="<<width;throw Failure(message.str());}
    integrate(lo,mid,depth+1);integrate(mid,hi,depth+1);
  }
  KernelIntegral finish(const Values&v,double initial,double endpoint,double ws,double error) const {
    KernelIntegral k;for(int a=0;a<3;++a) {
      k.force_i[a]=v[a];k.force_j[a]=-v[a];k.torque_i[a]=v[3+a];k.torque_j[a]=v[6+a];k.orbital_moment[a]=v[9+a];
    }
    for(int a=0;a<9;++a)k.symmetric_virial[a]=v[12+a];
    k.approximate_normal_tangent=v[21];k.average_energy=v[22];
    k.initial_energy=initial;k.endpoint_energy=endpoint;k.work=work(v,p);
    k.work_defect=k.work+endpoint-initial;k.work_scale=ws;k.force_error_estimate=error;return k;
  }
public:
  Integrator(const Path&path,const sc::Parameters&cohesion,const sb::Parameters&background,
             const Options&options):p(path),c(cohesion),b(background),o(options) {
    if(!(o.relative_work_tolerance>0.)||!(o.relative_force_tolerance>0.)||
       o.maximum_evaluations<5||o.maximum_depth<0)throw std::invalid_argument("invalid surface AVF options");
    length=std::max(max_axis(p.initial_i),max_axis(p.initial_j));
    if(!(length>0.)||!gr::finite(p.relative_displacement)||!gr::finite(p.rotation_i)||!gr::finite(p.rotation_j))
      throw std::invalid_argument("invalid surface AVF path");
    for(bool first:{true,false}) {
      const auto&a=first?p.initial_i:p.initial_j;const auto&z=first?p.final_i:p.final_j;
      if(a.axes!=z.axes||!(min_axis(a)>0.))throw std::invalid_argument("surface AVF requires unchanged positive axes");
      const auto theta=first?p.rotation_i:p.rotation_j;
      if(gr::norm(theta)>=std::acos(-1.))throw Failure("surface AVF trial rotation is not a short path");
      const auto rotation=gr::rotateLaboratory(a.rotation,theta);
      for(int k=0;k<9;++k)if(std::abs(rotation[k]-z.rotation[k])>2e-10)
        throw std::invalid_argument("surface AVF rotation increment disagrees with endpoint");
      const double ma=max_axis(a),mi=min_axis(a);
      gap_second_bound+=(ma*ma/mi+ma)*gr::dot(theta,theta);
    }
    const auto actual=gr::sub(gr::sub(p.final_j.position,p.final_i.position),
                               gr::sub(p.initial_j.position,p.initial_i.position));
    if(gr::norm(gr::sub(actual,p.relative_displacement))>2e-10*length)
      throw std::invalid_argument("surface AVF relative displacement disagrees with endpoint");
  }
  Result run(const Sample*known_initial,const Sample*known_endpoint) {
    Result out;
    if(c.work_of_separation==0.&&b.hamaker==0.){out.converged=true;return out;}
    nodes.emplace(0.,make(0.,known_initial));nodes.emplace(1.,make(1.,known_endpoint));
    const auto&begin=at(0.);const auto&end=at(1.);
    const double lower=std::max(
      std::min(begin.state.geometry.gap,begin.state.geometry.gap+begin.gap_rate-.5*gap_second_bound),
      std::min(end.state.geometry.gap,end.state.geometry.gap-end.gap_rate-.5*gap_second_bound));
    const double roundoff=32.*std::numeric_limits<double>::epsilon()*length;
    const bool c_absent=c.work_of_separation==0.||(!begin.state.cohesion.active&&!end.state.cohesion.active&&
      lower>=c.opening_range-roundoff);
    const bool b_absent=b.hamaker==0.||(!begin.state.background.active&&!end.state.background.active&&
      lower>=b.far_cutoff-b.reference_gap-roundoff);
    if(c_absent&&b_absent) {
      out.initial=begin.state;out.endpoint=end.state;out.evaluations=evaluations;out.converged=true;
      return out; // rigorously excluded whole-path support; no interior kernel
    }
    const auto&middle=at(.5);
    const double radius_bound=1./(min_axis(p.initial_i)/std::pow(max_axis(p.initial_i),2)+
                                  min_axis(p.initial_j)/std::pow(max_axis(p.initial_j),2));
    const double c_bind=std::acos(-1.)*radius_bound*c.work_of_separation*c.opening_range;
    const double b_bind=b.hamaker>0.?2.*std::acos(-1.)*radius_bound*sb::work_of_separation(b)*b.reference_gap:0.;
    c_work_scale=std::max({c_bind,std::abs(begin.state.cohesion.energy),std::abs(end.state.cohesion.energy),
                          std::abs(middle.state.cohesion.energy),std::numeric_limits<double>::min()});
    b_work_scale=std::max({b_bind,std::abs(begin.state.background.energy),std::abs(end.state.background.energy),
                          std::abs(middle.state.background.energy),std::numeric_limits<double>::min()});
    c_force_scale=std::max({2.*std::acos(-1.)*radius_bound*c.work_of_separation,
      wrench_norm(begin.cohesion,length),wrench_norm(end.cohesion,length),wrench_norm(middle.cohesion,length),
      std::numeric_limits<double>::min()});
    b_force_scale=std::max({b_bind/b.reference_gap,wrench_norm(begin.background,length),
      wrench_norm(end.background,length),wrench_norm(middle.background,length),std::numeric_limits<double>::min()});
    // Locate the main branch transitions before adaptive integration. A few
    // scalar gap-Newton updates are far cheaper than repeatedly bisecting a
    // panel whose only defect is a misplaced kink. These are local geometry
    // queries, not global mechanical solves. Nonmonotone extra crossings are
    // still handled by panel quadrature and the support enclosure.
    std::vector<double>cuts{0.,1.};const double g0=begin.state.geometry.gap,g1=end.state.geometry.gap;
    std::vector<double>boundaries{0.};if(c.work_of_separation>0.)boundaries.push_back(c.opening_range);
    if(b.hamaker>0.)for(double boundary:{b.curvature_switch-b.reference_gap,b.curvature_cutoff-b.reference_gap,
        b.far_switch-b.reference_gap,b.far_cutoff-b.reference_gap})boundaries.push_back(boundary);
    if(g0!=g1)for(double boundary:boundaries) {
      double s=(boundary-g0)/(g1-g0);if(!(s>0.&&s<1.))continue;
      double lo=0.,hi=1.;
      for(int trial=0;trial<4;++trial) {
        const auto&node=at(s);const double defect=node.state.geometry.gap-boundary;
        if(std::abs(defect)<=32.*std::numeric_limits<double>::epsilon()*length)break;
        if((defect>0.)==(g1>g0))hi=s;else lo=s;
        double next=node.gap_rate!=0.?s-defect/node.gap_rate:.5*(lo+hi);
        if(!(next>lo&&next<hi))next=.5*(lo+hi);
        if(next==s)break;s=next;
      }
      cuts.push_back(s);
    }
    std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
    for(std::size_t k=1;k<cuts.size();++k)integrate(cuts[k-1],cuts[k],0);
    out.initial=begin.state;out.endpoint=end.state;
    out.cohesion=finish(total_c,begin.state.cohesion.energy,end.state.cohesion.energy,c_work_scale,c_error);
    out.background=finish(total_b,begin.state.background.energy,end.state.background.energy,b_work_scale,b_error);
    out.evaluations=evaluations;out.panels=panels;out.maximum_depth=deepest;out.converged=true;
    if(std::abs(out.cohesion.work_defect)>o.relative_work_tolerance*c_work_scale||
       std::abs(out.background.work_defect)>o.relative_work_tolerance*b_work_scale)
      throw Failure("surface AVF accumulated work check failed");
    return out;
  }
};
} // namespace detail

inline Result evaluate(const Path&path,const sc::Parameters&cohesion,const sb::Parameters&background,
                       const Options&options={},const Sample*known_initial=nullptr,
                       const Sample*known_endpoint=nullptr) {
  return detail::Integrator(path,cohesion,background,options).run(known_initial,known_endpoint);
}
} // namespace SlurrySurfaceDiscreteGradient
#endif
