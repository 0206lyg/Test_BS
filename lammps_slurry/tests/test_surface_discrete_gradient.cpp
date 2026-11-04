#include "../src/slurry_surface_discrete_gradient.h"
#include <chrono>
#include <iomanip>
#include <iostream>

namespace dg=SlurrySurfaceDiscreteGradient;
namespace gr=slurry::gr_re2::graphite;
namespace sc=SlurrySurfaceCohesion;
namespace sb=SlurrySurfaceBackground;
using gr::Body;using gr::Vec3;
void require(bool yes,const char*message){if(!yes)throw std::runtime_error(message);}
Body shape(double ar,const Vec3&angle={}){
  Body b;const double a=4e-6*std::cbrt(ar);b.axes={a,a,a/ar};b.rotation=gr::rotationIncrement(angle);return b;
}
Vec3 support(const Body&b,const Vec3&n){
  const auto d=gr::mul(gr::transpose(b.rotation),n);Vec3 v{};double s=0.;
  for(int k=0;k<3;++k){v[k]=b.axes[k]*b.axes[k]*d[k];s+=d[k]*v[k];}
  return gr::mul(b.rotation,gr::scale(v,1./std::sqrt(s)));
}
void place(Body&a,Body&b,const Vec3&n,double gap){
  b.position=gr::add(a.position,gr::add(gr::add(support(a,n),support(b,n)),gr::scale(n,gap)));
}
dg::Path path(Body a,Body b,const Vec3&dr,const Vec3&ti={},const Vec3&tj={}){
  dg::Path p;p.initial_i=a;p.initial_j=b;p.final_i=a;p.final_j=b;
  p.relative_displacement=dr;p.rotation_i=ti;p.rotation_j=tj;
  p.final_j.position=gr::add(b.position,dr);p.final_i.rotation=gr::rotateLaboratory(a.rotation,ti);
  p.final_j.rotation=gr::rotateLaboratory(b.rotation,tj);return p;
}
void inspect(const dg::KernelIntegral&r,double length,const char*name){
  require(gr::norm(gr::add(r.force_i,r.force_j))==0.,"AVF action reaction is not exact");
  const auto moment=gr::sub(gr::add(r.torque_i,r.torque_j),r.orbital_moment);
  require(gr::norm(moment)<3e-10*std::max(gr::norm(r.force_i)*length,1e-35),"AVF path angular balance failed");
  require(std::abs(r.work_defect)<=1.1e-7*r.work_scale,"AVF pair work does not match potential change");
  std::cout<<name<<" work_relative="<<r.work_defect/r.work_scale
           <<" force_error="<<r.force_error_estimate<<'\n';
}
void spherical_crossing(){
  sc::Parameters c;sb::Parameters b;b.hamaker=0.;
  for(const auto interval:std::array<std::array<double,2>,5>{{
      {-1e-9,3e-9},{3e-9,-1e-9},{.1e-9,.5e-9},{.67e-9,1.2e-9},{-8e-9,-3e-9}}}){
    auto i=shape(1.),j=shape(1.);place(i,j,{0,0,1},interval[0]);
    const auto p=path(i,j,{0,0,interval[1]-interval[0]});const auto r=dg::evaluate(p,c,b);
    const double dr=p.relative_displacement[2];
    const double exact=(r.cohesion.endpoint_energy-r.cohesion.initial_energy)/dr;
    require(std::abs(r.cohesion.force_i[2]-exact)<2e-8*2.*std::acos(-1.)*2e-6*c.work_of_separation,
      "normal crossing averaged force differs from exact primitive");
    const double exact_tangent=(r.endpoint.cohesion.force_i[2]-r.cohesion.force_i[2])/dr;
    if(std::min(interval[0],interval[1])>0.&&std::max(interval[0],interval[1])<c.opening_range)
      require(std::abs(r.cohesion.approximate_normal_tangent-exact_tangent)<
        3e-7*2.*std::acos(-1.)*2e-6*c.work_of_separation/c.opening_range,
        "smooth normal-path averaged tangent is inconsistent");
    inspect(r.cohesion,4e-6,"sphere");
    std::cout<<"sphere begin="<<interval[0]<<" end="<<interval[1]<<" evals="<<r.evaluations<<" panels="<<r.panels<<'\n';
    const auto initial=dg::sample(p.initial_i,p.initial_j,c,b),end=dg::sample(p.final_i,p.final_j,c,b);
    const auto reused=dg::evaluate(p,c,b,{},&initial,&end);
    require(reused.evaluations==r.evaluations-2,"provided endpoint samples were recomputed");
    require(gr::norm(gr::sub(reused.cohesion.force_i,r.cohesion.force_i))<1e-20,"cached endpoints changed AVF");
  }
}
void oblique_rotating(){
  sc::Parameters c;sb::Parameters b;
  auto i=shape(4.,{.2,-.3,.1}),j=shape(2.3,{-.3,.2,.15});
  const auto normal=gr::normalized(Vec3{.31,-.42,.85});place(i,j,normal,-1e-9);
  const Vec3 ti{1.4e-4,-.6e-4,.8e-4},tj{-.7e-4,1.2e-4,-.5e-4};
  auto p=path(i,j,gr::add(gr::scale(normal,2.1e-9),Vec3{.3e-9,-.4e-9,.1e-9}),ti,tj);
  auto r=dg::evaluate(p,c,b);inspect(r.cohesion,7e-6,"oblique_cohesion");inspect(r.background,7e-6,"oblique_background");
  std::cout<<"oblique evals="<<r.evaluations<<" panels="<<r.panels<<" depth="<<r.maximum_depth<<'\n';
  // Compare the entire wrench, including components orthogonal to motion,
  // against independently tightened quadrature.
  dg::Options fine;fine.relative_work_tolerance=2e-9;fine.relative_force_tolerance=2e-7;fine.maximum_evaluations=1025;
  const auto ref=dg::evaluate(p,c,b,fine);
  for(const auto pair:std::array<std::pair<const dg::KernelIntegral*,const dg::KernelIntegral*>,2>{{
      {&r.cohesion,&ref.cohesion},{&r.background,&ref.background}}}) {
    const double scale=std::max(gr::norm(pair.second->force_i),1e-30);
    require(gr::norm(gr::sub(pair.first->force_i,pair.second->force_i))<2e-5*scale,"AVF force quadrature did not converge");
    require(gr::norm(gr::sub(pair.first->torque_i,pair.second->torque_i))<2e-5*scale*7e-6,"AVF torque quadrature did not converge");
  }
  // Fixed change of laboratory frame: all forces and torques must rotate.
  const auto Q=gr::rotationIncrement({.37,-.24,.19});auto rotated=p;
  for(auto body:{&rotated.initial_i,&rotated.initial_j,&rotated.final_i,&rotated.final_j}){
    body->position=gr::mul(Q,body->position);body->rotation=gr::multiply(Q,body->rotation);
  }
  rotated.relative_displacement=gr::mul(Q,p.relative_displacement);
  rotated.rotation_i=gr::mul(Q,p.rotation_i);rotated.rotation_j=gr::mul(Q,p.rotation_j);
  const auto rr=dg::evaluate(rotated,c,b);
  for(const auto pair:std::array<std::pair<const dg::KernelIntegral*,const dg::KernelIntegral*>,2>{{
      {&r.cohesion,&rr.cohesion},{&r.background,&rr.background}}}) {
    const double scale=std::max(gr::norm(pair.first->force_i),1e-30);
    require(gr::norm(gr::sub(gr::mul(Q,pair.first->force_i),pair.second->force_i))<3e-7*scale,"AVF force is not objective");
    require(gr::norm(gr::sub(gr::mul(Q,pair.first->torque_i),pair.second->torque_i))<3e-7*scale*7e-6,"AVF torque is not objective");
  }
}
void hidden_window(){
  // A sphere travels tangentially past another. Endpoint and midpoint samples
  // are all outside the 0.67 nm window, yet the path touches it at s=0.3.
  sc::Parameters c;sb::Parameters b;b.hamaker=0.;
  auto i=shape(1.),j=shape(1.);const double offset=800e-9;
  j.position={-.3*offset,8e-6+.2e-9,0.};const auto p=path(i,j,{offset,0,0});
  const auto begin=dg::sample(p.initial_i,p.initial_j,c,b),end=dg::sample(p.final_i,p.final_j,c,b);
  auto middle_j=j;middle_j.position=gr::add(j.position,{.5*offset,0.,0.});
  const auto middle=dg::sample(i,middle_j,c,b);
  // This chosen scale has a narrow off-centre feature; the exact minimum is
  // known analytically, independently of any midpoint samples.
  require(!begin.cohesion.active&&!end.cohesion.active&&!middle.cohesion.active,"hidden-window fixture is not hidden");
  dg::Options options;options.maximum_evaluations=513;
  const auto r=dg::evaluate(p,c,b,options,&begin,&end);
  require(gr::norm(r.cohesion.force_i)>1e-10,"AVF missed entry into a hidden narrow support");
  inspect(r.cohesion,4e-6,"hidden");std::cout<<"hidden evals="<<r.evaluations<<'\n';
}
void budget_and_disabled(){
  auto i=shape(4.,{.2,-.3,.1}),j=shape(2.);place(i,j,gr::normalized(Vec3{.3,.2,.9}),-.2e-9);
  const auto p=path(i,j,{3e-9,-2e-9,6e-9},{1e-3,.7e-3,-.3e-3},{-.8e-3,.4e-3,.9e-3});
  dg::Options options;options.maximum_evaluations=5;options.relative_work_tolerance=1e-12;
  bool failed=false;try{dg::evaluate(p,{}, {},options);}catch(const dg::Failure&){failed=true;}
  require(failed,"quadrature silently exceeded a hard evaluation budget");
  sc::Parameters c;c.work_of_separation=0.;sb::Parameters b;b.hamaker=0.;
  const auto disabled=dg::evaluate(p,c,b);require(disabled.converged&&disabled.evaluations==0,"disabled AVF kernel is not exact zero");
  // Known endpoint samples certify that a dormant pair never enters either
  // support, so the local algorithm must not evaluate an interior kernel.
  c=sc::Parameters{};b=sb::Parameters{};auto a=shape(1.),z=shape(1.);
  place(a,z,{0,0,1},700e-9);const auto far=path(a,z,{0,0,30e-9});
  const auto first=dg::sample(far.initial_i,far.initial_j,c,b),last=dg::sample(far.final_i,far.final_j,c,b);
  const auto dormant=dg::evaluate(far,c,b,{},&first,&last);
  require(dormant.converged&&dormant.evaluations==0,"certified dormant pair performed interior evaluations");
  auto oversized=far;oversized.rotation_i={4.,0.,0.};
  oversized.final_i.rotation=gr::rotateLaboratory(oversized.initial_i.rotation,oversized.rotation_i);
  bool recoverable=false;try{dg::evaluate(oversized,c,b);}catch(const dg::Failure&){recoverable=true;}
  require(recoverable,"oversized trial rotation is not a recoverable failure");
}
int main(){std::cout<<std::setprecision(12);
  spherical_crossing();oblique_rotating();hidden_window();budget_and_disabled();
  std::cout<<"surface AVF work, wrench, objectivity, hidden support and budget PASS\n";
}
