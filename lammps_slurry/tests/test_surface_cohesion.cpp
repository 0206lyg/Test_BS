#include "../src/slurry_surface_cohesion.h"
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace c=SlurrySurfaceCohesion;
namespace g=slurry::gr_re2::graphite;
using g::Body;using g::Vec3;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void close(double actual,double expected,double tolerance,const char* message){
  if(std::abs(actual-expected)>tolerance){std::cerr<<message<<": actual="<<actual<<" expected="<<expected<<" tolerance="<<tolerance<<"\n";throw std::runtime_error(message);}
}
Body body(double ar,const Vec3& angle={}) {
  Body b;const double a=4e-6*std::cbrt(ar);
  b.axes={a,a,a/ar};b.rotation=g::rotationIncrement(angle);return b;
}
Vec3 support(const Body& b,const Vec3& n) {
  const auto local=g::mul(g::transpose(b.rotation),n);
  Vec3 p{};double s=0.;
  for(int k=0;k<3;++k){p[k]=b.axes[k]*b.axes[k]*local[k];s+=local[k]*p[k];}
  return g::mul(b.rotation,g::scale(p,1./std::sqrt(s)));
}
void place(Body& a,Body& b,const Vec3& n,double gap) {
  b.position=g::add(a.position,g::add(g::add(support(a,n),support(b,n)),g::scale(n,gap)));
}
void analytic_cases() {
  c::Parameters p;const double pi=std::acos(-1.);
  for(double ar:{1.,4.})for(Vec3 n:{Vec3{0,0,1},Vec3{1,0,0}}) {
    auto a=body(ar),b=body(ar);
    const double radius=n[2]!=0.?a.axes[0]*a.axes[0]/(2.*a.axes[2]):a.axes[2]/2.;
    const double F0=2.*pi*radius*p.work_of_separation;
    for(double gap:{-30e-9,-1e-12,0.,.2e-9,.67e-9,1e-9}) {
      place(a,b,n,gap);const auto r=c::evaluate(a,b,p);
      if(gap>p.opening_range) {
        require(r.energy==0.&&g::norm(r.force_i)==0.&&!r.active,"outside range is not exactly zero");continue;
      }
      if(gap==p.opening_range) {
        require(std::abs(r.energy)<1e-20*F0*p.opening_range&&g::norm(r.force_i)<2e-11*F0,"cutoff limit is discontinuous");
        auto geometry=g::closestEllipsoidGap(a,b);geometry.gap=p.opening_range;
        const auto exact=c::evaluate(a,b,geometry,p);
        require(exact.energy==0.&&g::norm(exact.force_i)==0.&&!exact.active,"exact cutoff is not zero");continue;
      }
      const double fraction=gap<0.?1.:1.-gap/p.opening_range;
      const double U=gap<0.?F0*(gap-.5*p.opening_range):-.5*F0*p.opening_range*fraction*fraction;
      close(r.derjaguin_radius,radius,2e-13*radius,"analytic curvature radius");
      close(r.energy,U,2e-11*std::max(std::abs(U),F0*p.opening_range),"analytic energy");
      close(g::dot(r.force_i,n),F0*fraction,2e-11*F0,"analytic normal attraction");
      require(g::norm(r.torque_i)<1e-13*F0*a.axes[0],"aligned pair has spurious torque");
    }
  }
  auto a=body(4.),b=body(4.);const Vec3 n{0,0,1};
  place(a,b,n,0.);const auto touch=c::evaluate(a,b,p);
  const double kn=300.,equilibrium=-touch.attraction_at_contact/kn;
  place(a,b,n,equilibrium);const auto eq=c::evaluate(a,b,p);
  close(kn*(-eq.gap)-g::dot(eq.force_i,n),0.,1e-12*touch.attraction_at_contact,"overlap normal-force equilibrium");
  close(eq.energy+.5*kn*eq.gap*eq.gap,
        touch.energy-touch.attraction_at_contact*touch.attraction_at_contact/(2.*kn),
        2e-12*std::abs(eq.energy),"combined soft-contact energy minimum");
  std::cout<<"AR4_FF F0_N="<<touch.attraction_at_contact<<" kn300_equilibrium_overlap_m="<<-equilibrium<<"\n";
  // Disabled means exact zero and no evaluation of invalid geometry/shape.
  c::Parameters off=p;off.work_of_separation=0.;a.axes[0]=-1.;
  const auto zero=c::evaluate(a,b,off);
  require(zero.energy==0.&&zero.gap_attraction==0.&&!zero.active&&g::norm(zero.force_i)==0.&&g::norm(zero.torque_i)==0.,"disabled path is not exactly zero");
}
void derivatives_and_symmetries() {
  const c::Parameters p;double maximum_force_error=0.,maximum_torque_error=0.;
  const auto rotation=g::rotationIncrement({.37,-.23,.19});
  for(int sample=0;sample<7;++sample)for(double gap:{-.017e-6,.13e-9,.44e-9}) {
    auto a=body(4.,{.13+.11*sample,-.29,.18}),b=body(2.7,{-.21,.17+.08*sample,.31});
    const Vec3 n=g::normalized(Vec3{.24+.07*sample,-.39,.77});place(a,b,n,gap);
    const auto r=c::evaluate(a,b,p);const double L=a.axes[0],F=r.attraction_at_contact;
    for(int k=0;k<3;++k) {
      const double h=1e-13;auto plus=b,minus=b;plus.position[k]+=h;minus.position[k]-=h;
      const double numerical=-(c::evaluate(a,plus,p).energy-c::evaluate(a,minus,p).energy)/(2.*h);
      maximum_force_error=std::max(maximum_force_error,std::abs(numerical-r.force_j[k])/F);
      close(r.force_j[k],numerical,3e-6*F,"force does not differentiate energy");
      for(int which=0;which<2;++which) {
        auto ap=a,am=a,bp=b,bm=b;Vec3 theta{};theta[k]=1e-8;
        if(which==0){ap.rotation=g::rotateLaboratory(a.rotation,theta);am.rotation=g::rotateLaboratory(a.rotation,g::scale(theta,-1.));}
        else {bp.rotation=g::rotateLaboratory(b.rotation,theta);bm.rotation=g::rotateLaboratory(b.rotation,g::scale(theta,-1.));}
        const double numerical_torque=-(c::evaluate(ap,bp,p).energy-c::evaluate(am,bm,p).energy)/(2e-8);
        const double actual=which==0?r.torque_i[k]:r.torque_j[k];
        maximum_torque_error=std::max(maximum_torque_error,std::abs(numerical_torque-actual)/(F*L));
        close(actual,numerical_torque,3e-6*F*L,"torque does not differentiate energy");
      }
    }
    const auto total_moment=g::add(g::add(r.torque_i,r.torque_j),g::cross(g::sub(b.position,a.position),r.force_j));
    require(g::norm(total_moment)<2e-12*F*L,"angular momentum is not conserved");
    require(g::norm(g::add(r.force_i,r.force_j))==0.,"pair forces are not exactly antisymmetric");
    const auto exchange=c::evaluate(b,a,p);
    close(exchange.energy,r.energy,2e-10*std::abs(r.energy),"particle exchange changed energy");
    require(g::norm(g::sub(exchange.force_i,r.force_j))<2e-10*F,"particle exchange changed force");
    require(g::norm(g::sub(exchange.torque_i,r.torque_j))<2e-10*F*L,"particle exchange changed torque");
    auto ar=a,br=b;ar.rotation=g::multiply(rotation,a.rotation);br.rotation=g::multiply(rotation,b.rotation);
    ar.position=g::mul(rotation,a.position);br.position=g::mul(rotation,b.position);
    const auto rotated=c::evaluate(ar,br,p);
    close(rotated.energy,r.energy,2e-9*std::abs(r.energy),"rigid rotation changed energy");
    require(g::norm(g::sub(rotated.force_i,g::mul(rotation,r.force_i)))<2e-9*F,"force is not objective");
    require(g::norm(g::sub(rotated.torque_i,g::mul(rotation,r.torque_i)))<2e-9*F*L,"torque is not objective");
  }
  std::cout<<"maximum_force_gradient_error/F0="<<maximum_force_error
           <<" maximum_torque_gradient_error/(F0*a)="<<maximum_torque_error<<"\n";
}
void curvature_derivative() {
  // Vary orientation while adjusting the center to maintain the same normal
  // and gap. This isolates the curvature contribution from the much larger
  // force times support-lever torque and fails if R_D is held constant.
  const c::Parameters p;const Vec3 n=g::normalized(Vec3{.4,-.2,.8});
  for(double gap:{-17e-9,.2e-9}) {
    auto a=body(4.,{.3,-.2,.1}),b=body(3.,{-.1,.4,.2});place(a,b,n,gap);
    const auto r=c::evaluate(a,b,p);const double theta=2e-5;
    auto ap=a,am=a,bp=b,bm=b;ap.rotation=g::rotateLaboratory(a.rotation,{theta,0,0});am.rotation=g::rotateLaboratory(a.rotation,{-theta,0,0});
    place(ap,bp,n,gap);place(am,bm,n,gap);
    const double numerical=(c::evaluate(ap,bp,p).energy-c::evaluate(am,bm,p).energy)/(2.*theta);
    const auto displacement=g::scale(g::sub(bp.position,bm.position),1./(2.*theta));
    const double analytic=-r.torque_i[0]-g::dot(r.force_j,displacement);
    close(analytic,numerical,3e-5*std::abs(numerical),"curvature derivative missing from force/torque");
  }
}
void benchmark() {
  const c::Parameters p;auto a=body(4.,{.2,-.3,.1}),b=body(4.,{-.4,.1,.2});
  place(a,b,g::normalized(Vec3{.3,.2,.8}),-10e-9);
  const auto geometry=g::closestEllipsoidGap(a,b);double sum=0.;const int count=20000;
  const auto start=std::chrono::steady_clock::now();
  for(int k=0;k<count;++k)sum+=c::evaluate(a,b,geometry,p).energy;
  const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  require(std::isfinite(sum),"benchmark became nonfinite");
  std::cout<<"cached_geometry_kernel_ns_per_pair="<<seconds*1e9/count<<" checksum="<<sum<<"\n";
}
int main() {
  analytic_cases();derivatives_and_symmetries();curvature_derivative();benchmark();
  std::cout<<"surface cohesion energy/force/torque PASS\n";
}
