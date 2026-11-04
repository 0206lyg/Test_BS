#include "../src/slurry_surface_background.h"
#include <chrono>
#include <iostream>

namespace bg=SlurrySurfaceBackground;
namespace g=slurry::gr_re2::graphite;
using g::Body;using g::Vec3;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void close(double actual,double expected,double tolerance,const char* message){
  if(std::abs(actual-expected)>tolerance){std::cerr<<message<<": actual="<<actual<<" expected="<<expected<<" tolerance="<<tolerance<<'\n';throw std::runtime_error(message);}
}
Body body(double ar,const Vec3& angle={}){Body b;const double a=4e-6*std::cbrt(ar);b.axes={a,a,a/ar};b.rotation=g::rotationIncrement(angle);return b;}
Vec3 support(const Body&b,const Vec3&n){const auto q=g::rotatedDiagonal(b.rotation,{b.axes[0]*b.axes[0],b.axes[1]*b.axes[1],b.axes[2]*b.axes[2]});const auto v=g::mul(q,n);return g::scale(v,1./std::sqrt(g::dot(n,v)));}
void place(Body&a,Body&b,const Vec3&n,double gap){b.position=g::add(a.position,g::add(g::add(support(a,n),support(b,n)),g::scale(n,gap)));}

void analytic_cases(){
  const bg::Parameters p;const double pi=std::acos(-1.);
  const double W=p.hamaker/(12.*pi*p.reference_gap*p.reference_gap)*(1.-std::pow(p.sigma/p.reference_gap,6)/30.);
  close(bg::work_of_separation(p),W,1e-15*W,"planar background work");
  for(double ar:{1.,4.})for(Vec3 n:{Vec3{0,0,1},Vec3{1,0,0}}){
    auto a=body(ar),b=body(ar);const double R=n[2]!=0.?a.axes[0]*a.axes[0]/(2.*a.axes[2]):a.axes[2]/2.;
    for(double gap:{-30e-9,-1e-12,0.,.2e-9,2.e-9}){
      place(a,b,n,gap);const auto r=bg::evaluate(a,b,p);
      const double h=p.reference_gap+std::max(0.,gap),s6=std::pow(p.sigma/h,6);
      const double U=2.*R*(-p.hamaker/(12.*h)+p.hamaker*s6/(2520.*h));
      const double F=2.*R*(p.hamaker/(12.*h*h)-p.hamaker*s6/(360.*h*h));
      close(r.energy,U+std::min(0.,gap)*F,2e-11*std::abs(U+std::min(0.,gap)*F),"near energy or overlap continuation");
      close(g::dot(r.force_i,n),F,2e-11*F,"near normal force is wrong");
      close(r.attraction_at_contact,2.*pi*R*W,2e-12*F,"background contact force");
      require(g::norm(r.torque_i)<1e-12*F*a.axes[0],"aligned background torque");
    }
  }
  auto a=body(4.),b=body(4.);place(a,b,{0,0,1},0.);const auto touch=bg::evaluate(a,b,p);
  std::cout<<"W_bg_J_m2="<<W<<" W_excess_J_m2="<<.0219-W<<" FF_background_F0_N="<<touch.attraction_at_contact<<'\n';
  // Both terms sum to the requested total planar work at contact.
  const SlurrySurfaceCohesion::Parameters cp{.0219-W,.67e-9};
  const auto cohesive=SlurrySurfaceCohesion::evaluate(a,b,cp);
  const double R=a.axes[0]*a.axes[0]/(2.*a.axes[2]);
  close(g::dot(g::add(touch.force_i,cohesive.force_i),Vec3{0,0,1}),2.*pi*R*.0219,2e-12*cohesive.attraction_at_contact,"background and excess double count total work");
  bg::Parameters off=p;off.hamaker=0.;a.axes[0]=-1.;
  const auto zero=bg::evaluate(a,b,off);require(!zero.active&&zero.energy==0.&&g::norm(zero.force_i)==0.,"disabled background is not exact zero");
}

void gradients(){
  const bg::Parameters p;double force_error=0.,torque_error=0.;
  const auto rigid=g::rotationIncrement({.31,-.17,.42});
  for(int sample=0;sample<4;++sample)for(double gap:{-17e-9,.3e-9,2.9e-9,3.1e-9,9e-9,17.9e-9,18.1e-9,80e-9,397.9e-9,398.1e-9,450e-9,497e-9}){
    auto a=body(4.,{.13+.11*sample,-.29,.18}),b=body(2.7,{-.21,.17+.08*sample,.31});
    const Vec3 n=g::normalized(Vec3{.24+.07*sample,-.39,.77});place(a,b,n,gap);
    const auto r=bg::evaluate(a,b,p);const double L=a.axes[0],F=std::max(g::norm(r.force_i),std::abs(r.energy)/L);
    for(int k=0;k<3;++k){
      const double dx=1e-13;auto bp=b,bm=b;bp.position[k]+=dx;bm.position[k]-=dx;
      const double fd=-(bg::evaluate(a,bp,p).energy-bg::evaluate(a,bm,p).energy)/(2.*dx);
      force_error=std::max(force_error,std::abs(fd-r.force_j[k])/F);
      close(fd,r.force_j[k],2e-5*F,"background force is not the energy gradient");
      for(int which=0;which<2;++which){
        auto ap=a,am=a;bp=b;bm=b;Vec3 theta{};theta[k]=1e-8;
        if(which==0){ap.rotation=g::rotateLaboratory(a.rotation,theta);am.rotation=g::rotateLaboratory(a.rotation,g::scale(theta,-1.));}
        else{bp.rotation=g::rotateLaboratory(b.rotation,theta);bm.rotation=g::rotateLaboratory(b.rotation,g::scale(theta,-1.));}
        const double fd_t=-(bg::evaluate(ap,bp,p).energy-bg::evaluate(am,bm,p).energy)/(2e-8);
        const double actual=which==0?r.torque_i[k]:r.torque_j[k];
        torque_error=std::max(torque_error,std::abs(fd_t-actual)/(F*L));
        close(actual,fd_t,2e-5*F*L,"background torque is not the energy gradient");
      }
    }
    const auto angular=g::add(g::add(r.torque_i,r.torque_j),g::cross(g::sub(b.position,a.position),r.force_j));
    require(g::norm(angular)<3e-11*F*L,"background angular momentum");
    const auto exchange=bg::evaluate(b,a,p);close(exchange.energy,r.energy,1e-10*std::abs(r.energy),"exchange energy");
    require(g::norm(g::sub(exchange.force_i,r.force_j))<1e-10*F,"exchange force");
    require(g::norm(g::sub(exchange.torque_i,r.torque_j))<1e-10*F*L,"exchange torque");
    auto ar=a,br=b;ar.position=g::mul(rigid,a.position);br.position=g::mul(rigid,b.position);ar.rotation=g::multiply(rigid,a.rotation);br.rotation=g::multiply(rigid,b.rotation);
    const auto rot=bg::evaluate(ar,br,p);close(rot.energy,r.energy,5e-9*std::abs(r.energy),"objectivity energy");
    require(g::norm(g::sub(rot.force_i,g::mul(rigid,r.force_i)))<5e-9*F,"objectivity force");
  }
  std::cout<<"background_max_force_gradient_relative_error="<<force_error<<" torque_scaled_error="<<torque_error<<'\n';
}

void switches_and_scaling(){
  const bg::Parameters p;auto a=body(4.,{.2,-.3,.1}),b=body(3.,{-.4,.1,.2});const Vec3 n=g::normalized(Vec3{.3,.2,.8});
  for(double h:{p.reference_gap,p.curvature_switch,p.curvature_cutoff,p.far_switch,p.far_cutoff}){
    const double gap=h-p.reference_gap,dx=1e-14;
    place(a,b,n,gap-dx);const auto left=bg::evaluate(a,b,p);
    place(a,b,n,gap+dx);const auto right=bg::evaluate(a,b,p);
    const double F=std::max({g::norm(left.force_i),g::norm(right.force_i),1e-15});
    close(left.energy,right.energy,3.*F*dx,"energy is discontinuous at a background join");
    require(g::norm(g::sub(left.force_i,right.force_i))<5e-5*F,"force is discontinuous at a background join");
  }
  place(a,b,n,p.far_cutoff-p.reference_gap+1e-12);const auto zero=bg::evaluate(a,b,p);
  require(!zero.active&&zero.energy==0.&&g::norm(zero.force_i)==0.,"background not zero beyond cutoff");
  place(a,b,n,10e-9);const auto si=bg::evaluate(a,b,p);
  const double L=4e-6,E=5.696e-18;auto an=a,bn=b;for(int k=0;k<3;++k){an.axes[k]/=L;bn.axes[k]/=L;an.position[k]/=L;bn.position[k]/=L;}
  auto pn=p;pn.hamaker/=E;pn.sigma/=L;pn.reference_gap/=L;pn.curvature_switch/=L;pn.curvature_cutoff/=L;pn.far_switch/=L;pn.far_cutoff/=L;
  const auto norm=bg::evaluate(an,bn,pn);close(norm.energy*E,si.energy,3e-10*std::abs(si.energy),"background energy unit conversion");
  require(g::norm(g::sub(g::scale(norm.force_i,E/L),si.force_i))<3e-10*g::norm(si.force_i),"background force unit conversion");
  require(g::norm(g::sub(g::scale(norm.torque_i,E),si.torque_i))<3e-10*g::norm(si.torque_i),"background torque unit conversion");
}
void pinned_lb_energy_values(){
  // Golden energies evaluated independently by the actual Test_LB
  // reSquaredPotential.h at commit 058daeee6854bfcb1c1ad1cc5cabd699446845db,
  // with surfaceAdhesion=true, adhesionWork=surfaceBackgroundWork, and the
  // virtual displacement r+h0*n. This checks the RE2 algebra and switches,
  // not merely whether the new implementation differentiates its own energy.
  const bg::Parameters p;auto a=body(4.,{.17,-.23,.1}),b=body(2.7,{-.2,.13,.3});
  const auto n=g::normalized(Vec3{.3,.2,.8});
  const std::array<double,9> gaps{0.,.3e-9,3e-9,10e-9,18e-9,30e-9,100e-9,450e-9,497e-9};
  const std::array<double,9> energies{-1.9915006601499432e-17,-1.7317401042525536e-17,
    -7.9660058668104296e-18,-2.5801644096578117e-18,-9.7368157297144511e-19,
    -6.0412593686886908e-19,-1.8167734072311276e-19,-1.5443653414644532e-20,-2.900380483861156e-25};
  for(std::size_t k=0;k<gaps.size();++k){place(a,b,n,gaps[k]);const auto r=bg::evaluate(a,b,p);
    close(r.energy,energies[k],2e-9*std::abs(energies[k]),"pinned Test_LB energy mismatch");}
}
void benchmark(){
  const bg::Parameters p;auto a=body(4.,{.2,-.3,.1}),b=body(4.,{-.4,.1,.2});
  for(double gap:{-10e-9,10e-9,100e-9}){
    place(a,b,g::normalized(Vec3{.3,.2,.8}),gap);const auto geometry=g::closestEllipsoidGap(a,b);
    double sum=0.;const int count=20000;const auto start=std::chrono::steady_clock::now();
    for(int k=0;k<count;++k)sum+=bg::evaluate(a,b,geometry,p).energy;
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"background_gap_m="<<gap<<" cached_ns_per_pair="<<seconds*1e9/count<<" checksum="<<sum<<'\n';
  }
}
int main(){analytic_cases();gradients();switches_and_scaling();pinned_lb_energy_values();benchmark();std::cout<<"surface background energy/force/torque PASS\n";}
