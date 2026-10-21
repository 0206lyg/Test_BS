#include "slurry_general_provider.h"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace sg=SlurryGeneral;
namespace gp=SlurryGrandPair;
namespace gr=SlurryFiniteGap::gr;
namespace {
double largest_error=0.;
void require(bool value,const char* label) {
  if(!value)throw std::runtime_error(label);
}
template<class A,class B>
void close(const Eigen::MatrixBase<A>& actual,const Eigen::MatrixBase<B>& expected,
           double tolerance,const char* label) {
  const double error=(actual-expected).norm()/std::max({actual.norm(),expected.norm(),1e-290});
  largest_error=std::max(largest_error,error);
  if(!(error<tolerance)) {
    std::cerr<<label<<" relative error="<<std::setprecision(17)<<error<<'\n';
    throw std::runtime_error(label);
  }
}
void close_scalar(double actual,double expected,double tolerance,const char* label) {
  Eigen::Matrix<double,1,1> a,b;a<<actual;b<<expected;close(a,b,tolerance,label);
}
template<size_t N> Eigen::Matrix<double,N,1> vector(const std::array<double,N>& a) {
  Eigen::Matrix<double,N,1> v;for(size_t k=0;k<N;++k)v[k]=a[k];return v;
}
gp::Mat3 array3(const sg::Mat3& matrix) {
  gp::Mat3 a{};for(int r=0;r<3;++r)for(int c=0;c<3;++c)a[3*r+c]=matrix(r,c);return a;
}
gp::Vec3 array3(const sg::Vec3& value) {return {value.x(),value.y(),value.z()};}
sg::Mat3 matrix3(const gp::Mat3& a) {
  sg::Mat3 m;for(int r=0;r<3;++r)for(int c=0;c<3;++c)m(r,c)=a[3*r+c];return m;
}
sg::Mat17 matrix17(const gp::Model& model) {
  sg::Mat17 m;for(int r=0;r<17;++r)for(int c=0;c<17;++c)m(r,c)=model.resistance[17*r+c];return m;
}
struct PhysicalPair {gr::Body i,j;gr::GapResult gap;double c=150e-9;};
PhysicalPair physical(const sg::Geometry& geometry,double c=150e-9) {
  PhysicalPair p;p.c=c;p.i.axes=p.j.axes={geometry.ar*c,geometry.ar*c,c};
  p.i.position=array3(sg::Vec3(c*geometry.centers[0]));
  p.j.position=array3(sg::Vec3(c*geometry.centers[1]));
  p.i.rotation=array3(geometry.rotations[0]);p.j.rotation=array3(geometry.rotations[1]);
  p.gap.gap=c*geometry.gap;p.gap.normal={0,0,1};p.gap.converged=true;
  p.gap.leverI=array3(sg::Vec3(-c*geometry.centers[0]));
  p.gap.leverJ=array3(sg::Vec3(c*(sg::Vec3(0,0,geometry.gap)-geometry.centers[1])));
  return p;
}
PhysicalPair rotated(const PhysicalPair& original,const sg::Mat3& rotation) {
  auto p=original;
  p.i.position=array3(sg::Vec3(rotation*vector(original.i.position)));
  p.j.position=array3(sg::Vec3(rotation*vector(original.j.position)));
  p.i.rotation=array3(sg::Mat3(rotation*matrix3(original.i.rotation)));
  p.j.rotation=array3(sg::Mat3(rotation*matrix3(original.j.rotation)));
  p.gap.normal=array3(sg::Vec3(rotation*vector(original.gap.normal)));
  p.gap.leverI=array3(sg::Vec3(rotation*vector(original.gap.leverI)));
  p.gap.leverJ=array3(sg::Vec3(rotation*vector(original.gap.leverJ)));return p;
}
gp::Inputs input(double solver_length) {
  gp::Inputs x;
  for(int body=0;body<2;++body)for(int d=0;d<3;++d) {
    x.velocity[6*body+d]=1e-5*std::cos(3*body+d+.2);
    x.background[6*body+d]=2e-6*std::sin(2*body+d+.7);
    x.velocity[6*body+3+d]=solver_length*80*std::sin(3*body+d+.4);
    x.background[6*body+3+d]=solver_length*13*std::cos(2*body+d+.6);
  }
  x.strain={17.,23.,-7.,23.,-9.,11.,-7.,11.,-8.};return x;
}
gp::Inputs rotated(const gp::Inputs& x,const sg::Mat3& rotation) {
  auto y=x;
  for(int block=0;block<4;++block) {
    sg::Vec3 velocity,background;
    for(int d=0;d<3;++d){velocity[d]=x.velocity[3*block+d];background[d]=x.background[3*block+d];}
    velocity=(rotation*velocity).eval();background=(rotation*background).eval();
    for(int d=0;d<3;++d){y.velocity[3*block+d]=velocity[d];y.background[3*block+d]=background[d];}
  }
  y.strain=array3(sg::Mat3(rotation*matrix3(x.strain)*rotation.transpose()));return y;
}
sg::Mat3 skew(const sg::Vec3& v) {
  sg::Mat3 m;m<<0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;return m;
}
// Independent midpoint relative-motion map in public normalized coordinates.
sg::Map6 relative_map(const sg::Geometry& g) {
  sg::Map6 b=sg::Map6::Zero();const sg::Vec3 midpoint(0,0,.5*g.gap);
  b.block<3,3>(0,0)=-sg::Mat3::Identity();b.block<3,3>(0,6)=sg::Mat3::Identity();
  b.block<3,3>(0,3)=skew((midpoint-g.centers[0])/g.ar);
  b.block<3,3>(0,9)=-skew((midpoint-g.centers[1])/g.ar);
  b.block<3,3>(3,3)=-sg::Mat3::Identity();b.block<3,3>(3,9)=sg::Mat3::Identity();return b;
}
Eigen::Matrix<double,5,5> free_strain(const sg::Mat17& r) {
  const Eigen::Matrix<double,12,12> velocity=r.topLeftCorner<12,12>();
  Eigen::LLT<Eigen::Matrix<double,12,12>> factor(velocity);
  require(factor.info()==Eigen::Success,"free response velocity block must be SPD");
  return r.bottomRightCorner<5,5>()-r.bottomLeftCorner<5,12>()*factor.solve(r.topRightCorner<12,5>());
}
}
int main() {
  try {
    constexpr double eta=.01,c=150e-9;
    // This exact support-point geometry has physical gap 3 nm. Its first
    // director has positive x projection, so the canonical frame is identity.
    const auto g=sg::geometry_from_axes(2.,.02,sg::Vec3(.5,0,std::sqrt(.75)),
      sg::Vec3(std::sqrt(.75)*std::cos(.7),std::sqrt(.75)*std::sin(.7),.5));
    const auto p=physical(g,c);const double ell=g.ar*c,solver_length=1.7*ell;
    sg::Provider provider;provider.relative_closure=false;
    const auto raw=sg::evaluate_model(g,provider.options);
    const auto additive=provider.make_pair(p.i,p.j,p.gap,eta,solver_length,10.,.8,41,97);
    require(additive.active&&additive.weight==1.,"interior pair must be active at unit weight");
    close(matrix3(additive.frame.world_from_local),sg::Mat3::Identity(),2e-14,"canonical physical frame");
    close_scalar(additive.frame.model_length,ell,2e-14,"model length uses equatorial semiaxis");
    close(matrix17(additive.model),(raw.grand-raw.isolated)/g.ar,2e-12,"isolated subtraction and c/a conversion");
    close(matrix17(additive.model)+raw.isolated/g.ar,raw.grand/g.ar,2e-12,"isolated plus additive recovers full pair");
    auto x=input(solver_length);
    const auto y=gp::evaluate(additive.model,additive.frame,x);
    Eigen::Matrix<double,17,1> z=Eigen::Matrix<double,17,1>::Zero();
    for(int body=0;body<2;++body)for(int d=0;d<3;++d) {
      z[6*body+d]=x.velocity[6*body+d]-x.background[6*body+d];
      z[6*body+3+d]=ell/solver_length*(x.velocity[6*body+3+d]-x.background[6*body+3+d]);
    }
    const auto e=gp::strain_coordinates(x.strain);for(int k=0;k<5;++k)z[12+k]=ell*e[k];
    const Eigen::Matrix<double,17,1> force=eta*c*(raw.grand-raw.isolated)*z;
    close(vector(y.force_i),-force.segment<3>(0),3e-12,"physical force normalization/sign");
    close(vector(y.force_j),-force.segment<3>(6),3e-12,"physical second force normalization/sign");
    close(vector(y.torque_i),-ell*force.segment<3>(3),3e-12,"physical torque normalization/sign");
    close(vector(y.torque_j),-ell*force.segment<3>(9),3e-12,"physical second torque normalization/sign");
    close(vector(y.stress_coordinates),ell*force.tail<5>(),3e-12,"physical intrinsic stresslet normalization/sign");
    close_scalar(y.power,z.dot(force),3e-12,"physical raw operator power");

    provider.relative_closure=true;
    const auto relative=provider.make_pair(p.i,p.j,p.gap,eta,solver_length,10.,.8);
    const sg::Mat17 combined=raw.isolated/g.ar+matrix17(relative.model);
    const auto b=relative_map(g);
    close(sg::reduced_response(combined,b),sg::reduced_response(raw.grand/g.ar,b),4e-11,
      "relative closure recovers reduced full pair without double counting");
    close(free_strain(combined),free_strain(raw.grand/g.ar),4e-11,
      "relative closure recovers force/torque-free common strain response");
    Eigen::Matrix<double,17,3> translation=Eigen::Matrix<double,17,3>::Zero();
    translation.block<3,3>(0,0)=translation.block<3,3>(6,0)=sg::Mat3::Identity();
    require((matrix17(relative.model)*translation).norm()<1e-12*matrix17(relative.model).norm(),
      "relative correction must leave collective translation drag isolated");

    const auto base=gp::evaluate(relative.model,relative.frame,x);
    const auto rotation=Eigen::AngleAxisd(.731,sg::Vec3(1,2,3).normalized()).toRotationMatrix();
    const auto rp=rotated(p,rotation);
    const auto rm=provider.make_pair(rp.i,rp.j,rp.gap,eta,solver_length,10.,.8);
    const auto ry=gp::evaluate(rm.model,rm.frame,rotated(x,rotation));
    close_scalar(ry.power,base.power,2e-10,"rotated physical pair power");
    close(vector(ry.force_i),rotation*vector(base.force_i),2e-10,"rotated physical pair force");
    close(vector(ry.torque_j),rotation*vector(base.torque_j),2e-10,"rotated physical pair torque");
    close(matrix3(ry.stress),rotation*matrix3(base.stress)*rotation.transpose(),2e-10,
      "rotated physical pair stresslet");

    constexpr double size_scale=4.,eta_scale=3.;
    auto scaled_provider=provider;scaled_provider.epsilon_m*=size_scale;
    const auto sp=physical(g,c*size_scale);
    const auto sm=scaled_provider.make_pair(sp.i,sp.j,sp.gap,eta*eta_scale,solver_length*size_scale,10.,.8);
    auto sx=x;for(double&v:sx.strain)v/=size_scale;
    const auto sy=gp::evaluate(sm.model,sm.frame,sx);
    close_scalar(sy.power,eta_scale*size_scale*base.power,3e-12,"size/viscosity scaled power");
    close(vector(sy.force_i),eta_scale*size_scale*vector(base.force_i),3e-12,"size/viscosity scaled force");
    close(vector(sy.torque_i),eta_scale*size_scale*size_scale*vector(base.torque_i),3e-12,
      "size/viscosity scaled torque");
    close(matrix3(sy.stress),eta_scale*size_scale*size_scale*matrix3(base.stress),3e-12,
      "size/viscosity scaled stresslet");

    // Vary +E with particle and background velocities held fixed: dQ/dE=S.
    Eigen::Matrix<double,5,1> finite_difference;
    constexpr double strain_step=.01;
    for(int k=0;k<5;++k) {
      auto plus=x,minus=x;
      for(int d=0;d<9;++d){plus.strain[d]+=strain_step*gp::strain_basis()[k][d];minus.strain[d]-=strain_step*gp::strain_basis()[k][d];}
      finite_difference[k]=(gp::evaluate(relative.model,relative.frame,plus).energy-
        gp::evaluate(relative.model,relative.frame,minus).energy)/(2*strain_step);
    }
    close(finite_difference,vector(base.stress_coordinates),2e-9,"stress is the strain energy derivative");
    double conjugate=0.;
    for(int d=0;d<12;++d)conjugate+=(x.velocity[d]-x.background[d])*base.gradient[d];
    for(int d=0;d<9;++d)conjugate+=x.strain[d]*base.stress[d];
    close_scalar(conjugate,base.power,3e-12,"force/torque/stress power conjugacy");

    // A floor applies to hydro geometry only. It must not mutate contact gap.
    const auto floor_geometry=sg::geometry_from_axes(g.ar,provider.epsilon_m/c,g.rotations[0].col(2),g.rotations[1].col(2));
    const auto below_geometry=sg::geometry_from_axes(g.ar,.5*provider.epsilon_m/c,g.rotations[0].col(2),g.rotations[1].col(2));
    const auto below=physical(below_geometry,c);
    auto floor_provider=provider;floor_provider.relative_closure=false;
    const auto floor_model=floor_provider.make_pair(below.i,below.j,below.gap,eta,solver_length,10.,.8);
    const auto floor_raw=sg::evaluate_model(floor_geometry,provider.options);
    close(matrix17(floor_model.model),(floor_raw.grand-floor_raw.isolated)/g.ar,4e-12,"fluid gap floor response");
    close_scalar(below.gap.gap,.5*provider.epsilon_m,1e-15,"mechanical contact gap remains unchanged");
    auto outside=p;outside.gap.gap=100*c;
    const auto zero=provider.make_pair(outside.i,outside.j,outside.gap,eta,solver_length,.5,.8);
    require(!zero.active&&zero.weight==0.&&matrix17(zero.model).norm()==0.,"outside range is inactive");

    const auto sphere=sg::geometry_from_axes(1.,.02,sg::Vec3::UnitZ(),sg::Vec3::UnitZ());
    const auto sphere_raw=sg::evaluate_model(sphere,provider.options);
    const double pi=std::acos(-1.);
    close_scalar(sphere_raw.isolated(0,0),6*pi,3e-12,"sphere isolated translation coefficient");
    close_scalar(sphere_raw.isolated(3,3),8*pi,3e-12,"sphere isolated rotation coefficient");
    close_scalar(sphere_raw.isolated(12,12),40*pi/3.,3e-12,"two spheres isolated common strain coefficient");
    const auto sph=physical(sphere,c);
    const auto sphere_pair=provider.make_pair(sph.i,sph.j,sph.gap,eta,c,10.,.8);
    require(sphere_pair.active&&matrix17(sphere_pair.model).allFinite(),"AR1 uses general provider");

    sg::Provider coarse;coarse.options.max_level=1;
    const auto difficult=physical(sg::geometry_from_axes(20.,.02,sg::Vec3::UnitZ(),sg::Vec3::UnitZ()),c);
    bool rejected=false;
    try{coarse.make_pair(difficult.i,difficult.j,difficult.gap,eta,20*c,10.,.8);}
    catch(const std::runtime_error& error){rejected=std::string(error.what()).find("numerically unresolved")!=std::string::npos;}
    require(rejected,"unresolved AR20 low-order quadrature must fail explicitly");
    std::cout<<std::setprecision(12)<<"PASS: general provider units/signs, isolated subtraction, reduced closure/free C5, rotation/similarity, stress-work, floor/window, sphere and unresolved rejection; largest relative check error="<<largest_error<<'\n';
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;
  }
}
