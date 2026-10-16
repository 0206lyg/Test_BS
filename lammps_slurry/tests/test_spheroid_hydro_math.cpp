#include "slurry_spheroid_hydro_math.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace sh=SlurrySpheroidHydro;
namespace ph=SlurryPairHydro;
namespace gr=slurry::gr_re2::graphite;
using ph::Vec3;using ph::Vec12;using ph::Mat3;
namespace {
void require(bool condition,const std::string&message)
{if(!condition)throw std::runtime_error(message);}
void close(double a,double b,double tol,const std::string&message,double floor=1e-25)
{if(std::abs(a-b)>tol*std::max({std::abs(a),std::abs(b),floor}))
  throw std::runtime_error(message+": "+std::to_string(a)+" != "+std::to_string(b));}
template<std::size_t N>void vector_close(const std::array<double,N>&a,const std::array<double,N>&b,
                                       double tol,const std::string&message)
{double scale=1e-25;for(std::size_t i=0;i<N;++i)scale=std::max({scale,std::abs(a[i]),std::abs(b[i])});
 for(std::size_t i=0;i<N;++i)if(std::abs(a[i]-b[i])>tol*scale)
   throw std::runtime_error(message+" at "+std::to_string(i)+": "+std::to_string(a[i])+" != "+std::to_string(b[i]));}
double contract(const Mat3&a,const Mat3&b)
{double v=0;for(int k=0;k<9;++k)v+=a[k]*b[k];return v;}
gr::GapResult support_gap(gr::Body&bi,gr::Body&bj,Vec3 n,double h)
{
  n=gr::normalized(n);gr::GapResult gap;gap.normal=n;gap.gap=h;gap.converged=true;
  auto support=[&](const gr::Body&body){const auto q=gr::rotatedDiagonal(body.rotation,
    {body.axes[0]*body.axes[0],body.axes[1]*body.axes[1],body.axes[2]*body.axes[2]});
    const auto qn=gr::mul(q,n);return gr::scale(qn,1/std::sqrt(gr::dot(n,qn)));};
  gap.leverI=support(bi);gap.leverJ=gr::scale(support(bj),-1);
  bi.position={};bj.position=gr::add(gr::sub(gap.leverI,gap.leverJ),gr::scale(n,h));return gap;
}
Vec12 rotate_velocity(const Vec12&v,const Mat3&r)
{Vec12 out{};for(int k=0;k<4;++k){const auto x=gr::mul(r,{v[3*k],v[3*k+1],v[3*k+2]});
 for(int j=0;j<3;++j)out[3*k+j]=x[j];}return out;}

void spheres_and_limit()
{
  const Vec3 n=gr::normalized({.3,.5,.7});const Mat3 e{.2,.3,.1,.3,-.4,.2,.1,.2,.2};
  const Vec12 q{.4,-.2,.1,.2,.8,-.4,.3,.7,-.1,.5,.1,.4},bg{};
  for(double ratio:{.25,1.,1.4,4.}){
    gr::Body a,b;a.axes={1,1,1};b.axes={ratio,ratio,ratio};
    auto gap=support_gap(a,b,n,.003);
    const auto p=sh::make_pair(a,b,gap,.7,.0003,.05);
    const auto sphere=ph::make_sphere_pair(1,ratio,n,.003,.7,.0003,.05);
    const auto op=sh::assemble(p,bg,{},e,1.7),expected=ph::assemble(sphere,bg,{},e,1.7);
    require(p.exact_sphere,"Exact spheres dispatch");
    vector_close(op.resistance,expected.resistance,0,"Exact Mari resistance");
    vector_close(op.affine,expected.affine,0,"Exact Mari affine");
    vector_close(sh::evaluate(p,q,bg,{},e,1.7).stress,ph::evaluate(sphere,q,bg,{},e,1.7).stress,0,"Exact Mari stress");
    // Do NOT rely on dispatch alone: approach the sphere through tilted oblate
    // geometries to test the nonspherical coefficients, arms and strain drive.
    a.axes={1+1e-7,1+1e-7,1};b.axes={ratio*(1+2e-7),ratio*(1+2e-7),ratio};
    a.rotation=gr::rotationIncrement({.2,.5,.1});b.rotation=gr::rotationIncrement({-.4,.1,.7});
    gap=support_gap(a,b,n,.003);const auto near=sh::make_pair(a,b,gap,.7,.0003,.05);
    require(!near.exact_sphere,"Near sphere exercises general kernel");
    const auto nearop=sh::assemble(near,bg,{},e,1.7);
    vector_close(nearop.resistance,expected.resistance,4e-6,"Unequal near-sphere resistance limit");
    vector_close(nearop.affine,expected.affine,4e-6,"Unequal near-sphere strain-drive limit");
    const auto ne=sh::evaluate(near,q,bg,{},e,1.7);
    const auto se=ph::evaluate(sphere,q,bg,{},e,1.7);
    vector_close(ne.stress,se.stress,4e-6,"Unequal near-sphere stress limit");
    close(ne.power_shear,se.power_shear,4e-6,"Near-sphere shear decomposition");
    close(ne.power_pump,se.power_pump,4e-6,"Near-sphere pump decomposition");
  }
}

void curvature_coefficients()
{
  gr::Body a,b;a.axes={2,2,1};b=a;const auto gap=support_gap(a,b,{0,0,1},.002);
  const double pi=std::acos(-1.),eta=.8,eps=.0001,R=4,log=std::log(R/(gap.gap+eps));
  const auto p=sh::make_pair(a,b,gap,eta,eps,.1);
  close(p.log_reference_length,R,1e-14,"Face harmonic curvature radius");
  close(p.normal_resistance,1.5*pi*eta*R*R/(gap.gap+eps),1e-14,"Reynolds normal curvature coefficient");
  close(p.shear[0],pi*eta*R*log,1e-14,"Identical-curvature shear coefficient");
  close(p.pump[21],.6*pi*eta*R*R*R*log,2e-14,"Identical-curvature pump coefficient");
  require(sh::detail::dot6(p.normal_cross,p.normal_cross)==0,"Symmetric face has no cubic normal coupling");
  auto inner=gap;inner.gap=-.1;const auto overlap=sh::make_pair(a,b,inner,eta,eps,.1);
  inner.gap=0;const auto zero=sh::make_pair(a,b,inner,eta,eps,.1);
  vector_close(overlap.resistance,zero.resistance,0,"Finite overlap regularization");
  inner.gap=.1;require(!sh::make_pair(a,b,inner,eta,eps,.1).active,"Exact physical outer cutoff");
  // No scalar effective-radius replacement can pass this anisotropic rim test.
  auto rim=support_gap(a,b,{1,0,0},.002);const auto anisotropic=sh::make_pair(a,b,rim,eta,eps,100);
  require(std::abs(anisotropic.pump[21]-anisotropic.pump[28])>.1*anisotropic.pump[21],
      "Anisotropic rolling/pump resistance retained");
  close(anisotropic.minimum_curvature_radius,.5,1e-14,"Minimum rim radius");
  close(anisotropic.cutoff,.5*anisotropic.log_reference_length,1e-14,"Local curvature cutoff");
  // An independent simplification of A3/A7 for identically oriented surfaces:
  // their summed cubic gap coefficient vanishes, but the individual cubic
  // surfaces still give b_N,t = -2*N*support*P*Q*n. A parabolic-only kernel
  // would incorrectly set this leading logarithmic coupling to zero.
  const Vec3 n=gr::normalized({1,0,2});auto tilted=support_gap(a,b,n,.002);
  const auto cubic=sh::make_pair(a,b,tilted,eta,eps,.1);
  const Mat3 Q=gr::rotatedDiagonal(a.rotation,{.25,.25,1});
  const auto curvature=gr::nfCurvature(a,n);
  const double k1=sh::detail::contract(cubic.tangent1,curvature,cubic.tangent1);
  const double k2=sh::detail::contract(cubic.tangent2,curvature,cubic.tangent2);
  const double support=std::sqrt(4*n[0]*n[0]+4*n[1]*n[1]+n[2]*n[2]);
  const double N=3*pi*eta*std::log(cubic.log_reference_length/(.002+eps))/(2*std::sqrt(k1*k2)*(k1+k2));
  const Vec3 expected=gr::scale(ph::tangent(gr::mul(Q,n),n),-2*N*support);
  const Vec3 actual=sh::detail::world(cubic,cubic.normal_cross);
  require(gr::norm(expected)>1.,"Tilted cubic coupling is appreciable in unit fixture");
  vector_close(actual,expected,2e-13,"Independent identical-surface cubic logarithm");
}

void covariance_and_quadratic()
{
  std::mt19937_64 rng(1729);std::uniform_real_distribution<double> u(-1,1);
  auto random3=[&](){return Vec3{u(rng),u(rng),u(rng)};};
  for(int trial=0;trial<100;++trial){
    gr::Body a,b;a.axes={1.4,1.4,.8};b.axes={1.7,1.7,.6};
    a.rotation=gr::rotationIncrement(random3());b.rotation=gr::rotationIncrement(random3());
    auto gap=support_gap(a,b,random3(),.002);
    const auto p=sh::make_pair(a,b,gap,.7,.0001,.05);
    Vec12 q{},bg{};for(int k=0;k<12;++k){q[k]=u(rng);bg[k]=u(rng);}
    Mat3 strain{};for(int k=0;k<9;++k)strain[k]=u(rng);const Vec3 image=random3();const double length=1.9;
    const auto out=sh::evaluate(p,q,bg,image,strain,length);
    const auto op=sh::assemble(p,bg,image,strain,length);
    require(out.power_squeeze>=-1e-12&&out.power_shear>=-1e-12&&out.power_pump>=-1e-12,
      "Nonnegative modal dissipation");
    long double energy=op.energy_constant;double identity=contract(out.stress,strain);
    for(int i=0;i<12;++i){double gradient=op.affine[i];energy+=static_cast<long double>(q[i])*op.affine[i];
      for(int j=0;j<12;++j){gradient+=op.resistance[12*i+j]*q[j];
        energy+=.5L*q[i]*op.resistance[12*i+j]*q[j];
        close(op.resistance[12*i+j],op.resistance[12*j+i],2e-12,"Reciprocal resistance",1.);}
      close(gradient,out.gradient[i],3e-11,"Matrix and evaluated gradient",1.);
      identity+=(q[i]-bg[i])*out.gradient[i];}
    for(int k=0;k<3;++k)identity+=image[k]*out.gradient[6+k];
    close(2*static_cast<double>(energy),out.power,4e-12,"Full affine quadratic");
    close(identity,out.power,4e-12,"Power identity including imposed strain/image");
    Mat3 direction{};for(auto&x:direction)x=u(rng);auto ep=strain,em=strain;const double delta=1e-5;
    for(int k=0;k<9;++k){ep[k]+=delta*direction[k];em[k]-=delta*direction[k];}
    const double derivative=(sh::evaluate(p,q,bg,image,ep,length).power-
      sh::evaluate(p,q,bg,image,em,length).power)/(4*delta);
    close(derivative,contract(out.stress,direction),1e-7,"Stress is strain derivative",1.);
    const Mat3 rotation=gr::rotationIncrement(random3());gr::Body ar=a,br=b;
    ar.rotation=gr::multiply(rotation,a.rotation);br.rotation=gr::multiply(rotation,b.rotation);
    auto rotatedgap=support_gap(ar,br,gr::mul(rotation,gap.normal),gap.gap);
    const auto rotated=sh::make_pair(ar,br,rotatedgap,.7,.0001,.05);
    const auto rotatedstrain=gr::multiply(gr::multiply(rotation,strain),gr::transpose(rotation));
    const auto ro=sh::evaluate(rotated,rotate_velocity(q,rotation),rotate_velocity(bg,rotation),
      gr::mul(rotation,image),rotatedstrain,length);
    close(ro.power,out.power,4e-12,"Rigid laboratory rotation covariance");
    vector_close(ro.force_i,gr::mul(rotation,out.force_i),4e-12,"Rotated force covariance");
    vector_close(ro.torque_i,gr::mul(rotation,out.torque_i),4e-12,"Rotated torque covariance");
    vector_close(ro.stress,gr::multiply(gr::multiply(rotation,out.stress),gr::transpose(rotation)),4e-12,"Rotated stress covariance");
    Vec12 qs{},bs{};for(int k=0;k<6;++k){qs[k]=q[6+k];qs[6+k]=q[k];bs[k]=bg[6+k];bs[6+k]=bg[k];}
    auto reversegap=gap;reversegap.normal=gr::scale(gap.normal,-1);reversegap.leverI=gap.leverJ;reversegap.leverJ=gap.leverI;
    const auto reverse=sh::make_pair(b,a,reversegap,.7,.0001,.05);
    const auto swapped=sh::evaluate(reverse,qs,bs,gr::scale(image,-1),strain,length);
    close(swapped.power,out.power,4e-12,"Particle exchange power");
    vector_close(swapped.force_i,out.force_j,4e-12,"Particle exchange force");
    vector_close(swapped.torque_i,out.torque_j,4e-12,"Particle exchange torque");
    vector_close(swapped.stress,out.stress,4e-12,"Particle exchange stress");
  }
}
}
int main()
{
  spheres_and_limit();curvature_coefficients();covariance_and_quadratic();
  std::cout<<"Spheroid lubrication implementation checks passed: exact/near-sphere Mari limit, anisotropic curvature coefficients, finite regularization, reciprocity, modal dissipation, affine force/stress energy identities, rotation covariance and particle exchange.\n";
}
