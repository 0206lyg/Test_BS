#include "slurry_pair_hydro_math.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace SlurryPairHydro;
namespace {
void require(bool ok,const std::string &message)
{ if (!ok) throw std::runtime_error(message); }
void close(double actual,double expected,double tolerance,const std::string &message)
{
  const double scale=std::max({std::abs(actual),std::abs(expected),1.e-24});
  require(std::abs(actual-expected)<=tolerance*scale,message+": "+std::to_string(actual)+" != "+std::to_string(expected));
}
double contraction(const Mat3 &a,const Mat3 &b)
{ double out=0; for(int k=0;k<9;++k) out+=a[k]*b[k]; return out; }
Vec12 solve(Mat12 a,Vec12 b)
{
  for(int k=0;k<12;++k) {
    int pivot=k;
    for(int r=k+1;r<12;++r) if(std::abs(a[12*r+k])>std::abs(a[12*pivot+k])) pivot=r;
    require(std::abs(a[12*pivot+k])>1.e-16,"Singular test matrix");
    for(int c=0;c<12;++c) std::swap(a[12*k+c],a[12*pivot+c]);
    std::swap(b[k],b[pivot]);
    for(int r=k+1;r<12;++r) {
      const double ratio=a[12*r+k]/a[12*k+k];
      for(int c=k;c<12;++c) a[12*r+c]-=ratio*a[12*k+c];
      b[r]-=ratio*b[k];
    }
  }
  Vec12 x{};
  for(int r=11;r>=0;--r) {
    double rhs=b[r]; for(int c=r+1;c<12;++c) rhs-=a[12*r+c]*x[c];
    x[r]=rhs/a[12*r+r];
  }
  return x;
}

// Independent scalar coefficients from Mari Eq. (22), Eq. (23) and the
// LF_DEM leading M coefficients. This checks numerical prefactors, unequal
// radius asymmetry and the force/torque/strain contraction conventions.
void paper_coefficients()
{
  const double pi=std::acos(-1.0),a=1.3,eta=.7,h=.017,eps=.001;
  for(double lambda:{.1,.5,1.,1.4,2.,10.}) {
    const double b=lambda*a,s=1+lambda,log=std::log((a+b)/(2*(h+eps)));
    const double pref=6*pi*eta,invxi=(a+b)/(2*(h+eps));
    const auto p=make_sphere_pair(a,b,{1,0,0},h,eta,eps,.5*(a+b));
    const auto op=assemble(p,{}, {}, {},1);
    const double XA=pref*2*a*lambda*lambda/(s*s*s)*invxi;
    const double YA=pref*(4*a/15)*lambda*(2+lambda+2*lambda*lambda)/(s*s*s)*log;
    const double Bi=pref*(-2*a*a/15)*lambda*(4+lambda)/(s*s)*log;
    const double Bj=pref*(-2*a*a/15)*lambda*lambda*(1+4*lambda)/(s*s)*log;
    const double Cii=pref*(8*a*a*a/15)*lambda/s*log;
    const double Cij=pref*(2*a*a*a/15)*lambda*lambda/s*log;
    const double Cjj=lambda*lambda*Cii;
    close(op.resistance[0],XA,3.e-14,"Mari XA_ii");
    close(op.resistance[6],-XA,3.e-14,"Mari XA_ij");
    close(op.resistance[12+1],YA,3.e-14,"Mari YA_ii");
    close(op.resistance[12+7],-YA,3.e-14,"Mari YA_ij");
    close(op.resistance[12+5],-Bi,3.e-14,"Mari YB_ii");
    close(op.resistance[12+11],-Bj,3.e-14,"Mari YB_ji");
    close(op.resistance[12*5+5],Cii,3.e-14,"Mari YC_ii");
    close(op.resistance[12*5+11],Cij,3.e-14,"Mari YC_ij");
    close(op.resistance[12*11+11],Cjj,3.e-14,"Mari YC_jj");
    close(p.shear_lever_i+p.shear_lever_j,a+b,2.e-15,"Contact-limit lever sum");
    const Mat3 strain{.2,.4,.1,.4,-.3,.2,.1,.2,.1};
    const auto e=evaluate(p,{}, {}, {},strain,1);
    const double XGi=pref*2*a*a*lambda*lambda/(s*s*s)*invxi;
    const double XGji=lambda*XGi;
    const double YGi=pref*a*a/15*lambda*(4-lambda+7*lambda*lambda)/(s*s*s)*log;
    const double YGji=pref*a*a/15*lambda*lambda*(4*lambda*lambda-lambda+7)/(s*s*s)*log;
    const double Hi=pref*2*a*a*a/15*lambda*(2-lambda)/(s*s)*log;
    const double Hji=pref*a*a*a/15*lambda*lambda*(lambda+7)/(s*s)*log;
    close(e.force_i[0],(XGi+XGji)*strain[0],3.e-14,"Mari XG uniform strain");
    close(e.force_i[1],2*(YGi+YGji)*strain[3],3.e-14,"Mari YG uniform strain");
    close(e.torque_i[2],2*(Hi+Hji)*strain[3],3.e-14,"Mari YH uniform strain (LF_DEM contraction)");
    auto xm=[](double x,double l){return (4.0/3)*x*x*x*l*l/std::pow(1+l,3);};
    auto ym=[](double x,double l){return (4.0/15)*x*x*x*l*(1-l+4*l*l)/std::pow(1+l,3);};
    const double XMsum=pref*invxi*(xm(a,lambda)+xm(b,1/lambda)+2*(4.0/3)*a*a*a*std::pow(lambda,3)/std::pow(s,3));
    const double YMsum=pref*log*(ym(a,lambda)+ym(b,1/lambda)+2*a*a*a/15*lambda*lambda*(7-10*lambda+7*lambda*lambda)/std::pow(s,3));
    const double expected_power=1.5*XMsum*strain[0]*strain[0]+2*YMsum*(strain[3]*strain[3]+strain[6]*strain[6]);
    close(contraction(e.stress,strain),expected_power,3.e-14,"Mari M uniform-strain stress");
  }
}

void regularization_and_modes()
{
  const double a=300.e-9,eta=.01,eps=.001*a,cut=.5*a,pi=std::acos(-1.0);
  const auto p=make_sphere_pair(a,a,{1,0,0},3.e-9,eta,eps,cut);
  close(p.squeeze_resistance,1.5*pi*eta*a*a/(3.e-9+eps),2.e-15,"Equal-sphere squeeze coefficient");
  close(p.shear_resistance,pi*eta*a*std::log(a/(3.e-9+eps)),2.e-15,"Equal-sphere shear coefficient");
  close(p.pump_resistance,.6*pi*eta*a*a*a*std::log(a/(3.e-9+eps)),2.e-15,"Equal-sphere pump coefficient");
  const auto inner=make_sphere_pair(a,a,{1,0,0},1.e-9,eta,eps,cut);
  require(inner.squeeze_resistance>p.squeeze_resistance,"Lubrication must increase below mechanical 3 nm onset");
  const auto zero=make_sphere_pair(a,a,{1,0,0},0,eta,eps,cut);
  const auto overlap=make_sphere_pair(a,a,{1,0,0},-.1*a,eta,eps,cut);
  close(zero.squeeze_resistance,overlap.squeeze_resistance,0,"Finite overlap regularization");
  const auto inactive=make_sphere_pair(a,a,{1,0,0},cut,eta,eps,cut);
  require(!inactive.active && inactive.squeeze_resistance==0,"Exact outer cutoff");
  Vec12 counter{}; counter[5]=a; counter[11]=-a;
  const auto pump=evaluate(p,counter,{}, {}, {},a);
  close(pump.power_pump,4*p.pump_resistance,2.e-15,"Counterrotation pump power");
  require(pump.power_squeeze==0 && pump.power_shear==0,"Counterrotation is pure pump");
  for(double x:pump.stress) require(x==0,"Pump has no direct uniform-strain stress");
  Vec12 twist{};twist[3]=a;twist[9]=-a;
  require(evaluate(p,twist,{}, {}, {},a).power==0,"Nonsingular twist mode is excluded");
}

void quadratic_consistency()
{
  std::mt19937_64 rng(1729); std::uniform_real_distribution<double> random(-1,1);
  for(int sample=0;sample<120;++sample) {
    Vec3 n{random(rng),random(rng),random(rng)};
    const double a=.8+.3*random(rng),b=1.4*a,L=1.7;
    const auto p=make_sphere_pair(a,b,n,.01,.4,.001,.5);
    Vec12 q{},bg{},direction{};
    for(int k=0;k<12;++k) {q[k]=random(rng);bg[k]=random(rng);direction[k]=random(rng);}
    Vec3 image{random(rng),random(rng),random(rng)};
    Mat3 strain{}; for(int r=0;r<3;++r) for(int c=r;c<3;++c) strain[3*r+c]=strain[3*c+r]=random(rng);
    const auto op=assemble(p,bg,image,strain,L);
    const auto out=evaluate(p,q,bg,image,strain,L);
    require(out.power>=0 && out.power_squeeze>=0 && out.power_shear>=0 && out.power_pump>=0,"Nonnegative modal dissipation");
    double identity=contraction(out.stress,strain),quadratic=0,scale=0;
    long double energy=op.energy_constant;
    for(int r=0;r<12;++r) {
      energy+=static_cast<long double>(q[r])*op.affine[r];
      for(int c=0;c<12;++c) energy+=.5L*q[r]*op.resistance[12*r+c]*q[c];
    }
    close(2*static_cast<double>(energy),out.power,3.e-12,"Complete affine quadratic equals modal dissipation");
    for(int r=0;r<12;++r) {
      double gradient=op.affine[r];
      for(int c=0;c<12;++c) {
        gradient+=op.resistance[12*r+c]*q[c];
        close(op.resistance[12*r+c],op.resistance[12*c+r],1.e-14,"Reciprocal R_UU");
        quadratic+=direction[r]*op.resistance[12*r+c]*direction[c];
      }
      scale=std::max(scale,std::abs(gradient));
      require(std::abs(gradient-out.gradient[r])<2.e-12*std::max(scale,1.),"Matrix and modal force agree");
      identity+=(q[r]-bg[r])*out.gradient[r];
    }
    for(int k=0;k<3;++k) identity+=image[k]*out.gradient[6+k];
    require(quadratic>=-1.e-12,"Positive semidefinite resistance");
    close(out.power,identity,8.e-13,"Power identity including image and strain");
    Vec12 plus=q,minus=q;
    const double delta=1.e-5;
    double derivative=0;
    for(int k=0;k<12;++k) {plus[k]+=delta*direction[k];minus[k]-=delta*direction[k];derivative+=out.gradient[k]*direction[k];}
    const double numerical=(evaluate(p,plus,bg,image,strain,L).power-evaluate(p,minus,bg,image,strain,L).power)/(4*delta);
    require(std::abs(numerical-derivative)<1.e-7*std::max(std::abs(derivative),1.),"Force is quadratic derivative");
    Mat3 esplus=strain,esminus=strain,dstrain{};
    for(int r=0;r<3;++r) for(int c=r;c<3;++c) dstrain[3*r+c]=dstrain[3*c+r]=random(rng);
    for(int k=0;k<9;++k) {esplus[k]+=delta*dstrain[k];esminus[k]-=delta*dstrain[k];}
    const double dstress=(evaluate(p,q,bg,image,esplus,L).power-evaluate(p,q,bg,image,esminus,L).power)/(4*delta);
    const double stress_work=contraction(out.stress,dstrain);
    require(std::abs(dstress-stress_work)<1.e-7*std::max(std::abs(stress_work),1.),"Stress is quadratic strain derivative");
    Vec12 shifted=q,shifted_bg=bg;
    for(int k=0;k<12;++k) {const double change=random(rng);shifted[k]+=change;shifted_bg[k]+=change;}
    close(evaluate(p,shifted,shifted_bg,image,strain,L).power,out.power,2.e-14,"Coherent superposed ambient motion");
    Vec12 reverse{},reverse_bg{};
    for(int k=0;k<6;++k){reverse[k]=q[6+k];reverse[6+k]=q[k];reverse_bg[k]=bg[6+k];reverse_bg[6+k]=bg[k];}
    const auto swapped=make_sphere_pair(b,a,{-n[0],-n[1],-n[2]},.01,.4,.001,.5);
    const auto other=evaluate(swapped,reverse,reverse_bg,{-image[0],-image[1],-image[2]},strain,L);
    close(other.power,out.power,2.e-14,"Particle label exchange");
    for(int k=0;k<9;++k) require(std::abs(other.stress[k]-out.stress[k])<2.e-12*std::max(out.power,1.),"Stress invariant under particle exchange");
  }
}

void force_free_pair()
{
  const double pi=std::acos(-1.),a=1,eta=1,h=.01,L=1;
  for(const Vec3 n: {Vec3{1,0,0},Vec3{0,1,0},Vec3{.6,.8,0},Vec3{.3,.4,std::sqrt(.75)}}) {
    const auto p=make_sphere_pair(a,a,n,h,eta,.001,.5);
    const Mat3 strain{0,.5,0,.5,0,0,0,0,0};
    const auto op=assemble(p,{}, {},strain,L);
    Mat12 total=op.resistance; Vec12 rhs{}; std::array<double,12> self{};
    for(int k=0;k<12;++k) {self[k]=(k%6<3 ? 6*pi*eta*a : 8*pi*eta*a*a*a/(L*L));total[12*k+k]+=self[k];rhs[k]=-op.affine[k];}
    const Vec12 q=solve(total,rhs);
    const auto out=evaluate(p,q,{}, {},strain,L);
    double drag=0;for(int k=0;k<12;++k) drag+=self[k]*q[k]*q[k];
    close(drag+out.power,contraction(out.stress,strain),3.e-13,"Force-free total power and stress");
    Vec3 en{};for(int r=0;r<3;++r) for(int c=0;c<3;++c) en[r]+=strain[3*r+c]*n[c];
    const double enn=dot(en,n),et2=dot(en,en)-enn*enn;
    const double normal=p.squeeze_resistance/(1+p.squeeze_resistance/(3*pi*eta*a));
    const double shear=p.shear_resistance/(1+p.shear_resistance/(3*pi*eta*a)+p.shear_resistance*a*a/(4*pi*eta*a*a*a));
    const double analytic=4*a*a*(normal*enn*enn+shear*et2);
    close(drag+out.power,analytic,3.e-13,"Equal-sphere force-free analytic power");
    require(out.power_pump<1.e-24,"Identical sphere symmetry leaves pump unexcited");
  }
}
}

int main()
{
  paper_coefficients(); regularization_and_modes(); quadratic_consistency(); force_free_pair();
  std::cout<<"Mari sphere lubrication checks passed: independent unequal-sphere A/B/C/G/H/M coefficients, regularization, pump/twist, reciprocity, PSD, strain-stress derivative, periodic-image power and force-free pair.\n";
}
