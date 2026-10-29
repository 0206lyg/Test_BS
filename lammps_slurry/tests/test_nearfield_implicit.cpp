#include "slurry_nearfield_implicit_solver.h"
#include "slurry_nearfield_hydro.h"
#include "slurry_compact_bridge.h"
#include <iostream>
#include <iomanip>

namespace bs=SlurryBridgeSolver;
namespace ns=SlurryNearfieldSolver;
namespace ni=SlurryNearfieldImplicitSolver;
namespace nh=SlurryNearfieldHydro;
namespace cb=SlurryCompactBridge;
namespace gr=slurry::gr_re2::graphite;
using V3=Eigen::Vector3d;using M3=Eigen::Matrix3d;
using V5=Eigen::Matrix<double,5,1>;using M5=Eigen::Matrix<double,5,5>;
using G5=Eigen::Matrix<double,5,12>;
void require(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
gr::Vec3 arr(const V3&v){return {v[0],v[1],v[2]};}
V3 vec(const gr::Vec3&v){return {v[0],v[1],v[2]};}
bs::Options accuracy(){bs::Options o;o.relative_tolerance=1e-11;o.absolute_tolerance=1e-14;return o;}
bs::Problem simple(){
  bs::Problem p;p.count=2;p.dt=.02;p.prescribed=bs::Vector::Zero(22);
  p.external=bs::Vector::Zero(12);p.near_resistance.resize(22,22);
  p.self_mobility.assign(2,bs::Matrix11::Identity());return p;
}

void coulomb_branches_and_jacobian(){
  auto p=simple();bs::Contact c;c.i=0;c.j=1;c.normal=V3::UnitZ();c.gap=-.02;
  c.kn=500.;c.kt=300.;c.kr=100.;c.mu=.2;c.mu_roll=.3;c.roll_length=.7;
  c.history=V3(.2,.1,0.);c.rolling_history=V3(-.15,.08,0.);
  c.Bi.leftCols<3>()=-M3::Identity();c.Bj.leftCols<3>()=M3::Identity();
  p.contacts={c};ni::Workspace workspace;
  const auto result=ni::solve(p,accuracy(),workspace);
  const double fn=-c.kn*c.gap/(1+2*p.dt*c.kn);
  const V3 tangent=c.mu*fn*c.history.normalized();
  const V3 rolling=c.mu_roll*fn*c.roll_length*c.rolling_history.normalized();
  const V3 traction=tangent-fn*c.normal;
  require(result.contacts[0].active&&result.contacts[0].sliding&&result.contacts[0].rolling,
          "finite-friction reference did not exercise both Coulomb caps");
  require(std::abs(result.contacts[0].normal_load-fn)<1e-10&&
          (result.contacts[0].traction-traction).norm()<1e-10&&
          (result.contacts[0].torque-rolling).norm()<1e-10,
          "implicit Coulomb loads differ from independent closed form");
  require((result.velocity.head<3>()-traction).norm()<1e-10&&
          (result.velocity.segment<3>(6)+traction).norm()<1e-10&&
          (result.velocity.segment<3>(3)-rolling).norm()<1e-10&&
          (result.velocity.segment<3>(9)+rolling).norm()<1e-10,
          "implicit contact force or couple sign is incorrect");
  ni::Workspace direct_workspace;direct_workspace.use_krylov=false;
  const auto direct=ni::solve(p,accuracy(),direct_workspace);
  require((direct.velocity-result.velocity).norm()<1e-9&&direct.linear_iterations==0,
          "sparse Krylov changed the independently solved implicit contact root");
  require(workspace.maximum_true_relative_linear_residual<=workspace.linear_relative_target,
          "accepted sparse Newton direction failed its independent true residual check");
  const auto at=ni::detail::evaluate(p,result.velocity,workspace);
  const Eigen::MatrixXd analytic=ni::detail::jacobian(p,at,workspace);
  Eigen::MatrixXd numerical(12,12);const double epsilon=1e-7;
  for(int j=0;j<12;++j){auto plus=result.velocity,minus=result.velocity;plus[j]+=epsilon;minus[j]-=epsilon;
    numerical.col(j)=(ni::detail::evaluate(p,plus,workspace).residual-
                     ni::detail::evaluate(p,minus,workspace).residual)/(2*epsilon);}
  require((analytic-numerical).norm()<2e-7*analytic.norm(),
          "Coulomb generalized Jacobian differs from independent finite differences");
  require((analytic-analytic.transpose()).norm()>1.,
          "normal-load Coulomb coupling was incorrectly symmetrized");
  const auto metric=ni::acceptance_metric(p,result,result.contacts,workspace,accuracy());
  require(metric.residual_norm<=metric.tolerance&&result.mobility_applications==0&&
          result.linear_iterations>0&&workspace.krylov_calls>0,
          "implicit stage bypassed physical residual gate or skipped sparse Krylov");
  auto wrong=result.contacts;wrong[0].traction[0]+=.01;
  require(ni::acceptance_metric(p,result,wrong,workspace,accuracy()).residual_norm>metric.tolerance,
          "endpoint gate accepted a force inconsistent with the full physical moments");
  // Pull the surfaces apart; stale warm spring forces must not survive opening.
  p.external[2]=-20.;p.external[8]=20.;
  const auto opened=ni::solve(p,accuracy(),workspace,&result);
  require(!opened.contacts[0].active&&opened.contacts[0].tangent_history.isZero(0.)&&
          opened.contacts[0].rolling_history.isZero(0.),"opening contact retained implicit spring history");
  ni::Workspace fallback_workspace;fallback_workspace.krylov_limit=1;
  const auto fallback=ni::solve(p,accuracy(),fallback_workspace,&result);
  require(fallback_workspace.krylov_fallbacks>0&&fallback_workspace.newton_numeric_factorizations>0&&
          (fallback.velocity-opened.velocity).norm()<1e-9,
          "bounded sparse Krylov failure did not recover through direct LU");
  std::cout<<"Coulomb sliding/rolling Newton="<<result.newton_iterations
           <<" residual="<<result.relative_residual<<"; opening Newton="<<opened.newton_iterations<<'\n';
}

struct Relaxation {
  bs::Problem p;bs::Contact base;ni::Workspace workspace;
  G5 map;V5 stiffness,initial;V3 t1,t2;double lambda=0.;
  Relaxation(double ar,bool edge){
    p=simple();p.dt=100*2.5e-6;
    const double a=std::cbrt(ar),c=a/ar;
    std::vector<gr::Body>b(2);
    for(auto&body:b){body.axes={a,a,c};body.rotation={1,0,0,0,1,0,0,0,1};}
    b[0].position={0,0,0};b[1].position=edge?gr::Vec3{2*a-2.5e-5,0,0}:gr::Vec3{0,0,2*c-2.5e-5};
    const auto gap=gr::closestEllipsoidGap(b[0],b[1]);
    p.near_resistance=nh::evaluate(b[0],b[1],gap,1.,1.,.001,.5).resistance.sparseView();
    p.self_mobility=ns::build_self_mobility(b,workspace);
    auto&x=base;x.i=0;x.j=1;x.normal=vec(gap.normal);x.gap=gap.gap;
    // eta=.01 Pa s,L=4um,rate=100/s: kn=3,kt=kr=2.4 N/m.
    x.kn=750000.;x.kt=600000.;x.kr=600000.;x.mu=x.mu_roll=10.;x.roll_length=1.;
    const V3 d=vec(b[1].position)-vec(b[0].position);
    const V3 li=.5*(vec(gap.leverI)+d+vec(gap.leverJ)),lj=li-d;
    x.Bi.leftCols<3>()=-M3::Identity();x.Bi.rightCols<3>()=cb::cross_matrix(li);
    x.Bj.leftCols<3>()=M3::Identity();x.Bj.rightCols<3>()=-cb::cross_matrix(lj);
    const auto preload=bs::contact_response(x,bs::Vector::Zero(12),0.);
    p.external.segment<6>(0)=bs::generalized_contact_map(x,true).transpose()*bs::contact_wrench(preload);
    p.external.segment<6>(6)=bs::generalized_contact_map(x,false).transpose()*bs::contact_wrench(preload);
    p.contacts={x};ns::solve(p,accuracy(),workspace);
    t1=x.normal.unitOrthogonal();t2=x.normal.cross(t1);map.setZero();
    for(int k=0;k<3;++k){const V3 axis=k==0?t1:k==1?t2:x.normal;
      map.block<1,6>(k,0)=axis.transpose()*x.Bi;map.block<1,6>(k,6)=axis.transpose()*x.Bj;}
    map.block<1,3>(3,3)=-t1.transpose();map.block<1,3>(3,9)=t1.transpose();
    map.block<1,3>(4,3)=-t2.transpose();map.block<1,3>(4,9)=t2.transpose();
    stiffness<<x.kt,x.kt,x.kn,x.kr,x.kr;
    const M5 rate=stiffness.cwiseSqrt().asDiagonal()*map*workspace.factor.solve(map.transpose())*stiffness.cwiseSqrt().asDiagonal();
    Eigen::SelfAdjointEigenSolver<M5> eigen(.5*(rate+rate.transpose()));lambda=eigen.eigenvalues().maxCoeff();
    initial=stiffness.cwiseSqrt().cwiseInverse().asDiagonal()*eigen.eigenvectors().col(4);
    initial*=1e-8/initial.cwiseAbs().maxCoeff();
  }
  double energy(const V5&h){return .5*(stiffness.array()*h.array().square()).sum();}
  V5 advance(const V5&h,double physical_dt,int*iterations=nullptr){
    p.dt=100*physical_dt;p.contacts[0]=base;
    p.contacts[0].gap=base.gap+h[2];p.contacts[0].history=t1*h[0]+t2*h[1];
    p.contacts[0].rolling_history=t1*h[3]+t2*h[4];
    const auto result=ni::solve(p,accuracy(),workspace);
    if(iterations)*iterations=result.newton_iterations;
    const auto&r=result.contacts[0];
    require(r.active&&!r.sliding&&!r.rolling,"finite relaxation unexpectedly hit a Coulomb cap");
    V5 next;next<<t1.dot(r.tangent_history),t2.dot(r.tangent_history),r.predicted_gap-base.gap,
                   t1.dot(r.rolling_history),t2.dot(r.rolling_history);
    return next;
  }
  double error(int steps){
    const double duration=2./(100*lambda),dt=duration/steps;V5 h=initial;
    for(int i=0;i<steps;++i)h=advance(h,dt);
    return (h-initial*std::exp(-2.)).norm()/initial.norm();
  }
};

void finite_prestress_decay_and_convergence(){
  for(double ar:{1.,2.,5.})for(bool edge:{false,true}){
    Relaxation test(ar,edge);const double z=test.lambda*100*2.5e-6;
    int iterations=0;const V5 next=test.advance(test.initial,2.5e-6,&iterations);
    const double ratio=test.energy(next)/test.energy(test.initial);
    const double expected=1./((1+z)*(1+z));
    require(ratio<1.&&(next-test.initial/(1+z)).norm()<2e-6*test.initial.norm(),
            "finite-prestress implicit update failed exact backward-Euler relaxation");
    const double coarse=test.error(20),fine=test.error(40);
    require(coarse/fine>1.8&&coarse/fine<2.2,"implicit relaxation failed first-order exact-ODE convergence");
    std::cout<<"AR="<<ar<<" placement="<<(edge?"edge":"face")<<" dt_lambda="<<z
      <<" Heun_energy_factor="<<std::pow(1-z+.5*z*z,2)
      <<" BE_energy_factor="<<ratio<<" expected="<<expected<<" Newton="<<iterations
      <<" convergence_ratio="<<coarse/fine<<'\n';
  }
}
int main(){std::cout<<std::setprecision(12);coulomb_branches_and_jacobian();
  finite_prestress_decay_and_convergence();std::cout<<"nearfield implicit Coulomb and finite-prestress checks passed\n";}
