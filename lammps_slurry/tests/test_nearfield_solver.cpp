#include "slurry_nearfield_solver.h"
#include "slurry_hydro_math.h"
#include "slurry_stress_math.h"
#include <iostream>
#include <random>

namespace bs = SlurryBridgeSolver;
namespace ns = SlurryNearfieldSolver;
namespace gr = slurry::gr_re2::graphite;
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;
void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F call, const char *message) {
  bool threw = false;
  try { call(); } catch (const std::exception &) { threw = true; }
  require(threw, message);
}
bs::Problem simple(int n) {
  bs::Problem p; p.count=n; p.dt=.01;
  p.external=bs::Vector::Zero(6*n); p.prescribed=bs::Vector::Zero(11*n);
  p.self_mobility.assign(n, bs::Matrix11::Identity());
  p.near_resistance.resize(11*n, 11*n);
  return p; // No p.mobility: linking/executing an FFT is unnecessary.
}
bs::Options accuracy() {
  bs::Options o; o.relative_tolerance=1e-10; o.absolute_tolerance=1e-12;
  return o;
}

void dense_force_balance_and_cache() {
  auto p=simple(3); const int nm=33, nu=18;
  std::mt19937 random(3917); std::normal_distribution<double> value;
  for (auto &m:p.self_mobility) {
    bs::Matrix11 b;
    for(int i=0;i<b.size();++i) b.data()[i]=value(random);
    m=bs::Matrix11::Identity()+.03*b*b.transpose();
  }
  Eigen::MatrixXd b(nm,nm);
  for(int i=0;i<b.size();++i) b.data()[i]=value(random);
  p.near_resistance=(.02*b*b.transpose()).sparseView();
  for(int i=0;i<3;++i) for(int k=6;k<11;++k) p.prescribed[11*i+k]=value(random);
  for(int k=0;k<nu;++k) p.external[k]=.1*value(random);
  bs::Contact c;c.i=0;c.j=1;c.normal=V3::UnitZ();c.gap=-.04;
  c.kn=100.;c.kt=50.;c.mu=.4;c.kr=30.;c.mu_roll=.2;c.roll_length=.7;
  c.history=V3(.1,.06,0.);c.rolling_history=V3(.08,.05,0.);
  c.Bi.leftCols<3>()=-M3::Identity();c.Bj.leftCols<3>()=M3::Identity();
  c.Bi(0,4)=.8;c.Bj(1,3)=-.4;c.affine=V3(2.,-.3,-4.);
  p.contacts.push_back(c);
  auto inactive=c;inactive.i=1;inactive.j=2;inactive.gap=.01;inactive.affine[2]=-1e6;
  p.contacts.push_back(inactive);

  // Assemble loads independently of contact_response, including saturated
  // tangent/rolling caps, lever torques, and a closing but separated pair.
  const double fn=-c.kn*c.gap;
  const V3 traction=c.mu*fn*c.history.normalized()-fn*c.normal;
  const V3 torque=c.mu_roll*fn*c.roll_length*c.rolling_history.normalized();
  bs::Vector force=p.external;
  force.segment<6>(0)-=c.Bi.transpose()*traction;
  force.segment<6>(6)-=c.Bj.transpose()*traction;
  force.segment<3>(3)+=torque;force.segment<3>(9)-=torque;
  Eigen::MatrixXd R=Eigen::MatrixXd(p.near_resistance);
  for(int i=0;i<3;++i)R.block<11,11>(11*i,11*i)+=p.self_mobility[i].inverse();
  Eigen::MatrixXd P=Eigen::MatrixXd::Zero(nm,nu);
  for(int i=0;i<3;++i)P.block<6,6>(11*i,6*i).setIdentity();
  const Eigen::MatrixXd A=P.transpose()*R*P;
  const bs::Vector reference=A.ldlt().solve(force-P.transpose()*R*p.prescribed);
  ns::Workspace workspace;auto options=accuracy();double reported_tolerance=-1.;
  options.trace_callback=[&](const bs::IterationTrace &trace){
    require(std::string(trace.phase)=="nearfield_direct_accepted", "incorrect direct trace phase");
    reported_tolerance=trace.tolerance;
  };
  const auto result=ns::solve(p,options,workspace);
  require((result.velocity-reference).norm()<1e-11, "6N solution differs from dense full11 resistance");
  require((result.total_moment-R*(P*reference+p.prescribed)).norm()<1e-10,
          "reconstructed full moments differ from dense resistance");
  require((P.transpose()*result.total_moment-force).norm()<1e-11,
          "loaded contact force/torque balance failed");
  require((result.contacts[0].traction-traction).norm()<1e-13 &&
          (result.contacts[0].torque-torque).norm()<1e-13 && !result.contacts[1].active,
          "known contact loads were changed by the direct solve");
  require(result.linear_iterations==0 && result.newton_iterations==0 &&
          result.mobility_applications==0 && result.continuation_stages==0,
          "direct solve reported nonexistent iterations");
  require(result.hydro_power>0. && std::abs(result.hydro_power-result.q.dot(R*result.q))<1e-10,
          "hydrodynamic power lost a self or near contribution");
  const V3 relative=c.Bi*reference.segment<6>(0)+c.Bj*reference.segment<6>(6)+c.affine;
  const V3 relative_rotation=reference.segment<3>(9)-reference.segment<3>(3);
  require(std::abs(result.contact_power-traction.dot(relative)-torque.dot(relative_rotation))<1e-11,
          "contact power uses the wrong velocity or couple sign");
  // Independently compare the residual normalization to the old mixed metric.
  auto hydro=p;hydro.contacts.clear();
  bs::detail::Preconditioner old_monitor(hydro,bs::Vector::Zero(nu));
  bs::Vector old_rhs(nm+nu);
  old_rhs.head(nm)=-p.prescribed;
  old_rhs.tail(nu)=force-P.transpose()*p.near_resistance*p.prescribed;
  const double expected_tolerance=options.absolute_tolerance+options.relative_tolerance*old_monitor.left(old_rhs).norm();
  require(std::abs(reported_tolerance-expected_tolerance)<1e-12*expected_tolerance,
          "nearfield residual tolerance changed the mixed physical norm");

  require(workspace.symbolic_analyses==1 && workspace.numeric_factorizations==1 && workspace.solves==1,
          "initial factor counters incorrect");
  p.external[0]+=.2;
  const auto changed_load=ns::solve(p,options,workspace,&result);
  require(workspace.symbolic_analyses==1 && workspace.numeric_factorizations==1 &&
          workspace.numeric_reuses==1 && (changed_load.velocity-result.velocity).norm()>1e-3,
          "unchanged matrix did not reuse factors with a changed load");
  p.near_resistance*=1.01;
  ns::solve(p,options,workspace);
  require(workspace.symbolic_analyses==1 && workspace.numeric_factorizations==2,
          "changed matrix values failed symbolic-only reuse");
  p.near_resistance.coeffRef(0,11)=0.;p.near_resistance.coeffRef(11,0)=0.;
  p.near_resistance.prune(0.);
  ns::solve(p,options,workspace);
  require(workspace.symbolic_analyses==2 && workspace.numeric_factorizations==3,
          "changed pattern reused an obsolete symbolic factor");
  p.dt=1.7;
  const auto long_dt=ns::solve(p,options,workspace);
  p.dt=.001;
  const auto short_dt=ns::solve(p,options,workspace);
  require((long_dt.velocity-short_dt.velocity).norm()==0., "dt entered known-state contact force balance");
}

void isolated_spheroids() {
  const M3 Q=Eigen::AngleAxisd(.71,V3(.3,-.7,.2).normalized()).toRotationMatrix();
  M3 E;E<<.2,.17,-.31,.17,-.13,.09,-.31,.09,-.07;
  const V3 omega_fluid(.1,-.2,.3);
  for(double ar:{1.,2.,5.}) {
    const double a=std::cbrt(ar),c=a/ar;
    gr::Body body;body.axes={a,a,c};
    SlurryStress::Mat3 rotation{},strain{};
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
      body.rotation[3*i+j]=rotation[3*i+j]=Q(i,j);strain[3*i+j]=E(i,j);
    }
    ns::Workspace workspace;auto p=simple(1);
    p.self_mobility=ns::build_self_mobility({body},workspace);
    const auto again=ns::build_self_mobility({body},workspace);
    require(workspace.self_quadrature_evaluations==1 && (again[0]-p.self_mobility[0]).norm()==0.,
            "identical axes did not reuse body-frame self quadrature");
    for(int k=0;k<5;++k)p.prescribed[6+k]=-(E.array()*SlurryPeriodic::stf_basis()[k].array()).sum();
    const auto resistance=SlurryHydro::unit_resistance(body.axes);
    const auto stress=SlurryStress::unit_stresslet(body.axes);
    const V3 axis=Q.col(2);const double lambda=(c*c-a*a)/(c*c+a*a);
    const V3 jeffery_slip=lambda*axis.cross(E*axis);
    M3 translation=M3::Zero(),rotation_drag=M3::Zero();
    for(int k=0;k<3;++k){translation(k,k)=resistance.translation[k];rotation_drag(k,k)=resistance.rotation[k];}
    translation=Q*translation*Q.transpose();rotation_drag=Q*rotation_drag*Q.transpose();
    for(bool loaded:{false,true}) {
      p.external.setZero();
      if(loaded)p.external<<.31,-.11,.07,.12,.22,-.09;
      const V3 expected_velocity=translation.ldlt().solve(p.external.head<3>());
      const V3 expected_omega=omega_fluid+jeffery_slip+rotation_drag.ldlt().solve(p.external.tail<3>());
      const auto result=ns::solve(p,accuracy(),workspace);
      require((result.velocity.head<3>()-expected_velocity).norm()<2e-10,
              "isolated anisotropic translational Stokes drag mismatch");
      require((result.velocity.tail<3>()+omega_fluid-expected_omega).norm()<2e-10,
              "isolated Jeffery rotation or rotational drag mismatch");
      const auto exact=SlurryStress::stresslet(stress,rotation,1.,strain,
          {omega_fluid[0],omega_fluid[1],omega_fluid[2]},
          {expected_omega[0],expected_omega[1],expected_omega[2]});
      M3 actual=M3::Zero(),expected,free_stress;
      for(int k=0;k<5;++k)actual-=result.total_moment[6+k]*SlurryPeriodic::stf_basis()[k];
      for(int i=0;i<3;++i)for(int j=0;j<3;++j){expected(i,j)=exact.total[3*i+j];free_stress(i,j)=exact.free[3*i+j];}
      require((actual-expected).norm()<2e-9*expected.norm(),
              "isolated stresslet lost strain/rotation coupling or has wrong sign");
      const double independent_power=expected_velocity.dot(p.external.head<3>())+
          (expected_omega-omega_fluid-jeffery_slip).dot(p.external.tail<3>())+
          (E.array()*free_stress.array()).sum();
      require(std::abs(result.hydro_power-independent_power)<2e-9*independent_power,
              "isolated grand resistance has incorrect dissipation");
    }
    const auto twice_eta=ns::build_self_mobility({body},workspace,2.);
    require(workspace.self_quadrature_evaluations==2 &&
            (2.*twice_eta[0]-p.self_mobility[0]).norm()<1e-13,
            "self cache ignored viscosity changes");
  }
}

void invalid_inputs_and_recovery() {
  auto p=simple(1);p.external[0]=1.;ns::Workspace workspace;
  p.self_mobility[0](0,0)=-1.;
  rejects([&]{ns::solve(p,accuracy(),workspace);}, "indefinite self accepted");
  p.self_mobility[0].setIdentity();
  p.near_resistance=bs::Matrix11(-2.*bs::Matrix11::Identity()).sparseView();
  rejects([&]{ns::solve(p,accuracy(),workspace);}, "indefinite 6N resistance accepted");
  p.near_resistance.setZero();
  const auto recovered=ns::solve(p,accuracy(),workspace);
  require(std::abs(recovered.velocity[0]-1.)<1e-14, "workspace failed after rejected factorization");
  p.near_resistance.coeffRef(0,1)=.1;
  rejects([&]{ns::solve(p,accuracy(),workspace);}, "nonsymmetric near resistance accepted");
}

void unequal_isolated_spheres() {
  // A bidisperse dilute pair must retain each body's own translation and
  // rotation resistance; using one nominal radius would fail both checks.
  gr::Body a,b;a.axes={1.,1.,1.};b.axes={1.4,1.4,1.4};
  ns::Workspace workspace;auto p=simple(2);
  p.self_mobility=ns::build_self_mobility({a,b},workspace);
  require(workspace.self_quadrature_evaluations==2,"Different sizes shared a self cache entry");
  const auto reordered=ns::build_self_mobility({b,a},workspace);
  require(workspace.self_quadrature_evaluations==2&&
          (reordered[0]-p.self_mobility[1]).norm()==0.&&
          (reordered[1]-p.self_mobility[0]).norm()==0.,"Bidisperse self cache ordering");
  p.external[0]=p.external[6]=1.;p.external[3]=p.external[9]=1.;
  const auto result=ns::solve(p,accuracy(),workspace);
  const double pi=std::acos(-1.);
  for(int i=0;i<2;++i) {
    const double radius=i?1.4:1.;
    require(std::abs(result.velocity[6*i]-1./(6*pi*radius))<2e-12,
            "Unequal sphere translation drag");
    require(std::abs(result.velocity[6*i+3]-1./(8*pi*radius*radius*radius))<2e-12,
            "Unequal sphere rotation drag");
  }
}

int main() {
  dense_force_balance_and_cache();
  isolated_spheroids();
  unequal_isolated_spheres();
  invalid_inputs_and_recovery();
  std::cout<<"nearfield sparse/dense balance, cache invalidation, isolated Jeffery/stress and failure recovery passed\n";
}
