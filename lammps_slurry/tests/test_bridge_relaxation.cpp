#include "slurry_bridge_contact_solver.h"
#include "slurry_bridge_linear_solver.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace bs = SlurryBridgeSolver;
namespace cs = SlurryBridgeContactSolver;
namespace ls = SlurryBridgeLinearSolver;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

// Known linear contact ODE with independent, finite sliding and rolling
// prestretch. Normal load is externally balanced. The two relaxation rates
// match representative AR1 physical rates; they do not depend on mu once
// the Coulomb bounds are inactive. Time is scaled by rate=100 s^-1.
constexpr double rate = 100.;
constexpr double lambda_t = 61433229.9113121;
constexpr double lambda_r = 78244676.7332778;
constexpr double stiffness = 2.e7;
bs::Problem fixture(double dt_seconds) {
  bs::Problem p;
  p.count=2; p.dt=rate*dt_seconds;
  p.prescribed=bs::Vector::Zero(22);
  p.external=bs::Vector::Zero(12);
  p.near_resistance.resize(22,22);
  bs::Matrix11 mobility=bs::Matrix11::Identity();
  mobility.topLeftCorner<3,3>() *= lambda_t/(2.*rate*stiffness);
  mobility.block<3,3>(3,3) *= lambda_r/(2.*rate*stiffness);
  p.self_mobility.assign(2,mobility);
  p.mobility=[mobility](const bs::Vector &force) -> bs::Vector {
    bs::Vector velocity(22);
    velocity.head<11>()=mobility*force.head<11>();
    velocity.tail<11>()=mobility*force.tail<11>();
    return velocity;
  };
  bs::Contact c;
  c.i=0; c.j=1; c.normal=Vec3::UnitZ(); c.gap=-.01;
  c.kn=2.5e7; c.kt=stiffness; c.kr=stiffness;
  c.mu=c.mu_roll=1.e6; c.roll_length=1.;
  c.Bi.leftCols<3>()=-Mat3::Identity();
  c.Bj.leftCols<3>()=Mat3::Identity();
  c.history=Vec3(1.e-5,-2.e-5,0.);
  c.rolling_history=Vec3(-3.e-5,1.e-5,0.);
  p.external[2]=-c.kn*c.gap;
  p.external[8]=c.kn*c.gap;
  p.contacts.push_back(c);
  return p;
}
bs::Options accuracy() {
  bs::Options o;
  o.relative_tolerance=1.e-12;
  o.absolute_tolerance=1.e-12;
  o.linear_relative_tolerance=1.e-12;
  o.maximum_linear_iterations=1000;
  return o;
}

void large_step_and_infinite_friction_limit() {
  const double physical_dt=1.e-6;
  auto p=fixture(physical_dt);
  const auto initial=p.contacts[0];
  const double tangent_factor=1./(1.+lambda_t*physical_dt);
  const double rolling_factor=1./(1.+lambda_r*physical_dt);
  const auto implicit=cs::solve(p,accuracy());
  const auto &c=implicit.contacts.at(0);
  require(c.active && !c.sliding && !c.rolling,
          "high-friction relaxation unexpectedly activated a Coulomb cap");
  require((c.tangent_history-initial.history*tangent_factor).norm()<1.e-12*initial.history.norm(),
          "implicit sliding relaxation differs from analytic backward Euler");
  require((c.rolling_history-initial.rolling_history*rolling_factor).norm()<1.e-12*initial.rolling_history.norm(),
          "implicit rolling relaxation differs from analytic backward Euler");
  require(c.tangent_energy < .5*initial.kt*initial.history.squaredNorm() &&
          c.rolling_energy < .5*initial.kr*initial.rolling_history.squaredNorm(),
          "implicit undriven spring relaxation generated elastic energy");
  require(std::abs(c.normal_load+initial.kn*initial.gap)<1.e-10*c.normal_load,
          "balanced normal load drifted during uncoupled relaxation");

  // Increasing an already inactive cap must change neither stiffness nor
  // relaxation: this is the finite-spring mu->infinity limit.
  p.contacts[0].mu=p.contacts[0].mu_roll=1.e12;
  const auto unbounded=cs::solve(p,accuracy());
  require((unbounded.velocity-implicit.velocity).norm()<1.e-10*implicit.velocity.norm(),
          "inactive friction coefficient changed elastic relaxation");

  const auto explicit_current=ls::solve(p,accuracy());
  const auto explicit_next=ls::advance_contacts(p,explicit_current,{initial.gap}).at(0);
  const double tangent_growth=explicit_next.tangent_energy/
      (.5*initial.kt*initial.history.squaredNorm());
  const double rolling_growth=explicit_next.rolling_energy/
      (.5*initial.kr*initial.rolling_history.squaredNorm());
  require(std::abs(tangent_growth-std::pow(1.-lambda_t*physical_dt,2))<1.e-8*tangent_growth &&
          std::abs(rolling_growth-std::pow(1.-lambda_r*physical_dt,2))<1.e-8*rolling_growth,
          "explicit reference differs from analytic unstable amplification");
  require(tangent_growth>1000. && rolling_growth>1000.,
          "prestretched reference failed to expose explicit large-step instability");
  std::cout<<"dt=1e-6 s implicit spring factors "<<tangent_factor<<" "<<rolling_factor
           <<"; explicit energy factors "<<tangent_growth<<" "<<rolling_growth<<"\n";
}

double temporal_error(int steps) {
  const double physical_time=1./lambda_t;
  auto p=fixture(physical_time/steps);
  const auto initial=p.contacts[0];
  for(int k=0;k<steps;++k){
    const auto s=cs::solve(p,accuracy());
    require(!s.contacts[0].sliding && !s.contacts[0].rolling,
            "temporal convergence fixture yielded");
    p.contacts[0].history=s.contacts[0].tangent_history;
    p.contacts[0].rolling_history=s.contacts[0].rolling_history;
  }
  const Vec3 exact_t=initial.history*std::exp(-lambda_t*physical_time);
  const Vec3 exact_r=initial.rolling_history*std::exp(-lambda_r*physical_time);
  return (p.contacts[0].history-exact_t).norm()/initial.history.norm()+
         (p.contacts[0].rolling_history-exact_r).norm()/initial.rolling_history.norm();
}
int main(){
  large_step_and_infinite_friction_limit();
  const double coarse=temporal_error(16),fine=temporal_error(32);
  require(coarse/fine>1.9 && coarse/fine<2.1,
          "implicit finite-history relaxation is not first-order convergent to exact ODE");
  std::cout<<"finite prestretch, high-friction cap independence, energy decay and exact-ODE convergence passed; error ratio "<<coarse/fine<<"\n";
}
