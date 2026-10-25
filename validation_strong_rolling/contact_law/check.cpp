#include "slurry_bridge_contact_solver.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace SlurryBridgeSolver;
namespace pc=SlurryBridgeContactSolver;
static void check(bool ok,const char *m){if(!ok)throw std::runtime_error(m);}
static bool close(double a,double b,double r=1e-8){return std::abs(a-b)<=r*std::max({std::abs(a),std::abs(b),1e-15});}
int main(){
  // Production nondimensionalization: c, gamma^-1, eta*gamma*c^2.
  const double csi=150e-9,eta=.01,gamma=100.,
      fs=eta*gamma*csi*csi,ms=fs*csi,dt=1e-6*gamma;
  Contact c; c.i=0;c.j=1;c.normal=V3::UnitZ();
  c.Bi.leftCols<3>()=-Eigen::Matrix3d::Identity();
  c.Bj.leftCols<3>()=Eigen::Matrix3d::Identity();
  c.onset=10e-9/csi;c.gap=c.onset;
  c.kn=100*csi/fs;c.kt=80*csi/fs;c.mu=.3;
  c.roll_length=3e-6/csi;c.mu_roll=.3;c.kr=100*std::pow(3e-6,2)/ms;
  c.twist_length=45e-9/csi;c.mu_twist=.1;c.kw=80*std::pow(45e-9,2)/ms;
  const double imposed_N=1e-9/fs,delta=imposed_N/c.kn;
  Contact local=c;local.gap=local.onset-delta;
  const double sy=c.mu*imposed_N/c.kt,ry=c.mu_roll*c.roll_length*imposed_N/c.kr;
  Vector u=Vector::Zero(12);u[6]=.5*sy/dt;u[9]=.5*ry/dt;
  const auto elastic=contact_response(local,u,dt);
  check(!elastic.sliding&&!elastic.rolling,"new settings should remain elastic below exact thresholds");
  check(close(elastic.tangent_force[0],.5*c.mu*imposed_N),"tangential elastic force wrong");
  check(close(elastic.torque[0],.5*c.mu_roll*c.roll_length*imposed_N),"rolling elastic torque wrong");
  u[6]=2*sy/dt;u[9]=2*ry/dt;
  const auto yielded=contact_response(local,u,dt);
  check(yielded.sliding&&yielded.rolling,"new settings should yield beyond exact thresholds");
  check(close(yielded.tangent_force[0],c.mu*imposed_N),"new tangential cap wrong");
  check(close(yielded.torque[0],c.mu_roll*c.roll_length*imposed_N),"new rolling cap wrong");
  check(yielded.sliding_work>0&&yielded.rolling_work>0,"new sliding/rolling plastic work not positive");
  Contact old=local;old.mu=.1;old.mu_roll=.1;old.roll_length=45e-9/csi;old.kr=100*std::pow(45e-9,2)/ms;
  const auto previous=contact_response(old,u*1000,dt);
  check(close(yielded.tangent_force[0]/previous.tangent_force[0],3),"sliding capacity ratio is not 3");
  check(close(yielded.torque[0]/previous.torque[0],200),"rolling capacity ratio is not 200");
  Options options;options.relative_tolerance=1e-9;options.absolute_tolerance=1e-11;options.linear_relative_tolerance=1e-11;
  std::cout<<std::setprecision(17)<<"{\n  \"kn_N_per_m\":100,\n  \"kt_N_per_m\":80,\n  \"mu_sliding\":0.3,\n  \"mu_rolling\":0.3,\n  \"rolling_length_m\":3e-6,\n  \"rolling_stiffness_Nm_per_rad\":"<<c.kr*ms
    <<",\n  \"force_cap_at_1nN_N\":"<<yielded.tangent_force[0]*fs
    <<",\n  \"rolling_cap_at_1nN_Nm\":"<<yielded.torque[0]*ms
    <<",\n  \"sliding_yield_displacement_at_1nN_m\":"<<sy*csi
    <<",\n  \"rolling_yield_angle_at_1nN_rad\":"<<ry
    <<",\n  \"sliding_cap_ratio\":3,\n  \"rolling_cap_ratio\":200,\n  \"solver_cases\":[\n";
  const double loads[3][3]={{.01,.01,.001},{2.,100.,1.},{.01,100.,.001}};
  for(int i=0;i<3;++i){
    Problem p;p.count=2;p.dt=dt;p.self_mobility.assign(2,Matrix11::Identity());p.near_resistance.resize(22,22);
    p.prescribed=Vector::Zero(22);p.external=Vector::Zero(12);p.mobility=[](const Vector&x)->Vector{return x;};p.contacts={c};
    p.external[2]=1;p.external[8]=-1;
    p.external[0]=loads[i][0];p.external[6]=-loads[i][0];
    p.external[3]=loads[i][1];p.external[9]=-loads[i][1];
    p.external[5]=loads[i][2];p.external[11]=-loads[i][2];
    const double normal=2*c.kn*dt/(1+2*c.kn*dt);
    const double tangent=std::min(2*c.kt*dt*loads[i][0]/(1+2*c.kt*dt),c.mu*normal);
    const double roll=std::min(2*c.kr*dt*loads[i][1]/(1+2*c.kr*dt),c.mu_roll*c.roll_length*normal);
    const double twist=std::min(2*c.kw*dt*loads[i][2]/(1+2*c.kw*dt),c.mu_twist*c.twist_length*normal);
    const auto result=pc::solve_proximal(p,options);
    const auto&r=result.contacts.at(0);
    double error=std::max({std::abs(r.normal_load-normal),std::abs(r.tangent_force[0]+tangent),std::abs(r.torque[0]+roll),std::abs(r.torque[2]+twist)});
    check(error<2e-7,"strong contact solver differs from independent analytical force/torque balance");
    const double supplied=p.external.dot(result.velocity),defect=std::abs(supplied-result.hydro_power-result.contact_power);
    check(defect<1e-7*std::max(std::abs(supplied),1.),"strong contact solver violates work balance");
    const auto warm=pc::solve_proximal(p,options,&result);
    check(warm.linear_iterations==0,"converged strong contact root unnecessarily iterated");
    std::cout<<"    {\"case\":"<<i<<",\"maximum_absolute_wrench_error\":"<<error<<",\"power_defect\":"<<defect<<",\"relative_residual\":"<<result.relative_residual
      <<",\"newton_iterations\":"<<result.newton_iterations<<",\"linear_iterations\":"<<result.linear_iterations<<",\"sliding\":"<<(r.sliding?"true":"false")<<",\"rolling\":"<<(r.rolling?"true":"false")<<",\"twisting\":"<<(r.twisting?"true":"false")<<"}"<<(i<2?",":"")<<"\n";
  }
  std::cout<<"  ],\n  \"status\":\"passed\",\n  \"scope\":\"Constitutive law and controlled two-body diagonal-mobility solve; not bulk viscosity.\"\n}\n";
}
