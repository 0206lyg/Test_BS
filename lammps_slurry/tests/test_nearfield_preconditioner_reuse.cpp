#include "slurry_nearfield_imex_solver.h"
#include <iostream>
#include <chrono>

namespace bs=SlurryBridgeSolver;
namespace im=SlurryNearfieldImexSolver;
using V=Eigen::Vector3d;
using M=Eigen::Matrix3d;
static void require(bool okay,const char* text){if(!okay)throw std::runtime_error(text);}
static bs::Problem problem(int step,int count=4) {
  bs::Problem p;p.count=count;p.dt=.001*(1.+.1*(step%7));
  p.prescribed=bs::Vector::Zero(11*count);p.external=bs::Vector::Zero(6*count);
  p.near_resistance.resize(11*count,11*count);p.self_mobility.assign(count,bs::Matrix11::Identity());
  for(int i=0;i<count;++i) {
    p.external[6*i]=.1*std::sin(.1*step+i);
    p.external[6*i+1]=.2*std::cos(.1*step+i);
    p.external[6*i+3]=.05*std::cos(.17*step+i);
  }
  for(int i=0;i<count-1;++i) {
    bs::Contact c;c.i=i;c.j=i+1;c.normal=V::UnitZ();c.gap=-.001;
    c.kn=1e7;c.kt=3e6;c.kr=3e6;c.mu=c.mu_roll=10.;c.roll_length=1.;
    c.history=V(.00001*std::sin(.01*step+i),.00002,0.);
    c.rolling_history=V(.00001,-.00001*std::cos(.1*step+i),0.);
    c.Bi.leftCols<3>()=-M::Identity();c.Bj.leftCols<3>()=M::Identity();
    p.contacts.push_back(c);
  }
  return p;
}
int main() {
  try {
    bs::Options options;options.relative_tolerance=1e-10;options.absolute_tolerance=1e-12;
    im::Workspace reference,reused;reference.use_contact_preconditioner=reused.use_contact_preconditioner=true;
    reused.reuse_contact_preconditioner_across_solves=true;
    bs::Result warm_reference,warm_reused;double max_difference=0.;
    for(int step=0;step<40;++step) {
      auto p=problem(step);const auto caps=im::freeze_caps(p);
      const auto a=im::solve(p,caps,options,reference,step?&warm_reference:nullptr);
      const auto b=im::solve(p,caps,options,reused,step?&warm_reused:nullptr);
      const double difference=(a.velocity-b.velocity).norm()/std::max(1.,a.velocity.norm());
      max_difference=std::max(max_difference,difference);
      require(difference<1e-9,"previous-factor preconditioner changed the converged velocity");
      require(im::acceptance_metric(p,caps,b,b.contacts,reused,options).residual_norm<=
              im::acceptance_metric(p,caps,b,b.contacts,reused,options).tolerance,
              "reused preconditioner bypassed physical residual acceptance");
      warm_reference=a;warm_reused=b;
    }
    require(reused.contact_preconditioner_cross_solve_reuses>0,"test did not reuse a previous-call factor");
    require(reused.contact_preconditioner_refreshes<reference.contact_preconditioner_refreshes,
            "previous-call factor did not reduce refreshes");
    const auto resized=problem(40,3);const auto caps=im::freeze_caps(resized);
    const auto answer=im::solve(resized,caps,options,reused);
    require(answer.velocity.size()==18,"dimension change reused a wrong-sized factor");
    require(reused.maximum_true_relative_linear_residual<=reused.linear_relative_target,
            "reused preconditioner violated current-J true residual");
    std::cout<<"maximum_relative_velocity_difference="<<max_difference
             <<" baseline_refreshes="<<reference.contact_preconditioner_refreshes
             <<" reused_refreshes_including_resize="<<reused.contact_preconditioner_refreshes
             <<" cross_solve_reuses="<<reused.contact_preconditioner_cross_solve_reuses
             <<" max_true_linear_residual="<<reused.maximum_true_relative_linear_residual<<'\n';
    std::cout<<"nearfield preconditioner reuse PASS\n";return 0;
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
