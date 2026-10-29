// Replay one captured failing INNER convex equation without LAMMPS or MPI.
#include "slurry_nearfield_imex_solver.h"
#include "nearfield_snapshot.h"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

namespace bs=SlurryBridgeSolver;
namespace imex=SlurryNearfieldImexSolver;
void require(bool condition,const char*message){if(!condition)throw std::runtime_error(message);}

int main(int argc,char**argv) {
  if(argc!=3){std::cerr<<"usage: replay_ar5_recovery INNER.snapshot CAPS.txt\n";return 2;}
  try {
    bs::Problem p;bs::Options original;bs::Result warm;bool haswarm;
    SlurryNearfieldSnapshot::load(argv[1],p,original,warm,haswarm);
    // These are already INNER tolerances; do not multiply by 0.1 again.
    original.maximum_linear_iterations=1000;
    std::ifstream input(argv[2]);std::size_t count=0;input>>count;
    require(count==p.contacts.size(),"frozen-cap count mismatch");
    imex::FrozenCaps caps(count);for(auto&cap:caps)input>>cap;
    require(bool(input),"truncated frozen-cap file");
    const auto unchanged_caps=caps;
    std::map<std::string,bs::Vector> velocities;
    struct Case {const char*name;bool hybrid;int primary,recovery;double tolerance_scale;bool expected_failure;};
    const Case cases[]={
      {"original_limit40",false,40,0,1.,true},
      {"hybrid_limit40",true,40,0,1.,false},
      {"production_default",true,40,80,1.,false},
      {"original_extended_reference",false,200,0,1.,false},
      {"original_tight_reference",false,200,0,.001,false},
      {"forced_recovery",true,4,80,1.,false}
    };
    std::cout<<std::setprecision(17);
    for(const auto&test:cases) {
      bs::Options options=original;options.maximum_newton_iterations=test.primary;
      options.relative_tolerance*=test.tolerance_scale;options.absolute_tolerance*=test.tolerance_scale;
      imex::Workspace workspace;workspace.use_contact_preconditioner=true;
      workspace.use_huber_globalization=test.hybrid;workspace.maximum_recovery_iterations=test.recovery;
      const auto start=std::chrono::steady_clock::now();const auto cpu_start=std::clock();
      try {
        const auto result=imex::solve(p,caps,options,workspace,haswarm?&warm:nullptr);
        require(!test.expected_failure,"original 40-update failure was not reproduced");
        const auto metric=imex::acceptance_metric(p,caps,result,result.contacts,workspace,options);
        require(metric.residual_norm<=metric.tolerance,"exact physical acceptance failed");
        require(workspace.maximum_true_relative_linear_residual<=workspace.linear_relative_target,
                "linear direction failed its true residual gate");
        if(test.hybrid)require(workspace.huber_majorized_directions>0,"hybrid direction was not exercised");
        if(std::string(test.name)=="forced_recovery")
          require(workspace.nonlinear_recoveries==1&&workspace.nonlinear_recovery_iterations>0,
                  "bounded recovery was not exercised");
        velocities[test.name]=result.velocity;
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        const double cpu_seconds=double(std::clock()-cpu_start)/CLOCKS_PER_SEC;
        std::cout<<"PASS "<<test.name<<" newton="<<result.newton_iterations
          <<" linear="<<result.linear_iterations<<" residual="<<metric.residual_norm
          <<" tolerance="<<metric.tolerance<<" relative_residual="<<result.relative_residual
          <<" majorized="<<workspace.huber_majorized_directions
          <<" recoveries="<<workspace.nonlinear_recoveries
          <<" recovery_iterations="<<workspace.nonlinear_recovery_iterations
          <<" contact_factorizations="<<workspace.newton_numeric_factorizations
          <<" max_linear_relative="<<workspace.maximum_true_relative_linear_residual
          <<" wall_seconds="<<seconds<<" cpu_seconds="<<cpu_seconds<<'\n';
      } catch(const std::runtime_error&e) {
        if(test.expected_failure&&std::string(e.what()).find("IMEX: convex Newton limit;")==0)
          std::cout<<"EXPECTED_FAILURE "<<test.name<<' '<<e.what()<<'\n';
        else throw;
      }
    }
    const auto&reference=velocities.at("original_tight_reference");
    for(const char*name:{"hybrid_limit40","production_default","forced_recovery"}) {
      const double difference=(velocities.at(name)-reference).norm()/std::max(1.,reference.norm());
      std::cout<<"ROOT_AGREEMENT "<<name<<" relative_velocity_difference="<<difference<<'\n';
      require(difference<1e-8,"accepted velocity differs from tighter original-Newton reference");
    }
    require(caps==unchanged_caps,"frozen caps changed during replay");
    std::cout<<"PASS same equation, stiffness, timestep, frozen caps and physical residual gates\n";
  }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
