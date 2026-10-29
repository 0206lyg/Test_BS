// Frozen production step replay: no LAMMPS runtime required.
#include "slurry_nearfield_imex_solver.h"
#include "nearfield_snapshot.h"
#include <iostream>
#include <chrono>
#include <iomanip>
int main(int argc,char **argv) {
  namespace imex=SlurryNearfieldImexSolver;namespace bs=SlurryBridgeSolver;
  if(argc!=2)return 2;
  try {
    bs::Problem p;bs::Options o;bs::Result warm;bool haswarm;
    SlurryNearfieldSnapshot::load(argv[1],p,o,warm,haswarm);
    // The snapshot is at the outer physical-step entry. Match the first
    // inner solve's tighter tolerances without changing its 40-update limit.
    o.relative_tolerance*=.1;o.absolute_tolerance*=.1;
    const auto caps=imex::freeze_caps(p);
    bs::Result improved,reference;bool old_failed=false;
    for(int run=0;run<3;++run) {
      imex::Workspace w;w.use_contact_preconditioner=true;
      // Preserve this historical directional-search regression independently
      // of the later adaptive Huber globalization tested in the AR5 recovery.
      w.use_huber_globalization=false;
      w.maximum_recovery_iterations=0;
      w.refine_damped_line_search=run==1;
      bs::Options options=o;options.maximum_newton_iterations=run==2?100:40;
      const auto start=std::chrono::steady_clock::now();
      try {
        auto result=imex::solve(p,caps,options,w,haswarm?&warm:nullptr);
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        const auto m=imex::acceptance_metric(p,caps,result,result.contacts,w,options);
        if(m.residual_norm>m.tolerance)throw std::runtime_error("physical acceptance failed");
        std::cout<<std::setprecision(17)<<"run="<<run<<" refinements="<<w.damped_line_refinements
          <<" newton="<<result.newton_iterations<<" residual="<<m.residual_norm
          <<" tolerance="<<m.tolerance<<" seconds="<<elapsed<<'\n';
        if(run==0)throw std::runtime_error("original 40-update failure not reproduced");
        if(run==1){if(!w.damped_line_refinements)throw std::runtime_error("fallback not exercised");improved=std::move(result);}
        else reference=std::move(result);
      } catch(const std::runtime_error&e) {
        if(run==0&&std::string(e.what()).find("IMEX: convex Newton limit;")==0) {
          old_failed=true;std::cout<<"EXPECTED_BASELINE_FAILURE "<<e.what()<<'\n';
        } else throw;
      }
    }
    const double velocity_difference=(improved.velocity-reference.velocity).norm()/std::max(1.,reference.velocity.norm());
    std::cout<<"relative_velocity_difference="<<velocity_difference<<'\n';
    if(!old_failed||velocity_difference>1e-5)throw std::runtime_error("replay branch/reference mismatch");
    // Re-evaluation must leave the frozen inputs unchanged.
    if(caps!=imex::freeze_caps(p))throw std::runtime_error("frozen normal caps mutated");
    std::cout<<"PASS original failure recovered under unchanged timestep, stiffness, tolerances and 40-update budget\n";
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
