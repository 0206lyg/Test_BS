#include "slurry_nearfield_imex_solver.h"
#include "nearfield_snapshot.h"
#include <fstream>
#include <iomanip>
#include <iostream>
namespace bs=SlurryBridgeSolver;
namespace imex=SlurryNearfieldImexSolver;
int main(int argc,char**argv){
  if(argc!=4)return 2;
  bs::Problem p;bs::Options o;bs::Result warm;bool haswarm;
  SlurryNearfieldSnapshot::load(argv[1],p,o,warm,haswarm);
  o.maximum_linear_iterations=1000;
  std::ifstream input(argv[2]);std::size_t n;input>>n;
  if(n!=p.contacts.size())return 3;
  imex::FrozenCaps caps(n);for(auto&v:caps)input>>v;
  if(!input)return 4;
  std::cout<<std::setprecision(17);
  for(int repeat=0;repeat<3;++repeat)for(int order=0;order<2;++order){
    const bool hybrid=(repeat%2==0)?order:!order;
    auto opts=o;opts.maximum_newton_iterations=hybrid?40:200;
    imex::Workspace w;w.use_contact_preconditioner=true;w.use_huber_globalization=hybrid;
    w.maximum_recovery_iterations=hybrid?80:0;
    CostProfile::entries.clear();const auto cpu=std::clock();
    auto r=imex::solve(p,caps,opts,w,haswarm?&warm:nullptr);
    const double elapsed=double(std::clock()-cpu)/CLOCKS_PER_SEC;
    std::cout<<"{\"state\":\""<<argv[3]<<"\",\"hybrid\":"<<(hybrid?"true":"false")
      <<",\"repeat\":"<<repeat<<",\"cpu_seconds\":"<<elapsed
      <<",\"newton\":"<<r.newton_iterations<<",\"linear\":"<<r.linear_iterations
      <<",\"relative_residual\":"<<r.relative_residual
      <<",\"hydro_matrix_nnz\":"<<w.matrix.nonZeros()
      <<",\"hydro_factor_L_nnz\":"<<w.factor.matrixL().nestedExpression().nonZeros()
      <<",\"contact_matrix_nnz\":"<<w.jacobian.nonZeros()
      <<",\"contact_factor_L_nnz\":"<<w.newton_factor.matrixL().nestedExpression().nonZeros()
      <<",\"hydro_numeric_factorizations\":"<<(w.numeric_factorizations-w.newton_numeric_factorizations)
      <<",\"contact_numeric_factorizations\":"<<w.newton_numeric_factorizations
      <<",\"contact_symbolic_analyses\":"<<w.newton_symbolic_analyses
      <<",\"krylov_fallbacks\":"<<w.krylov_fallbacks
      <<",\"timers\":{";
    bool first=true;for(const auto&[name,e]:CostProfile::entries){
      if(!first)std::cout<<',';first=false;
      std::cout<<'"'<<name<<"\":{\"cpu_seconds\":"<<e.seconds<<",\"calls\":"<<e.count<<'}';}
    std::cout<<"}}\n"<<std::flush;
  }
}
