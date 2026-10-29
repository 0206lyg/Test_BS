#include "slurry_general_model.h"
#include "slurry_general_outer.h"
#include "slurry_general_near.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
using namespace SlurryGeneral;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
Geometry pose(double ar,double h,double ti,double tj,double phi) {
  return geometry_from_axes(ar,h,{std::sin(ti),0,std::cos(ti)},
    {std::sin(tj)*std::cos(phi),std::sin(tj)*std::sin(phi),std::cos(tj)});
}
void golden(const char* path) {
  std::ifstream in(path);check(bool(in),"Cannot open model Python golden file");
  std::string magic;int count;in>>magic>>count;check(magic=="GENERAL_MODEL_PYTHON_V1","Invalid model golden header");
  double worst=0.;
  for(int c=0;c<count;++c) {
    double ar,h,ti,tj,phi;int nr,np;in>>ar>>h>>ti>>tj>>phi>>nr>>np;
    ModelOptions raw;raw.energy_completion=false; // Historical Python evaluation fixture.
    const auto result=evaluate_model(pose(ar,h,ti,tj,phi),raw);
    if(!result.far)check(result.nrad==nr&&result.nphi==np,"Adaptive level differs from Python reference");
    for(const Mat17* actual:{&result.grand,&result.outer,&result.near,&result.isolated}) {
      Mat17 expected;for(int i=0;i<17;++i)for(int j=0;j<17;++j)in>>expected(i,j);
      check(bool(in),"Truncated model golden file");
      const double error=(*actual-expected).norm()/std::max(expected.norm(),1.);
      worst=std::max(worst,error);check(error<5e-10,"C++ combined model differs from Python reference");
    }
    check(result.linear_residual<1e-8,"Resolved model residual");
    check(result.outer_relative_change<=.005&&result.full_relative_change<=.005,"Resolved model norm gate");
    std::cout<<"AR="<<ar<<" h="<<h<<" quadrature="<<result.nrad<<'x'<<result.nphi<<" far="<<result.far<<'\n';
  }
  std::cout<<"Python model golden cases="<<count<<" max_matrix_relative_error="<<std::setprecision(17)<<worst<<'\n';
}
}
int main(int argc,char** argv) {
  try {
    const double pi=std::acos(-1.);
    const auto far_geometry=pose(2.,40.,pi/6,pi/3,pi/5);
    const auto far=evaluate_model(far_geometry);
    check(far.far&&far.levels==1&&far.nrad==0&&far.nphi==0,"Far branch should skip disk refinement");
    const auto direct=evaluate_outer(far_geometry,4,16,true);
    check((far.outer-direct.grand).norm()==0.,"Far response changed during composition");
    check((far.grand-far.outer-far.near).norm()<1e-13,"Model decomposition");
    ModelOptions limited;limited.max_level=1;bool failed=false;
    try{evaluate_model(pose(20.,.02,0,0,0),limited);}
    catch(const std::runtime_error& e){failed=std::string(e.what()).find("numerically unresolved")!=std::string::npos;}
    check(failed,"Low-order AR20 face case must not be silently accepted");
    ModelOptions invalid;invalid.max_level=0;failed=false;
    try{evaluate_model(far_geometry,invalid);}catch(const std::invalid_argument&){failed=true;}
    check(failed,"Invalid adaptive limit accepted");
    if(argc>1)golden(argv[1]);
    std::cout<<"General model adaptive resolution, far branch, and explicit failure checks passed\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
