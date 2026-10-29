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
void check_golden(const char* path) {
  std::ifstream in(path);check(bool(in),"Cannot open near Python golden file");
  std::string magic;int count;in>>magic>>count;check(magic=="GENERAL_NEAR_PYTHON_V1","Invalid near golden header");
  double worst=0.;
  for(int c=0;c<count;++c) {
    double ar,h,ti,tj,phi,factor;int derived;in>>ar>>h>>ti>>tj>>phi>>factor>>derived;
    const auto result=evaluate_near(pose(ar,h,ti,tj,phi),factor,derived!=0);
    for(const Mat17* actual:{&result.grand,&result.inverse_gap,&result.logarithm}) {
      Mat17 expected;for(int i=0;i<17;++i)for(int j=0;j<17;++j)in>>expected(i,j);
      check(bool(in),"Truncated near golden file");
      const double error=(*actual-expected).norm()/std::max(expected.norm(),1.);
      worst=std::max(worst,error);check(error<3e-12,"C++ near matrix differs from Python reference");
    }
    double reference,cutoff,candidate;in>>reference>>cutoff>>candidate;
    check(std::abs(result.reference_length-reference)<2e-12*reference,"Near curvature reference mismatch");
    check(std::abs(result.cutoff-cutoff)<2e-12*cutoff,"Near cutoff mismatch");
    check(std::abs(result.derived_normal_log_candidate-candidate)<3e-11*std::max(std::abs(candidate),1.),"Derived normal-log candidate mismatch");
  }
  std::cout<<"Python golden cases="<<count<<" max_matrix_relative_error="<<std::setprecision(17)<<worst<<'\n';
}
}
int main(int argc,char** argv) {
  try {
    const double pi=std::acos(-1.);
    const auto sphere=evaluate_near(pose(1,.01,pi/6,pi/3,pi/5));
    check(std::abs(sphere.inverse_gap(8,8)/(1.5*pi)-1)<1e-13,"Sphere squeeze coefficient");
    check(sphere.normal_log_coefficient==0.,"Default normal logarithm must be omitted");
    const auto derived=evaluate_near(pose(1,.01,pi/6,pi/3,pi/5),1,true);
    check(std::abs(derived.normal_log_coefficient/(27*pi/20)-1)<1e-13,"Sphere derived log coefficient");
    for(double ar:{1.,2.,5.,10.,20.}) {
      const auto a=evaluate_near(pose(ar,.002,pi/6,pi/3,pi/5));
      const auto b=evaluate_near(pose(ar,2.,pi/6,pi/3,pi/5));
      check((a.inverse_gap-b.inverse_gap).norm()==0.&&(a.logarithm-b.logarithm).norm()==0.,"Singular coefficients depend on gap");
      check(b.grand.norm()==0.,"Correction must vanish outside cutoff");
      for(const Mat17* coefficient:{&a.inverse_gap,&a.logarithm}) {
        check((*coefficient-coefficient->transpose()).norm()<1e-12*std::max(coefficient->norm(),1.),"Near reciprocity");
        for(int k=0;k<3;++k){Eigen::Matrix<double,17,1> v=Eigen::Matrix<double,17,1>::Zero();v[k]=v[k+6]=1.;
          check((*coefficient*v).norm()<1e-12*std::max(coefficient->norm(),1.),"Common translation nullspace");}
      }
    }
    const auto below=compact_near_factors(1-1e-6,1),at=compact_near_factors(1,1),above=compact_near_factors(1+1e-6,1);
    for(int k=0;k<2;++k){check(at[k]==0&&above[k]==0,"C1 cutoff value");check(std::abs(below[k]/1e-6)<2e-6,"C1 cutoff slope");}
    const auto face=evaluate_near(pose(20,.02,0,0,0));check(face.cutoff==1.,"Face-face cutoff cap");
    if(argc>1)check_golden(argv[1]);
    std::cout<<"General near algebra, continuation and strain-coordinate checks passed\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
