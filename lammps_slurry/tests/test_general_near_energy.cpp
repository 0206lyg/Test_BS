#include "slurry_general_model.h"
#include "slurry_general_near.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
using namespace SlurryGeneral;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
double minimum(const Mat17& a){Eigen::SelfAdjointEigenSolver<Mat17> e(.5*(a+a.transpose()).eval());
  check(e.info()==Eigen::Success,"Energy test eigensolve failed");return e.eigenvalues()[0];}
void factor_identity(const Geometry& g,bool derived=false) {
  const auto raw=evaluate_near(g,1.,derived,false),n=evaluate_near(g,1.,derived,true);
  check((n.inverse_gap-raw.inverse_gap).norm()==0.&&(n.logarithm-raw.logarithm).norm()==0.,"Energy completion changed singular coefficients");
  check((n.grand-raw.grand-n.completion).norm()<2e-12*std::max(n.grand.norm(),1.),"Energy completion decomposition");
  const auto f=compact_near_factors(g.gap,n.cutoff);
  const double k=f[0]*n.inverse_gap6(2,2)+f[1]*n.logarithm6(2,2);
  Eigen::Matrix<double,6,1> c=f[1]*n.logarithm6.col(2);c[2]=0.;
  auto y=n.logarithm6;y.row(2).setZero();y.col(2).setZero();
  Eigen::Matrix<double,6,1> square=c/k;square[2]=1.;
  const Eigen::Matrix<double,6,6> gram=k*square*square.transpose()+f[1]*y;
  const Eigen::Matrix<double,6,6> completed=f[0]*n.inverse_gap6+f[1]*n.logarithm6+n.completion6;
  check((gram-completed).norm()<2e-12*std::max(completed.norm(),1.),"Positive-square identity does not hold");
  check(minimum(n.grand)>-2e-12*std::max(n.grand.norm(),1.),"Completed near response is not PSD within roundoff");
  Eigen::Matrix<double,17,1> common=Eigen::Matrix<double,17,1>::Zero();common[0]=common[6]=1.;
  check((n.completion*common).norm()<1e-12*std::max(n.completion.norm(),1.),"Completion violates common translation nullspace");
}
void python_fixture(const char* path) {
  std::ifstream in(path);check(bool(in),"Cannot open Python near-energy fixture");
  std::string magic;double ar,h,ti,tj,phi;in>>magic>>ar>>h>>ti>>tj>>phi;
  check(magic=="GENERAL_NEAR_ENERGY_PYTHON_V1","Invalid near-energy fixture header");
  const auto g=geometry_from_axes(ar,h,{std::sin(ti),0,std::cos(ti)},
      {std::sin(tj)*std::cos(phi),std::sin(tj)*std::sin(phi),std::cos(tj)});
  const auto result=evaluate_near(g,1.,false,true);double worst=0.;
  for(const Mat17* actual:{&result.grand,&result.completion}) {
    Mat17 expected;for(int i=0;i<17;++i)for(int j=0;j<17;++j)in>>expected(i,j);
    check(bool(in),"Truncated near-energy fixture");
    const double error=(*actual-expected).norm()/std::max(expected.norm(),1.);
    worst=std::max(worst,error);check(error<5e-11,"NumPy/C++ near-energy construction differs");
  }
  std::cout<<"Python positive-square fixture max_relative_error="<<std::setprecision(17)<<worst<<'\n';
}
}
int main(int argc,char** argv) {
  try {
    // Actual first failed pair from AR20 N64 integration, tags 1 and 64.
    const auto g=geometry_from_axes(20.,.007800524472922343,
      {.0459959091219549,0.,.9989416280964796},
      {.18424127321483846,.029663839054549584,.9824333106611978});
    factor_identity(g);factor_identity(g,true);
    const auto sphere=geometry_from_axes(1.,.02,{0,0,1},{0,0,1});factor_identity(sphere);
    check(evaluate_near(sphere,1.,false,true).completion.norm()==0.,"Spheres need no cubic energy completion");
    const auto raw=evaluate_near(g);
    const auto completed=evaluate_model(g);
    check(minimum(raw.grand+completed.outer)<0.,"Historical failure fixture no longer demonstrates raw failure");
    check(minimum(completed.grand)>0.,"Energy-completed actual failure pair is not SPD");
    Eigen::LLT<Mat17> iso(completed.isolated);check(iso.info()==Eigen::Success,"Isolated fixture not SPD");
    const Mat17 left=iso.matrixL().solve(completed.grand);
    const Mat17 scaled=iso.matrixL().solve(left.transpose()).transpose();
    const double scaled_min=minimum(scaled);
    check(scaled_min>.65&&scaled_min<.68,"Failure geometry completed response changed unexpectedly");
    // Cutoff and contact limits are model identities, independent of sampling.
    auto near_cut=g;near_cut.gap=1-1e-6;const auto c1=evaluate_near(near_cut,1.,false,true);
    near_cut.gap=1-2e-6;const auto c2=evaluate_near(near_cut,1.,false,true);
    check(std::abs(c2.completion.norm()/c1.completion.norm()-4)<1e-4,"Completion does not vanish quadratically at cutoff");
    near_cut.gap=1.;check(evaluate_near(near_cut,1.,false,true).completion.norm()==0.,"Completion not zero at cutoff");
    auto tiny=g;tiny.gap=1e-8;const auto t1=evaluate_near(tiny,1.,false,true);
    tiny.gap=1e-9;const auto t2=evaluate_near(tiny,1.,false,true);
    check(t2.completion.norm()<.15*t1.completion.norm(),"Completion does not tend to zero at contact");
    if(argc>1)python_fixture(argv[1]);
    std::cout<<std::setprecision(17)<<"Near positive-square identity and actual AR20 integration failure passed; completed_scaled_min="<<scaled_min
      <<" quadrature="<<completed.nrad<<'x'<<completed.nphi<<" completion_full_Frobenius_ratio="<<completed.near_completion.norm()/completed.grand.norm()<<'\n';
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
