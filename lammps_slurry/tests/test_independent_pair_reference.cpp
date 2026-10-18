// Tests actual runtime predictions against frozen independent Stokes data.
// Usage: test_independent_pair_reference model.ssb independent_pair_reference.txt
// No Python or external eigensolver is required at build/run time.
#include "slurry_scattering_provider.h"
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>
using SlurryScattering::Mat11;
using Mat5=Eigen::Matrix<double,5,5>;

static Mat5 free_strain(const Mat11& h) {
  Eigen::LLT<Eigen::Matrix<double,6,6>> factor(h.topLeftCorner<6,6>());
  if(factor.info()!=Eigen::Success)throw std::runtime_error("Velocity block is not SPD");
  const Mat5 s=h.bottomRightCorner<5,5>()-h.bottomLeftCorner<5,6>()*factor.solve(h.topRightCorner<6,5>());
  return .5*(s+s.transpose()).eval();
}
template<int N> static double max_power_error(const Eigen::Matrix<double,N,N>& predicted,
                                             const Eigen::Matrix<double,N,N>& reference) {
  if(!predicted.allFinite()||!reference.allFinite())throw std::runtime_error("Nonfinite matrix");
  Eigen::LLT<Eigen::Matrix<double,N,N>> check(reference);
  if(check.info()!=Eigen::Success)throw std::runtime_error("Reference is not SPD");
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double,N,N>> eig(predicted,reference);
  if(eig.info()!=Eigen::Success)throw std::runtime_error("Generalized eigensolve failed");
  if(eig.eigenvalues().minCoeff()<=0)throw std::runtime_error("Predicted pair energy is not positive");
  return (eig.eigenvalues().array()-1.).abs().maxCoeff();
}
int main(int argc,char**argv) {
  try {
    if(argc!=3)throw std::runtime_error("Usage: test_independent_pair_reference model.ssb reference.txt");
    SlurryScattering::Provider provider;provider.load(argv[1]);provider.film_grid=161;
    if(!provider.hybrid)throw std::runtime_error("A matched hybrid bundle is required");
    std::ifstream input(argv[2]);std::string magic;int count=0;
    if(!(input>>magic>>count)||magic!="SLURRY_INDEPENDENT_REFERENCE_V1"||count!=2)
      throw std::runtime_error("Invalid independent reference fixture");
    for(int k=0;k<count;++k) {
      std::string name;double h,tolerance,stress_tolerance;
      input>>name>>h>>tolerance>>stress_tolerance;
      std::array<SlurryScattering::Vec3,2> axes;
      for(auto& a:axes)for(int j=0;j<3;++j)input>>a[j];
      Mat11 reference,self;
      for(int i=0;i<11;++i)for(int j=0;j<11;++j)input>>reference(i,j);
      for(int i=0;i<11;++i)for(int j=0;j<11;++j)input>>self(i,j);
      if(!input)throw std::runtime_error("Truncated independent reference fixture");
      if(!(h>0&&tolerance>0&&tolerance<.1&&stress_tolerance>0&&stress_tolerance<.1))
        throw std::runtime_error("Invalid fixture limits");
      const auto bodies=SlurryScattering::geometry(axes,h,provider.basis.aspect_ratio);
      const auto B=SlurryScattering::midpoint_map(bodies,h,provider.basis.aspect_ratio);
      const Mat11 runtime_self=SlurryScattering::reduced(provider.isolated(bodies),B);
      const double self_error=(runtime_self-self).norm()/self.norm();
      if(self_error>1e-10)throw std::runtime_error("Isolated baseline/frame mismatch");
      const Mat11 predicted=provider.correction(axes,h,0.,2*k,2*k+1)+self;
      const double error=max_power_error<11>(predicted,reference);
      const double stress_error=max_power_error<5>(free_strain(predicted),free_strain(reference));
      const double normal_error=predicted(2,2)/reference(2,2)-1.;
      std::cout<<std::setprecision(12)<<name<<" all11_error="<<error<<" limit="<<tolerance
        <<" free_strain_error="<<stress_error<<" limit="<<stress_tolerance
        <<" normal_error="<<normal_error<<" isolated_error="<<self_error<<"\n";
      if(error>tolerance||stress_error>stress_tolerance)
        throw std::runtime_error("Independent Stokes comparison exceeds documented per-case tolerance: "+name);
    }
    std::string extra;if(input>>extra)throw std::runtime_error("Unexpected trailing fixture data");
    std::cout<<"PASS: two independent Stokes comparisons; this is not a universal accuracy bound.\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
  return 0;
}
