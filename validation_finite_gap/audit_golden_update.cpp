// Audit an intentional source-algorithm change before updating a regression
// baseline. This does not generate or replace independent Stokes reference data.
#include "slurry_scattering_provider.h"
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using Mat11=SlurryScattering::Mat11;
using Mat5=Eigen::Matrix<double,5,5>;
using Axes=std::array<SlurryScattering::Vec3,2>;
template<int N> double energy_error(const Eigen::Matrix<double,N,N>& a,const Eigen::Matrix<double,N,N>& b){
  Eigen::LLT<Eigen::Matrix<double,N,N>> first(a),second(b);
  if(first.info()!=Eigen::Success||second.info()!=Eigen::Success)throw std::runtime_error("Non-SPD source total");
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double,N,N>> spectrum(a,b);
  if(spectrum.info()!=Eigen::Success||spectrum.eigenvalues().minCoeff()<=0.)throw std::runtime_error("Invalid source spectrum");
  return (spectrum.eigenvalues().array()-1.).abs().maxCoeff();
}
Mat5 free_strain(const Mat11& m){
  Eigen::LLT<Eigen::Matrix<double,6,6>> factor(m.topLeftCorner<6,6>());
  if(factor.info()!=Eigen::Success)throw std::runtime_error("Non-SPD velocity block");
  const Mat5 f=m.bottomRightCorner<5,5>()-m.bottomLeftCorner<5,6>()*factor.solve(m.topRightCorner<6,5>());
  return .5*(f+f.transpose()).eval();
}
int main(int argc,char**argv){try{
  if(argc!=4)throw std::runtime_error("Usage: audit_golden_update basis old_fixture candidate_fixture");
  SlurryScattering::Provider selected,fine;selected.load(argv[1]);fine.load(argv[1]);selected.film_grid=161;fine.film_grid=321;
  std::ifstream input(argv[2]);int count=0;input>>count;
  if(!input||count!=3)throw std::runtime_error("Invalid old golden fixture");
  std::vector<Axes> poses;std::vector<double> gaps;std::vector<Mat11> candidates;
  bool pass=true;std::cout<<std::setprecision(17);
  for(int k=0;k<count;++k){Axes axes;for(auto& a:axes)for(int j=0;j<3;++j)input>>a[j];double h;input>>h;
    Mat11 old;for(int i=0;i<11;++i)for(int j=0;j<11;++j)input>>old(i,j);
    if(!input||!old.allFinite())throw std::runtime_error("Invalid old matrix");
    // The first two fixtures share a pose and can reuse exactly keyed anchors.
    const auto actual=selected.correction(axes,h,0.,0,1),reference=fine.correction(axes,h,0.,0,1);
    const auto bodies=SlurryScattering::geometry(axes,h,selected.basis.aspect_ratio);
    const auto B=SlurryScattering::midpoint_map(bodies,h,selected.basis.aspect_ratio);
    const Mat11 self=SlurryScattering::reduced(selected.isolated(bodies),B);
    const Mat11 actual_total=self+actual,fine_total=self+reference,old_total=self+old;
    const double difference=(actual-old).norm()/old.norm();
    const double source_error=energy_error<11>(actual_total,fine_total);
    const double source_strain=energy_error<5>(free_strain(actual_total),free_strain(fine_total));
    const double changed_energy=energy_error<11>(actual_total,old_total);
    const bool converged=source_error<=.005&&source_strain<=.005;
    pass=pass&&converged;
    std::cout<<"GOLDEN_AUDIT index="<<k<<" h_over_c="<<h
      <<" old_vs_new_correction_frobenius="<<difference
      <<" old_vs_new_all11="<<changed_energy
      <<" source161_vs321_all11="<<source_error
      <<" source161_vs321_free_strain="<<source_strain<<" source_gate_pass="<<converged<<'\n';
    poses.push_back(axes);gaps.push_back(h);candidates.push_back(actual);
  }
  std::string extra;if(input>>extra)throw std::runtime_error("Unexpected old-fixture data");
  if(!pass)throw std::runtime_error("Source convergence gate failed; candidate not written");
  std::ofstream output(argv[3]);if(!output)throw std::runtime_error("Cannot write candidate fixture");
  output<<std::setprecision(17)<<count<<'\n';
  for(int k=0;k<count;++k){for(const auto&a:poses[k])for(int j=0;j<3;++j)output<<a[j]<<' ';output<<gaps[k]<<'\n';
    for(int i=0;i<11;++i){for(int j=0;j<11;++j)output<<candidates[k](i,j)<<' ';output<<'\n';}}
  if(!output)throw std::runtime_error("Candidate write failed");
  std::cout<<"CANDIDATE_WRITTEN "<<argv[3]<<"; regression baseline only, not independent physical evidence\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
