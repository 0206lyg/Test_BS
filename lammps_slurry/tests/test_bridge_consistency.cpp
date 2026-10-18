// Held-out numerical consistency only. This is not an independent physical
// reference and does not replace test_basis_physical.cpp.
#include "slurry_scattering_provider.h"
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#ifndef SLURRY_NUMERICAL_SOURCE_FINGERPRINT
#define SLURRY_NUMERICAL_SOURCE_FINGERPRINT "unbound"
#endif
namespace sc=SlurryScattering;
using Mat11=sc::Mat11;
using Mat5=Eigen::Matrix<double,5,5>;
static constexpr double convergence_limit=.005;
static constexpr double bridge_limit=.05;
static void save_matrix(std::ofstream &output,const char *slot,double h,const Mat11 &m) {
  if(!output.is_open())return;
  output<<"{\"slot\":\""<<slot<<"\",\"h_over_c\":"<<h<<",\"matrix\":[";
  for(int i=0;i<11;++i) { if(i)output<<",";output<<"[";for(int j=0;j<11;++j){if(j)output<<",";output<<m(i,j);}output<<"]"; }
  output<<"]}\n"<<std::flush;
}
static std::string quote(const std::string &s) {
  std::string out="\"";
  for(char c:s) { if(c=='"'||c=='\\'){out+='\\';out+=c;}else if(c=='\n')out+="\\n";else out+=c; }
  return out+'"';
}
template<int N> static double power_error(const Eigen::Matrix<double,N,N> &a,const Eigen::Matrix<double,N,N> &b) {
  if(!a.allFinite()||!b.allFinite())throw std::runtime_error("Nonfinite energy matrix");
  Eigen::LLT<Eigen::Matrix<double,N,N>> af(a),bf(b);
  if(af.info()!=Eigen::Success||bf.info()!=Eigen::Success)throw std::runtime_error("Energy comparison requires SPD matrices");
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double,N,N>> e(a,b);
  if(e.info()!=Eigen::Success)throw std::runtime_error("Generalized energy eigensolve failed");
  return (e.eigenvalues().array()-1.).abs().maxCoeff();
}
static Mat5 free_strain(const Mat11 &r) {
  Eigen::LLT<Eigen::Matrix<double,6,6>> f(r.topLeftCorner<6,6>());
  if(f.info()!=Eigen::Success)throw std::runtime_error("Velocity block is not SPD");
  const Mat5 s=r.bottomRightCorner<5,5>()-r.bottomLeftCorner<5,6>()*f.solve(r.topRightCorner<6,5>());
  return .5*(s+s.transpose()).eval();
}
static bool same_matrix(const sc::Matrix &a,const sc::Matrix &b) {
  return a.rows()==b.rows()&&a.cols()==b.cols()&&(a.array()==b.array()).all();
}
static bool same_coefficients(const sc::Basis &a,const sc::Basis &b) {
  return a.aspect_ratio==b.aspect_ratio&&a.degree==b.degree&&a.use_fourier&&b.use_fourier&&
    same_matrix(a.targets,b.targets)&&same_matrix(a.projection,b.projection)&&
    same_matrix(a.rigid,b.rigid)&&same_matrix(a.strain,b.strain)&&
    same_matrix(a.force_torque,b.force_torque)&&same_matrix(a.stresslet,b.stresslet)&&
    same_matrix(a.fourier_rings,b.fourier_rings)&&same_matrix(a.fourier_density,b.fourier_density);
}
static bool compare(const Mat11 &a,const Mat11 &b,const char *stage,double h,double limit) {
  const double error=power_error<11>(a,b),strain=power_error<5>(free_strain(a),free_strain(b));
  const bool passed=error<=limit&&strain<=limit;
  std::cout<<"{\"stage\":"<<quote(stage)<<",\"h_over_c\":"<<h
    <<",\"passed\":"<<(passed?"true":"false")<<",\"all11_error\":"<<error
    <<",\"free_strain_error\":"<<strain<<",\"limit\":"<<limit<<"}\n"<<std::flush;
  return passed;
}
static bool isolated_check(const sc::Basis &b,const Mat11 &reference,const char *slot) {
  Mat11 r;r.topLeftCorner<6,6>()=b.force_torque*b.rigid;r.topRightCorner<6,5>()=b.force_torque*b.strain;
  r.bottomLeftCorner<5,6>()=b.stresslet*b.rigid;r.bottomRightCorner<5,5>()=b.stresslet*b.strain;
  const double reciprocal=(r-r.transpose()).norm()/r.norm();
  const double error=power_error<11>(.5*(r+r.transpose()).eval(),reference);
  const bool passed=error<=.001&&reciprocal<=.001;
  std::cout<<"{\"stage\":\"isolated_units\",\"slot\":"<<quote(slot)
    <<",\"passed\":"<<(passed?"true":"false")<<",\"all11_error\":"<<error
    <<",\"raw_reciprocity\":"<<reciprocal<<",\"limit\":0.001}\n"<<std::flush;
  return passed;
}
static Mat11 direct(const sc::Basis &b,const std::array<sc::Vec3,2> &axes,double h,const char *slot) {
  const auto bodies=sc::geometry(axes,h,b.aspect_ratio);const auto raw=b.evaluate(bodies);
  const sc::Mat17 r=.5*(raw.grand+raw.grand.transpose()).eval();Eigen::LLT<sc::Mat17> positive(r);
  const bool passed=raw.reciprocity<=.01&&raw.linear_residual<=1e-8&&positive.info()==Eigen::Success;
  std::cout<<"{\"stage\":\"raw_full17\",\"slot\":"<<quote(slot)<<",\"h_over_c\":"<<h
    <<",\"passed\":"<<(passed?"true":"false")<<",\"raw_reciprocity\":"<<raw.reciprocity
    <<",\"linear_residual\":"<<raw.linear_residual<<",\"positive\":"<<(positive.info()==Eigen::Success?"true":"false")<<"}\n"<<std::flush;
  if(!passed)throw std::runtime_error("Raw direct source failed SPD/reciprocity/solve gate");
  return sc::reduced(r,sc::midpoint_map(bodies,h,b.aspect_ratio));
}
int main(int argc,char **argv) {
  try {
    if(argc!=6&&argc!=7)throw std::runtime_error("Usage: bridge_consistency candidate128.ssb candidate256.ssb refined256.ssb refined512.ssb film_grid [matrix_output]");
    std::cout<<std::setprecision(17);
    std::ofstream matrix_output;
    if(argc==7) {matrix_output.open(argv[6]);if(!matrix_output)throw std::runtime_error("Cannot open matrix output");matrix_output<<std::setprecision(17);}
    sc::Provider model,quadrature,refined,refined_quadrature;
    model.load(argv[1]);quadrature.load(argv[2]);refined.load(argv[3]);refined_quadrature.load(argv[4]);
    model.film_grid=std::stoi(argv[5]);
    quadrature.film_grid=model.film_grid;
    if(model.film_grid<21||model.film_grid%2==0)throw std::runtime_error("Film grid must be odd and at least21");
    if(!model.hybrid||!quadrature.hybrid)
      throw std::runtime_error("Expected two hybrid candidate models");
    const sc::Basis &refined_basis=refined.hybrid?refined.near_basis:refined.basis;
    const sc::Basis &refined_quadrature_basis=refined_quadrature.hybrid?refined_quadrature.near_basis:refined_quadrature.basis;
    if(model.basis.aspect_ratio!=10.||quadrature.basis.aspect_ratio!=10.||refined.basis.aspect_ratio!=10.||refined_quadrature.basis.aspect_ratio!=10.)
      throw std::runtime_error("Held-out bridge trial is AR10 only");
    if(!same_coefficients(model.near_basis,quadrature.near_basis)||!same_coefficients(model.basis,quadrature.basis)||
       !same_coefficients(refined_basis,refined_quadrature_basis))
      throw std::runtime_error("Quadrature refinements must retain identical source coefficients");
    if(quadrature.near_basis.fourier_nphi<2*model.near_basis.fourier_nphi||
       refined_quadrature_basis.fourier_nphi<2*refined_basis.fourier_nphi||refined_basis.degree<=model.near_basis.degree)
      throw std::runtime_error("Refinement inputs do not increase quadrature/order");
    std::cout<<"{\"stage\":\"metadata\",\"evaluated_path\":\"bridge_consistency\",\"aspect_ratio\":10"
      <<",\"compiled_source_fingerprint\":"<<quote(SLURRY_NUMERICAL_SOURCE_FINGERPRINT)
      <<",\"film_grid\":"<<model.film_grid<<",\"candidate_degree\":"<<model.near_basis.degree
      <<",\"refined_degree\":"<<refined_basis.degree<<",\"convergence_limit\":"<<convergence_limit
      <<",\"bridge_limit\":"<<bridge_limit<<",\"independent_physical_reference\":false}\n"<<std::flush;
    if(!isolated_check(model.near_basis,model.isolated_local,"candidate")||
       !isolated_check(refined_basis,model.isolated_local,"refined"))return 2;
    const sc::Vec3 axis(.025811286645980556,0.,.99966683324079475);
    const std::array<sc::Vec3,2> axes{axis,axis};
    // These held-out gaps are frozen before the matching implementation changes.
    for(double h:{.25,.75,1.5}) {
      const Mat11 r128=direct(model.near_basis,axes,h,"candidate128");
      const Mat11 r256=direct(quadrature.near_basis,axes,h,"candidate256");
      save_matrix(matrix_output,"candidate128",h,r128);save_matrix(matrix_output,"candidate256",h,r256);
      if(!compare(r128,r256,"candidate_runtime_refinement",h,convergence_limit))return 2;
      const Mat11 ref256=direct(refined_basis,axes,h,"refined256");
      const Mat11 ref512=direct(refined_quadrature_basis,axes,h,"refined512");
      save_matrix(matrix_output,"refined256",h,ref256);save_matrix(matrix_output,"refined512",h,ref512);
      if(!compare(ref256,ref512,"reference_runtime_refinement",h,convergence_limit))return 2;
      if(!compare(r256,ref512,"basis_order_refinement",h,convergence_limit))return 2;
      const auto bodies=sc::geometry(axes,h,10.);const auto map=sc::midpoint_map(bodies,h,10.);
      const Mat11 predicted=model.correction(axes,h,0.)+sc::reduced(model.isolated(bodies),map);
      const Mat11 predicted_refined=quadrature.correction(axes,h,0.)+sc::reduced(quadrature.isolated(bodies),map);
      save_matrix(matrix_output,"matched128",h,predicted);save_matrix(matrix_output,"matched256",h,predicted_refined);
      if(!compare(predicted,predicted_refined,"matched_runtime_refinement",h,convergence_limit))return 2;
      if(!compare(predicted,ref512,"heldout_bridge",h,bridge_limit))return 2;
    }
    return 0;
  }catch(const std::exception &e) {
    std::cout<<"{\"stage\":\"exception\",\"passed\":false,\"error\":"<<quote(e.what())<<"}\n";return 2;
  }
}
