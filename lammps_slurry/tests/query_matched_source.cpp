// One explicitly requested AR10 offset query for an independent MFS comparison.
#include "slurry_scattering_provider.h"
#include <iomanip>
#include <iostream>
#ifndef SLURRY_NUMERICAL_SOURCE_FINGERPRINT
#define SLURRY_NUMERICAL_SOURCE_FINGERPRINT "unbound"
#endif
int main(int argc,char **argv) {
  try {
    if(argc!=4&&argc!=5&&argc!=9)throw std::runtime_error("Usage: query_matched_source model.ssb h_over_c film_grid [matched_provider|direct_basis [epsilon theta_i theta_j phi]]");
    namespace sc=SlurryScattering;
    const std::string mode=argc>=5?argv[4]:"matched_provider";
    if(mode!="matched_provider"&&mode!="direct_basis")throw std::runtime_error("Invalid query evaluation mode");
    sc::Provider p;p.load(argv[1]);const double h=std::stod(argv[2]);p.film_grid=std::stoi(argv[3]);
    if((mode=="matched_provider"&&!p.hybrid)||p.basis.aspect_ratio!=10.||!(h>0.)||!std::isfinite(h)||p.film_grid<21||p.film_grid%2==0)
      throw std::runtime_error("Expected AR10 source, positive gap, and odd film grid>=21");
    const sc::Vec3 axis(.025811286645980556,0.,.99966683324079475);
    std::array<sc::Vec3,2> axes{axis,axis};double epsilon=0.;
    if(argc==9) {
      epsilon=std::stod(argv[5]);const double ti=std::stod(argv[6]),tj=std::stod(argv[7]),phi=std::stod(argv[8]);
      if(mode!="matched_provider"||epsilon<0.||!std::isfinite(epsilon)||!std::isfinite(ti)||!std::isfinite(tj)||!std::isfinite(phi))
        throw std::runtime_error("Custom pose requires matched_provider and finite nonnegative regularization");
      axes={sc::Vec3(std::sin(ti),0.,std::cos(ti)),sc::Vec3(std::sin(tj)*std::cos(phi),std::sin(tj)*std::sin(phi),std::cos(tj))};
    }
    const auto bodies=sc::geometry(axes,h,10.);const auto map=sc::midpoint_map(bodies,h,10.);
    const sc::Basis &source=p.hybrid?p.near_basis:p.basis;
    sc::Mat11 r;double reciprocity=0.,linear_residual=0.;
    if(mode=="direct_basis") {
      const auto raw=source.evaluate(bodies);reciprocity=raw.reciprocity;linear_residual=raw.linear_residual;
      const sc::Mat17 symmetric=.5*(raw.grand+raw.grand.transpose()).eval();Eigen::LLT<sc::Mat17> factor(symmetric);
      if(reciprocity>.01||linear_residual>1e-8||factor.info()!=Eigen::Success)throw std::runtime_error("Direct source failed reciprocity/residual/SPD gate");
      r=sc::reduced(symmetric,map);
    }else r=p.correction(axes,h,epsilon)+sc::reduced(p.isolated(bodies),map);
    Eigen::LLT<sc::Mat11> positive(r);if(positive.info()!=Eigen::Success)throw std::runtime_error("Actual matched total is not SPD");
    std::cout<<std::setprecision(17)<<"{\"compiled_source_fingerprint\":\""<<SLURRY_NUMERICAL_SOURCE_FINGERPRINT
      <<"\",\"evaluated_path\":\""<<mode<<"\",\"aspect_ratio\":10,\"hybrid_version\":"<<p.hybrid_version
      <<",\"film_grid\":"<<p.film_grid<<",\"h_over_c\":"<<h<<",\"epsilon_over_c\":"<<epsilon
      <<",\"matching_policy\":\""<<(p.hybrid_version==3?"direct_near_at_heff_ge_0.5; unchanged_first_film_segment_below_0.5":"serialized_anchor_matching")
      <<"\",\"positive_total\":true,\"anchor_gaps\":[";
    for(int k=0;k<p.anchor_count;++k){if(k)std::cout<<",";std::cout<<p.anchor_gaps[k];}
    std::cout<<"],\"source_degree\":"<<source.degree<<",\"source_nphi\":"<<source.fourier_nphi
      <<",\"source_rings\":"<<source.fourier_rings.rows();
    if(mode=="direct_basis")std::cout<<",\"raw_reciprocity\":"<<reciprocity<<",\"linear_residual\":"<<linear_residual;
    std::cout<<",\"axes\":[["<<axes[0][0]<<","<<axes[0][1]<<","<<axes[0][2]<<"],["<<axes[1][0]<<","<<axes[1][1]<<","<<axes[1][2]
      <<"]],\"units\":\"c=eta=1; input relative midpoint U,aOmega,aE; conjugate resistance\",\"reduced11\":[";
    for(int i=0;i<11;++i){if(i)std::cout<<",";std::cout<<"[";for(int j=0;j<11;++j){if(j)std::cout<<",";std::cout<<r(i,j);}std::cout<<"]";}
    std::cout<<"]}\n";return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 2;}
}
