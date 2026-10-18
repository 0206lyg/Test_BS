#include "slurry_scattering_provider.h"
#include <iomanip>
#include <iostream>
// The 300 nm thickness gives c=150 nm, so h/c=.02 is a 3 nm gap.
// This pose previously exposed the finite-film quadrature defect.
int main(int argc,char**argv) {
  try {
    if(argc!=3)throw std::runtime_error("usage: test_v3_preservation V2.ssb V3.ssb");
    SlurryScattering::Provider old_model,new_model;old_model.load(argv[1]);new_model.load(argv[2]);
    if(old_model.hybrid_version!=2||new_model.hybrid_version!=3||new_model.anchor_count!=2)
      throw std::runtime_error("Expected V2 and V3 source bundles");
    const double ti=.03490658503988659,tj=.46583237567450148,phi=.52359877559829882;
    const std::array<SlurryScattering::Vec3,2> axes{{
      {std::sin(ti),0.,std::cos(ti)},
      {std::sin(tj)*std::cos(phi),std::sin(tj)*std::sin(phi),std::cos(tj)}}};
    const auto a=old_model.correction(axes,.02,.001),b=new_model.correction(axes,.02,.001);
    const double absolute=(a-b).cwiseAbs().maxCoeff(),relative=(a-b).norm()/std::max(a.norm(),1e-300);
    const bool passed=absolute==0.;
    std::cout<<std::setprecision(17)<<"{\"gate\":\"V3_near_gap_exact_preservation\",\"gap_nm\":3,\"h_over_c\":0.02,\"epsilon_over_c\":0.001,\"film_grid\":161,\"maximum_absolute_matrix_difference\":"<<absolute<<",\"relative_matrix_difference\":"<<relative<<",\"passed\":"<<(passed?"true":"false")<<"}\n";
    return passed?0:2;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
