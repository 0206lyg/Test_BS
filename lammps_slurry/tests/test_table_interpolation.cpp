// Measures lookup error against its direct source, not independent Stokes error.
#include "slurry_spd_table.h"
#include "slurry_scattering_provider.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#ifndef SLURRY_NUMERICAL_SOURCE_FINGERPRINT
#define SLURRY_NUMERICAL_SOURCE_FINGERPRINT "unbound"
#endif
namespace sc=SlurryScattering;
namespace st=SlurrySPDTable;
using Mat11=sc::Mat11;
using Mat5=Eigen::Matrix<double,5,5>;
using Axes=std::array<sc::Vec3,2>;
static std::string quote(const std::string &s) {
  std::string out="\"";
  for(char c:s) {if(c=='\"'||c=='\\'){out+='\\';out+=c;}
    else if(c=='\n')out+="\\n";else if(c=='\r')out+="\\r";
    else if(c=='\t')out+="\\t";else out+=c;}
  return out+'\"';
}
static Axes axes_at(double ti,double tj,double phi) {
  return {{sc::Vec3(std::sin(ti),0.,std::cos(ti)),
           sc::Vec3(std::sin(tj)*std::cos(phi),std::sin(tj)*std::sin(phi),std::cos(tj))}};
}
template<int N> static double energy_error(const Eigen::Matrix<double,N,N> &a,
                                          const Eigen::Matrix<double,N,N> &b) {
  Eigen::LLT<Eigen::Matrix<double,N,N>> fa(a),fb(b);
  if(!a.allFinite()||!b.allFinite()||fa.info()!=Eigen::Success||fb.info()!=Eigen::Success)
    throw std::runtime_error("Nonfinite or nonpositive total resistance");
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double,N,N>> eig(a,b);
  if(eig.info()!=Eigen::Success)throw std::runtime_error("Energy error eigensolve failed");
  return (eig.eigenvalues().array()-1.).abs().maxCoeff();
}
static Mat5 free_strain(const Mat11 &r) {
  Eigen::LLT<Eigen::Matrix<double,6,6>> f(r.topLeftCorner<6,6>());
  if(f.info()!=Eigen::Success)throw std::runtime_error("Velocity block is not positive");
  const Mat5 s=r.bottomRightCorner<5,5>()-r.bottomLeftCorner<5,6>()*f.solve(r.topRightCorner<6,5>());
  return .5*(s+s.transpose()).eval();
}
struct Pose{const char *name;double ti,tj,phi,h;};
int main(int argc,char **argv) {
  try {
    if(argc!=4)throw std::runtime_error("Usage: slurry_table_interpolation_probe basis.ssb table.sft film_grid");
    sc::Provider direct;direct.load(argv[1]);direct.film_grid=std::stoi(argv[3]);
    st::Provider fast;fast.load(argv[2]);
    if(direct.film_grid<21||direct.film_grid%2==0)throw std::runtime_error("Invalid film grid");
    if(std::abs(direct.basis.aspect_ratio-fast.ar)>1e-10*fast.ar)throw std::runtime_error("Basis/table AR mismatch");
    const double pi=std::acos(-1.),epsilon=fast.hmin;
    const double upper=fast.hmax-epsilon;
    auto h=[&](double requested){return std::min(requested,.97*upper);};
    // Frozen before looking at the regenerated basis/table. Includes both sides
    // of the azimuth seam, poles and the direct-model outer branch if covered.
    const std::vector<Pose> poses={
      {"asymmetric_3nm",.073,.137,.79,h(.020)},
      {"near_face",std::atan(.267/fast.ar),std::atan(.913/fast.ar),-.417,h(.0027)},
      {"oblique",.5279,.1733,-1.437,h(.0937)},
      {"face_edge",.0397,1.4893,-.931,h(.0719)},
      {"near_edges",1.5297,1.3979,-2.641,h(.0039)},
      {"azimuth_seam_left",.3911,1.1739,-pi+1e-7,h(.137)},
      {"azimuth_seam_right",.3911,1.1739,pi-1e-7,h(.137)},
      {"north_pole",0.,.2831,1.843,h(.273)},
      {"outer_gap",.3911,1.1739,2.287,fast.hmax>2.5 ? .5*(2.5+fast.hmax)-epsilon : .827*upper}};
    std::cout<<std::setprecision(17)
      <<"{\"stage\":\"metadata\",\"aspect_ratio\":"<<fast.ar
      <<",\"film_grid\":"<<direct.film_grid<<",\"limit\":0.01,\"case_count\":"<<poses.size()
      <<",\"compiled_source_fingerprint\":"<<quote(SLURRY_NUMERICAL_SOURCE_FINGERPRINT)
      <<",\"reference\":\"direct_matched_source_not_independent_Stokes\"}\n"<<std::flush;
    int index=0;
    for(const auto &p:poses) {
      const Axes axes=axes_at(p.ti,p.tj,p.phi);
      const Mat11 self=fast.shape->self(axes,p.h);
      const Mat11 reference=self+direct.correction(axes,p.h,epsilon,100+index,0);
      const Mat11 predicted=self+fast.correction(axes,p.h,epsilon);
      const double error=energy_error<11>(predicted,reference);
      const double strain=energy_error<5>(free_strain(predicted),free_strain(reference));
      const Mat11 periodic=self+fast.correction(axes_at(p.ti,p.tj,p.phi+2*pi),p.h,epsilon);
      Axes flipped=axes;flipped[0]=-flipped[0];flipped[1]=-flipped[1];
      const Mat11 reversed=self+fast.correction(flipped,p.h,epsilon);
      const double symmetry=std::max((periodic-predicted).norm(),(reversed-predicted).norm())/predicted.norm();
      const bool passed=error<=.01&&strain<=.01&&symmetry<=1e-10;
      std::cout<<"{\"stage\":\"interpolation\",\"name\":"<<quote(p.name)
        <<",\"passed\":"<<(passed?"true":"false")<<",\"theta_i\":"<<p.ti
        <<",\"theta_j\":"<<p.tj<<",\"phi\":"<<p.phi<<",\"h_over_c\":"<<p.h
        <<",\"epsilon_over_c\":"<<epsilon<<",\"all11_error\":"<<error
        <<",\"free_strain_error\":"<<strain<<",\"symmetry_error\":"<<symmetry
        <<",\"limit\":0.01}\n"<<std::flush;
      if(!passed)return 2;
      ++index;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cout<<"{\"stage\":\"exception\",\"passed\":false,\"error\":"<<quote(e.what())<<"}\n";
    return 2;
  }
}
