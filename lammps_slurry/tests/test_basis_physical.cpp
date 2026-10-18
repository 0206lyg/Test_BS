// Independent source gate. The two reference fixtures are never regenerated or
// fitted here. Direct basis accuracy and matched-provider accuracy are distinct.
// The separate pair_numerical mode supplies generic numerical checks only.
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
using Axes=std::array<sc::Vec3,2>;
static constexpr double physical_limit=.05;
static constexpr double isolated_limit=.001;
static constexpr double reciprocity_limit=.01;
static std::string quote(const std::string &s) {
  std::string out="\"";
  for(char c:s) {
    if(c=='"'||c=='\\'){out+='\\';out+=c;}
    else if(c=='\n')out+="\\n";
    else if(c=='\r')out+="\\r";
    else if(c=='\t')out+="\\t";
    else out+=c;
  }
  return out+'"';
}
template<int N> static double power_error(const Eigen::Matrix<double,N,N>& a,
                                         const Eigen::Matrix<double,N,N>& b) {
  if(!a.allFinite()||!b.allFinite())throw std::runtime_error("Nonfinite energy matrix");
  Eigen::LLT<Eigen::Matrix<double,N,N>> baseline(b);
  if(baseline.info()!=Eigen::Success)throw std::runtime_error("Reference energy matrix is not SPD");
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double,N,N>> eig(a,b);
  if(eig.info()!=Eigen::Success)throw std::runtime_error("Generalized energy eigensolve failed");
  return (eig.eigenvalues().array()-1.).abs().maxCoeff();
}
static Mat5 free_strain(const Mat11 &r) {
  Eigen::LLT<Eigen::Matrix<double,6,6>> factor(r.topLeftCorner<6,6>());
  if(factor.info()!=Eigen::Success)throw std::runtime_error("Prediction velocity block is not SPD");
  const Mat5 s=r.bottomRightCorner<5,5>()-r.bottomLeftCorner<5,6>()*factor.solve(r.topRightCorner<6,5>());
  return .5*(s+s.transpose()).eval();
}
static bool isolated_check(const sc::Basis &basis,const Mat11 &reference,const char *slot) {
  Mat11 raw;
  raw.topLeftCorner<6,6>()=basis.force_torque*basis.rigid;
  raw.topRightCorner<6,5>()=basis.force_torque*basis.strain;
  raw.bottomLeftCorner<5,6>()=basis.stresslet*basis.rigid;
  raw.bottomRightCorner<5,5>()=basis.stresslet*basis.strain;
  const Mat11 symmetric=.5*(raw+raw.transpose()).eval();
  const double reciprocal=(raw-raw.transpose()).norm()/raw.norm();
  const double error=power_error<11>(symmetric,reference);
  const double stress=(raw.bottomRightCorner<5,5>()-reference.bottomRightCorner<5,5>()).norm()/reference.bottomRightCorner<5,5>().norm();
  Eigen::LLT<Mat11> factor(symmetric);
  const bool pass=factor.info()==Eigen::Success && error<=isolated_limit && reciprocal<=isolated_limit;
  std::cout<<"{\"stage\":\"isolated_units\",\"slot\":"<<quote(slot)
    <<",\"passed\":"<<(pass?"true":"false")<<",\"aspect_ratio\":"<<basis.aspect_ratio
    <<",\"all11_error\":"<<error<<",\"strain_block_error\":"<<stress
    <<",\"raw_reciprocity\":"<<reciprocal<<",\"limit\":"<<isolated_limit
    <<",\"units\":\"c=eta=1; input U,aOmega,aE; output F,T/a,S/a\"}\n"<<std::flush;
  return pass;
}
struct CheckedGrand {sc::Mat17 symmetric;double reciprocity,linear_residual,min_eigenvalue;bool pass;};
static CheckedGrand raw_check(const sc::Basis &basis,const Axes &axes,double h) {
  const auto raw=basis.evaluate(sc::geometry(axes,h,basis.aspect_ratio));
  CheckedGrand result;result.symmetric=.5*(raw.grand+raw.grand.transpose()).eval();
  result.reciprocity=raw.reciprocity;result.linear_residual=raw.linear_residual;
  Eigen::SelfAdjointEigenSolver<sc::Mat17> spectrum(result.symmetric);
  if(spectrum.info()!=Eigen::Success)throw std::runtime_error("Raw full17 eigensolve failed");
  result.min_eigenvalue=spectrum.eigenvalues().minCoeff();
  result.pass=result.reciprocity<=reciprocity_limit && result.linear_residual<=1e-8 && result.min_eigenvalue>0.;
  return result;
}
static bool print_raw_gate(const CheckedGrand &r,const std::string &name,const char *slot,double h) {
  std::cout<<"{\"stage\":\"raw_full17\",\"name\":"<<quote(name)<<",\"slot\":"<<quote(slot)
    <<",\"h_over_c\":"<<h<<",\"passed\":"<<(r.pass?"true":"false")
    <<",\"raw_reciprocity\":"<<r.reciprocity<<",\"reciprocity_limit\":"<<reciprocity_limit
    <<",\"linear_residual\":"<<r.linear_residual<<",\"min_eigenvalue\":"<<r.min_eigenvalue<<"}\n"<<std::flush;
  return r.pass;
}
// These are numerical consistency checks at selected generic poses. They do
// not compare with an independent Stokes reference or certify physical accuracy.
static bool pair_numerical_check(sc::Provider &provider,std::ostream *matrix_output) {
  if(provider.hybrid_version!=3)
    throw std::runtime_error("pair_numerical requires a V3 hybrid bundle");
  const double pi=std::acos(-1.),ar=provider.basis.aspect_ratio;
  struct Pose {const char *name;double theta_i,theta_j,phi;};
  const std::array<Pose,5> poses{{
    {"failed_orientation",std::atan(std::tan(pi/32.)/ar),std::atan(std::tan(7*pi/32.)/ar),13*pi/24.},
    {"face_face",0.,0.,0.},
    {"face_edge",0.,pi/2.,0.},
    {"tilted",.4,.8,1.1},
    {"edge_edge",pi/2.,pi/2.,pi/2.}
  }};
  for(const auto &pose:poses) {
    const Axes axes{sc::Vec3(std::sin(pose.theta_i),0.,std::cos(pose.theta_i)),
      sc::Vec3(std::sin(pose.theta_j)*std::cos(pose.phi),std::sin(pose.theta_j)*std::sin(pose.phi),std::cos(pose.theta_j))};
    provider.anchor_cache.clear();
    for(const char *component:{"raw_full17","matched_total11"}) {
      const bool raw_component=std::string(component)=="raw_full17";
      const std::array<double,2> gaps=raw_component?std::array<double,2>{{.5,1.}}:std::array<double,2>{{.001,.02}};
      for(double h:gaps) {
        const std::string prefix="{\"stage\":\"pair_numerical\",\"category\":\"numerical\",\"component\":"+
          quote(component)+",\"name\":"+quote(pose.name);
        try {
          double reciprocity=0.,residual=0.,minimum=0.;bool pass=false;
          if(raw_component) {
            const auto checked=raw_check(provider.near_basis,axes,h);
            reciprocity=checked.reciprocity;residual=checked.linear_residual;
            minimum=checked.min_eigenvalue;pass=checked.pass;
          } else {
            const auto bodies=sc::geometry(axes,h,ar);
            const Mat11 self=sc::reduced(provider.isolated(bodies),sc::midpoint_map(bodies,h,ar));
            const Mat11 total=self+provider.correction(axes,h,0.,0,1);
            if(!total.allFinite())throw std::runtime_error("Nonfinite matched total11");
            const Mat11 symmetric=.5*(total+total.transpose()).eval();
            Eigen::SelfAdjointEigenSolver<Mat11> spectrum(symmetric);
            if(spectrum.info()!=Eigen::Success)throw std::runtime_error("Matched total11 eigensolve failed");
            minimum=spectrum.eigenvalues().minCoeff();
            Eigen::LLT<Mat11> factor(symmetric);
            pass=minimum>0.&&factor.info()==Eigen::Success;
            if(matrix_output)*matrix_output<<pose.name<<" matched_total11 h="<<h<<"\n"<<symmetric<<"\n"<<std::flush;
          }
          std::cout<<prefix<<",\"h_over_c\":"<<h<<",\"theta_i\":"<<pose.theta_i
            <<",\"theta_j\":"<<pose.theta_j<<",\"phi\":"<<pose.phi
            <<",\"passed\":"<<(pass?"true":"false")<<",\"positive_total\":"<<(minimum>0.?"true":"false")
            <<",\"min_eigenvalue\":"<<minimum;
          if(raw_component)std::cout<<",\"raw_reciprocity\":"<<reciprocity<<",\"reciprocity_limit\":"<<reciprocity_limit
            <<",\"linear_residual\":"<<residual<<",\"residual_limit\":1e-8";
          std::cout<<"}\n"<<std::flush;
          if(!pass)return false;
        } catch(const std::exception &e) {
          std::cout<<prefix<<",\"h_over_c\":"<<h<<",\"passed\":false,\"error\":"<<quote(e.what())<<"}\n"<<std::flush;
          return false;
        }
      }
    }
  }
  return true;
}
int main(int argc,char **argv) {
  try {
    if(argc!=5&&argc!=6)throw std::runtime_error("Usage: basis_physical_probe basis.ssb reference.txt direct_basis|matched_provider|isolated_only|pair_numerical film_grid [matrix_output]");
    std::ofstream matrix_output;
    if(argc==6) {
      matrix_output.open(argv[5]);
      if(!matrix_output)throw std::runtime_error("Cannot open optional matrix output");
      matrix_output<<std::setprecision(17);
    }
    const std::string mode=argv[3];
    if(mode!="direct_basis"&&mode!="matched_provider"&&mode!="isolated_only"&&mode!="pair_numerical")throw std::runtime_error("Invalid evaluation mode");
    std::cout<<std::setprecision(17);
    sc::Provider provider;provider.load(argv[1]);provider.film_grid=std::stoi(argv[4]);
    if(provider.film_grid<21||provider.film_grid%2==0)throw std::runtime_error("Film grid must be odd and at least21");
    if(mode=="matched_provider"&&!provider.hybrid)throw std::runtime_error("matched_provider requires an actual matched hybrid bundle");
    const sc::Basis &direct_basis=provider.hybrid?provider.near_basis:provider.basis;
    const char *direct_slot=provider.hybrid?"near":"primary";
    std::cout<<"{\"stage\":\"metadata\",\"evaluated_path\":"<<quote(mode)
      <<",\"gate_category\":"<<quote(mode=="isolated_only"||mode=="pair_numerical"?"numerical":"physical")
      <<",\"compiled_source_fingerprint\":"<<quote(SLURRY_NUMERICAL_SOURCE_FINGERPRINT)
      <<",\"aspect_ratio\":"<<provider.basis.aspect_ratio<<",\"film_grid\":"<<provider.film_grid
      <<",\"hybrid\":"<<(provider.hybrid?"true":"false")
      <<",\"hybrid_version\":"<<provider.hybrid_version
      <<",\"matching_policy\":"<<quote(provider.hybrid_version==3?"direct_near_at_heff_ge_0.5; unchanged_first_film_segment_below_0.5":"serialized_anchor_matching")
      <<",\"basis_slot\":"<<quote(mode=="direct_basis"?direct_slot:"matched")
      <<",\"physical_limit\":"<<physical_limit<<"}\n";
    if(provider.hybrid_version!=3&&!isolated_check(provider.basis,provider.isolated_local,"primary"))return 2;
    if(provider.hybrid&&!isolated_check(provider.near_basis,provider.isolated_local,"near"))return 2;
    if(mode=="isolated_only")return 0;
    if(mode=="pair_numerical")return pair_numerical_check(provider,matrix_output.is_open()?&matrix_output:nullptr)?0:2;
    if(std::abs(provider.basis.aspect_ratio-10.)>1e-12)
      throw std::runtime_error("The frozen independent pair fixtures cover AR10 only; no substitution allowed");
    std::ifstream input(argv[2]);std::string magic;int count=0;
    if(!(input>>magic>>count)||magic!="SLURRY_INDEPENDENT_REFERENCE_V1"||count!=2)
      throw std::runtime_error("Invalid independent reference fixture");
    for(int k=0;k<count;++k) {
      std::string name;double h,legacy_limit,legacy_stress_limit;Axes axes;Mat11 reference,self;
      input>>name>>h>>legacy_limit>>legacy_stress_limit;
      for(auto &axis:axes)for(int j=0;j<3;++j)input>>axis[j];
      for(int i=0;i<11;++i)for(int j=0;j<11;++j)input>>reference(i,j);
      for(int i=0;i<11;++i)for(int j=0;j<11;++j)input>>self(i,j);
      if(!input)throw std::runtime_error("Truncated reference fixture");
      const auto bodies=sc::geometry(axes,h,provider.basis.aspect_ratio);
      const auto map=sc::midpoint_map(bodies,h,provider.basis.aspect_ratio);
      const Mat11 runtime_self=sc::reduced(provider.isolated(bodies),map);
      const double self_error=(runtime_self-self).norm()/self.norm();
      if(self_error>1e-10)throw std::runtime_error("Fixture unit/frame/isolated-self mismatch");
      Mat11 predicted;
      if(mode=="direct_basis") {
        const auto raw=raw_check(direct_basis,axes,h);
        if(!print_raw_gate(raw,name,direct_slot,h))return 2;
        predicted=sc::reduced(raw.symmetric,map);
      } else {
        if(provider.hybrid_version==3) {
          if(h>=.5) {
            const auto raw=raw_check(provider.near_basis,axes,h);
            if(!print_raw_gate(raw,name,"near_direct",h))return 2;
          } else {
            for(double anchor_gap:{.5,1.}) {
              const auto raw=raw_check(provider.near_basis,axes,anchor_gap);
              if(!print_raw_gate(raw,name,"near_film_anchor",anchor_gap))return 2;
            }
          }
        } else {
        // Check every source anchor used by this serialized matching model.
        // A direct check at the fixture gap alone would not check this path.
        for(int anchor=0;anchor<provider.anchor_count;++anchor) {
          const bool far=anchor==provider.anchor_count-1;
          const double anchor_gap=provider.anchor_gaps[anchor];
          const auto raw=raw_check(far?provider.basis:provider.near_basis,axes,anchor_gap);
          if(!print_raw_gate(raw,name,far?"far_anchor":"near_anchor",anchor_gap))return 2;
        }
        }
        predicted=provider.correction(axes,h,0.,2*k,2*k+1)+runtime_self;
      }
      Eigen::LLT<Mat11> positive(predicted);
      const double error=power_error<11>(predicted,reference);
      const double stress=positive.info()==Eigen::Success?power_error<5>(free_strain(predicted),free_strain(reference)):1e300;
      const double normal=predicted(2,2)/reference(2,2)-1.;
      const bool pass=positive.info()==Eigen::Success&&error<=physical_limit&&stress<=physical_limit;
      if(matrix_output.is_open()) {
        matrix_output<<name<<"\n"<<predicted<<"\n"<<std::flush;
      }
      std::cout<<"{\"stage\":\"independent_physical\",\"name\":"<<quote(name)
        <<",\"h_over_c\":"<<h<<",\"passed\":"<<(pass?"true":"false")
        <<",\"all11_error\":"<<error<<",\"free_strain_error\":"<<stress
        <<",\"normal_error\":"<<normal<<",\"isolated_self_error\":"<<self_error
        <<",\"positive_total\":"<<(positive.info()==Eigen::Success?"true":"false")
        <<",\"limit\":"<<physical_limit<<"}\n"<<std::flush;
      if(!pass)return 2;
    }
    std::string extra;if(input>>extra)throw std::runtime_error("Unexpected trailing reference data");
    return 0;
  } catch(const std::exception &e) {
    std::cout<<"{\"stage\":\"exception\",\"passed\":false,\"error\":"<<quote(e.what())<<"}\n";
    return 2;
  }
}
