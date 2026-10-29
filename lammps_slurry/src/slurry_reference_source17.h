#ifndef SLURRY_REFERENCE_SOURCE17_H
#define SLURRY_REFERENCE_SOURCE17_H

#include "slurry_spd_coordinates17.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Explicit imported reference matrices, not a Stokes Basis or a physical
// qualification certificate. All matrices use c=eta=1 and the existing full17
// work-conjugate convention. The caller must retain the original evidence.
namespace SlurryReferenceSource17 {
namespace sc=SlurryScattering;
using sc::Mat17;
using sc::Mat3;
using sc::Vec3;
using Axes=std::array<Vec3,2>;
inline constexpr const char* source_magic="SLURRY_REFERENCE_SOURCE17_V1";
inline constexpr const char* model_revision="JANOSCHEK_MATCHED_HARDFLOOR_FULL17_V4";
inline constexpr const char* convention="slurry_full17_c_eta_1_v1";
inline constexpr const char* dissipative_model_revision="JANOSCHEK_DISSIPATIVE_INCREMENT_FULL17_V1";
inline constexpr double reciprocity_limit=1e-3;
inline constexpr double pose_tolerance=1e-10;
enum class MatchedModel {ConstantResidual,DissipativeIncrement};
inline const char* model_revision_for(MatchedModel model) {
  return model==MatchedModel::DissipativeIncrement?dissipative_model_revision:model_revision;
}

inline std::string fnv1a64(const std::string& bytes) {
  std::uint64_t hash=14695981039346656037ull;
  for(unsigned char c:bytes){hash^=c;hash*=1099511628211ull;}
  std::ostringstream out;out<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;return out.str();
}
inline bool sha256_token(const std::string& value) {
  return value.size()==64&&std::all_of(value.begin(),value.end(),[](unsigned char c){return std::isxdigit(c)!=0;});
}
inline void word(std::istream& in,const char* expected) {
  std::string value;if(!(in>>value)||value!=expected)
    throw std::runtime_error(std::string("Reference source expected token: ")+expected);
}
inline Mat17 whiten(const Mat17& matrix,const Mat17& lower) {
  const Mat17 left=lower.triangularView<Eigen::Lower>().solve(matrix);
  return lower.triangularView<Eigen::Lower>().solve(left.transpose()).transpose();
}
inline double positive_minimum(const Mat17& matrix) {
  if(!matrix.allFinite())throw std::runtime_error("Nonfinite reference resistance");
  const Mat17 symmetric=.5*(matrix+matrix.transpose()).eval();
  Eigen::LLT<Mat17> llt(symmetric);
  Eigen::SelfAdjointEigenSolver<Mat17> eig(symmetric,Eigen::EigenvaluesOnly);
  if(llt.info()!=Eigen::Success||eig.info()!=Eigen::Success||
     !eig.eigenvalues().allFinite()||!(eig.eigenvalues().minCoeff()>0.))
    throw std::runtime_error("Reference/matched total is not SPD; eigenvalue clipping is disabled");
  return eig.eigenvalues().minCoeff();
}

struct Pose {
  double theta_i=0.,theta_j=0.,phi=0.,anchor_h=0.;
  std::string source_json_sha256,solution_sha256;
  Axes axes;
  Mat17 raw=Mat17::Zero(),anchor_total=Mat17::Zero(),anchor_residual=Mat17::Zero();
  SlurryFastSingular::LocalMap contact_map=SlurryFastSingular::LocalMap::Zero();
  Eigen::Matrix<double,6,6> shear_coefficient=Eigen::Matrix<double,6,6>::Zero();
  Eigen::Matrix<double,6,1> cross_coefficient=Eigen::Matrix<double,6,1>::Zero();
  double anchor_normal_resistance=0.;
  double raw_reciprocity=0.,metric_reciprocity=0.,minimum_metric_eigenvalue=0.;
};

inline Axes canonical_axes(double ti,double tj,double azimuth,bool require_sparse_gauge=true) {
  const double pi=std::acos(-1.);
  if(!std::isfinite(ti)||!std::isfinite(tj)||!std::isfinite(azimuth)||
     ti<0.||tj<0.||ti>.5*pi||tj>.5*pi||azimuth< -pi||azimuth>=pi)
    throw std::runtime_error("Reference pose is outside the canonical angular domain");
  // A polar first director leaves the second director to fix the z gauge.
  // A polar second director has no azimuth. Do not rotate a supplied matrix
  // silently to repair either convention.
  if(require_sparse_gauge&&(std::sin(ti)<=1e-13||std::sin(tj)<=1e-13)&&std::abs(azimuth)>1e-13)
    throw std::runtime_error("Reference polar pose must have canonical phi=0");
  return Axes{{{std::sin(ti),0.,std::cos(ti)},
               {std::sin(tj)*std::cos(azimuth),std::sin(tj)*std::sin(azimuth),std::cos(tj)}}};
}
inline bool equivalent_pose(const Pose& a,const Pose& b) {
  for(double sign:{1.,-1.}){
    Mat3 rotation=Mat3::Identity();rotation(0,0)=rotation(1,1)=sign;
    bool equal=true;
    for(int k=0;k<2;++k){const Vec3 axis=rotation*b.axes[k];
      if((a.axes[k]*a.axes[k].transpose()-axis*axis.transpose()).norm()>pose_tolerance)equal=false;
    }
    if(equal)return true;
  }
  return false;
}

class Source {
 public:
  double ar=0.,c_m=0.,floor_m=0.,hmin=0.,hmax=0.;
  std::string source_fnv1a64;
  std::vector<Pose> poses;
  std::unique_ptr<SlurrySPDCoordinates17::Shape> shape;

  void load(const std::string& path,bool require_sparse_gauge=true) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file)throw std::runtime_error("Cannot open reference source: "+path);
    const auto size=file.tellg();
    if(size<=0||size>std::streamoff(128*1024*1024))throw std::runtime_error("Reference source must be between 1 byte and 128 MiB");
    std::string bytes(std::size_t(size),'\0');file.seekg(0);file.read(&bytes[0],size);
    if(!file)throw std::runtime_error("Cannot read complete reference source");
    source_fnv1a64=fnv1a64(bytes);std::istringstream in(bytes);
    word(in,source_magic);word(in,"MODEL");word(in,model_revision);
    word(in,"CONVENTION");word(in,convention);word(in,"STATUS");word(in,"PROVISIONAL");word(in,"0");
    word(in,"UNITS");in>>c_m>>floor_m;word(in,"AR");in>>ar;word(in,"POSES");
    int count=0;in>>count;
    if(!in||!std::isfinite(c_m)||!std::isfinite(floor_m)||!std::isfinite(ar)||
       !(c_m>0.&&floor_m>0.&&ar>=1.)||count<1||count>16384)
      throw std::runtime_error("Invalid reference units, shape or pose count");
    hmin=floor_m/c_m;
    if(!std::isfinite(hmin)||!(hmin>0.))throw std::runtime_error("Invalid reference hard floor");
    shape=std::make_unique<SlurrySPDCoordinates17::Shape>(ar);poses.clear();poses.reserve(count);
    for(int k=0;k<count;++k){
      Pose p;word(in,"POSE");in>>p.theta_i>>p.theta_j>>p.phi>>p.anchor_h>>p.source_json_sha256>>p.solution_sha256;
      if(!in||!std::isfinite(p.anchor_h)||!(p.anchor_h>hmin)||
         !sha256_token(p.source_json_sha256)||(p.solution_sha256!="-"&&!sha256_token(p.solution_sha256)))
        throw std::runtime_error("Invalid reference anchor or provenance hash");
      p.axes=canonical_axes(p.theta_i,p.theta_j,p.phi,require_sparse_gauge);
      if(k==0)hmax=p.anchor_h;
      else if(std::abs(p.anchor_h-hmax)>1e-12*std::max(p.anchor_h,hmax))
        throw std::runtime_error("Sparse reference V1 requires one common anchor gap; differing anchors need a new format");
      if(require_sparse_gauge)for(const auto& previous:poses)if(equivalent_pose(previous,p))
        throw std::runtime_error("Duplicate equivalent canonical reference pose");
      word(in,"RAW17");for(int i=0;i<17;++i)for(int j=0;j<17;++j)in>>p.raw(i,j);word(in,"END_POSE");
      if(!in||!p.raw.allFinite()||!(p.raw.norm()>0.))throw std::runtime_error("Invalid raw17 reference matrix");
      const Mat17 self=shape->self(p.axes,p.anchor_h);Eigen::LLT<Mat17> self_llt(self);
      if(self_llt.info()!=Eigen::Success)throw std::runtime_error("Reference isolated metric is not SPD");
      const Mat17 scaled=whiten(p.raw,self_llt.matrixL());
      p.raw_reciprocity=(p.raw-p.raw.transpose()).norm()/p.raw.norm();
      p.metric_reciprocity=(scaled-scaled.transpose()).norm()/scaled.norm();
      if(!std::isfinite(p.raw_reciprocity)||!std::isfinite(p.metric_reciprocity)||
         p.raw_reciprocity>reciprocity_limit||p.metric_reciprocity>reciprocity_limit)
        throw std::runtime_error("Raw reference reciprocity exceeds 1e-3; symmetrization cannot hide this failure");
      p.anchor_total=.5*(p.raw+p.raw.transpose()).eval();positive_minimum(p.anchor_total);
      p.minimum_metric_eigenvalue=positive_minimum(.5*(scaled+scaled.transpose()).eval());
      p.anchor_residual=p.anchor_total-self-SlurryFastSingular::leading17(p.axes,p.anchor_h,ar);
      if(!p.anchor_residual.allFinite())throw std::runtime_error("Nonfinite matched reference residual");
      const auto anchor_pair=SlurryFastSingular::canonical_pair(p.axes,p.anchor_h,ar).pair;
      p.anchor_normal_resistance=anchor_pair.normal_resistance;
      p.contact_map=SlurryFastSingular::local_map(anchor_pair,ar);
      const double unit_gap=anchor_pair.log_reference_length/std::exp(1.);
      const auto unit_pair=SlurryFastSingular::canonical_pair(p.axes,unit_gap,ar).pair;
      const double unit_log=std::log(unit_pair.log_reference_length/unit_gap);
      if(!(unit_log>0.)||!std::isfinite(unit_log)||!(p.anchor_normal_resistance>0.)||
         !std::isfinite(p.anchor_normal_resistance))throw std::runtime_error("Invalid dissipative increment coefficients");
      for(int i=0;i<6;++i){p.cross_coefficient[i]=unit_pair.normal_cross[i]/unit_log;
        for(int j=0;j<6;++j)p.shear_coefficient(i,j)=unit_pair.shear[6*i+j]/unit_log;}
      if(!p.contact_map.allFinite()||!p.shear_coefficient.allFinite()||!p.cross_coefficient.allFinite()||p.cross_coefficient[2]!=0.)
        throw std::runtime_error("Invalid dissipative increment local modes");
      const auto unit_map=SlurryFastSingular::local_map(unit_pair,ar);
      if((unit_map-p.contact_map).norm()>1e-12*std::max(1.,p.contact_map.norm()))
        throw std::runtime_error("Dissipative increment requires a gap-independent contact map");
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> shear_eig(p.shear_coefficient,Eigen::EigenvaluesOnly);
      if(shear_eig.info()!=Eigen::Success||!shear_eig.eigenvalues().allFinite()||
         shear_eig.eigenvalues().minCoeff()< -1e-12*std::max(1.,p.shear_coefficient.norm()))
        throw std::runtime_error("Unit-log tangential Gram matrix lost positive semidefiniteness; clipping disabled");
      poses.push_back(std::move(p));
    }
    word(in,"END");std::string extra;if(in>>extra)throw std::runtime_error("Extra data after reference source END");
  }

  Mat17 constant_candidate(std::size_t index,double actual_h)const {
    if(!shape||index>=poses.size()||!std::isfinite(actual_h))throw std::runtime_error("Invalid reference total query");
    const auto& p=poses[index];const double h=std::max(actual_h,hmin);
    if(h>p.anchor_h*(1.+1e-12))throw std::runtime_error("Reference continuation above its anchor is disabled");
    // Retain the supplied anchor exactly; continuation is provisional at every
    // other gap, regardless of how accurately it is stored/interpolated.
    if(h==p.anchor_h)return p.anchor_total;
    const Mat17 raw=shape->self(p.axes,h)+SlurryFastSingular::leading17(p.axes,h,ar)+p.anchor_residual;
    const Mat17 result=.5*(raw+raw.transpose()).eval();
    if(!result.allFinite())throw std::runtime_error("Nonfinite constant-residual candidate");
    return result;
  }

  Mat17 total(std::size_t index,double actual_h,MatchedModel model=MatchedModel::ConstantResidual)const {
    if(model==MatchedModel::ConstantResidual){const Mat17 result=constant_candidate(index,actual_h);positive_minimum(result);return result;}
    if(!shape||index>=poses.size()||!std::isfinite(actual_h))throw std::runtime_error("Invalid reference total query");
    const auto& p=poses[index];const double h=std::max(actual_h,hmin);
    if(h>p.anchor_h)throw std::runtime_error("Dissipative continuation above its anchor is disabled");
    if(h==p.anchor_h)return p.anchor_total;
    // This is a separately named finite-gap closure, not an alteration of the
    // raw Janoschek operator. Its extra db*db^T/dn term vanishes at the anchor
    // and is O(h log^2(h)) at small h; retained singular coefficients survive.
    const double relative_decrease=(p.anchor_h-h)/p.anchor_h;
    const double t=relative_decrease<.5?-std::log1p(-relative_decrease):std::log(p.anchor_h)-std::log(h);
    const double dn=p.anchor_normal_resistance*std::expm1(t);
    if(!(t>0.)||!(dn>0.)||!std::isfinite(t)||!std::isfinite(dn))
      throw std::runtime_error("Invalid dissipative normal increment");
    Eigen::Matrix<double,6,1> mode=p.cross_coefficient*(t/std::sqrt(dn));
    mode[2]=std::sqrt(dn);
    const Eigen::Matrix<double,6,6> increment=t*p.shear_coefficient+mode*mode.transpose();
    const Mat17 raw=p.anchor_total+p.contact_map.transpose()*increment*p.contact_map;
    const Mat17 result=.5*(raw+raw.transpose()).eval();positive_minimum(result);return result;
  }
};
}
#endif
