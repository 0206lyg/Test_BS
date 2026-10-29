#ifndef SLURRY_SPD_COORDINATES17_H
#define SLURRY_SPD_COORDINATES17_H
#include "slurry_spd_coordinates.h"

// Full grand resistance: no common-motion minimization or eleven-mode lift.
// The positive local completion below conditions coordinates ONLY. It never
// changes the raw physical matrix that the table encodes.
namespace SlurrySPDCoordinates17 {
namespace sc=SlurryScattering;
using sc::Mat17;
using sc::Vec3;
using Axes=std::array<Vec3,2>;
class Shape {
 public:
  double ar;
  SlurrySPDCoordinates::Shape isolated;
  explicit Shape(double aspect_ratio):ar(aspect_ratio),isolated(aspect_ratio){}
  Mat17 self(const Axes& axes,double h)const {
    SlurrySPDCoordinates::require_geometry(axes,h,ar);
    const auto bodies=sc::geometry(axes,h,ar);Mat17 G=Mat17::Zero();
    for(int k=0;k<2;++k){
      sc::Mat11 Q=sc::Mat11::Zero();Q.block<3,3>(0,0)=bodies[k].rotation;
      Q.block<3,3>(3,3)=bodies[k].rotation;Q.bottomRightCorner<5,5>()=sc::strain_rotation(bodies[k].rotation);
      const sc::Mat11 H=Q*isolated.isolated_local*Q.transpose();
      G.block<6,6>(6*k,6*k)=H.topLeftCorner<6,6>();
      G.block<6,5>(6*k,12)=H.topRightCorner<6,5>();G.block<5,6>(12,6*k)=H.bottomLeftCorner<5,6>();
      G.bottomRightCorner<5,5>()+=H.bottomRightCorner<5,5>();
    }
    return .5*(G+G.transpose()).eval();
  }
  Mat17 preconditioner(const Axes& axes,double h)const {
    SlurrySPDCoordinates::require_geometry(axes,h,ar);
    const auto canonical=SlurryFastSingular::canonical_pair(axes,h,ar);
    const auto& pair=canonical.pair;
    Eigen::Matrix<double,6,6> K=Eigen::Matrix<double,6,6>::Zero();
    Eigen::Matrix<double,6,1> normal=Eigen::Matrix<double,6,1>::Zero();normal[2]=1.;
    if(pair.log_reference_length>h){
      for(int i=0;i<6;++i)for(int j=0;j<6;++j)K(i,j)=pair.shear[6*i+j];
      for(int k=0;k<6;++k)if(k!=2)normal[k]=pair.normal_cross[k]/pair.normal_resistance;
    }
    K+=pair.normal_resistance*normal*normal.transpose();
    const auto T=SlurryFastSingular::local_map(pair,ar);
    const Mat17 P=self(axes,h)+T.transpose()*K*T;
    return .5*(P+P.transpose()).eval();
  }
  Mat17 cholesky(const Axes& axes,double h)const {
    const Mat17 P=preconditioner(axes,h);Eigen::LLT<Mat17> factor(P);
    if(!P.allFinite()||factor.info()!=Eigen::Success)
      throw std::runtime_error("Full17 conditioning metric is not SPD");
    return factor.matrixL();
  }
};
}
#endif
