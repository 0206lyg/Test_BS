#ifndef SLURRY_SPD_COORDINATES_H
#define SLURRY_SPD_COORDINATES_H

#include "slurry_singular_pair.h"
#include <Eigen/LU>

// Coordinate conditioning only. These helpers do not replace, clip, or add
// dissipation to the physical target matrix being tabulated.
namespace SlurrySPDCoordinates {
namespace sc=SlurryScattering;
using sc::Vec3;
using sc::Mat3;
using sc::Mat11;
using Axes=std::array<Vec3,2>;
using Mat6=Eigen::Matrix<double,6,6>;
using Map611=Eigen::Matrix<double,6,11>;

inline Mat3 cross_matrix(const Vec3& v) {
  Mat3 c;c<<0.,-v.z(),v.y(),v.z(),0.,-v.x(),-v.y(),v.x(),0.;return c;
}

inline void require_geometry(const Axes& axes,double h,double ar) {
  if(!(ar>=1.)||!std::isfinite(ar)||!std::isfinite(h))
    throw std::invalid_argument("Invalid SPD coordinate geometry");
  for(const auto& axis:axes)if(!axis.allFinite()||std::abs(axis.norm()-1.)>1e-10)
    throw std::invalid_argument("SPD coordinate directors must be unit vectors");
}

// y=M*x, with x=[midpoint slip, a*relative angular velocity, a*strain].
// y uses contact slip in the fixed canonical x,y,z frame, a*relative angular
// velocity in that same frame, and unchanged strain. No curvature-eigenvector
// gauge enters this transform. The virtual fluid geometry must be used for both
// M and the target whose singular contribution is to be transported.
inline Mat11 contact_map(const Axes& axes,double h,double ar) {
  require_geometry(axes,h,ar);
  const auto bodies=sc::geometry(axes,h,ar);
  const auto B=sc::midpoint_map(bodies,h,ar);
  const Mat6 BB=B*B.transpose();
  const Eigen::Matrix<double,12,6> P=B.transpose()*BB.ldlt().solve(Mat6::Identity());
  const Vec3 ri=-bodies[0].center,rj=Vec3(0.,0.,h)-bodies[1].center;
  Eigen::Matrix<double,6,17> C=Eigen::Matrix<double,6,17>::Zero();
  C.block<3,3>(0,0)=-Mat3::Identity();C.block<3,3>(0,6)=Mat3::Identity();
  C.block<3,3>(0,3)=cross_matrix(ri)/ar;C.block<3,3>(0,9)=-cross_matrix(rj)/ar;
  C.block<3,3>(3,3)=-Mat3::Identity();C.block<3,3>(3,9)=Mat3::Identity();
  const auto strain=sc::strain_basis();
  for(int k=0;k<5;++k)C.block<3,1>(0,12+k)=strain[k]*(ri-rj)/ar;
  Mat11 M=Mat11::Identity();
  M.topLeftCorner<6,6>()=C.leftCols<12>()*P;
  M.topRightCorner<6,5>()=C.rightCols<5>();
  if(!M.allFinite())throw std::runtime_error("Nonfinite contact coordinate map");
  return M;
}

inline Mat11 inverse_contact_map(const Mat11& M) {
  Eigen::FullPivLU<Mat11> lu(M);lu.setThreshold(1e-12);
  if(!lu.isInvertible())throw std::runtime_error("Singular contact coordinate map");
  const Mat11 inverse=lu.inverse();
  if((M*inverse-Mat11::Identity()).norm()>1e-10)
    throw std::runtime_error("Inaccurate inverse contact coordinate map");
  return inverse;
}

class Shape {
 public:
  double ar;
  Mat11 isolated_local;
  explicit Shape(double aspect_ratio):ar(aspect_ratio) {
    if(!(ar>=1.)||!std::isfinite(ar))throw std::invalid_argument("Invalid SPD shape aspect ratio");
    isolated_local.setZero();
    const auto coeff=SlurryStress::unit_stresslet({ar,ar,1.});
    const auto drag=SlurryHydro::unit_resistance({ar,ar,1.});
    for(int k=0;k<3;++k){isolated_local(k,k)=drag.translation[k];isolated_local(k+3,k+3)=drag.rotation[k]/(ar*ar);}
    const SlurryStress::Mat3 Q{1.,0.,0.,0.,1.,0.,0.,0.,1.};
    const auto es=sc::strain_basis();
    for(int k=0;k<5;++k) {
      SlurryStress::Mat3 E{};
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)E[3*i+j]=es[k](i,j)/ar;
      const auto s=SlurryStress::stresslet(coeff,Q,1.,E,{0.,0.,0.},{0.,0.,0.});
      for(int i=0;i<3;++i)isolated_local(3+i,6+k)=isolated_local(6+k,3+i)=-s.hydro_torque[i]/ar;
      for(int i=0;i<5;++i)for(int r=0;r<3;++r)for(int c=0;c<3;++c)
        isolated_local(6+i,6+k)+=es[i](r,c)*s.total[3*r+c]/ar;
    }
  }
  Mat11 self(const Axes& axes,double actual_h) const {
    require_geometry(axes,actual_h,ar);
    const auto bodies=sc::geometry(axes,actual_h,ar);
    sc::Mat17 G=sc::Mat17::Zero();
    for(int k=0;k<2;++k) {
      Mat11 Q=Mat11::Zero();Q.block<3,3>(0,0)=bodies[k].rotation;Q.block<3,3>(3,3)=bodies[k].rotation;
      Q.bottomRightCorner<5,5>()=sc::strain_rotation(bodies[k].rotation);
      const Mat11 H=Q*isolated_local*Q.transpose();
      G.block<6,6>(6*k,6*k)=H.topLeftCorner<6,6>();
      G.block<6,5>(6*k,12)=H.topRightCorner<6,5>();G.block<5,6>(12,6*k)=H.bottomLeftCorner<5,6>();
      G.bottomRightCorner<5,5>()+=H.bottomRightCorner<5,5>();
    }
    return sc::reduced(G,sc::midpoint_map(bodies,actual_h,ar));
  }
  Mat11 contact_map(const Axes& axes,double effective_h) const {
    return SlurrySPDCoordinates::contact_map(axes,effective_h,ar);
  }
  // Positive analytic conditioning metric. The nonnegative logarithm and normal
  // square completion are used ONLY in P; the tabulated physical target remains
  // the unmodified matched matrix. P is SPD because exact isolated self is SPD.
  Mat11 preconditioner(const Axes& axes,double actual_h,double effective_h) const {
    require_geometry(axes,effective_h,ar);
    if(!(effective_h>0.))throw std::invalid_argument("Positive fluid gap required for SPD conditioning");
    namespace gr=slurry::gr_re2::graphite;
    namespace sh=SlurrySpheroidHydro;
    const auto bodies=sc::geometry(axes,effective_h,ar);
    gr::Body bi,bj;gr::Body* body[]={&bi,&bj};
    for(int k=0;k<2;++k){body[k]->axes={ar,ar,1.};for(int i=0;i<3;++i){
      body[k]->position[i]=bodies[k].center[i];
      for(int j=0;j<3;++j)body[k]->rotation[3*i+j]=bodies[k].rotation(i,j);}}
    gr::GapResult gap;gap.normal={0.,0.,1.};gap.gap=effective_h;gap.converged=true;
    for(int k=0;k<3;++k){gap.leverI[k]=-bodies[0].center[k];gap.leverJ[k]=(k==2?effective_h:0.)-bodies[1].center[k];}
    const auto pair=sh::make_pair(bi,bj,gap,1.,0.,1.,true);
    Mat6 K=Mat6::Zero();Eigen::Matrix<double,6,1> normal=Eigen::Matrix<double,6,1>::Zero();normal[2]=1.;
    if(pair.log_reference_length>effective_h) {
      for(int i=0;i<6;++i)for(int j=0;j<6;++j)K(i,j)=pair.shear[6*i+j];
      for(int k=0;k<6;++k)if(k!=2)normal[k]=pair.normal_cross[k]/pair.normal_resistance;
    }
    K+=pair.normal_resistance*normal*normal.transpose();
    Mat3 rotation;
    for(int k=0;k<3;++k){rotation(0,k)=pair.tangent1[k];rotation(1,k)=pair.tangent2[k];rotation(2,k)=pair.normal[k];}
    Mat6 local=Mat6::Zero();local.topLeftCorner<3,3>()=rotation;local.bottomRightCorner<3,3>()=rotation/ar;
    const Map611 A=local*contact_map(axes,effective_h).topRows<6>();
    const Mat11 P=self(axes,actual_h)+A.transpose()*K*A;
    return .5*(P+P.transpose()).eval();
  }
};
}
#endif
