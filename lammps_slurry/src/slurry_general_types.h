// Geometry-only general oblate hydrodynamics; internal units c=eta=1.
#ifndef SLURRY_GENERAL_TYPES_H
#define SLURRY_GENERAL_TYPES_H
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <stdexcept>
namespace SlurryGeneral {
using Vec3=Eigen::Vector3d;
using Mat3=Eigen::Matrix3d;
using Mat11=Eigen::Matrix<double,11,11>;
using Mat17=Eigen::Matrix<double,17,17>;
using Mat22=Eigen::Matrix<double,22,22>;
struct Geometry {
  double ar=1.,gap=1.;
  std::array<Vec3,2> centers{{Vec3::Zero(),Vec3(0,0,3)}};
  std::array<Mat3,2> rotations{{Mat3::Identity(),Mat3::Identity()}};
};
// Native 11 order: U, omega, +E in the orthonormal STF basis.
// Public 17 order: Ui,a*omega_i,Uj,a*omega_j,a*E (a=AR here).
// Public matrices multiply eta*c, not eta*a. The LAMMPS adapter converts it.
inline const std::array<Mat3,5>& strain_basis() {
  static const auto b=[](){std::array<Mat3,5> e;
    for(auto& v:e)v.setZero();
    const double s2=1/std::sqrt(2.),s6=1/std::sqrt(6.);
    e[0](0,0)=s2;e[0](1,1)=-s2;
    e[1](0,0)=e[1](1,1)=s6;e[1](2,2)=-2*s6;
    e[2](0,1)=e[2](1,0)=s2;e[3](0,2)=e[3](2,0)=s2;
    e[4](1,2)=e[4](2,1)=s2;return e;}();return b;
}
inline Mat3 axis_rotation(Vec3 p) {
  if(!p.allFinite()||p.norm()==0)throw std::invalid_argument("Invalid spheroid director");
  p.normalize();Vec3 x=std::abs(p.x())<.9?Vec3::UnitX():Vec3::UnitY();
  x=(x-p*p.dot(x)).normalized();Mat3 q;q.col(0)=x;q.col(1)=p.cross(x);q.col(2)=p;return q;
}
inline Mat3 shape_tensor(double a,const Mat3& q) {
  return a*a*Mat3::Identity()+(1-a*a)*q.col(2)*q.col(2).transpose();
}
// Canonical closest points are (0,0,0) and (0,0,h), normal +z.
inline Geometry geometry_from_axes(double ar,double h,const Vec3& pi,const Vec3& pj) {
  if(!(ar>=1&&h>0)||!std::isfinite(ar)||!std::isfinite(h))
    throw std::invalid_argument("General spheroid model needs AR>=1 and positive gap");
  Geometry g;g.ar=ar;g.gap=h;g.rotations={axis_rotation(pi),axis_rotation(pj)};
  for(int b=0;b<2;++b){const Mat3 a=shape_tensor(ar,g.rotations[b]);
    const Vec3 lever=a.col(2)/std::sqrt(a(2,2));
    if(b==0)g.centers[b]=-lever;else g.centers[b]=Vec3(0,0,h)+lever;}
  return g;
}
}
#endif
