#ifndef SLURRY_GR_RE2_GRAPHITE_PARTICLE_MATH_H
#define SLURRY_GR_RE2_GRAPHITE_PARTICLE_MATH_H

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

// SLURRY SCOPE BEGIN
namespace slurry { namespace gr_re2 {

namespace graphite {
using Vec3 = std::array<double,3>;
using Mat3 = std::array<double,9>; // Row-major, body coordinates -> laboratory.
inline constexpr Mat3 identity3{1.,0.,0.,0.,1.,0.,0.,0.,1.};
inline Vec3 add(const Vec3& a,const Vec3& b) {return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
inline Vec3 sub(const Vec3& a,const Vec3& b) {return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
inline Vec3 scale(const Vec3& a,double s) {return {s*a[0],s*a[1],s*a[2]};}
inline double dot(const Vec3& a,const Vec3& b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline Vec3 cross(const Vec3& a,const Vec3& b) {return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline double norm(const Vec3& a) {return std::sqrt(dot(a,a));}
inline Vec3 normalized(const Vec3& a) {const double n=norm(a);if (!(n>0.)||!std::isfinite(n)) throw std::domain_error("Cannot normalize particle vector");return scale(a,1./n);}
inline Vec3 mul(const Mat3& m,const Vec3& a) {return {m[0]*a[0]+m[1]*a[1]+m[2]*a[2],m[3]*a[0]+m[4]*a[1]+m[5]*a[2],m[6]*a[0]+m[7]*a[1]+m[8]*a[2]};}
inline Mat3 transpose(const Mat3& m) {return {m[0],m[3],m[6],m[1],m[4],m[7],m[2],m[5],m[8]};}
inline Mat3 multiply(const Mat3& a,const Mat3& b) {Mat3 c{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)c[3*i+j]+=a[3*i+k]*b[3*k+j];return c;}
inline Mat3 rotatedDiagonal(const Mat3& r,const Vec3& d) {Mat3 q{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)q[3*i+j]+=r[3*i+k]*d[k]*r[3*j+k];return q;}
inline Mat3 rotationIncrement(const Vec3& angle) {
  const double t=norm(angle); const double a=t>1e-7?std::sin(t)/t:1.-t*t/6.+t*t*t*t/120.;
  const double b=t>1e-7?(1.-std::cos(t))/(t*t):.5-t*t/24.+t*t*t*t/720.;
  const Mat3 k{0.,-angle[2],angle[1],angle[2],0.,-angle[0],-angle[1],angle[0],0.};
  const Mat3 k2=multiply(k,k);Mat3 r=identity3;for(int j=0;j<9;++j)r[j]+=a*k[j]+b*k2[j];return r;
}
inline Mat3 rotateLaboratory(const Mat3& r,const Vec3& angle) {return multiply(rotationIncrement(angle),r);}
inline bool finite(const Vec3& x) {return std::isfinite(x[0])&&std::isfinite(x[1])&&std::isfinite(x[2]);}
struct Body {
  Vec3 position{},velocity{},omega{};
  Vec3 axes{1.65e-6,1.65e-6,.20e-6};
  Mat3 rotation=identity3;
  double mass=1.;
  Vec3 inertiaBody{1.,1.,1.};
};
} // namespace graphite

} } // SLURRY SCOPE END
#endif
