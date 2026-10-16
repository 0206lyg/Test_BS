#ifndef SLURRY_HISTORY_MATH_H
#define SLURRY_HISTORY_MATH_H
#include "particleMath.h"

namespace SlurryHistory {
namespace g = slurry::gr_re2::graphite;
using Vec3 = g::Vec3;
inline Vec3 tangent(const Vec3 &x,const Vec3 &n) {
  return g::sub(x,g::scale(n,g::dot(x,n)));
}
// Transport through common rigid rotation, including spin about the contact
// normal, then the minimum rotation taking that normal to the new geometry.
// The input state is immutable; nonlinear residual evaluations never commit it.
inline Vec3 transport(const Vec3 &history,const Vec3 &old_normal,
                      const Vec3 &new_normal,const Vec3 &common_angle) {
  const auto common=g::rotationIncrement(common_angle);
  const auto a=g::normalized(g::mul(common,old_normal));
  const auto b=g::normalized(new_normal);
  const auto axis=g::cross(a,b);
  const double sine=g::norm(axis),cosine=std::clamp(g::dot(a,b),-1.,1.);
  auto correction=g::identity3;
  if(sine>1.e-14)
    correction=g::rotationIncrement(g::scale(axis,std::atan2(sine,cosine)/sine));
  else if(cosine<0.) {
    const auto pi_axis=g::normalized(g::cross(a,std::abs(a[0])<.8?Vec3{1,0,0}:Vec3{0,1,0}));
    correction=g::rotationIncrement(g::scale(pi_axis,std::acos(-1.)));
  }
  return tangent(g::mul(correction,g::mul(common,history)),b);
}
}
#endif
