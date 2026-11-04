#ifndef SLURRY_SUPPORT_BOUND_H
#define SLURRY_SUPPORT_BOUND_H

// A one-sided separating-plane certificate for two ellipsoids. For ANY unit
// direction n, n.(x_j-x_i)-h_i(n)-h_j(n) is a lower bound on the exact signed
// support gap. A positive certificate can skip the expensive gap maximization;
// a negative/uncertain result says nothing and must retain the original solve.
// The second body must already carry its CURRENT periodic image shift. This
// helper does not apply to finite cylinders or other non-ellipsoidal shapes.
#include "../vendor/gr_re2/ellipsoidGap.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace SlurrySupportBound {
namespace gr = slurry::gr_re2::graphite;

inline bool separated(const gr::Body &first, const gr::Body &second,
                      const gr::Vec3 &direction, double cutoff) {
  const double length = std::hypot(direction[0], direction[1], direction[2]);
  // A subnormal norm can have order-one relative rounding error, so its
  // quotient need not be approximately unit length. Decline certification.
  if (!(length >= std::numeric_limits<double>::min()) ||
      !std::isfinite(length) || !std::isfinite(cutoff))
    return false;
  const gr::Vec3 normal = {direction[0]/length, direction[1]/length, direction[2]/length};
  auto radius = [&](const gr::Body &body) {
    gr::Vec3 projected{};
    for (int axis = 0; axis < 3; ++axis) {
      if (!(body.axes[axis] > 0.) || !std::isfinite(body.axes[axis]))
        return std::numeric_limits<double>::quiet_NaN();
      double local = 0.;
      for (int coordinate = 0; coordinate < 3; ++coordinate)
        local += body.rotation[3*coordinate+axis]*normal[coordinate];
      projected[axis] = body.axes[axis]*local;
    }
    return std::hypot(projected[0], projected[1], projected[2]);
  };
  const double ri = radius(first), rj = radius(second);
  const gr::Vec3 displacement = {second.position[0]-first.position[0],
                                second.position[1]-first.position[1],
                                second.position[2]-first.position[2]};
  const double projected = normal[0]*displacement[0] + normal[1]*displacement[1] +
                           normal[2]*displacement[2];
  const double lower = projected-ri-rj;
  auto shape_scale = [](const gr::Body &body) {
    double rotation_sum=0.;
    for (double entry:body.rotation) rotation_sum+=std::abs(entry);
    return *std::max_element(body.axes.begin(),body.axes.end())*rotation_sum;
  };
  // Bound cancellation in current coordinates as well as the small support
  // dot products. Use the full semiaxes, not just a small projected radius.
  // The generous factor covers normalization, products/sums and hypot roundoff.
  const double scale = std::hypot(first.position[0],first.position[1],first.position[2]) +
      std::hypot(second.position[0],second.position[1],second.position[2]) +
      std::hypot(displacement[0],displacement[1],displacement[2]) +
      shape_scale(first)+shape_scale(second)+std::abs(cutoff);
  const double margin = 512.*std::numeric_limits<double>::epsilon()*
      std::max(scale,std::numeric_limits<double>::min());
  return std::isfinite(lower) && std::isfinite(margin) && lower > cutoff+margin;
}
} // namespace SlurrySupportBound
#endif
