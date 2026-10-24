#ifndef SLURRY_CYLINDER_GEOMETRY_H
#define SLURRY_CYLINDER_GEOMETRY_H

// Continuous, closed circular cylinders; no bead or polyhedral approximation.
// Distances are nonnegative. Interpenetration is reported, not assigned an
// invented penetration depth. tolerance is an ABSOLUTE distance in input units.
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace SlurryCylinderGeometry {
using V3 = Eigen::Vector3d;
struct Body {
  V3 center = V3::Zero(), axis = V3::UnitZ();
  double radius = 1., half_thickness = .05;
};
struct Gap {
  double gap = 0.;
  V3 normal = V3::UnitZ(), point_i = V3::Zero(), point_j = V3::Zero();
  bool overlap = false, converged = false;
  double residual = 0.; // certified upper minus supporting-plane lower bound
  int iterations = 0;
};
inline void validate(const Body &b) {
  if (!b.center.allFinite() || !b.axis.allFinite() || !(b.axis.norm() > 0.) ||
      !std::isfinite(b.radius) || !std::isfinite(b.half_thickness) ||
      !(b.radius > 0.) || !(b.half_thickness > 0.))
    throw std::invalid_argument("Cylinder needs finite center/axis and positive dimensions");
}
inline double volume(const Body &b) {
  validate(b);
  return 2. * std::acos(-1.) * b.radius * b.radius * b.half_thickness;
}
inline V3 support(const Body &b, const V3 &d) {
  validate(b);
  if (!d.allFinite()) throw std::invalid_argument("Nonfinite cylinder support direction");
  const V3 n = b.axis.normalized();
  const double dn = d.dot(n);
  // Cross products retain orthogonality even for an almost axial direction;
  // subtracting d_n*n can leave a roundoff-sized axial vector which is then
  // magnified to the full disk radius by normalization.
  const V3 tangent = n.cross(d.cross(n));
  V3 p = b.center;
  // Dot products close to an axial direction may leave a tiny spurious radial
  // vector. Keeping it is valid: any point on that cap supports an axial ray.
  if (tangent.norm() > 0.) p += b.radius * tangent.normalized();
  if (dn != 0.) p += std::copysign(b.half_thickness, dn) * n;
  return p;
}

namespace detail {
struct Vertex { V3 w, a, b; };
struct Closest {
  V3 v = V3::Zero();
  std::vector<Vertex> vertices;
  std::vector<double> weights;
};
// Enumerating the at most 15 simplex faces avoids hand-coded Voronoi cases.
// All accepted barycentric coordinates are nonnegative, so each witness stays
// inside the exact cylinder. Degenerate affine faces are skipped, not inverted.
inline Closest closest_simplex(const std::vector<Vertex> &s) {
  Closest result;
  double best = std::numeric_limits<double>::infinity();
  for (unsigned mask = 1; mask < (1u << s.size()); ++mask) {
    std::array<int, 4> index{};
    int m = 0;
    for (unsigned k = 0; k < s.size(); ++k) if (mask & (1u << k)) index[m++] = k;
    std::array<double, 4> w{{1., 0., 0., 0.}};
    const V3 a = s[index[0]].w;
    if (m == 2) {
      const V3 d = s[index[1]].w - a;
      if (d.squaredNorm() <= 1e-30) continue;
      w[1] = -a.dot(d) / d.squaredNorm(); w[0] = 1. - w[1];
    } else if (m == 3) {
      const V3 d = s[index[1]].w - a, e = s[index[2]].w - a;
      const double dd = d.dot(d), ee = e.dot(e), de = d.dot(e);
      const double det = d.cross(e).squaredNorm();
      if (!(det > 1e-26 * dd * ee) || !(det > 1e-300)) continue;
      // Resolve the projection in an orthonormal frame. The equivalent
      // normal-equation cofactors subtract nearly equal terms when GJK's
      // triangle becomes thin, magnifying roundoff near rim/rim contact.
      const double length=std::sqrt(dd);
      const V3 t1=d/length, t2=d.cross(e).normalized().cross(t1);
      const double height=e.dot(t2);
      if (!(std::abs(height)>0.)) continue;
      w[2]=-a.dot(t2)/height;
      w[1]=(-a.dot(t1)-w[2]*e.dot(t1))/length;
      w[0] = 1. - w[1] - w[2];
    } else if (m == 4) {
      Eigen::Matrix3d d;
      for (int k = 0; k < 3; ++k) d.col(k) = s[index[k + 1]].w - a;
      Eigen::FullPivLU<Eigen::Matrix3d> lu(d);
      lu.setThreshold(1e-13);
      if (lu.rank() != 3) continue;
      const V3 u = lu.solve(-a);
      w[0] = 1. - u.sum();
      for (int k = 0; k < 3; ++k) w[k + 1] = u[k];
    }
    bool admissible = true;
    double sum = 0.;
    for (int k = 0; k < m; ++k) {
      if (w[k] < -2e-12 || !std::isfinite(w[k])) admissible = false;
      w[k] = std::max(0., w[k]); sum += w[k];
    }
    if (!admissible || !(sum > 0.)) continue;
    V3 p = V3::Zero();
    for (int k = 0; k < m; ++k) p += (w[k] /= sum) * s[index[k]].w;
    const double distance = p.squaredNorm();
    if (distance < best) {
      best = distance; result.v = p; result.vertices.clear(); result.weights.clear();
      for (int k = 0; k < m; ++k) if (w[k] > 0.) {
        result.vertices.push_back(s[index[k]]); result.weights.push_back(w[k]);
      }
    }
  }
  if (!std::isfinite(best)) throw std::runtime_error("Cylinder GJK has no finite simplex");
  return result;
}
inline Gap parallel_gap(const Body &a, const Body &b) {
  Gap out;
  const V3 n = a.axis.normalized(), d = b.center - a.center;
  const double z = d.dot(n);
  const V3 radial = d - z * n;
  const double rho = radial.norm();
  const V3 t = rho > 0. ? V3(radial / rho) : n.unitOrthogonal();
  const double dz = std::abs(z) - a.half_thickness - b.half_thickness;
  const double dr = rho - a.radius - b.radius;
  double zi, zj;
  if (dz > 0.) {
    zi = std::copysign(a.half_thickness, z);
    zj = z - std::copysign(b.half_thickness, z);
  } else {
    zi = zj = .5 * (std::max(-a.half_thickness, z - b.half_thickness) +
                    std::min(a.half_thickness, z + b.half_thickness));
  }
  double ri, rj;
  if (dr > 0.) { ri = a.radius; rj = rho - b.radius; }
  else {
    ri = rj = .5 * (std::max(-a.radius, rho - b.radius) +
                    std::min(a.radius, rho + b.radius));
  }
  out.point_i = a.center + zi * n + ri * t;
  out.point_j = a.center + zj * n + rj * t;
  const V3 separation = out.point_j - out.point_i;
  out.gap = std::hypot(std::max(0., dz), std::max(0., dr));
  if (out.gap > 0.) out.normal = separation / separation.norm();
  else out.normal = dz >= dr ? V3(std::copysign(1., z) * n) : t;
  out.overlap = dz < 0. && dr < 0.; out.converged = true; out.iterations = 1;
  return out;
}
// A cap-normal supporting plane is exact whenever the opposite support point
// projects inside that circular cap. This is a witness/lower-bound equality,
// not an angular approximation. It avoids simplex stagnation for flat faces.
inline bool cap_gap(const Body &a,const Body &b,Gap &out) {
  for (double sign : {-1.,1.}) {
    const V3 n=sign*a.axis.normalized();
    const V3 cap=a.center+a.half_thickness*n;
    const V3 point_b=support(b,-n);
    const double gap=n.dot(point_b-cap);
    if (!(gap>0.)) continue;
    const V3 point_a=point_b-gap*n;
    const V3 radial=point_a-cap;
    if (radial.squaredNorm()>a.radius*a.radius) continue;
    out.gap=gap;out.normal=n;out.point_i=point_a;out.point_j=point_b;
    out.converged=true;out.iterations=1;out.residual=0.;out.overlap=false;
    return true;
  }
  return false;
}
} // namespace detail

inline Gap closest_gap(const Body &original_a, const Body &original_b,
                       double tolerance = 1e-10) {
  validate(original_a); validate(original_b);
  if (!(tolerance > 0.) || !std::isfinite(tolerance))
    throw std::invalid_argument("Cylinder gap tolerance must be positive and finite");
  const V3 na = original_a.axis.normalized(), nb = original_b.axis.normalized();
  // Only truly parallel axes take the product-of-intervals/disks shortcut.
  if (na.cross(nb).norm() <= 16. * std::numeric_limits<double>::epsilon())
    return detail::parallel_gap(original_a, original_b);
  Gap cap;
  if (detail::cap_gap(original_a,original_b,cap)) return cap;
  if (detail::cap_gap(original_b,original_a,cap)) {
    std::swap(cap.point_i,cap.point_j);cap.normal=-cap.normal;return cap;
  }
  const V3 origin = original_a.center + .5 * (original_b.center - original_a.center);
  const double scale = std::max({original_a.radius, original_a.half_thickness,
                                 original_b.radius, original_b.half_thickness});
  Body a = original_a, b = original_b;
  a.center = (a.center - origin) / scale; b.center = (b.center - origin) / scale;
  a.axis = na; b.axis = nb;
  a.radius /= scale; a.half_thickness /= scale;
  b.radius /= scale; b.half_thickness /= scale;
  const double tol = std::max(tolerance / scale, 128. * std::numeric_limits<double>::epsilon());
  const auto vertex = [&](const V3 &direction) {
    detail::Vertex p; p.a = support(a, direction); p.b = support(b, -direction);
    p.w = p.a - p.b; return p;
  };
  V3 direction = b.center - a.center;
  if (direction.norm() == 0.) direction = V3::UnitX();
  std::vector<detail::Vertex> simplex{vertex(direction)};
  detail::Closest nearest;
  Gap out;
  for (int iteration = 0; iteration < 2048; ++iteration) {
    nearest = detail::closest_simplex(simplex);
    const double upper = nearest.v.norm();
    out.iterations = iteration + 1;
    if (upper <= 128. * std::numeric_limits<double>::epsilon()) {
      out.overlap = true; out.converged = true; out.residual = upper * scale; break;
    }
    const auto next = vertex(-nearest.v);
    const double lower = std::max(0., nearest.v.dot(next.w) / upper);
    out.residual = std::max(0., upper - lower) * scale;
    if (upper - lower <= tol) { out.converged = true; break; }
    simplex = nearest.vertices;
    // A nonzero closest point cannot need an interior tetrahedron. The branch
    // protects array bounds if a nearly singular tetrahedron retained 4 nodes.
    if (simplex.size() >= 4) {
      const auto m = std::min_element(nearest.weights.begin(), nearest.weights.end());
      simplex.erase(simplex.begin() + std::distance(nearest.weights.begin(), m));
    }
    simplex.push_back(next);
  }
  V3 ai = V3::Zero(), bj = V3::Zero();
  for (std::size_t k = 0; k < nearest.vertices.size(); ++k) {
    ai += nearest.weights[k] * nearest.vertices[k].a;
    bj += nearest.weights[k] * nearest.vertices[k].b;
  }
  out.point_i = origin + scale * ai; out.point_j = origin + scale * bj;
  const V3 separation = bj - ai;
  out.gap = out.overlap ? 0. : scale * separation.norm();
  if (separation.norm() > 128. * std::numeric_limits<double>::epsilon())
    out.normal = separation.normalized();
  else out.normal = direction.normalized();
  return out;
}
} // namespace SlurryCylinderGeometry
#endif
