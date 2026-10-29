#ifndef SLURRY_ROUGH_SURFACE_H
#define SLURRY_ROUGH_SURFACE_H

// Isotropic, statistically homogenized rough-surface SLIDING EXCESS.
// This is a finite-patch model extension, NOT an exact rough-spheroid solution.
// Reference mechanism: Minten & Rallabandi, arXiv:2506.21799, equations (2)-(7).
// Each surface carries spherical caps of height A and lateral radius w, with
// areal number density coverage/(pi*w*w). The leading hard-core pair closure is
// g(d)=Theta(d): geometrically overlapping cap pairs are excluded, with no density
// renormalization. This is NOT the exact correlated distribution of a dense rough
// surface. At high coverage, the actual spatial correlations must be calibrated.
// Their squeeze resistance is projected onto tangential relative motion and
// averaged over azimuth. The odd normal/tangential coupling averages to zero.
// Only that tangential excess is added; normal roughness resistance, shadowing,
// cap elasticity, spatial correlations and a resolved surface realization are
// NOT modeled. The existing smooth normal hydrodynamics/contact law is retained.
//
// The smooth contact patch has H(x)=h+x^T Lambda x. Positive quadrature integrates
// A<=H<3A inside the existing finite curvature patch. H<A would make a cap
// intersect the opposing flat base and is excluded as well. These exclusions
// add no mechanical force. Cap-pair offset s in [0,2w]
// gives the geometrical microscopic gap
//   d=sqrt((2b+H-2A)^2+s^2)-2b, b=(w*w+A*A)/(2A).
// The floor is applied to d itself. The microscopic squeeze law uses the positive
// excess 3*pi*eta*b^2/2 * [1/max(d,floor)-1/A]_+. Choosing d_cut=A is an explicit
// near-field cutoff model, not a measured parameter or an exact finite-cap law.
// It retains hydrodynamic sliding before solid contact at H=2A. Geometry then
// gives finite support H<3A without a hard force jump. No contact-onset shift of
// all lubrication gaps is made. Only actual d>=0 is integrated; positive d below
// the numerical floor is regularized. A negative d is NEVER replaced by a small
// positive gap and counted as lubrication. No additional elastic repulsion is
// introduced by this hydrodynamic correction.
// The cap approximation is best for A<<w and w small compared with local body
// curvature radii. At sharp spheroid rims it remains an explicit local model.
//
// Common-point velocities throughout the patch give rigid-body objectivity.
// With q=(U,Omega,-E), R=integral B(x)^T k(x)*(I-n*n^T)*B(x) dA maps q to
// ON-FLUID F,T,S, so q^T R q >=0 exactly. The elliptic patch's first moment
// vanishes and its second moment is analytic; no angular quadrature is needed.
// All arguments share one consistent length unit; eta is the matching viscosity.

#include "slurry_compact_bridge.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace SlurryRoughSurface {
namespace cb = SlurryCompactBridge;
namespace gr = slurry::gr_re2::graphite;
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;
using Mat22 = Eigen::Matrix<double, 22, 22>;
using Map322 = Eigen::Matrix<double, 3, 22>;

struct Options {
  bool enabled = false;
  double height = 0.;        // A, per surface; not an RMS height
  double width = 0.;         // w, projected cap radius
  double coverage = .2;      // rho*pi*w^2, separately on each surface
  double gap_floor = 0.;
  int quadrature_order = 8;  // supported: 4, 8, 16; azimuth is analytic
};

struct Result {
  Mat22 resistance = Mat22::Zero();
  double coefficient = 0.;             // zeta, force / tangential velocity
  double height_moment = 0.;           // .5*integral (H-h) k(H) dA
  double effective_patch_area = 0.;
  double equivalent_pairs = 0.;        // expected cap pairs, not an integer
  double minimum_micro_gap = 0.;       // minimum admitted geometric cap gap >=0
  double effective_minimum_micro_gap = 0.;
  bool active = false;
};

inline void validate(const Options &o) {
  if (!o.enabled)
    return;
  if (!(o.height > 0.) || !std::isfinite(o.height) ||
      !(o.width >= o.height) || !std::isfinite(o.width) ||
      !(o.coverage >= 0. && o.coverage <= 1.) || !std::isfinite(o.coverage) ||
      !(o.gap_floor > 0.) || !std::isfinite(o.gap_floor) ||
      (o.quadrature_order != 4 && o.quadrature_order != 8 && o.quadrature_order != 16))
    throw std::invalid_argument("rough surface needs A>0, w>=A, coverage in [0,1], floor>0 and order 4/8/16");
}

namespace detail {
// x-log(1+x), avoiding cancellation for broad/shallow caps.
inline double x_minus_log1p(double x) {
  if (std::abs(x) >= 1e-4)
    return x - std::log1p(x);
  const double x2 = x * x;
  return x2 * (.5 + x * (-1. / 3. + x * (.25 + x * (-.2 + x / 6.))));
}

template <class Function>
inline double integrate(double lo, double hi, int order, Function f) {
  static constexpr double x4[] = {.3399810435848562648, .8611363115940525752};
  static constexpr double w4[] = {.6521451548625461426, .3478548451374538574};
  static constexpr double x8[] = {.1834346424956498049, .5255324099163289858,
                                 .7966664774136267396, .9602898564975362317};
  static constexpr double w8[] = {.36268378337836198297, .3137066458778872873,
                                 .2223810344533744705, .1012285362903762592};
  static constexpr double x16[] = {.09501250983763744019, .2816035507792589132,
      .4580167776572273863, .6178762444026437484, .7554044083550030339,
      .8656312023878317439, .9445750230732325761, .9894009349916499326};
  static constexpr double w16[] = {.1894506104550684963, .1826034150449235889,
      .1691565193950025382, .1495959888165767321, .1246289712555338721,
      .09515851168249278481, .06225352393864789286, .02715245941175409485};
  const double *x = order == 4 ? x4 : order == 8 ? x8 : x16;
  const double *w = order == 4 ? w4 : order == 8 ? w8 : w16;
  const double mid = .5 * (lo + hi), half = .5 * (hi - lo);
  double sum = 0.;
  for (int k = 0; k < order / 2; ++k)
    sum += w[k] * (f(mid - half * x[k]) + f(mid + half * x[k]));
  return half * sum;
}

// Integral (t-z*z/t)*[1/max(t-2b,floor)-1/cutoff]_+ dt over t>=max(z,2b).
// The t>=2b bound excludes overlapping cap pairs instead of flooring overlap.
// The floored portion
// is analytic. The free part uses positive quadrature in log(d), resolving the
// smallest microscopic gaps without cancellation when z>2b or d approaches cut.
inline double offset_integral(double z, double b, double width, double floor,
                              double cutoff, int order = 8) {
  if (!(cutoff > floor))
    return 0.;
  const double c = 2. * b;
  const double lower = std::max(z, c);
  const double tmax = std::min(std::hypot(z, 2. * width), c + cutoff);
  if (!(tmax > lower))
    return 0.;
  const double floored_end = std::min(tmax, c + floor);
  double result = 0.;
  if (floored_end > lower) {
    const double dt = floored_end - lower, x = dt / lower;
    const double first = (lower - z) * (lower + z) / lower * dt;
    result += (first + .5 * dt * dt + z * z * x_minus_log1p(x)) *
              (cutoff - floor) / (floor * cutoff);
  }
  const double t0 = std::max(z, c + floor);
  if (tmax > t0) {
    const double u0 = t0 - c, u1 = tmax - c;
    const double vmax = std::log1p((u1 - u0) / u0);
    result += integrate(0., vmax, order, [&](double v) {
      const double u = u0 * std::exp(v), t = c + u;
      return std::max(0., t - z) * (t + z) / t *
             std::max(0., cutoff - u) / cutoff;
    });
  }
  return result;
}

// Split at the analytically known switches of the microscopic floor and cap
// footprint/cutoff. Positive quadrature then sees smooth pieces; refinement
// does not need to discover tiny asperity gaps by increasing arbitrary samples.
template <class Function>
inline double integrate_height(double lo, double hi, double A, double b,
                               double w, double floor, int order, Function f) {
  std::array<double, 6> points{{lo, hi, hi, hi, hi, hi}};
  int count = 2;
  auto add = [&](double H) {
    if (H > lo && H < hi)
      points[count++] = H;
  };
  add(2. * A);
  add(2. * A + floor);
  for (double d : {floor, A}) {
    const double c = 2. * b;
    const double disc = (c + d) * (c + d) - 4. * w * w;
    if (disc >= 0.) {
      // Rationalize sqrt(disc)-c to avoid cancellation for shallow caps.
      const double shift = (2. * c * d + d * d - 4. * w * w) /
                           (std::sqrt(disc) + c);
      add(2. * A + shift);
    }
  }
  std::sort(points.begin(), points.begin() + count);
  double result = 0.;
  for (int k = 1; k < count; ++k)
    result += integrate(points[k - 1], points[k], order, f);
  return result;
}

inline Map322 common_point_map(const V3 &li, const V3 &lj) {
  Map322 B = Map322::Zero();
  B.block<3, 3>(0, 0) = -M3::Identity();
  B.block<3, 3>(0, 3) = cb::cross_matrix(li);
  B.block<3, 3>(0, 11) = M3::Identity();
  B.block<3, 3>(0, 14) = -cb::cross_matrix(lj);
  for (int k = 0; k < 5; ++k) {
    B.col(6 + k) = -cb::strain_basis()[k] * li;
    B.col(17 + k) = cb::strain_basis()[k] * lj;
  }
  return B;
}
} // namespace detail

inline Result evaluate(const gr::Body &bi, const gr::Body &bj, const gr::GapResult &gap,
                       const cb::Result &smooth, double eta, const Options &o) {
  Result out;
  if (!o.enabled)
    return out;
  validate(o);
  if (!(eta > 0.) || !std::isfinite(eta) || !std::isfinite(gap.gap))
    throw std::invalid_argument("rough surface needs finite eta>0 and finite gap");
  // The scalar-gap floor follows the existing smooth bridge convention. The
  // actual closest points/normals used in B and the body contact law are unchanged.
  const double h = std::max(gap.gap, o.gap_floor), cap_height = 2. * o.height;
  const double reach = cap_height + o.height; // microscopic d_cut=A
  const double lo = std::max(h, o.height);
  out.minimum_micro_gap = std::max(0., lo - cap_height);
  out.effective_minimum_micro_gap = std::max(out.minimum_micro_gap, o.gap_floor);
  if (!(h < reach) || !(smooth.ell_n > 0.) || o.coverage == 0. ||
      !(o.height > o.gap_floor))
    return out;
  const double hi = std::min(reach, h + smooth.ell_n);
  const double deltaH = hi - lo;
  if (!(deltaH > 0.))
    return out;
  const V3 n = cb::vector(gap.normal).normalized();
  const V3 ref = std::abs(n[0]) < .8 ? V3::UnitX() : V3::UnitY();
  Eigen::Matrix<double, 3, 2> T;
  T.col(0) = n.cross(ref).normalized();
  T.col(1) = n.cross(T.col(0));
  const M3 Ai = cb::shape(bi, 1.), Aj = cb::shape(bj, 1.);
  const M3 Ki = std::sqrt(n.dot(Ai * n)) * Ai.inverse();
  const M3 Kj = std::sqrt(n.dot(Aj * n)) * Aj.inverse();
  const Eigen::Matrix2d lambda = .5 * T.transpose() * (Ki + Kj) * T;
  const double determinant = lambda.determinant();
  if (!(determinant > 0.) || !std::isfinite(determinant))
    throw std::runtime_error("rough surface needs positive local curvature");
  const double area_per_height = cb::pi / std::sqrt(determinant);
  const double b = (o.width * o.width + o.height * o.height) / (2. * o.height);
  const double density = o.coverage / (cb::pi * o.width * o.width);
  out.effective_patch_area = area_per_height * deltaH;
  out.equivalent_pairs = density * density * area_per_height * cb::pi *
      detail::integrate_height(lo, hi, o.height, b, o.width, o.gap_floor,
                               o.quadrature_order, [&](double H) {
        const double z = 2. * b + H - cap_height;
        const double upper2 = std::min(4. * o.width * o.width,
                                      (reach - H) * (2. * b + o.height + z));
        const double excluded2 = std::max(0., (cap_height - H) * (2. * b + z));
        return std::max(0., upper2 - excluded2);
      });
  // zeta_n=3*pi*eta*b^2/2*[1/d_e-1/A]_+. Azimuth contributes sin(theta)^2/2;
  // integrating offsets 2*pi*s*ds gives 3*pi^2*eta*b^2/2 times offset_integral.
  const double integral = detail::integrate_height(lo, hi, o.height, b,
      o.width, o.gap_floor, o.quadrature_order,
      [&](double H) {
        const double z = 2. * b + H - cap_height;
        return detail::offset_integral(z, b, o.width, o.gap_floor,
                                       o.height, o.quadrature_order);
      });
  out.coefficient = density * density * area_per_height *
                    (1.5 * cb::pi * cb::pi * eta * b * b) * integral;
  const double moment_integral = detail::integrate_height(lo, hi, o.height, b,
      o.width, o.gap_floor, o.quadrature_order,
      [&](double H) {
        const double z = 2. * b + H - cap_height;
        return .5 * (H - h) * detail::offset_integral(z, b, o.width, o.gap_floor,
                                                    o.height, o.quadrature_order);
      });
  out.height_moment = density * density * area_per_height *
                      (1.5 * cb::pi * cb::pi * eta * b * b) * moment_integral;
  if (!(out.coefficient >= 0.) || !std::isfinite(out.coefficient))
    throw std::runtime_error("nonfinite rough surface coefficient");
  if (out.coefficient == 0.)
    return out;
  const V3 d = cb::vector(bj.position) - cb::vector(bi.position);
  const V3 li = .5 * (cb::vector(gap.leverI) + d + cb::vector(gap.leverJ));
  const V3 lj = li - d;
  const auto B = detail::common_point_map(li, lj);
  const M3 tangent = M3::Identity() - n * n.transpose();
  out.resistance = out.coefficient * B.transpose() * tangent * B;
  // For each contour H-h=s, <x*x^T> = .5*s*T*Lambda^{-1}*T^T.
  // Preserve rotation/strain variation across the finite patch rather than
  // collapsing every asperity to one point (which would lose twisting drag).
  const Eigen::Matrix2d covariance = lambda.inverse();
  const Eigen::LLT<Eigen::Matrix2d> chol(covariance);
  if (chol.info() != Eigen::Success)
    throw std::runtime_error("rough surface patch moment needs positive curvature");
  const Eigen::Matrix<double, 3, 2> directions = T * chol.matrixL().toDenseMatrix();
  const auto Bzero = detail::common_point_map(V3::Zero(), V3::Zero());
  for (int k = 0; k < 2; ++k) {
    const V3 shift = directions.col(k);
    const Map322 gradient = detail::common_point_map(shift, shift) - Bzero;
    out.resistance += out.height_moment * gradient.transpose() * tangent * gradient;
  }
  out.resistance = .5 * (out.resistance + out.resistance.transpose()).eval();
  out.active = true;
  return out;
}
} // namespace SlurryRoughSurface
#endif
