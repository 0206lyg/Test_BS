#include "slurry_compact_bridge.h"
#include <iomanip>
#include <iostream>
#include <stdexcept>

// compact_bridge_v4.dat is a historical hard-gap-floor fixture. The additive
// model is checked against sphere formulas and geometric invariants instead.
namespace cb = SlurryCompactBridge;
using cb::Mat3;
using cb::Mat11;
using cb::Mat22;
using cb::Vec3;

namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void close(double got, double expected, double tolerance, const char *message) {
  require(std::abs(got - expected) <= tolerance * std::max(1., std::abs(expected)), message);
}
cb::gr::Body body(double ar, double length, const Mat3 &rotation = Mat3::Identity()) {
  cb::gr::Body b;
  const double major = length * std::cbrt(ar), minor = major / ar;
  b.axes = {{major, major, minor}}; // fixed volume-equivalent radius
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      b.rotation[3 * i + j] = rotation(i, j);
  b.position = {{0., 0., 0.}};
  return b;
}
void place_at_gap(const cb::gr::Body &bi, cb::gr::Body &bj, const Vec3 &n, double h) {
  // For a prescribed common outward normal, the support points give an
  // independent geometric construction of a separated or shallow-overlap pair.
  const Mat3 Ai = cb::shape(bi, 1.), Aj = cb::shape(bj, 1.);
  const Vec3 li = Ai * n / std::sqrt(n.dot(Ai * n));
  const Vec3 lj = -Aj * n / std::sqrt(n.dot(Aj * n));
  const Vec3 p = cb::vector(bi.position) + li - lj + h * n;
  for (int k = 0; k < 3; ++k)
    bj.position[k] = p[k];
}
Mat22 normalized(const cb::Result &result, double eta, double length) {
  Eigen::Matrix<double, 22, 1> d = Eigen::Matrix<double, 22, 1>::Ones();
  d.head<3>().setConstant(length);
  d.segment<3>(11).setConstant(length);
  return d.asDiagonal() * result.resistance * d.asDiagonal() / (eta * length * length * length);
}
Mat22 exchanged(const Mat22 &matrix) {
  Mat22 out;
  out.topLeftCorner<11, 11>() = matrix.bottomRightCorner<11, 11>();
  out.bottomRightCorner<11, 11>() = matrix.topLeftCorner<11, 11>();
  out.topRightCorner<11, 11>() = matrix.bottomLeftCorner<11, 11>();
  out.bottomLeftCorner<11, 11>() = matrix.topRightCorner<11, 11>();
  return out;
}
cb::gr::Body rotate(const cb::gr::Body &b, const Mat3 &rotation) {
  cb::gr::Body out = b;
  Mat3 orientation;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      orientation(i, j) = b.rotation[3 * i + j];
  orientation = (rotation * orientation).eval();
  const Vec3 p = rotation * cb::vector(b.position);
  for (int i = 0; i < 3; ++i) {
    out.position[i] = p[i];
    for (int j = 0; j < 3; ++j)
      out.rotation[3 * i + j] = orientation(i, j);
  }
  return out;
}
Mat22 rotation_map(const Mat3 &rotation) {
  Mat11 single = Mat11::Zero();
  single.topLeftCorner<3, 3>() = rotation;
  single.block<3, 3>(3, 3) = rotation;
  for (int k = 0; k < 5; ++k) {
    const Mat3 e = rotation * cb::strain_basis()[k] * rotation.transpose();
    for (int j = 0; j < 5; ++j)
      single(6 + j, 6 + k) = (cb::strain_basis()[j].array() * e.array()).sum();
  }
  Mat22 out = Mat22::Zero();
  out.topLeftCorner<11, 11>() = single;
  out.bottomRightCorner<11, 11>() = single;
  return out;
}
} // namespace

int main() {
  try {
    // Equal unit spheres have ell_n=ell_t=patch_width=1. The normal
    // coefficient is (3*pi/2)/he/(1+he)^2; the tangential one is pi*log(1+1/he).
    // In particular h>0 below epsilon must not produce the old floor plateau.
    constexpr double epsilon = 1e-3;
    const auto sphere_i = body(1., 1.);
    double previous_normal = 1e100;
    int analytic_cases = 0;
    for (double h : {-epsilon, 0., .25 * epsilon, .5 * epsilon, 2 * epsilon, .05, .5, 1.01}) {
      auto sphere_j = body(1., 1.);
      sphere_j.position[2] = 2. + h;
      const auto got = cb::evaluate(sphere_i, sphere_j, 1., 1., epsilon);
      const double physical = std::max(h, 0.), he = physical + epsilon;
      // Independent quintic support window, fixed by the physical gap.
      const double t = std::min(1., std::max(0., (physical - .1) / .9));
      const double weight = std::pow(1. - t, 3) * (1. + 3. * t + 6. * t * t);
      const double normal = weight * (1.5 * cb::pi) / (he * std::pow(1. + he, 2));
      const double tangent = weight * cb::pi * std::log1p(1. / he);
      close(got.gap, h, 2e-13, "sphere physical gap");
      close(got.effective_gap, he, 2e-13, "additive effective gap");
      close(got.ell_n, 1., 2e-13, "sphere normal length");
      close(got.ell_t, 1., 2e-13, "sphere tangent length");
      close(got.patch_width, 1., 2e-13, "sphere patch width");
      close(got.weight, weight, 2e-13, "physical support weight");
      close(got.resistance(2, 2), normal, 5e-12, "sphere normal resistance");
      close(got.resistance(0, 0), tangent, 5e-12, "sphere tangential resistance");
      require(got.active == (h < 1.), "physical support boundary");
      if (h > 0. && h < .01)
        require(got.resistance(2, 2) < previous_normal, "sub-cutoff gap must remain resolved");
      previous_normal = got.resistance(2, 2);
      ++analytic_cases;
    }
    // Changing regularization changes the resistance, not its outer support.
    for (double h : {.5, .9999, 1.0001}) {
      auto bj = body(1., 1.);
      bj.position[2] = 2. + h;
      const auto small = cb::evaluate(sphere_i, bj, 1., 1., epsilon);
      const auto large = cb::evaluate(sphere_i, bj, 1., 1., .1);
      close(small.weight, large.weight, 1e-14, "regularization moved physical support");
      require(small.active == large.active, "regularization changed support activation");
    }

    const Mat3 qi = Eigen::AngleAxisd(.53, Vec3(1., 2., -.4).normalized()).toRotationMatrix();
    const Mat3 qj = Eigen::AngleAxisd(-.71, Vec3(-.2, 1., 3.).normalized()).toRotationMatrix();
    const Mat3 global = Eigen::AngleAxisd(1.23, Vec3(2., -.7, 1.).normalized()).toRotationMatrix();
    const Mat22 transform = rotation_map(global);
    const Vec3 normal = Vec3(.37, -.51, .78).normalized();
    double worst_gap = 0., worst_exchange = 0., worst_rotation = 0., lowest_eigenvalue = 0.;
    int shape_cases = 0;
    for (double length : {1., 4e-6}) {
      constexpr double eta = .23;
      for (double ar : {1., 2., 5.}) {
        const auto bi = body(ar, length, qi);
        auto contact_bj = body(ar, length, qj);
        place_at_gap(bi, contact_bj, normal, 0.);
        const auto contact = cb::evaluate(bi, contact_bj, eta, length, epsilon * length);
        for (double scaled_gap : {-1e-4, 0., 5e-4, .03}) {
          auto bj = body(ar, length, qj);
          place_at_gap(bi, bj, normal, scaled_gap * length);
          const auto gap = cb::gr::closestEllipsoidGap(bi, bj);
          const auto got = cb::evaluate(bi, bj, gap, eta, length, epsilon * length, true);
          const Mat22 matrix = normalized(got, eta, length);
          const double denominator = std::max(matrix.norm(), 1e-300);
          const double gap_error = std::abs(got.gap / length - scaled_gap);
          worst_gap = std::max(worst_gap, gap_error);
          require(gap_error < 2e-12, "spheroid signed gap");
          close(got.effective_gap / length, std::max(scaled_gap, 0.) + epsilon, 2e-12,
                "spheroid additive effective gap");
          require(got.active && matrix.allFinite(), "overlap/contact lubrication must remain active");
          require((matrix - matrix.transpose()).norm() / denominator < 2e-13,
                  "resistance symmetry");
          Eigen::SelfAdjointEigenSolver<Mat22> eig(matrix);
          require(eig.info() == Eigen::Success, "resistance eigensolve");
          const double negative = eig.eigenvalues()[0] / denominator;
          lowest_eigenvalue = std::min(lowest_eigenvalue, negative);
          require(negative >= -2e-12 && got.schur_min_scaled >= -2e-12,
                  "nonnegative viscous dissipation");
          const auto reverse = cb::evaluate(bj, bi, eta, length, epsilon * length);
          const double exchange = (matrix - exchanged(normalized(reverse, eta, length))).norm() /
                                  denominator;
          worst_exchange = std::max(worst_exchange, exchange);
          require(exchange < 2e-10, "body-exchange symmetry");
          const auto rotated = cb::evaluate(rotate(bi, global), rotate(bj, global), eta, length,
                                            epsilon * length);
          const double covariance = (normalized(rotated, eta, length) -
                                     transform * matrix * transform.transpose()).norm() / denominator;
          worst_rotation = std::max(worst_rotation, covariance);
          require(covariance < 2e-10, "coordinate-frame covariance");
          if (scaled_gap < 0.) {
            require(got.gap < 0., "signed overlap geometry was discarded");
            require((matrix - normalized(contact, eta, length)).norm() / denominator < 2e-10,
                    "shallow overlap must use contact-limit scalar resistance");
          }
          ++shape_cases;
        }
      }
    }
    std::cout << std::setprecision(12) << "PASS " << analytic_cases << " sphere limits, "
              << shape_cases << " AR=1/2/5 geometry cases; gap=" << worst_gap
              << " exchange=" << worst_exchange << " covariance=" << worst_rotation
              << " min_scaled=" << lowest_eigenvalue << "\n";
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
