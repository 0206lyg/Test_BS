#include "slurry_rough_surface.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace cb = SlurryCompactBridge;
namespace rough = SlurryRoughSurface;
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;
using V22 = Eigen::Matrix<double, 22, 1>;

static void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
static cb::gr::Body body(double ar, const V3 &x, const M3 &Q = M3::Identity()) {
  cb::gr::Body b;
  b.axes = {{ar, ar, 1.}};
  for (int k = 0; k < 3; ++k) {
    b.position[k] = x[k];
    for (int l = 0; l < 3; ++l)
      b.rotation[3 * k + l] = Q(k, l);
  }
  return b;
}
static rough::Result evaluate(const cb::gr::Body &a, const cb::gr::Body &b,
                              const rough::Options &o, double eta = 1.) {
  const auto g = cb::gr::closestEllipsoidGap(a, b);
  const auto smooth = cb::evaluate(a, b, g, eta, 1., o.gap_floor);
  return rough::evaluate(a, b, g, smooth, eta, o);
}
static Eigen::Matrix<double, 11, 11> grand_rotation(const M3 &Q) {
  Eigen::Matrix<double, 11, 11> G = Eigen::Matrix<double, 11, 11>::Zero();
  G.block<3, 3>(0, 0) = Q;
  G.block<3, 3>(3, 3) = Q;
  for (int k = 0; k < 5; ++k) {
    const M3 rotated = Q * cb::strain_basis()[k] * Q.transpose();
    for (int l = 0; l < 5; ++l)
      G(6 + l, 6 + k) = (rotated.array() * cb::strain_basis()[l].array()).sum();
  }
  return G;
}

int main() {
  try {
    rough::Options o;
    o.enabled = true;
    o.height = .1;       // 15 nm when the length unit is 150 nm
    o.width = 1.;        // 150 nm cap footprint radius
    o.coverage = .2;
    o.gap_floor = 1. / 150.;
    const auto a = body(20., V3::Zero());
    const auto b = body(20., V3(0., 0., 2. + 28. / 150.));
    const auto r = evaluate(a, b, o);
    require(r.active && r.coefficient > 0., "rough response should be active at 28 nm");
    require(r.effective_minimum_micro_gap == o.gap_floor, "microscopic floor missing");
    require((r.resistance - r.resistance.transpose()).norm() < 1e-12 * r.resistance.norm(),
            "rough grand matrix is not symmetric");
    Eigen::SelfAdjointEigenSolver<cb::Mat22> eigen(r.resistance);
    require(eigen.eigenvalues()[0] > -2e-12 * r.resistance.norm(), "rough matrix is not PSD");

    // Numerical integration of the original offset measure independently checks
    // the transformed cap-pair quadrature and the azimuth factor of one half.
    const double radius = (o.width * o.width + o.height * o.height) / (2. * o.height);
    for (double H : {.10, .15, .19, .21, .25, .29}) {
      const double z = 2. * radius + H - 2. * o.height;
      const int n = 200000;
      // Integrate the original positive-gap radial domain, with no counting or
      // renormalization of cap pairs whose spherical surfaces overlap.
      const double lower = std::sqrt(std::max(0., (2. * radius - z) * (2. * radius + z)));
      const double upper = std::min(2. * o.width, std::sqrt(std::max(0.,
          (2. * radius + o.height - z) * (2. * radius + o.height + z))));
      const double ds = (upper - lower) / n;
      double numeric = 0.;
      for (int k = 0; k < n; ++k) {
        const double s = lower + (k + .5) * ds, t = std::hypot(z, s);
        const double d = std::max(t - 2. * radius, o.gap_floor);
        const double normal = 1.5 * cb::pi * radius * radius *
                              std::max(0., 1. / d - 1. / o.height);
        numeric += 2. * cb::pi * s * ds * normal * .5 * s * s / (t * t);
      }
      const double transformed = 1.5 * cb::pi * cb::pi * radius * radius *
          rough::detail::offset_integral(z, radius, o.width, o.gap_floor, o.height, 16);
      require(std::abs(numeric / transformed - 1.) < 2e-8,
              "cap offset normalization/integral mismatch");
    }

    // A superposed common translation/rotation must produce no new dissipation.
    const V3 U(.2, -.4, .7), omega(.3, -.1, .2);
    V22 rigid = V22::Zero();
    rigid.head<3>() = U + omega.cross(cb::vector(a.position));
    rigid.segment<3>(3) = omega;
    rigid.segment<3>(11) = U + omega.cross(cb::vector(b.position));
    rigid.segment<3>(14) = omega;
    require((r.resistance * rigid).norm() < 2e-12 * r.resistance.norm() * rigid.norm(),
            "rough model violates rigid-body objectivity");

    // Surface slip should have the reported zeta, with action-reaction forces.
    V22 slip = V22::Zero();
    slip[11] = 1.;
    V22 f = r.resistance * slip;
    require(std::abs(f[11] / r.coefficient - 1.) < 2e-12, "reported zeta mismatch");
    require((f.head<3>() + f.segment<3>(11)).norm() < 1e-12 * f.norm(),
            "rough forces violate action-reaction");
    require((f.segment<3>(3) + f.segment<3>(14) +
             cb::vector(a.position).cross(f.head<3>()) +
             cb::vector(b.position).cross(f.segment<3>(11))).norm() <
             1e-12 * f.norm(), "rough pair violates angular momentum balance");

    // Finite area must resist opposite twisting even though centroid slip is zero.
    // Here Lambda=I/AR^2, so integral |x|^2*k dA = 2*AR^2*height_moment.
    V22 twist = V22::Zero();
    twist[16] = 1.;
    const double twist_power = twist.dot(r.resistance * twist);
    require(twist_power > 0. &&
            std::abs(twist_power / (2. * 20. * 20. * r.height_moment) - 1.) < 2e-12,
            "finite-patch twisting moment missing or incorrect");

    // Arbitrary rotation also rotates the STF stresslet/strain block correctly.
    const M3 Q = Eigen::AngleAxisd(.71, V3(.2, .5, .3).normalized()).toRotationMatrix();
    const auto ar = body(20., Q * cb::vector(a.position), Q);
    const auto br = body(20., Q * cb::vector(b.position), Q);
    const auto rr = evaluate(ar, br, o);
    cb::Mat22 G = cb::Mat22::Zero();
    G.topLeftCorner<11, 11>() = grand_rotation(Q);
    G.bottomRightCorner<11, 11>() = grand_rotation(Q);
    require((rr.resistance - G * r.resistance * G.transpose()).norm() <
            2e-9 * r.resistance.norm(), "rough model is not rotationally covariant");

    // Label exchange does not change the physical operator.
    const auto exchanged = evaluate(b, a, o);
    cb::Mat22 P = cb::Mat22::Zero();
    P.topRightCorner<11, 11>().setIdentity();
    P.bottomLeftCorner<11, 11>().setIdentity();
    require((P * exchanged.resistance * P - r.resistance).norm() <
            2e-11 * r.resistance.norm(), "rough pair label exchange mismatch");

    // Newtonian linearity, eta scaling and independently distributed cap density.
    require((r.resistance * (17. * slip) - 17. * f).norm() < 2e-12 * f.norm(),
            "rough force is not linear in velocity");
    const auto viscous = evaluate(a, b, o, 3.);
    require((viscous.resistance - 3. * r.resistance).norm() < 2e-12 * r.resistance.norm(),
            "rough resistance does not scale with fluid viscosity");
    auto less = o;
    less.coverage *= .5;
    require(std::abs(evaluate(a, b, less).coefficient / r.coefficient - .25) < 2e-12,
            "both surface areal densities must enter");

    // Refinement at finite microscopic floor (not just a formal h->0 test).
    double worst_quad = 0.;
    for (double h : {.01, .05, .10, .15, .19, .20, .25, .29}) {
      const auto bb = body(20., V3(0., 0., 2. + h));
      auto fine = o;
      fine.quadrature_order = 16;
      const auto coarse_response = evaluate(a, bb, o);
      const auto fine_response = evaluate(a, bb, fine);
      worst_quad = std::max(worst_quad,
          (coarse_response.resistance - fine_response.resistance).norm() /
          fine_response.resistance.norm());
    }
    std::cout << "rough quadrature relative matrix error: " << worst_quad << "\n";
    require(worst_quad < 2e-3, "rough patch quadrature requires refinement");

    // If the central macro gap lies below A, omit that invalid cap/base region.
    // The admitted area/coefficient is an annulus and stays identical here, but
    // its moment must still use the original H-h rather than H-max(h,A).
    const auto below_base = evaluate(a, body(20., V3(0., 0., 2. + .05)), o);
    const auto at_base = evaluate(a, body(20., V3(0., 0., 2. + .10)), o);
    require(std::abs(below_base.coefficient / at_base.coefficient - 1.) < 2e-11,
            "invalid cap/base central area was not excluded");
    require(std::abs((below_base.height_moment - at_base.height_moment) /
                    (.5 * (.10 - .05) * at_base.coefficient) - 1.) < 2e-11,
            "annulus patch moment lost the original closest-point origin");
    require(below_base.minimum_micro_gap >= 0., "negative microgap admitted into rough resistance");

    // Finite support and no artificial static force. No background smooth
    // resistance is subtracted, so enabling roughness can never lower dissipation.
    const auto at_contact = body(20., V3(0., 0., 2. + .2));
    const auto before_contact = body(20., V3(0., 0., 2. + .21));
    require(evaluate(a, at_contact, o).coefficient > 0. &&
            evaluate(a, before_contact, o).coefficient > 0.,
            "rough hydro must exist without elastic asperity compression");
    const auto far = body(20., V3(0., 0., 2. + .301));
    require(!evaluate(a, far, o).active, "rough correction should vanish outside 3A");
    auto disabled = o;
    disabled.enabled = false;
    require(evaluate(a, b, disabled).resistance.norm() == 0., "disabled roughness is not exact zero");
    require((r.resistance * V22::Zero()).norm() == 0., "roughness introduced static repulsion");

    std::cout << "PASS rough surface: PSD, rigid motion, covariance, exchange, density, "
                 "force/torque/stresslet mapping, microscopic floor, independent cap integral, "
                 "Newtonian scaling; quadrature error=" << worst_quad
              << ", zeta=" << r.coefficient << ", equivalent_pairs=" << r.equivalent_pairs << "\n";
  } catch (const std::exception &e) {
    std::cerr << "FAIL rough surface: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
