#include "slurry_nearfield_imex_solver.h"
#include <iostream>

namespace bs = SlurryBridgeSolver;
namespace ns = SlurryNearfieldSolver;
namespace im = SlurryNearfieldImexSolver;
static void require(bool okay, const char *message) {
  if (!okay) throw std::runtime_error(message);
}

static bs::Problem problem(int index) {
  bs::Problem p; p.count = 3; p.dt = .001 * (1. + .05 * (index % 3));
  p.external = bs::Vector::Zero(18); p.prescribed = bs::Vector::Zero(33);
  p.self_mobility.assign(3, (1. + .02 * (index / 7)) * bs::Matrix11::Identity());
  bs::Vector v = bs::Vector::Zero(33);
  v[0] = 1.; v[11] = -1.; v[6] = .15; v[15] = .2;
  const Eigen::MatrixXd near = (.1 + .03 * (index / 4)) * v * v.transpose();
  p.near_resistance = near.sparseView();
  for (int i = 0; i < 3; ++i) {
    p.external[6*i] = .1 * std::sin(.1 * index + i);
    p.external[6*i+1] = .2 * std::cos(.17 * index + i);
    p.external[6*i+3] = .05 * std::sin(.23 * index + i);
    p.prescribed[11*i+6] = .1 * std::cos(.07 * index + i);
  }
  for (int i = 0; i < 2; ++i) {
    bs::Contact c; c.i = i; c.j = i+1;
    c.normal = Eigen::Vector3d::UnitZ(); c.gap = -.001;
    c.kn = 1e4; c.kt = 3e3; c.kr = 1e3; c.mu = .5; c.mu_roll = .3;
    c.roll_length = 1.;
    c.history = Eigen::Vector3d(.0001 * std::sin(.1 * index), .0002, 0.);
    c.rolling_history = Eigen::Vector3d(.0001, -.0001 * std::cos(.1 * index), 0.);
    c.Bi.leftCols<3>() = -Eigen::Matrix3d::Identity();
    c.Bj.leftCols<3>() = Eigen::Matrix3d::Identity();
    p.contacts.push_back(c);
  }
  return p;
}

int main() {
  try {
    bs::Options options; options.relative_tolerance = 1e-10; options.absolute_tolerance = 1e-12;
    im::Workspace optimized, old_style;
    optimized.use_contact_preconditioner = old_style.use_contact_preconditioner = true;
    optimized.reuse_contact_preconditioner_across_solves =
        old_style.reuse_contact_preconditioner_across_solves = true;
    bs::Result warm_fast, warm_old;
    double maximum_velocity_difference = 0.;
    constexpr int count = 32;
    for (int index = 0; index < count; ++index) {
      const auto p = problem(index); const auto caps = im::freeze_caps(p);
      // Reproduce the discarded explicit-contact solve formerly run before
      // every valid warm IMEX start. Its velocity is deliberately not used.
      if (index) (void)ns::solve(p, options, old_style);
      const auto reference = im::solve(p, caps, options, old_style, index ? &warm_old : nullptr);
      const auto result = im::solve(p, caps, options, optimized, index ? &warm_fast : nullptr);
      const double difference = (result.velocity - reference.velocity).norm();
      maximum_velocity_difference = std::max(maximum_velocity_difference, difference);
      require(difference <= 1e-13 * std::max(1., reference.velocity.norm()),
              "Removing unused cold solve changed converged warm velocity");
      require(bs::detail::same_matrix(optimized.matrix, old_style.matrix),
              "Current hydrodynamic resistance differs from independently prepared operator");
      const auto metric = im::acceptance_metric(p, caps, result, result.contacts, optimized, options);
      require(metric.residual_norm <= metric.tolerance, "Warm preparation bypassed physical residual");
      warm_fast = result; warm_old = reference;
    }
    require(old_style.solves == optimized.solves + count - 1,
            "Valid warm starts still execute an unused explicit force solve");
    require(optimized.cold_start_solves == 1 && optimized.warm_start_preparations == count - 1,
            "Warm/cold initialization counters are inconsistent");
    require(optimized.resistance_preparation_reuses > 0 &&
            optimized.resistance_assemblies < optimized.resistance_preparations,
            "Exact unchanged-operator snapshot did not avoid assembly");

    auto invalid = problem(count - 1); invalid.dt = 0.;
    bool bad_dt = false;
    try { ns::prepare_resistance(invalid, optimized, options); }
    catch (const std::invalid_argument &) { bad_dt = true; }
    require(bad_dt, "Exact operator cache hid invalid current timestep");
    invalid = problem(count - 1); invalid.near_resistance.coeffRef(0, 1) += .01;
    invalid.near_resistance.makeCompressed();
    bool bad_matrix = false;
    try { ns::prepare_resistance(invalid, optimized, options); }
    catch (const std::invalid_argument &) { bad_matrix = true; }
    require(bad_matrix, "Changed non-symmetric operator was accepted from stale cache");

    std::cout << "nearfield_warm_prepare PASS: max_velocity_difference="
              << maximum_velocity_difference << " old_explicit_solves=" << old_style.solves
              << " optimized_explicit_solves=" << optimized.solves
              << " exact_operator_reuses=" << optimized.resistance_preparation_reuses
              << " operator_assemblies=" << optimized.resistance_assemblies << '\n';
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
