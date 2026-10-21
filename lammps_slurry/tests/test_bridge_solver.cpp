#include "slurry_bridge_solver.h"
#include "slurry_bridge_contact_solver.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
using namespace SlurryBridgeSolver;
using SlurryBridgeContactSolver::solve_proximal;
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
Problem simple(int n) {
  Problem p;
  p.count = n;
  p.dt = .1;
  p.prescribed = Vector::Zero(11 * n);
  p.external = Vector::Zero(6 * n);
  p.near_resistance.resize(11 * n, 11 * n);
  p.self_mobility.assign(n, Matrix11::Identity());
  p.mobility = [](const Vector &x) { return x; };
  return p;
}
Contact pair() {
  Contact c;
  c.i = 0;
  c.j = 1;
  c.Bi.leftCols<3>() = -Eigen::Matrix3d::Identity();
  c.Bj.leftCols<3>() = Eigen::Matrix3d::Identity();
  c.normal = V3::UnitZ();
  c.kn = 100.;
  return c;
}
void check_proximal_derivatives(const Contact &contact, const Vector &velocity, double dt) {
  using SlurryBridgeContactSolver::proximal_response;
  const auto original = contact_response(contact, velocity, dt);
  const SlurryBridgeContactSolver::Scale scale{.7, .4};
  const auto response =
      proximal_response(contact, original.relative, original.traction, dt, scale);
  require(response.residual.norm() < 1e-12,
          "original Coulomb root is not a proximal root");
  require((response.traction_inverse * response.traction_derivative -
           Eigen::Matrix3d::Identity()).norm() < 1e-10,
          "analytic proximal traction inverse is incorrect");
  require((-response.traction_inverse * response.velocity_derivative -
           original.derivative).norm() < 1e-9,
          "condensed proximal Jacobian changed the original contact derivative");
  Eigen::Matrix3d velocity_difference, traction_difference;
  const double eps = 1e-6;
  for (int axis = 0; axis < 3; ++axis) {
    const V3 delta = eps * V3::Unit(axis);
    velocity_difference.col(axis) =
        (proximal_response(contact, original.relative + delta, original.traction, dt, scale).residual -
         proximal_response(contact, original.relative - delta, original.traction, dt, scale).residual) /
        (2. * eps);
    traction_difference.col(axis) =
        (proximal_response(contact, original.relative, original.traction + delta, dt, scale).residual -
         proximal_response(contact, original.relative, original.traction - delta, dt, scale).residual) /
        (2. * eps);
  }
  require((velocity_difference - response.velocity_derivative).norm() < 1e-7 &&
              (traction_difference - response.traction_derivative).norm() < 1e-7,
          "proximal contact Jacobian disagrees with finite differences");
}
int main() {
  Options options;
  options.relative_tolerance = 1e-10;
  options.absolute_tolerance = 1e-12;
  options.linear_relative_tolerance = 1e-11;
  // Entire coupled 11N/6N solve against an independently assembled dense reference.
  auto p = simple(3);
  const int nm = 33, nu = 18;
  std::mt19937 generator(92);
  std::normal_distribution<double> random;
  Eigen::MatrixXd a(nm, nm), b(nm, nm);
  for (int i = 0; i < a.size(); ++i) {
    a.data()[i] = random(generator);
    b.data()[i] = random(generator);
  }
  Eigen::MatrixXd m = Eigen::MatrixXd::Identity(nm, nm) + .015 * a * a.transpose();
  Eigen::MatrixXd l = .02 * b * b.transpose();
  p.near_resistance = l.sparseView();
  p.mobility = [&](const Vector &x) -> Vector { return m * x; };
  for (int i = 0; i < 3; ++i)
    p.self_mobility[i] = m.block<11, 11>(11 * i, 11 * i);
  for (int i = 0; i < 3; ++i)
    for (int j = 6; j < 11; ++j)
      p.prescribed[11 * i + j] = random(generator);
  for (int i = 0; i < nu; ++i)
    p.external[i] = random(generator) * .1;
  Eigen::MatrixXd projection = Eigen::MatrixXd::Zero(nm, nu);
  for (int i = 0; i < 3; ++i)
    projection.block<6, 6>(11 * i, 6 * i).setIdentity();
  const Eigen::MatrixXd resistance = m.inverse() + l;
  const Vector reference =
      (projection.transpose() * resistance * projection)
          .ldlt()
          .solve(p.external - projection.transpose() * resistance * p.prescribed);
  const auto r = solve(p, options);
  const Problem dense_problem = p;
  require((r.velocity - reference).norm() < 1e-8, "mixed solve disagrees with dense reference");
  require((r.mid_moment - m.ldlt().solve(r.q)).norm() < 1e-8, "mid moments wrong");
  require(r.hydro_power >= 0., "negative hydrodynamic dissipation");
  const auto proximal_dense = solve_proximal(p, options);
  require((proximal_dense.velocity - reference).norm() < 1e-8 &&
              (proximal_dense.mid_moment - m.ldlt().solve(proximal_dense.q)).norm() < 1e-8,
          "proximal solver disagrees with independent dense hydrodynamic reference");
  // Inexact forcing can reduce inner work, but its bound must not relax the
  // original outer stopping criterion or change this independent dense root.
  Options forcing_options = options;
  forcing_options.max_linear_forcing = .05;
  int forcing_traces = 0;
  forcing_options.trace_callback = [&](const IterationTrace &trace) {
    if (std::string(trace.phase) != "linear")
      return;
    ++forcing_traces;
    require(trace.linear_target >= options.linear_relative_tolerance * trace.residual &&
                trace.linear_target <= .05 * trace.residual * (1. + 1e-12),
            "inexact linear forcing escaped its configured bounds");
  };
  const auto forced = solve(p, forcing_options);
  require(forcing_traces > 0 && (forced.velocity - reference).norm() < 1e-8 &&
              forced.relative_residual <= options.relative_tolerance * 1.01,
          "inexact forcing changed the final physical root or stopping rule");
  const auto warm = solve(p, options, &r);
  require(warm.linear_iterations == 0, "warm start should already solve unchanged configuration");
  // Hydrodynamic candidates far outside contact must contribute no penalty
  // stiffness. Even enormous kn/kt cannot change that solve/preconditioner.
  const detail::Preconditioner no_candidates(p, Vector::Zero(nu));
  auto separated = pair();
  separated.kn = 1e20;
  separated.kt = 1e20;
  separated.mu = .1;
  separated.gap = 100.;
  p.contacts.push_back(separated);
  const detail::Preconditioner with_candidate(p, Vector::Zero(nu));
  const Vector preconditioner_probe = Vector::LinSpaced(nm + nu, -1., 1.);
  require((no_candidates.right(preconditioner_probe) - with_candidate.right(preconditioner_probe))
                  .norm() == 0.,
          "separated candidate added artificial preconditioner stiffness");
  Eigen::MatrixXd approximate = projection.transpose() * l * projection;
  for (int i = 0; i < p.count; ++i)
    approximate.block<6, 6>(6 * i, 6 * i) += p.self_mobility[i].inverse().topLeftCorner<6, 6>();
  Eigen::MatrixXd whitening(nu, nu);
  for (int j = 0; j < nu; ++j) {
    Vector unit = Vector::Zero(nm + nu);
    unit[nm + j] = 1.;
    whitening.col(j) = no_candidates.right(unit).tail(nu);
  }
  require(
      (whitening.transpose() * approximate * whitening - Eigen::MatrixXd::Identity(nu, nu)).norm() <
          1e-10,
      "global sparse whitening or permutation is incorrect");
  require(std::abs(preconditioner_probe.dot(no_candidates.right(preconditioner_probe)) -
                   no_candidates.left(preconditioner_probe).dot(preconditioner_probe)) < 1e-10,
          "left and right preconditioners are not transposes");
  require((no_candidates.inverse_left(no_candidates.left(preconditioner_probe)) -
           preconditioner_probe).norm() < 1e-10,
          "inverse fixed-metric scaling or permutation is incorrect");
  const auto separated_result = solve(p, options);
  require((separated_result.velocity - r.velocity).norm() < 1e-10,
          "separated candidate changed velocity");
  require(separated_result.linear_iterations == r.linear_iterations,
          "separated candidate changed Krylov cost");
  // The coupled inverse must preserve the nonsymmetric mu*N derivative and
  // the off-diagonal force/velocity blocks. Assemble an independent dense
  // mixed Jacobian; with block-diagonal mobility this inverse is exact.
  auto schur_problem = dense_problem;
  auto sliding_pair = pair();
  sliding_pair.gap = -.1;
  sliding_pair.kt = 35.;
  sliding_pair.mu = .2;
  sliding_pair.history = V3(.05, .02, 0.);
  sliding_pair.affine = V3(3., .2, -.5);
  sliding_pair.Bi(0, 4) = .7;
  sliding_pair.Bj(1, 3) = -.4;
  auto sliding_image = sliding_pair;
  sliding_image.i = sliding_image.j = 2;
  sliding_image.Bj(0, 4) = .6;
  schur_problem.contacts = {sliding_pair, sliding_image};
  const Vector schur_velocity = Vector::LinSpaced(nu, -.3, .4);
  std::vector<ContactResponse> schur_state;
  Eigen::MatrixXd schur_a = projection.transpose() * l * projection;
  for (const auto &contact : schur_problem.contacts) {
    const auto response = contact_response(contact, schur_velocity, schur_problem.dt);
    require(response.active && response.sliding, "Schur fixture must exercise sliding contact");
    schur_state.push_back(response);
    Eigen::MatrixXd map = Eigen::MatrixXd::Zero(3, nu);
    map.block<3, 6>(0, 6 * contact.i) += contact.Bi;
    map.block<3, 6>(0, 6 * contact.j) += contact.Bj;
    schur_a += map.transpose() * response.derivative * map;
  }
  require((schur_a - schur_a.transpose()).norm() > 1e-3,
          "Schur fixture lost the nonsymmetric Coulomb coupling");
  Eigen::MatrixXd mixed_jacobian = Eigen::MatrixXd::Zero(nm + nu, nm + nu);
  for (int i = 0; i < schur_problem.count; ++i)
    mixed_jacobian.block<11, 11>(11 * i, 11 * i) = schur_problem.self_mobility[i];
  mixed_jacobian.topRightCorner(nm, nu) = -projection;
  mixed_jacobian.bottomLeftCorner(nu, nm) = -projection.transpose();
  mixed_jacobian.bottomRightCorner(nu, nu) = -schur_a;
  const detail::CoupledPreconditioner coupled(schur_problem, schur_state);
  const Vector dense_correction = mixed_jacobian.fullPivLu().solve(preconditioner_probe);
  const Vector schur_correction = coupled.apply(preconditioner_probe);
  require((schur_correction - dense_correction).norm() < 1e-10,
          "coupled Schur inverse disagrees with the full nonsymmetric dense Jacobian");
  require((mixed_jacobian * schur_correction - preconditioner_probe).norm() < 1e-10,
          "coupled Schur inverse fails the true mixed residual");
  const Apply mixed_apply = [&](const Vector &value) -> Vector {
    return no_candidates.left(mixed_jacobian * value);
  };
  const Apply mixed_inverse = [&](const Vector &value) -> Vector {
    return coupled.apply(no_candidates.inverse_left(value));
  };
  const Vector scaled_rhs = no_candidates.left(preconditioner_probe);
  const auto one_step = detail::fgmres(mixed_apply, mixed_inverse, scaled_rhs,
                                      1e-11 * scaled_rhs.norm(), options);
  require(one_step.iterations == 1 && (one_step.x - dense_correction).norm() < 1e-10,
          "right FGMRES did not use the exact coupled inverse in one iteration");
  // Contact activated by the evolving iterate, rather than initial affine
  // flow, must still be found (all candidates stay in the residual).
  auto activation = simple(2);
  auto ac = pair();
  ac.gap = .05;
  ac.onset = 0.;
  ac.kn = 1e5;
  activation.contacts.push_back(ac);
  activation.external[2] = 1.;
  activation.external[8] = -1.;
  const auto activated = solve(activation, options);
  require(activated.contacts[0].active, "future contact activation was missed");
  require(activated.force_residual_norm < 1e-8, "activated contact force balance failed");
  // Backward Euler contact, with a closed-form normal load.
  p = simple(2);
  auto c = pair();
  c.onset = .1;
  c.affine.z() = -1.;
  p.contacts.push_back(c);
  auto normal = solve(p, options);
  const double load = 20. / 21.;
  require(std::abs(normal.contacts[0].normal_load - load) < 1e-9,
          "implicit normal contact load incorrect");
  require(std::abs(normal.velocity[2] + load) < 1e-9 && std::abs(normal.velocity[8] - load) < 1e-9,
          "contact force sign incorrect");
  const auto proximal_normal = solve_proximal(p, options);
  require(std::abs(proximal_normal.contacts[0].normal_load - load) < 1e-9 &&
              std::abs(proximal_normal.velocity[2] + load) < 1e-9 &&
              std::abs(proximal_normal.velocity[8] - load) < 1e-9,
          "proximal normal contact disagrees with the independent backward-Euler root");
  // Tangential stick/slip and mu*N derivative.
  p.contacts[0].affine.x() = 1.;
  p.contacts[0].kt = 10.;
  p.contacts[0].mu = 1.;
  auto stick = solve(p, options);
  require(!stick.contacts[0].sliding, "sticking contact classified as sliding");
  require(std::abs(stick.contacts[0].tangent_force.x() - 1. / 3.) < 1e-9,
          "sticking force incorrect");
  check_proximal_derivatives(p.contacts[0], stick.velocity, p.dt);
  const auto proximal_stick = solve_proximal(p, options);
  require(!proximal_stick.contacts[0].sliding &&
              std::abs(proximal_stick.contacts[0].tangent_force.x() - 1. / 3.) < 1e-9 &&
              (proximal_stick.velocity - stick.velocity).norm() < 1e-9,
          "proximal sticking contact changed the independent elastic root");
  p.contacts[0].mu = .1;
  auto slide = solve(p, options);
  require(slide.contacts[0].sliding, "sliding contact classified as sticking");
  require(std::abs(slide.contacts[0].tangent_force.x() - .1 * load) < 1e-9,
          "Coulomb limit incorrect");
  require(slide.contacts[0].sliding_work >= 0., "sliding dissipation negative");
  check_proximal_derivatives(p.contacts[0], slide.velocity, p.dt);
  const auto proximal_slide = solve_proximal(p, options);
  require(proximal_slide.contacts[0].sliding &&
              std::abs(proximal_slide.contacts[0].tangent_force.x() - .1 * load) < 1e-9 &&
              (proximal_slide.velocity - slide.velocity).norm() < 1e-9,
          "proximal sliding contact changed the independent Coulomb root");
  // Gradient verification with varying normal load, not a frozen mu*N approximation.
  Vector u = Vector::Zero(12);
  u[0] = .15;
  u[8] = .1;
  const auto state = contact_response(p.contacts[0], u, p.dt);
  Vector direction = Vector::Zero(12);
  direction[0] = .3;
  direction[8] = .2;
  const double eps = 1e-6;
  const V3 finite_difference =
      (contact_response(p.contacts[0], u + eps * direction, p.dt).traction -
       contact_response(p.contacts[0], u - eps * direction, p.dt).traction) /
      (2. * eps);
  const V3 dv = p.contacts[0].Bi * direction.head<6>() + p.contacts[0].Bj * direction.tail<6>();
  require((finite_difference - state.derivative * dv).norm() < 1e-7,
          "sliding contact Jacobian incorrect");
  // Newton trials must use the same committed history. Reassemble the final
  // physical equations and repeat the solve without committing a time step.
  auto history_problem = p;
  const V3 initial_history(.03, .02, 0.);
  history_problem.contacts[0].history = initial_history;
  const auto history_result = solve(history_problem, options);
  Vector history_q = history_problem.prescribed;
  for (int i = 0; i < history_problem.count; ++i)
    history_q.segment<6>(11 * i) += history_result.velocity.segment<6>(6 * i);
  const Vector history_moment =
      history_result.mid_moment + history_problem.near_resistance * history_q;
  Vector force_balance = history_problem.external;
  for (int i = 0; i < history_problem.count; ++i)
    force_balance.segment<6>(6 * i) -= history_moment.segment<6>(11 * i);
  for (const auto &contact : history_problem.contacts) {
    const auto response = contact_response(contact, history_result.velocity, history_problem.dt);
    force_balance.segment<6>(6 * contact.i) -= contact.Bi.transpose() * response.traction;
    force_balance.segment<6>(6 * contact.j) -= contact.Bj.transpose() * response.traction;
  }
  require(force_balance.norm() < 1e-9 &&
              (history_problem.mobility(history_result.mid_moment) - history_q).norm() < 1e-9,
          "reported convergence does not satisfy the physical contact/mobility equations");
  require((history_problem.contacts[0].history - initial_history).norm() == 0.,
          "Newton iterations mutated committed contact history");
  const auto history_warm = solve(history_problem, options, &history_result);
  require(history_warm.newton_iterations == 0 &&
              (history_warm.contacts[0].tangent_history -
               history_result.contacts[0].tangent_history).norm() < 1e-12 &&
              (history_problem.contacts[0].history - initial_history).norm() == 0.,
          "repeated solve advanced contact history without a committed time step");
  // Cold-start tangent-stiffness continuation is an initial-guess strategy, not a
  // change to the final force law. This sliding pair has an independent
  // closed-form root even with nonzero, oblique committed tangent history.
  auto continuation_problem = history_problem;
  int mobility_calls = 0;
  continuation_problem.mobility = [&](const Vector &value) -> Vector {
    ++mobility_calls;
    return value;
  };
  Options continuation_options = options;
  continuation_options.cold_start_continuation = true;
  int traced_newton = 0, traced_linear = 0, last_total_newton = 0, last_total_linear = 0;
  bool traced_zero_stiffness = false, traced_full_stiffness = false;
  continuation_options.trace_callback = [&](const IterationTrace &trace) {
    require(trace.friction_scale == 1., "stiffness continuation changed the Coulomb coefficient");
    require(trace.stiffness_scale >= 0. && trace.stiffness_scale <= 1.,
            "continuation evaluated an invalid stiffness scale");
    require(trace.total_newton_iterations >= last_total_newton &&
                trace.total_linear_iterations >= last_total_linear,
            "continuation diagnostic totals reset between stages");
    last_total_newton = trace.total_newton_iterations;
    last_total_linear = trace.total_linear_iterations;
    if (std::string(trace.phase) == "accepted")
      ++traced_newton;
    if (std::string(trace.phase) == "linear")
      traced_linear += trace.linear_iterations;
    if (std::string(trace.phase) == "stage") {
      traced_zero_stiffness |= trace.stiffness_scale == 0.;
      traced_full_stiffness |= trace.stiffness_scale == 1.;
    }
  };
  const auto continued = solve(continuation_problem, continuation_options);
  const V3 tangent_drive = initial_history + V3(.1, 0., 0.);
  V3 exact_traction = .1 * load * tangent_drive.normalized();
  exact_traction.z() = -load;
  Vector exact_velocity = Vector::Zero(12);
  exact_velocity.head<3>() = exact_traction;
  exact_velocity.segment<3>(6) = -exact_traction;
  // A saturated sliding force can already be the final root before kt reaches
  // its original value, so require stage-entry events rather than Newton
  // iterations at both endpoints. The final tangent history also checks kt.
  require(continued.continuation_stages >= 2 && traced_zero_stiffness && traced_full_stiffness,
          "cold continuation did not reach zero and original tangent-stiffness stages");
  require((continued.velocity - exact_velocity).norm() < 1e-9 &&
              (continued.velocity - history_result.velocity).norm() < 1e-9 &&
              (continued.contacts[0].traction - exact_traction).norm() < 1e-9,
          "continuation did not solve the original Coulomb force law");
  require(continued.mobility_residual_norm < 1e-9 && continued.force_residual_norm < 1e-9,
          "continuation accepted a surrogate residual at the final stage");
  require(continued.newton_iterations == traced_newton &&
              continued.linear_iterations == traced_linear &&
              continued.mobility_applications == mobility_calls,
          "continuation diagnostics omit work from intermediate stages");
  require((continuation_problem.contacts[0].history - initial_history).norm() == 0. &&
              (continued.contacts[0].tangent_history -
               history_result.contacts[0].tangent_history).norm() < 1e-12,
          "continuation stages advanced committed contact history");
  continuation_options.trace_callback = {};
  const auto continued_warm = solve(continuation_problem, continuation_options, &continued);
  require(continued_warm.continuation_stages == 0 && continued_warm.continuation_retries == 0 &&
              continued_warm.newton_iterations == 0 && continued_warm.linear_iterations == 0 &&
              (continued_warm.velocity - continued.velocity).norm() < 1e-12,
          "valid warm solve unnecessarily restarted stiffness continuation");
  mobility_calls = 0;
  const auto proximal_history = solve_proximal(continuation_problem, options);
  require((proximal_history.velocity - exact_velocity).norm() < 1e-9 &&
              proximal_history.force_residual_norm < 1e-9 &&
              proximal_history.mobility_residual_norm < 1e-9 &&
              (proximal_history.contacts[0].traction - exact_traction).norm() < 1e-9 &&
              (proximal_history.contacts[0].tangent_history -
               history_result.contacts[0].tangent_history).norm() < 1e-12,
          "proximal formulation failed the original oblique-contact root or physical residual");
  require((continuation_problem.contacts[0].history - initial_history).norm() == 0. &&
              proximal_history.mobility_applications == mobility_calls,
          "proximal solve changed committed history or omitted mobility work");
  const auto proximal_warm = solve_proximal(continuation_problem, options, &proximal_history);
  require(proximal_warm.newton_iterations == 0 && proximal_warm.linear_iterations == 0 &&
              (proximal_warm.velocity - proximal_history.velocity).norm() < 1e-12 &&
              (proximal_warm.contacts[0].tangent_history -
               proximal_history.contacts[0].tangent_history).norm() < 1e-12 &&
              (continuation_problem.contacts[0].history - initial_history).norm() == 0.,
          "proximal warm solve advanced time or committed contact history");
  // A stale but finite warm guess with a rotated friction direction cannot
  // reach this nonlinear root in one Newton step. Soft-stiffness stages can;
  // their rescue must include the failed direct work and preserve history.
  auto stale_warm = history_result;
  stale_warm.velocity[1] += .75;
  stale_warm.mid_moment.head<6>() = stale_warm.velocity.head<6>();
  Options rescue_options = options;
  rescue_options.maximum_newton_iterations = 1;
  rescue_options.max_linear_forcing = .05;
  rescue_options.max_breakpoint_intervals = 0;
  bool direct_nonlinear_failure = false;
  try {
    solve(history_problem, rescue_options, &stale_warm);
  } catch (const NonlinearFailure &) {
    direct_nonlinear_failure = true;
  }
  require(direct_nonlinear_failure, "warm-rescue fixture no longer exercises nonlinear failure");
  rescue_options.cold_start_continuation = true;
  int warm_retry_traces = 0, rescue_newton = 0, rescue_linear = 0;
  rescue_options.trace_callback = [&](const IterationTrace &trace) {
    warm_retry_traces += std::string(trace.phase) == "warm_retry";
    rescue_newton += std::string(trace.phase) == "accepted";
    if (std::string(trace.phase) == "linear")
      rescue_linear += trace.linear_iterations;
  };
  mobility_calls = 0;
  const auto rescued = solve(continuation_problem, rescue_options, &stale_warm);
  require(warm_retry_traces == 1 && rescued.continuation_retries >= 1 &&
              rescued.continuation_stages >= 2,
          "failed warm solve did not enter bounded stiffness continuation");
  require((rescued.velocity - exact_velocity).norm() < 1e-9 &&
              rescued.force_residual_norm < 1e-9 && rescued.mobility_residual_norm < 1e-9 &&
              (rescued.contacts[0].tangent_history -
               history_result.contacts[0].tangent_history).norm() < 1e-12 &&
              (continuation_problem.contacts[0].history - initial_history).norm() == 0.,
          "warm rescue changed the original root or advanced committed history");
  require(rescued.newton_iterations == rescue_newton && rescued.linear_iterations == rescue_linear &&
              rescued.mobility_applications == mobility_calls,
          "warm rescue diagnostics omit the failed direct solve");
  rescue_options.trace_callback = {};
  const auto rescue_warm = solve(continuation_problem, rescue_options, &rescued);
  require(rescue_warm.newton_iterations == 0 && rescue_warm.linear_iterations == 0 &&
              rescue_warm.continuation_stages == 0 && rescue_warm.continuation_retries == 0,
          "already converged warm guess needlessly ran rescue continuation");
  // A scalar Coulomb return map with a narrow sticking interval stalls under
  // repeated halving. Locate that interval without smoothing the force law.
  auto narrow_problem = simple(2);
  auto narrow_contact = pair();
  narrow_contact.onset = .1;
  narrow_contact.affine = V3(.8, 0., -1.);
  narrow_contact.history = V3(.02, 0., 0.);
  narrow_contact.kt = 1e7;
  narrow_contact.mu = 1.;
  narrow_problem.contacts = {narrow_contact};
  Result preloaded;
  preloaded.velocity = Vector::Zero(12);
  preloaded.velocity[2] = -load;
  preloaded.velocity[8] = load;
  preloaded.mid_moment = Vector::Zero(22);
  preloaded.mid_moment.head<6>() = preloaded.velocity.head<6>();
  preloaded.mid_moment.segment<6>(11) = preloaded.velocity.tail<6>();
  Options breakpoint_options = options;
  breakpoint_options.maximum_newton_iterations = 8;
  breakpoint_options.max_breakpoint_intervals = 0;
  double minimum_halving_step = 1.;
  int halving_iterations = 0;
  breakpoint_options.trace_callback = [&](const IterationTrace &trace) {
    if (std::string(trace.phase) == "accepted") {
      minimum_halving_step = std::min(minimum_halving_step, trace.step_length);
      ++halving_iterations;
    }
  };
  bool halving_stalled = false;
  try {
    solve(narrow_problem, breakpoint_options, &preloaded);
  } catch (const NonlinearFailure &) {
    halving_stalled = true;
  }
  require(halving_stalled && minimum_halving_step < 1e-3,
          "breakpoint fixture no longer exercises tiny-step Coulomb halving");
  breakpoint_options.max_breakpoint_intervals = 4;
  int breakpoint_traces = 0;
  breakpoint_options.trace_callback = [&](const IterationTrace &trace) {
    if (std::string(trace.phase) == "accepted" && trace.breakpoint)
      ++breakpoint_traces;
  };
  const auto narrow_result = solve(narrow_problem, breakpoint_options, &preloaded);
  const double exact_stick = narrow_contact.kt * .1 / (1. + 2. * narrow_contact.kt * .1);
  Vector narrow_exact = preloaded.velocity;
  narrow_exact[0] = exact_stick;
  narrow_exact[6] = -exact_stick;
  const auto narrow_response =
      contact_response(narrow_contact, narrow_result.velocity, narrow_problem.dt);
  require(narrow_result.breakpoint_steps > 0 &&
              narrow_result.breakpoint_steps == breakpoint_traces &&
              narrow_result.newton_iterations < halving_iterations,
          "breakpoint search did not resolve the narrow sticking interval");
  require(!narrow_response.sliding && (narrow_result.velocity - narrow_exact).norm() < 1e-9 &&
              (narrow_result.velocity.head<3>() - narrow_response.traction).norm() < 1e-8 &&
              (narrow_result.velocity.segment<3>(6) + narrow_response.traction).norm() < 1e-8 &&
              narrow_result.mobility_residual_norm < 1e-9,
          "breakpoint search changed the exact contact root or final physical residual");
  require((narrow_problem.contacts[0].history - narrow_contact.history).norm() == 0.,
          "breakpoint trials mutated committed contact history");
  // Expanding the yield polynomial around alpha=0 loses this 2e-11-wide
  // sticking interval through cancellation. Its two distinct roots matter.
  Contact tiny_interval;
  tiny_interval.i = tiny_interval.j = 0;
  tiny_interval.Bi.leftCols<3>().setIdentity();
  tiny_interval.gap = -.1;
  tiny_interval.kn = 1.;
  tiny_interval.kt = 1e10;
  tiny_interval.mu = 1.;
  tiny_interval.history = V3(-.5, 0., 0.);
  const Vector stationary = Vector::Zero(6);
  Vector crossing = Vector::Zero(6);
  crossing[0] = 1.;
  const auto tiny_ranges =
      detail::contact_line_intervals(tiny_interval, stationary, crossing, 1., .25, .75);
  require(tiny_ranges.size() == 1 &&
              std::abs(tiny_ranges[0].first - (.5 - 1e-11)) < 1e-15 &&
              std::abs(tiny_ranges[0].second - (.5 + 1e-11)) < 1e-15,
          "yield root cancellation discarded a narrow sticking interval");
  require(contact_response(tiny_interval, (.5 - 2e-11) * crossing, 1.).sliding &&
              !contact_response(tiny_interval, .5 * crossing, 1.).sliding &&
              contact_response(tiny_interval, (.5 + 2e-11) * crossing, 1.).sliding,
          "narrow interval disagrees with the original Coulomb branch law");
  // Exactly activating contact with severe stiffness must select the closing
  // derivative, not require billions of tiny line-search reductions.
  p = simple(2);
  c = pair();
  c.kn = 1e8;
  c.gap = c.onset = .02;
  p.contacts = {c};
  p.external[2] = 1.;
  p.external[8] = -1.;
  Options kink_options = options;
  kink_options.linear_relative_tolerance = 1e-8;
  auto kink = solve(p, kink_options);
  require(kink.kink_reselections > 0, "activation kink derivative was not selected");
  require(std::abs(kink.contacts[0].normal_load - 1.) < 1e-5, "stiff activation load incorrect");
  // Periodic self image: the same unknown appears at both endpoints.
  p = simple(1);
  c = pair();
  c.j = 0;
  c.onset = .1;
  c.affine = V3(.1, 0, -1.);
  c.kt = 10.;
  c.mu = .1;
  c.Bi(0, 4) = 1.;
  c.Bj(0, 4) = 1.;
  p.contacts = {c};
  auto self = solve(p, options);
  require(std::abs(self.velocity[4]) > 1e-8, "self-image torque was discarded");
  require(self.force_residual_norm < 1e-8, "self-image residual too large");
  // A solve cap must fail rather than report an unconverged velocity.
  bool failed = false;
  options.maximum_linear_iterations = 1;
  try {
    // The self-only fixture now has an exact coupled inverse and can converge
    // in one Krylov step. Use genuine interparticle mobility for this gate.
    solve(dense_problem, options);
  } catch (const std::runtime_error &) {
    failed = true;
  }
  require(failed, "unconverged solve was accepted");
  // Reach a positive-stiffness continuation stage before forcing a genuine
  // Krylov failure. The initial external load exactly balances frictionless
  // normal contact at rest, so the zero-stiffness stage needs no linear solve.
  auto capped_continuation = dense_problem;
  capped_continuation.prescribed.setZero();
  capped_continuation.external.setZero();
  capped_continuation.contacts = {sliding_pair};
  auto zero_stiffness_contact = sliding_pair;
  zero_stiffness_contact.kt = 0.;
  const auto initial_normal =
      contact_response(zero_stiffness_contact, Vector::Zero(nu), capped_continuation.dt);
  capped_continuation.external.segment<6>(6 * sliding_pair.i) +=
      sliding_pair.Bi.transpose() * initial_normal.traction;
  capped_continuation.external.segment<6>(6 * sliding_pair.j) +=
      sliding_pair.Bj.transpose() * initial_normal.traction;
  Options capped_options = options;
  capped_options.cold_start_continuation = true;
  int retry_traces = 0;
  bool reached_stiffness_stage = false;
  capped_options.trace_callback = [&](const IterationTrace &trace) {
    retry_traces += std::string(trace.phase) == "retry";
    reached_stiffness_stage |= trace.stiffness_scale > 0.;
  };
  bool linear_failure_propagated = false;
  try {
    solve(capped_continuation, capped_options);
  } catch (const NonlinearFailure &) {
    require(false, "continuation reclassified a linear failure as nonlinear");
  } catch (const std::runtime_error &error) {
    linear_failure_propagated =
        std::string(error.what()).find("GMRES did not converge") != std::string::npos;
  }
  require(reached_stiffness_stage && linear_failure_propagated && retry_traces == 0,
          "continuation masked or retried a failed linear solve");
  // Neither NaN tolerances nor finite vectors whose norm overflows may turn
  // a false stopping comparison into a successful zero-iteration result.
  auto invalid_problem = simple(1);
  invalid_problem.external[0] = 1.;
  Options invalid_options;
  invalid_options.relative_tolerance = std::numeric_limits<double>::quiet_NaN();
  bool invalid_tolerance_rejected = false;
  try {
    solve(invalid_problem, invalid_options);
  } catch (const std::invalid_argument &) {
    invalid_tolerance_rejected = true;
  }
  require(invalid_tolerance_rejected, "NaN solver tolerance was accepted");
  invalid_problem.external[0] = 1e200;
  bool overflowing_norm_rejected = false;
  try {
    solve(invalid_problem);
  } catch (const std::exception &) {
    overflowing_norm_rejected = true;
  }
  require(overflowing_norm_rejected, "nonfinite residual norm was reported as convergence");
  std::cout << "bridge mixed solve: dense reference, coupled nonsymmetric Schur inverse, "
               "right FGMRES, warm start, implicit contact, stick/slip, Jacobian, self images, "
               "continuation, proximal, history, failure gate PASS\n";
}
