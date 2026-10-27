#ifndef SLURRY_BRIDGE_LINEAR_SOLVER_H
#define SLURRY_BRIDGE_LINEAR_SOLVER_H

// Force balance at a known configuration and known spring histories.
// Contact forces are fixed data of this linear solve. Updating a spring for
// the next time step is a separate operation and never changes this root.
#include "slurry_bridge_solver.h"

namespace SlurryBridgeLinearSolver {
namespace bs = SlurryBridgeSolver;

inline bs::Result solve(const bs::Problem &p,
                        const bs::Options &options = bs::Options{},
                        const bs::Result *warm = nullptr) {
  bs::detail::validate(p, options);
  const int n = p.count, nm = 11 * n, nu = 6 * n;
  const auto started = std::chrono::steady_clock::now();
  bs::Result out;
  bs::Problem hydro = p;
  hydro.contacts.clear();
  out.contacts.reserve(p.contacts.size());
  const bs::Vector zero_velocity = bs::Vector::Zero(nu);
  for (const auto &c : p.contacts) {
    // dt=0 evaluates the current overlap and committed histories, including
    // a Coulomb projection if a changed normal load lowers the current cap.
    const auto response = bs::contact_response(c, zero_velocity, 0.);
    hydro.external.segment<6>(6 * c.i).noalias() -=
        bs::generalized_contact_map(c, true).transpose() * bs::contact_wrench(response);
    hydro.external.segment<6>(6 * c.j).noalias() -=
        bs::generalized_contact_map(c, false).transpose() * bs::contact_wrench(response);
    out.contacts.push_back(response);
  }

  auto workspace = options.workspace ? options.workspace
      : std::make_shared<bs::detail::SolverWorkspace>();
  bs::Options linear_options = options;
  linear_options.workspace = workspace;
  bs::detail::Preconditioner monitor(hydro, zero_velocity, workspace);
  auto mobility = [&](const bs::Vector &moment) -> bs::Vector {
    if (moment.isZero(0.)) return bs::Vector::Zero(nm);
    ++out.mobility_applications;
    return p.mobility(moment);
  };
  auto residual = [&](const bs::Vector &x) -> bs::Vector {
    const bs::Vector q = bs::detail::embed(x.tail(nu), n) + p.prescribed;
    bs::Vector r(nm + nu);
    r.head(nm) = mobility(x.head(nm)) - q;
    r.tail(nu) = -bs::detail::select(x.head(nm) + p.near_resistance * q, n)
                  + hydro.external;
    if (!r.allFinite())
      throw std::runtime_error("bridge linear solver: nonfinite residual");
    return r;
  };
  auto matrix = [&](const bs::Vector &x) -> bs::Vector {
    const bs::Vector q = bs::detail::embed(x.tail(nu), n);
    bs::Vector y(nm + nu);
    y.head(nm) = mobility(x.head(nm)) - q;
    y.tail(nu) = -bs::detail::select(x.head(nm) + p.near_resistance * q, n);
    return monitor.left(y);
  };
  const double rhs_norm = monitor.left(residual(bs::Vector::Zero(nm + nu))).norm();
  const double tolerance = options.absolute_tolerance + options.relative_tolerance * rhs_norm;
  bs::Vector x = bs::Vector::Zero(nm + nu);
  if (warm && warm->velocity.size() == nu && warm->mid_moment.size() == nm &&
      warm->velocity.allFinite() && warm->mid_moment.allFinite()) {
    x.head(nm) = warm->mid_moment;
    x.tail(nu) = warm->velocity;
  }
  bs::Vector raw = residual(x), weighted = monitor.left(raw);
  double norm = weighted.norm();
  if (!std::isfinite(rhs_norm) || !std::isfinite(tolerance) || !std::isfinite(norm))
    throw std::runtime_error("bridge linear solver: nonfinite convergence metric");
  bs::IterationTrace trace;
  trace.tolerance = tolerance;
  trace.residual = norm;
  trace.merit_residual = norm;
  for (const auto &c : out.contacts) {
    trace.active_contacts += c.active;
    trace.sliding_contacts += c.sliding;
  }
  auto emit = [&](const char *phase) {
    if (!options.trace_callback) return;
    trace.phase = phase;
    trace.total_linear_iterations = out.linear_iterations;
    trace.total_newton_iterations = 0;
    trace.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    options.trace_callback(trace);
  };
  emit("linear_begin");
  if (norm > tolerance) {
    const std::vector<bs::ContactResponse> no_contacts;
    bs::detail::CoupledPreconditioner preconditioner(hydro, no_contacts, workspace);
    auto inverse = [&](const bs::Vector &r) -> bs::Vector {
      return preconditioner.apply(monitor.inverse_left(r));
    };
    // Bound the solve by the physical residual tolerance. In particular an
    // almost converged warm start must not request sub-roundoff accuracy.
    const double target = std::min(.5 * tolerance,
        std::max(.1 * tolerance, options.linear_relative_tolerance * norm));
    const auto correction = bs::detail::fgmres(matrix, inverse, -weighted, target,
                                              linear_options);
    x += correction.x;
    out.linear_iterations = correction.iterations;
    trace.linear_iterations = correction.iterations;
    trace.linear_target = target;
    trace.linear_residual = correction.residual;
    raw = residual(x);
    weighted = monitor.left(raw);
    norm = weighted.norm();
  }
  if (!std::isfinite(norm) || norm > tolerance) {
    std::ostringstream message;
    message << std::scientific << std::setprecision(9)
            << "bridge linear solver: force/mobility residual did not converge; residual="
            << norm << ", tolerance=" << tolerance
            << ", linear iterations=" << out.linear_iterations;
    throw std::runtime_error(message.str());
  }
  out.velocity = x.tail(nu);
  out.mid_moment = x.head(nm);
  out.q = bs::detail::embed(out.velocity, n) + p.prescribed;
  out.total_moment = out.mid_moment + p.near_resistance * out.q;
  out.residual_norm = norm;
  out.relative_residual = norm / std::max({rhs_norm, options.absolute_tolerance,
                                          std::numeric_limits<double>::min()});
  out.force_residual_norm = raw.tail(nu).norm();
  out.mobility_residual_norm = raw.head(nm).norm();
  out.hydro_power = out.q.dot(out.total_moment);
  for (std::size_t k = 0; k < p.contacts.size(); ++k) {
    const auto &c = p.contacts[k];
    auto &r = out.contacts[k];
    r.relative = c.Bi * out.velocity.segment<6>(6 * c.i)
               + c.Bj * out.velocity.segment<6>(6 * c.j) + c.affine;
    r.relative_rotation = out.velocity.segment<3>(6 * c.j + 3)
                        - out.velocity.segment<3>(6 * c.i + 3);
    out.contact_power += bs::contact_power(r);
  }
  trace.residual = norm;
  trace.merit_residual = norm;
  emit("linear_accepted");
  return out;
}

inline std::vector<bs::ContactResponse>
advance_contacts(const bs::Problem &p, const bs::Result &current,
                 const std::vector<double> &endpoint_gaps) {
  if (current.contacts.size() != p.contacts.size() ||
      endpoint_gaps.size() != p.contacts.size() || current.velocity.size() != 6 * p.count ||
      !current.velocity.allFinite())
    throw std::invalid_argument("bridge linear history update: size mismatch");
  std::vector<bs::ContactResponse> next;
  next.reserve(p.contacts.size());
  for (std::size_t k = 0; k < p.contacts.size(); ++k) {
    if (!std::isfinite(endpoint_gaps[k]))
      throw std::invalid_argument("bridge linear history update: nonfinite endpoint gap");
    auto c = p.contacts[k];
    // Start from the current admissible history, not from an old history
    // that may have exceeded the cap after the normal load changed.
    c.history = current.contacts[k].tangent_history;
    c.rolling_history = current.contacts[k].rolling_history;
    c.twisting_history = current.contacts[k].twisting_history;
    // A new contact starts with an unstretched spring. Do not integrate
    // pre-contact sliding/rolling through a whole finite time step.
    const double history_dt = current.contacts[k].active ? p.dt : 0.;
    next.push_back(bs::contact_response(c, current.velocity, history_dt, endpoint_gaps[k]));
  }
  return next;
}
} // namespace SlurryBridgeLinearSolver
#endif
