#ifndef SLURRY_NEARFIELD_IMPLICIT_SOLVER_H
#define SLURRY_NEARFIELD_IMPLICIT_SOLVER_H

// Backward-Euler contact constitutive update at fixed hydrodynamic geometry.
// The same 6N force balance is solved with the exact elastic/Coulomb return map.
// Sliding/rolling cap dependence on normal load makes its generalized Jacobian
// nonsymmetric. Solve the sparse 6N Jacobian with the cached SPD hydrodynamic
// preconditioner and checked FGMRES; retain sparse LU as a bounded fallback.
#include "slurry_nearfield_solver.h"

namespace SlurryNearfieldImplicitSolver {
namespace bs = SlurryBridgeSolver;
namespace ns = SlurryNearfieldSolver;

struct Workspace : ns::Workspace {
  bool use_krylov = true;
  int krylov_limit = 160, krylov_restart = 80;
  double linear_relative_target = 1e-10;
  std::size_t krylov_calls = 0, krylov_iterations = 0, krylov_max_iterations = 0,
              krylov_fallbacks = 0, jacobian_products = 0;
  double maximum_true_relative_linear_residual = 0.;
  std::shared_ptr<bs::detail::SolverWorkspace> linear_workspace =
      std::make_shared<bs::detail::SolverWorkspace>();
  bs::Sparse jacobian;
  Eigen::SparseLU<bs::Sparse> newton_factor;
  bool jacobian_ready = false;
  std::size_t newton_symbolic_analyses = 0, newton_numeric_factorizations = 0,
              newton_numeric_reuses = 0, newton_solves = 0;
};
struct Metric {
  double residual_norm = 0., rhs_norm = 0., tolerance = 0.,
         force_norm = 0., mobility_norm = 0.;
};

inline bs::Vector contact_force(const bs::Problem &p,
                               const std::vector<bs::ContactResponse> &contacts) {
  if (contacts.size() != p.contacts.size())
    throw std::invalid_argument("nearfield implicit: contact response size mismatch");
  bs::Vector force = p.external;
  for (std::size_t k = 0; k < contacts.size(); ++k) {
    const auto &c = p.contacts[k];
    force.segment<6>(6*c.i).noalias() -=
        bs::generalized_contact_map(c, true).transpose()*bs::contact_wrench(contacts[k]);
    force.segment<6>(6*c.j).noalias() -=
        bs::generalized_contact_map(c, false).transpose()*bs::contact_wrench(contacts[k]);
  }
  return force;
}

inline double rhs_norm(const bs::Problem &original, const Workspace &workspace) {
  const bs::Vector zero = bs::Vector::Zero(6*original.count);
  std::vector<bs::ContactResponse> contacts;
  contacts.reserve(original.contacts.size());
  for (const auto &c : original.contacts)
    contacts.push_back(bs::contact_response(c, zero, original.dt));
  const bs::Vector force = contact_force(original, contacts)-
      bs::detail::select(original.near_resistance*original.prescribed, original.count);
  return std::sqrt(ns::detail::mobility_metric_squared(original.prescribed, workspace)+
                   ns::detail::force_metric_squared(force, workspace));
}

// For endpoint acceptance, pass the ORIGINAL unshifted problem and responses
// evaluated with actual endpoint gaps. The zero-state scale then stays fixed
// across geometry corrections and cannot make convergence artificially easier.
inline Metric acceptance_metric(const bs::Problem &original, const bs::Result &state,
                                const std::vector<bs::ContactResponse> &contacts,
                                const Workspace &workspace, const bs::Options &options) {
  const int n = original.count;
  if (state.q.size() != 11*n || state.mid_moment.size() != 11*n ||
      state.total_moment.size() != 11*n || !workspace.ready)
    throw std::invalid_argument("nearfield implicit: invalid residual state");
  bs::Vector mobility_error(11*n);
  for (int i = 0; i < n; ++i)
    mobility_error.segment<11>(11*i).noalias() =
        original.self_mobility[i]*state.mid_moment.segment<11>(11*i)-state.q.segment<11>(11*i);
  const bs::Vector force_error = contact_force(original, contacts)-
      bs::detail::select(state.total_moment, n);
  Metric out;
  out.force_norm = force_error.norm();
  out.mobility_norm = mobility_error.norm();
  out.residual_norm = std::sqrt(ns::detail::force_metric_squared(force_error, workspace)+
                                ns::detail::mobility_metric_squared(mobility_error, workspace));
  out.rhs_norm = rhs_norm(original, workspace);
  out.tolerance = options.absolute_tolerance+options.relative_tolerance*out.rhs_norm;
  if (!std::isfinite(out.residual_norm) || !std::isfinite(out.rhs_norm) || !std::isfinite(out.tolerance))
    throw std::runtime_error("nearfield implicit: nonfinite physical residual");
  return out;
}

namespace detail {
struct Evaluation {
  bs::Result state;
  bs::Vector residual;
  double norm = 0., merit = 0.;
};
inline Evaluation evaluate(const bs::Problem &p, const bs::Vector &velocity,
                           const Workspace &workspace) {
  Evaluation out;
  auto &state = out.state;
  const int n = p.count;
  state.velocity = velocity;
  state.q = bs::detail::embed(velocity, n)+p.prescribed;
  state.mid_moment.resize(11*n);
  bs::Vector mobility_error(11*n);
  for (int i = 0; i < n; ++i) {
    state.mid_moment.segment<11>(11*i).noalias() =
        workspace.self_inverse[i]*state.q.segment<11>(11*i);
    mobility_error.segment<11>(11*i).noalias() =
        p.self_mobility[i]*state.mid_moment.segment<11>(11*i)-state.q.segment<11>(11*i);
  }
  state.total_moment = state.mid_moment+p.near_resistance*state.q;
  state.contacts.reserve(p.contacts.size());
  for (const auto &c : p.contacts)
    state.contacts.push_back(bs::contact_response(c, velocity, p.dt));
  // Evaluate the reconstructed full physical equation, not just A*u-b.
  out.residual = bs::detail::select(state.total_moment, n)-contact_force(p, state.contacts);
  state.force_residual_norm = out.residual.norm();
  state.mobility_residual_norm = mobility_error.norm();
  const double force_squared = ns::detail::force_metric_squared(out.residual, workspace);
  out.norm = std::sqrt(force_squared+ns::detail::mobility_metric_squared(mobility_error, workspace));
  out.merit = .5*out.norm*out.norm;
  state.residual_norm = out.norm;
  if (!velocity.allFinite() || !std::isfinite(out.norm))
    throw std::runtime_error("nearfield implicit: nonfinite trial state");
  return out;
}

inline bs::Sparse jacobian(const bs::Problem &p, const Evaluation &at,
                           const Workspace &workspace) {
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(144*p.contacts.size());
  auto add = [&](int i, int j, const bs::Matrix6 &block) {
    for (int r = 0; r < 6; ++r)
      for (int c = 0; c < 6; ++c)
        if (block(r, c) != 0.) entries.emplace_back(6*i+r, 6*j+c, block(r, c));
  };
  for (std::size_t k = 0; k < p.contacts.size(); ++k) {
    const auto &c = p.contacts[k];
    const bs::Matrix6 derivative = bs::contact_jacobian(at.state.contacts[k]);
    if (derivative.isZero(0.)) continue;
    const bs::Matrix6 bi = bs::generalized_contact_map(c, true),
                      bj = bs::generalized_contact_map(c, false);
    add(c.i, c.i, bi.transpose()*derivative*bi);
    add(c.i, c.j, bi.transpose()*derivative*bj);
    add(c.j, c.i, bj.transpose()*derivative*bi);
    add(c.j, c.j, bj.transpose()*derivative*bj);
  }
  bs::Sparse contacts(6*p.count, 6*p.count);
  contacts.setFromTriplets(entries.begin(), entries.end());
  bs::Sparse result = workspace.matrix+contacts;
  result.makeCompressed();
  return result;
}

inline bool factorize(const bs::Sparse &matrix, Workspace &w) {
  if (w.jacobian_ready && bs::detail::same_matrix(matrix, w.jacobian)) {
    ++w.newton_numeric_reuses; ++w.numeric_reuses;
    return true;
  }
  const bool same = w.jacobian_ready && bs::detail::same_pattern(matrix, w.jacobian);
  w.jacobian_ready = false;
  w.jacobian = matrix;
  if (!same) {
    w.newton_factor.analyzePattern(w.jacobian);
    ++w.newton_symbolic_analyses; ++w.symbolic_analyses;
  }
  w.newton_factor.factorize(w.jacobian);
  ++w.newton_numeric_factorizations; ++w.numeric_factorizations;
  if (w.newton_factor.info() != Eigen::Success) return false;
  w.jacobian_ready = true;
  return true;
}
} // namespace detail

inline bs::Result solve_stage(const bs::Problem &p, const bs::Options &options,
                        Workspace &workspace, const bs::Result *warm = nullptr) {
  if (options.maximum_newton_iterations <= 0 || options.maximum_line_search_iterations <= 0 ||
      options.maximum_linear_iterations <= 0 || options.gmres_restart <= 0 ||
      workspace.krylov_limit <= 0 || workspace.krylov_restart <= 0 ||
      !(workspace.linear_relative_target > 0.) || workspace.linear_relative_target >= 1. ||
      !std::isfinite(workspace.linear_relative_target))
    throw std::invalid_argument("nearfield implicit: invalid iteration limits");
  const auto start = std::chrono::steady_clock::now();
  const auto initial_linear_iterations = workspace.krylov_iterations;
  // This also validates/assembles the self and pair operators, initializes the
  // SPD physical metric and supplies a useful cold velocity guess. Its contact
  // response is temporary; only the implicit accepted response is returned.
  bs::Options base_options = options;
  base_options.trace_callback = {};
  const auto current = ns::solve(p, base_options, workspace);
  bs::Vector velocity = current.velocity;
  if (warm && warm->velocity.size() == 6*p.count && warm->velocity.allFinite())
    velocity = warm->velocity;
  const double scale = rhs_norm(p, workspace);
  const double tolerance = options.absolute_tolerance+options.relative_tolerance*scale;
  auto at = detail::evaluate(p, velocity, workspace);
  int updates = 0, gradient_steps = 0, latest_linear_iterations = 0;
  double latest_linear_residual = 0., latest_linear_target = 0.;
  auto trace = [&](const char *phase, double alpha) {
    if (!options.trace_callback) return;
    bs::IterationTrace t;
    t.phase = phase; t.iteration = updates; t.total_newton_iterations = updates;
    t.linear_iterations = latest_linear_iterations;
    t.total_linear_iterations = int(workspace.krylov_iterations-initial_linear_iterations);
    t.linear_residual = latest_linear_residual; t.linear_target = latest_linear_target;
    t.residual = t.merit_residual = at.norm; t.tolerance = tolerance;
    t.step_length = alpha; t.descent = gradient_steps > 0;
    t.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    for (const auto &c : at.state.contacts) { t.active_contacts += c.active; t.sliding_contacts += c.sliding; }
    options.trace_callback(t);
  };
  trace("nearfield_implicit_begin", 0.);
  while (at.norm > tolerance) {
    if (updates >= options.maximum_newton_iterations) {
      std::ostringstream message;
      message << std::scientific << std::setprecision(9)
              << "nearfield implicit: Newton limit; residual=" << at.norm
              << ", tolerance=" << tolerance << ", iterations=" << updates;
      at.state.linear_iterations = int(workspace.krylov_iterations-initial_linear_iterations);
      at.state.newton_iterations = updates;
      at.state.descent_steps = gradient_steps;
      throw bs::NonlinearFailure(message.str(), at.state, at.norm, tolerance);
    }
    const bs::Sparse jac = detail::jacobian(p, at, workspace);
    const bs::Vector weighted_residual = workspace.factor.solve(at.residual);
    const bs::Vector gradient = jac.transpose()*weighted_residual;
    bs::Vector direction;
    bool newton_available = false;
    const double rhs_length = at.residual.norm();
    latest_linear_iterations = 0;
    latest_linear_target = workspace.linear_relative_target*rhs_length;
    if (workspace.use_krylov) {
      ++workspace.krylov_calls;
      bs::Options linear_options = options;
      linear_options.maximum_linear_iterations = std::min(options.maximum_linear_iterations,
                                                           workspace.krylov_limit);
      linear_options.gmres_restart = std::min(options.gmres_restart, workspace.krylov_restart);
      linear_options.workspace = workspace.linear_workspace;
      const bs::Apply apply = [&](const bs::Vector &value) {
        ++workspace.jacobian_products;
        bs::Vector product = jac*value;
        if (!product.allFinite())
          throw std::runtime_error("nearfield implicit: nonfinite sparse Jacobian product");
        return product;
      };
      const bs::Apply inverse = [&](const bs::Vector &value) {
        ++latest_linear_iterations; ++workspace.krylov_iterations;
        bs::Vector result = workspace.factor.solve(value);
        if (workspace.factor.info() != Eigen::Success || !result.allFinite())
          throw std::runtime_error("nearfield implicit: hydrodynamic preconditioner failed");
        return result;
      };
      try {
        auto solved = bs::detail::fgmres(apply, inverse, -at.residual,
                                        latest_linear_target, linear_options);
        direction = std::move(solved.x);
        latest_linear_residual = (jac*direction+at.residual).norm();
        if (!direction.allFinite() || !std::isfinite(latest_linear_residual))
          throw std::runtime_error("nearfield implicit: nonfinite Krylov Newton direction");
        newton_available = latest_linear_residual <= latest_linear_target;
      } catch (const std::runtime_error &failure) {
        // Only bounded Krylov stagnation/breakdown is recoverable. Invalid
        // operators, nonfinite products and failed SPD preconditioning propagate.
        const std::string message = failure.what();
        if (message != "bridge solver: singular Krylov system" &&
            message.find("bridge solver: GMRES did not converge;") != 0) throw;
      }
      workspace.krylov_max_iterations = std::max(workspace.krylov_max_iterations,
                                                std::size_t(latest_linear_iterations));
      if (!newton_available) ++workspace.krylov_fallbacks;
    }
    if (!newton_available) {
      newton_available = detail::factorize(jac, workspace);
      if (newton_available) {
        direction = workspace.newton_factor.solve(-at.residual);
        ++workspace.newton_solves; ++workspace.solves;
        newton_available = workspace.newton_factor.info() == Eigen::Success && direction.allFinite();
        if (newton_available) {
          bs::Vector error = jac*direction+at.residual;
          latest_linear_residual = error.norm();
          for (int refinement = 0; latest_linear_residual > latest_linear_target && refinement < 2;
               ++refinement) {
            direction -= workspace.newton_factor.solve(error);
            ++workspace.newton_solves; ++workspace.solves;
            error = jac*direction+at.residual;
            latest_linear_residual = error.norm();
          }
          if (!direction.allFinite() || !std::isfinite(latest_linear_residual) ||
              latest_linear_residual > latest_linear_target)
            throw std::runtime_error("nearfield implicit: direct Newton solve failed true residual check");
        }
      }
    }
    if (newton_available) workspace.maximum_true_relative_linear_residual = std::max(
        workspace.maximum_true_relative_linear_residual, latest_linear_residual/rhs_length);
    // A nonmonotone window can indefinitely accept opening/closing cycles.
    // A true merit kink is handled by the bounded friction continuation below.
    const double reference = at.merit;
    const double previous_norm = at.norm;
    bool accepted = false;
    double accepted_alpha = 0.;
    for (int attempt = 0; attempt < 2 && !accepted; ++attempt) {
      const bool use_gradient = attempt == 1 || !newton_available;
      if (use_gradient) {
        direction = -workspace.factor.solve(gradient);
        if (!direction.allFinite()) break;
      }
      const double slope = gradient.dot(direction);
      if (!(slope < 0.) || !std::isfinite(slope)) {
        if (use_gradient) break;
        continue;
      }
      double alpha = 1.;
      for (int backtrack = 0; backtrack < options.maximum_line_search_iterations; ++backtrack, alpha *= .5) {
        auto trial = detail::evaluate(p, at.state.velocity+alpha*direction, workspace);
        if (trial.norm <= tolerance || trial.merit <= reference+1e-4*alpha*slope) {
          at = std::move(trial); accepted = true; accepted_alpha = alpha;
          if (use_gradient) ++gradient_steps;
          break;
        }
      }
      if (!newton_available) break; // Gradient was already tried on attempt zero.
    }
    if (!accepted) {
      std::ostringstream message;
      message << std::scientific << std::setprecision(9)
              << "nearfield implicit: physical-residual line search failed; residual="
              << at.norm << ", tolerance=" << tolerance << ", iterations=" << updates;
      at.state.linear_iterations = int(workspace.krylov_iterations-initial_linear_iterations);
      at.state.newton_iterations = updates;
      at.state.descent_steps = gradient_steps;
      throw bs::NonlinearFailure(message.str(), at.state, at.norm, tolerance);
    }
    ++updates;
    trace("nearfield_implicit_iterate", accepted_alpha);
    if (at.norm > tolerance && accepted_alpha < 1e-6 &&
        previous_norm-at.norm < 1e-4*previous_norm) {
      at.state.linear_iterations = int(workspace.krylov_iterations-initial_linear_iterations);
      at.state.newton_iterations = updates;
      at.state.descent_steps = gradient_steps;
      throw bs::NonlinearFailure("nearfield implicit: opening/cap merit kink",
                                 at.state, at.norm, tolerance);
    }
  }
  trace("nearfield_implicit_accepted", 0.);
  auto result = std::move(at.state);
  const auto metric = acceptance_metric(p, result, result.contacts, workspace, options);
  if (metric.residual_norm > metric.tolerance)
    throw std::runtime_error("nearfield implicit: accepted state failed physical residual gate");
  result.residual_norm = metric.residual_norm;
  result.relative_residual = metric.residual_norm/std::max({metric.rhs_norm, options.absolute_tolerance,
                                                           std::numeric_limits<double>::min()});
  result.force_residual_norm = metric.force_norm;
  result.mobility_residual_norm = metric.mobility_norm;
  result.linear_iterations = int(workspace.krylov_iterations-initial_linear_iterations);
  result.newton_iterations = updates;
  result.descent_steps = gradient_steps;
  result.hydro_power = result.q.dot(result.total_moment);
  for (const auto &contact : result.contacts) result.contact_power += bs::contact_power(contact);
  if (!std::isfinite(result.hydro_power) || !std::isfinite(result.contact_power))
    throw std::runtime_error("nearfield implicit: nonfinite power diagnostic");
  // This parameter stage never advances physical time or committed history.
  return result;
}
// Recover only nonlinear globalization failures. The frictionless stage has
// a monotone normal elastic law; continuation follows its solution to the exact
// original Coulomb coefficients. Intermediate spring responses are discarded:
// every stage sees the same original histories, geometry and physical timestep.
inline bs::Result solve(const bs::Problem &p, const bs::Options &options,
                        Workspace &workspace, const bs::Result *warm = nullptr) {
  const auto start = std::chrono::steady_clock::now();
  int total_updates = 0, total_linear = 0, total_descent = 0, successful_stages = 0, retries = 0;
  double last_failed_residual = 0., last_failed_tolerance = 0., last_target = 1.;
  auto account_failure = [&](const bs::NonlinearFailure &failure) {
    total_updates += failure.newton_iterations;
    total_linear += failure.linear_iterations;
    total_descent += failure.descent_steps;
    ++retries;
    last_failed_residual = failure.residual;
    last_failed_tolerance = failure.tolerance;
  };
  try {
    return solve_stage(p, options, workspace, warm);
  } catch (const bs::NonlinearFailure &failure) {
    bool friction = false;
    for (const auto &c : p.contacts)
      friction = friction || (c.kt > 0. && c.mu > 0.) ||
          (c.kr > 0. && c.mu_roll > 0. && c.roll_length > 0.) ||
          (c.kw > 0. && c.mu_twist > 0. && c.twist_length > 0.);
    if (!friction) throw;
    account_failure(failure);
  }
  bs::Problem stage = p;
  bs::Result seed;
  bool initialized = false;
  double parameter = 0., increment = .25;
  constexpr int maximum_parameter_attempts = 32;
  for (int attempt = 0; attempt < maximum_parameter_attempts; ++attempt) {
    const double target = initialized ? std::min(1., parameter+increment) : 0.;
    last_target = target;
    for (std::size_t k = 0; k < stage.contacts.size(); ++k) {
      stage.contacts[k].mu = target*p.contacts[k].mu;
      stage.contacts[k].mu_roll = target*p.contacts[k].mu_roll;
      stage.contacts[k].mu_twist = target*p.contacts[k].mu_twist;
    }
    bs::Options local = options;
    local.maximum_newton_iterations = std::min(options.maximum_newton_iterations, 24);
    if (options.trace_callback) local.trace_callback = [&](const bs::IterationTrace &input) {
      bs::IterationTrace output = input;
      output.continuation_stage = attempt+1;
      output.friction_scale = target;
      output.total_newton_iterations += total_updates;
      output.total_linear_iterations += total_linear;
      output.elapsed_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now()-start).count();
      options.trace_callback(output);
    };
    try {
      auto result = solve_stage(stage, local, workspace, initialized ? &seed : nullptr);
      total_updates += result.newton_iterations;
      total_linear += result.linear_iterations;
      total_descent += result.descent_steps;
      ++successful_stages;
      seed = std::move(result);
      parameter = target;
      initialized = true;
      if (parameter == 1.) {
        // solve_stage has tested the original coefficients against the original
        // full force/mobility residual, never a continuation-stage surrogate.
        seed.linear_iterations = total_linear;
        seed.newton_iterations = total_updates;
        seed.descent_steps = total_descent;
        seed.continuation_stages = successful_stages;
        seed.continuation_retries = retries;
        return seed;
      }
      increment = std::min(.5, 1.5*increment);
    } catch (const bs::NonlinearFailure &failure) {
      account_failure(failure);
      increment *= .5;
      if (!initialized || increment < 1e-5) {
        seed.linear_iterations = total_linear;
        seed.newton_iterations = total_updates;
        seed.descent_steps = total_descent;
        throw bs::NonlinearFailure(
            "nearfield implicit: friction continuation cannot advance at scale="+
            std::to_string(target)+"; "+failure.what(), seed,
            failure.residual, failure.tolerance);
      }
    }
  }
  seed.linear_iterations = total_linear;
  seed.newton_iterations = total_updates;
  seed.descent_steps = total_descent;
  throw bs::NonlinearFailure(
      "nearfield implicit: friction continuation attempt limit at scale="+
      std::to_string(last_target), seed, last_failed_residual, last_failed_tolerance);
}
} // namespace SlurryNearfieldImplicitSolver
#endif
