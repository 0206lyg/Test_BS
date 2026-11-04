#ifndef SLURRY_NEARFIELD_SOLVER_H
#define SLURRY_NEARFIELD_SOLVER_H

// Explicit-contact force balance with isolated grand resistance plus sparse
// pair resistance. q=P*u+qE, g_self=M_self^-1*q, and
// P^T*(g_self+R_near*q)=F_external+F_contact. Eliminate the eleven moments:
// [P^T*(M_self^-1+R_near)*P] u=F-P^T*(M_self^-1+R_near)*qE.
// Only the resulting SPD 6N sparse matrix is factored. No FFT, Krylov or
// nonlinear contact solve is performed, and no history is advanced here.
#include "slurry_bridge_solver.h"
#include "slurry_periodic_mobility.h"
#include "../vendor/gr_re2/particleMath.h"
#include <map>
#include <tuple>

namespace SlurryNearfieldSolver {
namespace bs = SlurryBridgeSolver;
namespace pm = SlurryPeriodic;
namespace gr = slurry::gr_re2::graphite;
using SelfKey = std::tuple<std::array<double, 3>, double, int>;

struct Workspace {
  // Body-frame quadrature is independent of position and orientation. Cache
  // by exact axes/viscosity/order; changing a physical coefficient invalidates it.
  std::map<SelfKey, bs::Matrix11> body_mobility;
  std::vector<bs::Matrix11> self_snapshot, self_inverse, self_lower;
  bs::Sparse matrix;
  Eigen::SimplicialLDLT<bs::Sparse> factor;
  bool ready = false;
  std::size_t symbolic_analyses = 0, numeric_factorizations = 0,
              numeric_reuses = 0, solves = 0, refinement_solves = 0,
              self_quadrature_evaluations = 0, self_factorizations = 0;
  // Exact snapshots let a warm implicit solve prepare its CURRENT operator
  // without constructing and solving an unused explicit-contact problem.
  bs::Sparse prepared_near_resistance;
  bool prepared_operator_ready = false;
  std::size_t resistance_generation = 0, prepared_resistance_generation = 0,
              prepared_self_generation = 0, resistance_preparations = 0,
              resistance_assemblies = 0, resistance_preparation_reuses = 0;
};

inline std::vector<bs::Matrix11>
build_self_mobility(const std::vector<gr::Body> &bodies, Workspace &workspace,
                    double eta = 1., int order = 256) {
  if (!(eta > 0.) || !std::isfinite(eta) || order < 4)
    throw std::invalid_argument("nearfield self: invalid viscosity or quadrature order");
  std::vector<bs::Matrix11> result;
  result.reserve(bodies.size());
  for (const auto &body : bodies) {
    for (double a : body.axes)
      if (!(a > 0.) || !std::isfinite(a))
        throw std::invalid_argument("nearfield self: invalid semiaxis");
    Eigen::Matrix3d rotation;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) rotation(r, c) = body.rotation[3*r+c];
    if (!rotation.allFinite() || rotation.determinant() <= 0. ||
        (rotation.transpose()*rotation-Eigen::Matrix3d::Identity()).norm() > 1e-8)
      throw std::invalid_argument("nearfield self: invalid body rotation");
    const SelfKey key(body.axes, eta, order);
    auto found = workspace.body_mobility.find(key);
    if (found == workspace.body_mobility.end()) {
      const Eigen::Vector3d axes(body.axes[0], body.axes[1], body.axes[2]);
      bs::Matrix11 mobility = pm::self_tail(
          axes.array().square().matrix().asDiagonal(), 0., eta, order);
      found = workspace.body_mobility.emplace(key, std::move(mobility)).first;
      ++workspace.self_quadrature_evaluations;
    }
    const bs::Matrix11 transform = pm::rotation11(rotation);
    result.push_back(transform*found->second*transform.transpose());
  }
  return result;
}

namespace detail {
inline void validate(const bs::Problem &p, const bs::Options &o,
                     bool validate_operator = true) {
  const int n = p.count;
  // p.mobility is deliberately not required: the sole mobility is M_self.
  if (n <= 0 || int(p.self_mobility.size()) != n ||
      p.prescribed.size() != 11*n || p.external.size() != 6*n ||
      p.near_resistance.rows() != 11*n || p.near_resistance.cols() != 11*n ||
      !(p.dt > 0.) || !std::isfinite(p.dt) || !p.prescribed.allFinite() ||
      !p.external.allFinite())
    throw std::invalid_argument("nearfield solver: invalid dimensions or input");
  if (!(o.relative_tolerance > 0.) || !std::isfinite(o.relative_tolerance) ||
      o.absolute_tolerance < 0. || !std::isfinite(o.absolute_tolerance))
    throw std::invalid_argument("nearfield solver: invalid residual tolerance");
  if (validate_operator) {
  for (const auto &m : p.self_mobility)
    if (!m.allFinite() || (m-m.transpose()).norm() > 1e-12*std::max(1., m.norm()))
      throw std::invalid_argument("nearfield solver: self mobility must be finite and symmetric");
  for (int outer = 0; outer < p.near_resistance.outerSize(); ++outer)
    for (bs::Sparse::InnerIterator it(p.near_resistance, outer); it; ++it)
      if (!std::isfinite(it.value()))
        throw std::invalid_argument("nearfield solver: nonfinite pair resistance");
  const bs::Sparse antisymmetric = p.near_resistance-bs::Sparse(p.near_resistance.transpose());
  if (antisymmetric.norm() > 1e-12*std::max(1., p.near_resistance.norm()))
    throw std::invalid_argument("nearfield solver: pair resistance must be symmetric");
  }
  for (const auto &c : p.contacts)
    if (!c.normal.allFinite() || c.i < 0 || c.j < 0 || c.i >= n || c.j >= n ||
        std::abs(c.normal.norm()-1.) > 1e-8 || c.kn < 0. || c.kt < 0. || c.mu < 0. ||
        !std::isfinite(c.kn+c.kt+c.mu+c.gap+c.onset) || !c.Bi.allFinite() ||
        !c.Bj.allFinite() || !c.affine.allFinite() || !c.history.allFinite() ||
        !c.rolling_history.allFinite() || c.kr < 0. || c.kw < 0. || c.mu_roll < 0. ||
        c.mu_twist < 0. || c.roll_length < 0. || c.twist_length < 0. ||
        !std::isfinite(c.kr+c.kw+c.mu_roll+c.mu_twist+c.roll_length+c.twist_length+c.twisting_history))
      throw std::invalid_argument("nearfield solver: invalid contact");
}

inline void prepare_self(const bs::Problem &p, Workspace &w) {
  bool same = w.self_snapshot.size() == p.self_mobility.size();
  for (std::size_t i = 0; same && i < p.self_mobility.size(); ++i)
    same = (w.self_snapshot[i].array() == p.self_mobility[i].array()).all();
  if (same) return;
  std::vector<bs::Matrix11> inverse(p.count), lower(p.count);
  for (int i = 0; i < p.count; ++i) {
    Eigen::LLT<bs::Matrix11> factor(p.self_mobility[i]);
    if (factor.info() != Eigen::Success)
      throw std::runtime_error("nearfield solver: self mobility is not positive definite");
    lower[i] = factor.matrixL();
    inverse[i] = factor.solve(bs::Matrix11::Identity());
  }
  w.self_snapshot = p.self_mobility;
  w.self_inverse = std::move(inverse);
  w.self_lower = std::move(lower);
  ++w.self_factorizations;
}

inline void factorize(const bs::Sparse &a, Workspace &w) {
  if (w.ready && bs::detail::same_matrix(w.matrix, a)) {
    ++w.numeric_reuses;
    return;
  }
  const bool same_pattern = w.ready && bs::detail::same_pattern(w.matrix, a);
  w.ready = false;
  w.matrix = a;
  if (!same_pattern) {
    w.factor.analyzePattern(w.matrix);
    ++w.symbolic_analyses;
  }
  w.factor.factorize(w.matrix);
  ++w.numeric_factorizations;
  ++w.resistance_generation;
  if (w.factor.info() != Eigen::Success || !w.factor.vectorD().allFinite() ||
      w.factor.vectorD().minCoeff() <= 0.)
    throw std::runtime_error("nearfield solver: sparse resistance is not positive definite");
  w.ready = true;
}

inline double force_metric_squared(const bs::Vector &force, const Workspace &w) {
  // LDLT whitening gives r^T*A^-1*r without a second sparse factorization.
  const bs::Vector permuted = w.factor.permutationP()*force;
  const bs::Vector whitened = w.factor.matrixL().solve(permuted);
  return (whitened.array().square()/w.factor.vectorD().array()).sum();
}
inline double mobility_metric_squared(const bs::Vector &velocity, const Workspace &w) {
  double result = 0.;
  for (std::size_t i = 0; i < w.self_lower.size(); ++i) {
    const Eigen::Matrix<double, 11, 1> whitened =
        w.self_lower[i].template triangularView<Eigen::Lower>().solve(velocity.segment<11>(11*i));
    result += whitened.squaredNorm();
  }
  return result;
}
} // namespace detail

// Prepare/factor only the exact current hydrodynamic resistance. This function
// performs no force solve and no contact update. Snapshot hits are coefficient-
// exact, including the full eleven-coordinate pair operator and self mobility;
// there is no geometry tolerance or approximate/stale matrix reuse.
inline void prepare_resistance(const bs::Problem &p, Workspace &workspace,
                               const bs::Options &options = bs::Options{}) {
  ++workspace.resistance_preparations;
  bool same_self = workspace.self_snapshot.size() == p.self_mobility.size();
  for (std::size_t i = 0; same_self && i < p.self_mobility.size(); ++i)
    same_self = (workspace.self_snapshot[i].array() == p.self_mobility[i].array()).all();
  const bool same_operator = workspace.ready && workspace.prepared_operator_ready &&
      workspace.matrix.rows() == 6*p.count &&
      workspace.prepared_resistance_generation == workspace.resistance_generation &&
      workspace.prepared_self_generation == workspace.self_factorizations && same_self &&
      bs::detail::same_matrix(workspace.prepared_near_resistance, p.near_resistance);
  // Even a snapshot hit validates current dt, dimensions, applied forces,
  // contact geometry/history/coefficient inputs and solver tolerances.
  detail::validate(p, options, !same_operator);
  if (same_operator) {
    ++workspace.resistance_preparation_reuses;
    ++workspace.numeric_reuses;
    return;
  }
  workspace.prepared_operator_ready = false;
  detail::prepare_self(p, workspace);
  const int n = p.count;
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(36*n+p.near_resistance.nonZeros());
  for (int i = 0; i < n; ++i)
    for (int r = 0; r < 6; ++r)
      for (int c = 0; c < 6; ++c)
        if (workspace.self_inverse[i](r, c) != 0.)
          entries.emplace_back(6*i+r, 6*i+c, workspace.self_inverse[i](r, c));
  for (int outer = 0; outer < p.near_resistance.outerSize(); ++outer)
    for (bs::Sparse::InnerIterator it(p.near_resistance, outer); it; ++it)
      if (it.row()%11 < 6 && it.col()%11 < 6)
        entries.emplace_back(6*(it.row()/11)+it.row()%11,
                             6*(it.col()/11)+it.col()%11, it.value());
  bs::Sparse a(6*n, 6*n);
  a.setFromTriplets(entries.begin(), entries.end());
  // Remove only assembly roundoff; residuals use the full original operator.
  a = (.5*(a+bs::Sparse(a.transpose()))).eval();
  a.makeCompressed();
  ++workspace.resistance_assemblies;
  detail::factorize(a, workspace);
  workspace.prepared_near_resistance = p.near_resistance;
  workspace.prepared_resistance_generation = workspace.resistance_generation;
  workspace.prepared_self_generation = workspace.self_factorizations;
  workspace.prepared_operator_ready = true;
}

inline bs::Result solve(const bs::Problem &p, const bs::Options &options,
                        Workspace &workspace, const bs::Result *warm = nullptr) {
  (void)warm; // Direct factor solves do not depend on a velocity guess.
  const auto start = std::chrono::steady_clock::now();
  prepare_resistance(p, workspace, options);
  const int n = p.count;
  bs::Result out;
  out.contacts.reserve(p.contacts.size());
  bs::Vector force = p.external;
  const bs::Vector zero = bs::Vector::Zero(6*n);
  for (const auto &c : p.contacts) {
    const auto response = bs::contact_response(c, zero, 0.);
    force.segment<6>(6*c.i).noalias() -=
        bs::generalized_contact_map(c, true).transpose()*bs::contact_wrench(response);
    force.segment<6>(6*c.j).noalias() -=
        bs::generalized_contact_map(c, false).transpose()*bs::contact_wrench(response);
    out.contacts.push_back(response);
  }
  const bs::Vector near_prescribed = p.near_resistance*p.prescribed;
  bs::Vector prescribed_moment = near_prescribed;
  for (int i = 0; i < n; ++i) {
    prescribed_moment.segment<11>(11*i).noalias() +=
        workspace.self_inverse[i]*p.prescribed.segment<11>(11*i);
  }
  const bs::Vector rhs = force-bs::detail::select(prescribed_moment, n);
  out.velocity = workspace.factor.solve(rhs);
  ++workspace.solves;
  if (workspace.factor.info() != Eigen::Success || !out.velocity.allFinite())
    throw std::runtime_error("nearfield solver: sparse resistance solve failed");

  // Preserve the mixed physical convergence metric of the explicit bridge
  // stage: the zero mixed state has rM=-qE, rF=F-P^T*Rnear*qE. In particular,
  // the normalizer is not merely the condensed rhs, which may vanish in shear.
  const bs::Vector mixed_rhs_force = force-bs::detail::select(near_prescribed, n);
  const double rhs_norm = std::sqrt(detail::mobility_metric_squared(p.prescribed, workspace)+
                                    detail::force_metric_squared(mixed_rhs_force, workspace));
  const double tolerance = options.absolute_tolerance+options.relative_tolerance*rhs_norm;
  bs::Vector force_error, mobility_error(11*n);
  for (int refinement = 0; ; ++refinement) {
    out.q = bs::detail::embed(out.velocity, n)+p.prescribed;
    out.mid_moment.resize(11*n);
    for (int i = 0; i < n; ++i) {
      out.mid_moment.segment<11>(11*i).noalias() =
          workspace.self_inverse[i]*out.q.segment<11>(11*i);
      mobility_error.segment<11>(11*i).noalias() =
          p.self_mobility[i]*out.mid_moment.segment<11>(11*i)-out.q.segment<11>(11*i);
    }
    out.total_moment = out.mid_moment+p.near_resistance*out.q;
    force_error = force-bs::detail::select(out.total_moment, n);
    out.force_residual_norm = force_error.norm();
    out.mobility_residual_norm = mobility_error.norm();
    out.residual_norm = std::sqrt(detail::force_metric_squared(force_error, workspace)+
                                  detail::mobility_metric_squared(mobility_error, workspace));
    if (!std::isfinite(out.residual_norm) || !std::isfinite(rhs_norm) || !std::isfinite(tolerance))
      throw std::runtime_error("nearfield solver: nonfinite physical residual");
    if (out.residual_norm <= tolerance) break;
    // At most two bounded roundoff refinements; failure is never hidden by
    // accepting a proxy residual, changing dt or clipping physical overlap.
    if (refinement == 2) {
      std::ostringstream message;
      message << std::scientific << std::setprecision(9)
              << "nearfield solver: physical residual did not converge; residual="
              << out.residual_norm << ", tolerance=" << tolerance;
      throw std::runtime_error(message.str());
    }
    const bs::Vector correction = workspace.factor.solve(force_error);
    ++workspace.refinement_solves;
    if (workspace.factor.info() != Eigen::Success || !correction.allFinite())
      throw std::runtime_error("nearfield solver: residual refinement failed");
    out.velocity += correction;
  }
  out.relative_residual = out.residual_norm/std::max({rhs_norm, options.absolute_tolerance,
                                                    std::numeric_limits<double>::min()});
  out.hydro_power = out.q.dot(out.total_moment);
  for (std::size_t k = 0; k < p.contacts.size(); ++k) {
    const auto &c = p.contacts[k];
    auto &r = out.contacts[k];
    r.relative = c.Bi*out.velocity.segment<6>(6*c.i)+c.Bj*out.velocity.segment<6>(6*c.j)+c.affine;
    r.relative_rotation = out.velocity.segment<3>(6*c.j+3)-out.velocity.segment<3>(6*c.i+3);
    out.contact_power += bs::contact_power(r);
  }
  if (!std::isfinite(out.hydro_power) || !std::isfinite(out.contact_power))
    throw std::runtime_error("nearfield solver: nonfinite power diagnostic");
  if (options.trace_callback) {
    bs::IterationTrace trace;
    trace.phase = "nearfield_direct_accepted";
    trace.residual = trace.merit_residual = out.residual_norm;
    trace.tolerance = tolerance;
    trace.linear_residual = out.residual_norm;
    trace.linear_target = tolerance;
    trace.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    for (const auto &r : out.contacts) {
      trace.active_contacts += r.active;
      trace.sliding_contacts += r.sliding;
    }
    options.trace_callback(trace);
  }
  // All iteration/mobility counters remain zero: this is a direct sparse
  // solve. Workspace counters separately expose factor and solve costs.
  return out;
}
} // namespace SlurryNearfieldSolver
#endif
