#ifndef SLURRY_BRIDGE_SOLVER_H
#define SLURRY_BRIDGE_SOLVER_H

// Dimensionless mixed mobility/resistance/contact solve.  The caller chooses
// consistent length, time and viscosity units; no geometry is changed here.
// q_i = (U_i-u_inf, Omega_i-Omega_inf, -E_inf[STF5]); g is ON-FLUID.
// M g = q, P^T(g + L q) = external + contact_force.
// Only 11x11 self blocks are inverted. No nested many-body mobility inversion.
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#ifdef SLURRY_HAVE_MKL_PARDISO
#include <Eigen/PardisoSupport>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SlurryBridgeSolver {
namespace detail { struct SolverWorkspace; }
using Vector = Eigen::VectorXd;
using Matrix11 = Eigen::Matrix<double, 11, 11>;
using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Map36 = Eigen::Matrix<double, 3, 6>;
using V3 = Eigen::Vector3d;
using Sparse = Eigen::SparseMatrix<double>;
using Apply = std::function<Vector(const Vector &)>;
struct Contact {
  int i = 0, j = 0;
  Map36 Bi = Map36::Zero(), Bj = Map36::Zero();
  V3 affine = V3::Zero(), normal = V3::UnitZ(), history = V3::Zero();
  double gap = 0, onset = 0, kn = 0, kt = 0, mu = 0;
};
struct ContactResponse {
  V3 relative = V3::Zero(), traction = V3::Zero(), tangent_force = V3::Zero();
  V3 tangent_history = V3::Zero(), trial_history = V3::Zero();
  Eigen::Matrix3d derivative = Eigen::Matrix3d::Zero();
  double predicted_gap = 0, normal_load = 0, normal_energy = 0,
         tangent_energy = 0, sliding_work = 0;
  bool active = false, sliding = false;
};
struct Problem {
  int count = 0;
  Apply mobility;
  std::vector<Matrix11> self_mobility;
  Sparse near_resistance;
  Vector prescribed, external;
  std::vector<Contact> contacts;
  double dt = 0;
};
struct IterationTrace {
  const char *phase = "begin";
  int iteration = 0, linear_iterations = 0, total_linear_iterations = 0,
      total_newton_iterations = 0, continuation_stage = 0, active_contacts = 0,
      sliding_contacts = 0, kink_reselections = 0;
  double residual = 0, tolerance = 0, step_length = 0, linear_target = 0,
         linear_residual = 0, linear_defect = 0, minimum_yield_distance = 1,
         elapsed_seconds = 0;
  double friction_scale = 1., stiffness_scale = 1.;
  double merit_residual = std::numeric_limits<double>::quiet_NaN();
  bool descent = false, breakpoint = false;
};
struct Options {
  double relative_tolerance = 1e-7, absolute_tolerance = 1e-11;
  double linear_relative_tolerance = 1e-8;
  // Zero retains fixed linear accuracy. Otherwise bounded Eisenstat-Walker
  // forcing avoids oversolving early Newton systems in the fixed merit norm.
  double max_linear_forcing = 0.;
  int maximum_newton_iterations = 40, maximum_linear_iterations = 1200;
  int gmres_restart = 80, maximum_line_search_iterations = 30;
  int max_breakpoint_intervals = 4;
  std::function<void(const IterationTrace &)> trace_callback;
  // Initial-guess continuation for cold starts and failed warm solves. Every
  // accepted result solves the original coefficients/tolerance; no intermediate
  // parameter stage advances physical time or committed spring history.
  bool cold_start_continuation = false;
  // Shared by sequential Newton/endpoint solves. Exact comparisons invalidate
  // cached numeric factors; only unchanged sparsity reuses symbolic analysis.
  std::shared_ptr<detail::SolverWorkspace> workspace;
};
struct Result {
  Vector velocity, mid_moment, q, total_moment;
  std::vector<ContactResponse> contacts;
  int linear_iterations = 0, newton_iterations = 0, mobility_applications = 0,
      kink_reselections = 0, descent_steps = 0, breakpoint_steps = 0;
  int continuation_stages = 0, continuation_retries = 0;
  double residual_norm = 0, relative_residual = 0;
  double force_residual_norm = 0, mobility_residual_norm = 0;
  double hydro_power = 0, contact_power = 0;
};
// Only nonlinear globalization failures may trigger a continuation retry.
// Linear algebra, nonfinite inputs, and physical-evaluation errors propagate.
struct NonlinearFailure : std::runtime_error {
  int linear_iterations, newton_iterations, mobility_applications,
      kink_reselections, descent_steps, breakpoint_steps;
  double residual, tolerance;
  NonlinearFailure(const std::string &message, const Result &work, double r,
                   double t)
      : std::runtime_error(message), linear_iterations(work.linear_iterations),
        newton_iterations(work.newton_iterations),
        mobility_applications(work.mobility_applications),
        kink_reselections(work.kink_reselections),
        descent_steps(work.descent_steps),
        breakpoint_steps(work.breakpoint_steps), residual(r), tolerance(t) {}
};
inline ContactResponse contact_response(const Contact &c, const Vector &u,
                                        double dt,
                                        double endpoint_gap = std::numeric_limits<double>::quiet_NaN()) {
  ContactResponse out;
  out.relative =
      c.Bi * u.segment<6>(6 * c.i) + c.Bj * u.segment<6>(6 * c.j) + c.affine;
  const double vn = c.normal.dot(out.relative);
  out.predicted_gap = std::isnan(endpoint_gap) ? c.gap + dt * vn : endpoint_gap;
  const double compression = std::isnan(endpoint_gap)
      ? std::max((c.onset - c.gap) - dt * vn, 0.)
      : std::max(c.onset - endpoint_gap, 0.);
  out.normal_load = c.kn * compression;
  out.active = out.normal_load > 0.;
  out.normal_energy = .5 * c.kn * compression * compression;
  if (!out.active)
    return out;
  const Eigen::Matrix3d nn = c.normal * c.normal.transpose();
  const Eigen::Matrix3d tangent = Eigen::Matrix3d::Identity() - nn;
  out.traction = -out.normal_load * c.normal;
  out.derivative = c.kn * dt * nn;
  if (c.kt <= 0. || c.mu <= 0.)
    return out;
  out.trial_history = tangent * (c.history + dt * out.relative);
  const double ell = out.trial_history.norm(), limit = c.mu * out.normal_load;
  out.sliding = c.kt * ell > limit;
  if (out.sliding && ell > 0.) {
    const V3 direction = out.trial_history / ell;
    out.tangent_force = limit * direction;
    // Exact nonsymmetric normal-load derivative, including d(mu*N)/du.
    out.derivative +=
        (limit * dt / ell) * (tangent - direction * direction.transpose()) -
        c.mu * c.kn * dt * direction * c.normal.transpose();
  } else {
    out.tangent_force = c.kt * out.trial_history;
    out.derivative += c.kt * dt * tangent;
  }
  out.traction += out.tangent_force;
  out.tangent_history = out.tangent_force / c.kt;
  out.tangent_energy = .5 * c.kt * out.tangent_history.squaredNorm();
  if (out.sliding)
    out.sliding_work = std::max(
        0., out.tangent_force.dot(out.trial_history - out.tangent_history));
  return out;
}
namespace detail {
inline Vector embed(const Vector &u, int n) {
  Vector q = Vector::Zero(11 * n);
  for (int i = 0; i < n; ++i)
    q.segment<6>(11 * i) = u.segment<6>(6 * i);
  return q;
}
inline Vector select(const Vector &q, int n) {
  Vector u(6 * n);
  for (int i = 0; i < n; ++i)
    u.segment<6>(6 * i) = q.segment<6>(11 * i);
  return u;
}
#include "slurry_bridge_workspace.h"
struct Preconditioner {
  int n;
  std::shared_ptr<MonitorFactors> factors;
  explicit Preconditioner(const Problem &p, const Vector &velocity,
                          const std::shared_ptr<SolverWorkspace> &workspace = {})
      : n(p.count) {
    auto self = self_factors(p, workspace);
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(36 * n + p.near_resistance.nonZeros());
    auto add_block = [&](int i, int j, const Matrix6 &block) {
      for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
          if (block(r, c) != 0.)
            entries.emplace_back(6 * i + r, 6 * j + c, block(r, c));
    };
    for (int i = 0; i < n; ++i) {
      add_block(i, i, self->inverse[i].template topLeftCorner<6, 6>());
    }
    // Preserve the complete near-contact graph, not only its diagonal blocks.
    // Independent diagonal scaling cannot resolve nearly rigid clusters of
    // strongly lubricated particles. This is a sparse approximate resistance,
    // not an inverse of the global mobility.
    for (int outer = 0; outer < p.near_resistance.outerSize(); ++outer)
      for (Sparse::InnerIterator it(p.near_resistance, outer); it; ++it)
        if (it.row() % 11 < 6 && it.col() % 11 < 6)
          entries.emplace_back(6 * (it.row() / 11) + it.row() % 11,
                               6 * (it.col() / 11) + it.col() % 11, it.value());
    for (const auto &c : p.contacts) {
      const auto state = contact_response(c, velocity, p.dt);
      const double signed_compression =
          (c.onset - c.gap) - p.dt * c.normal.dot(state.relative);
      const double gap_scale = std::max({1., std::abs(c.gap), std::abs(c.onset),
                                         std::abs(state.predicted_gap)});
      const bool at_onset =
          std::abs(signed_compression) <=
          128. * std::numeric_limits<double>::epsilon() * gap_scale;
      // A hydrodynamic neighbor is not necessarily a mechanical contact.
      // Inactive separated candidates remain in the residual/Jacobian but
      // must not add artificial stiffness to the preconditioner. At onset
      // keep the possible closing derivative for the semismooth selection.
      if (!state.active && !at_onset)
        continue;
      const Eigen::Matrix3d nn = c.normal * c.normal.transpose();
      const Eigen::Matrix3d tangent = Eigen::Matrix3d::Identity() - nn;
      Eigen::Matrix3d stiffness = p.dt * c.kn * nn;
      if (c.kt > 0. && c.mu > 0.) {
        if (state.sliding && state.trial_history.norm() > 0.) {
          const double ell = state.trial_history.norm();
          const V3 direction = state.trial_history / ell;
          stiffness += (p.dt * c.mu * state.normal_load / ell) *
                       (tangent - direction * direction.transpose());
        } else
          stiffness += p.dt * c.kt * tangent;
      }
      if (c.i == c.j) {
        const Map36 b = c.Bi + c.Bj;
        add_block(c.i, c.i, b.transpose() * stiffness * b);
      } else {
        add_block(c.i, c.i, c.Bi.transpose() * stiffness * c.Bi);
        add_block(c.j, c.j, c.Bj.transpose() * stiffness * c.Bj);
        add_block(c.i, c.j, c.Bi.transpose() * stiffness * c.Bj);
        add_block(c.j, c.i, c.Bj.transpose() * stiffness * c.Bi);
      }
    }
    Sparse approximate(6 * n, 6 * n);
    approximate.setFromTriplets(entries.begin(), entries.end());
    approximate = (.5 * (approximate + Sparse(approximate.transpose()))).eval();
    approximate.makeCompressed();
    factors = monitor_factors(std::move(self), approximate, workspace);
  }
  Vector left(const Vector &x) const {
    Vector y(x.size());
    for (int i = 0; i < n; ++i) {
      y.segment<11>(11 * i) =
          factors->self->lower[i].template triangularView<Eigen::Lower>().solve(
              x.segment<11>(11 * i));
    }
    // Eigen factors P*A*P^-1=L*L^T. Retain the permutation on BOTH sides.
    const Vector permuted = factors->factor.permutationP() * x.tail(6 * n);
    y.tail(6 * n) = factors->factor.matrixL().solve(permuted);
    return y;
  }
  Vector inverse_left(const Vector &x) const {
    Vector y(x.size());
    for (int i = 0; i < n; ++i)
      y.segment<11>(11 * i) = factors->self->lower[i] * x.segment<11>(11 * i);
    const Vector triangular = factors->factor.matrixL() * x.tail(6 * n);
    y.tail(6 * n) = factors->factor.permutationPinv() * triangular;
    return y;
  }
  Vector right(const Vector &x) const {
    Vector y(x.size());
    for (int i = 0; i < n; ++i) {
      y.segment<11>(11 * i) =
          factors->self->lower[i].transpose().template triangularView<Eigen::Upper>().solve(
              x.segment<11>(11 * i));
    }
    const Vector triangular = factors->factor.matrixU().solve(x.tail(6 * n));
    y.tail(6 * n) = factors->factor.permutationPinv() * triangular;
    return y;
  }
};
// Exact inverse of the sparse approximate mixed Jacobian
// K0=[D,-P;-P^T,-A], D=blockdiag(M_ii), A=L_UU+B^T C' B.
// The full nonsymmetric Coulomb derivative is retained. Only the 6N Schur
// complement S=A+P^T D^-1 P is factorized; the global mobility stays
// matrix-free.
struct CoupledPreconditioner {
  int n;
  std::shared_ptr<SelfFactors> self;
  std::shared_ptr<CoupledFactors> factors;
  CoupledPreconditioner(const Problem &p,
                        const std::vector<ContactResponse> &state,
                        const std::shared_ptr<SolverWorkspace> &workspace = {})
      : n(p.count), self(self_factors(p, workspace)) {
    Sparse matrix = base_resistance(p, self, workspace);
    std::vector<Eigen::Triplet<double>> entries;
    std::size_t active = 0;
    for (const auto &response : state) active += !response.derivative.isZero(0.);
    entries.reserve(144 * active);
    auto add = [&](int i, int j, const Matrix6 &block) {
      for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
          if (block(r, c) != 0.)
            entries.emplace_back(6 * i + r, 6 * j + c, block(r, c));
    };
    for (std::size_t k = 0; k < p.contacts.size(); ++k) {
      const auto &c = p.contacts[k];
      const auto &derivative = state[k].derivative;
      if (derivative.isZero(0.))
        continue;
      if (c.i == c.j) {
        const Map36 b = c.Bi + c.Bj;
        add(c.i, c.i, b.transpose() * derivative * b);
      } else {
        add(c.i, c.i, c.Bi.transpose() * derivative * c.Bi);
        add(c.i, c.j, c.Bi.transpose() * derivative * c.Bj);
        add(c.j, c.i, c.Bj.transpose() * derivative * c.Bi);
        add(c.j, c.j, c.Bj.transpose() * derivative * c.Bj);
      }
    }
    if (!entries.empty()) {
      Sparse contacts(6 * n, 6 * n);
      contacts.setFromTriplets(entries.begin(), entries.end());
      matrix += contacts;
    }
    matrix.makeCompressed();
    factors = coupled_factors(matrix, workspace);
  }
  Vector apply(const Vector &rhs) const {
    Vector out(rhs.size()), free_m(11 * n);
    for (int i = 0; i < n; ++i)
      free_m.segment<11>(11 * i) = self->inverse[i] * rhs.segment<11>(11 * i);
    const Vector schur_rhs = -rhs.tail(6 * n) - select(free_m, n);
    const Vector u = factors->factor.solve(schur_rhs);
    if (factors->factor.info() != Eigen::Success || !u.allFinite())
      throw std::runtime_error(
          "bridge solver: coupled sparse Schur solve failed");
    out.tail(6 * n) = u;
    for (int i = 0; i < n; ++i)
      out.segment<11>(11 * i) =
          free_m.segment<11>(11 * i) +
          self->inverse[i].template leftCols<6>() * u.segment<6>(6 * i);
    return out;
  }
};
struct LinearResult {
  Vector x;
  int iterations = 0;
  double residual = 0;
};
inline LinearResult fgmres(const Apply &apply, const Apply &precondition,
                           const Vector &rhs, double target,
                           const Options &opt) {
  const int dimension = rhs.size(),
            restart = std::min(dimension, opt.gmres_restart);
  LinearResult out;
  out.x = Vector::Zero(dimension);
  Vector residual = rhs;
  KrylovWorkspace local;
  auto &storage = opt.workspace ? opt.workspace->krylov : local;
  storage.resize(dimension, restart);
  auto &basis = storage.basis;
  auto &directions = storage.directions;
  auto &h = storage.h;
  auto &cs = storage.cs;
  auto &sn = storage.sn;
  auto &g = storage.g;
  while (out.iterations < opt.maximum_linear_iterations) {
    const double beta = residual.norm();
    if (beta <= target) {
      out.residual = beta;
      return out;
    }
    const int m =
        std::min(restart, opt.maximum_linear_iterations - out.iterations);
    // Only Hessenberg and rotations need clearing. Basis/directions are
    // overwritten before use; avoid zeroing O(dimension*restart) each restart.
    h.topLeftCorner(m + 1, m).setZero();
    cs.head(m).setZero(); sn.head(m).setZero(); g.head(m + 1).setZero();
    basis.col(0) = residual / beta;
    g[0] = beta;
    int used = 0;
    for (int j = 0; j < m; ++j) {
      directions.col(j) = precondition(basis.col(j));
      Vector w = apply(directions.col(j));
      ++out.iterations;
      if (!w.allFinite())
        throw std::runtime_error("bridge solver: nonfinite Krylov product");
      // Reorthogonalization is material for large near-contact condition
      // numbers.
      for (int repeat = 0; repeat < 2; ++repeat)
        for (int i = 0; i <= j; ++i) {
          const double projection = basis.col(i).dot(w);
          h(i, j) += projection;
          w.noalias() -= projection * basis.col(i);
        }
      h(j + 1, j) = w.norm();
      const bool breakdown = h(j + 1, j) < 1e-28;
      if (!breakdown)
        basis.col(j + 1) = w / h(j + 1, j);
      for (int i = 0; i < j; ++i) {
        const double a = h(i, j), b = h(i + 1, j);
        h(i, j) = cs[i] * a + sn[i] * b;
        h(i + 1, j) = -sn[i] * a + cs[i] * b;
      }
      const double length = std::hypot(h(j, j), h(j + 1, j));
      if (length == 0.)
        throw std::runtime_error("bridge solver: singular Krylov system");
      cs[j] = h(j, j) / length;
      sn[j] = h(j + 1, j) / length;
      h(j, j) = length;
      h(j + 1, j) = 0.;
      g[j + 1] = -sn[j] * g[j];
      g[j] *= cs[j];
      used = j + 1;
      if (std::abs(g[j + 1]) <= target || breakdown)
        break;
    }
    const Vector y = h.topLeftCorner(used, used)
                         .template triangularView<Eigen::Upper>()
                         .solve(g.head(used));
    out.x.noalias() += directions.leftCols(used) * y;
    residual = rhs - apply(out.x);
    out.residual = residual.norm();
    if (out.residual <= target)
      return out;
  }
  std::ostringstream message;
  message << std::scientific << std::setprecision(6)
          << "bridge solver: GMRES did not converge; residual=" << out.residual
          << ", target=" << target << ", iterations=" << out.iterations;
  throw std::runtime_error(message.str());
}
// Exact contact branch intervals along a frozen Newton line. Long-double
// coefficients retain a narrow sticking interval between nearly equal roots.
inline std::vector<std::pair<double, double>>
contact_line_intervals(const Contact &c, const Vector &u, const Vector &du,
                       double dt, double lower, double upper) {
  std::vector<std::pair<double, double>> intervals;
  if (c.kn <= 0. || !(upper > lower))
    return intervals;
  const V3 relative =
      c.Bi * u.segment<6>(6 * c.i) + c.Bj * u.segment<6>(6 * c.j) + c.affine;
  const V3 direction =
      c.Bi * du.segment<6>(6 * c.i) + c.Bj * du.segment<6>(6 * c.j);
  const double compression = c.onset - c.gap - dt * c.normal.dot(relative);
  const double dcompression = -dt * c.normal.dot(direction);
  std::vector<double> cuts{lower, upper};
  if (dcompression != 0.) {
    const double onset = -compression / dcompression;
    if (onset > lower && onset < upper) {
      cuts.push_back(onset);
      intervals.emplace_back(lower, onset);
      intervals.emplace_back(onset, upper);
    }
  }
  if (c.kt <= 0. || c.mu <= 0.)
    return intervals;
  const Eigen::Matrix3d tangent =
      Eigen::Matrix3d::Identity() - c.normal * c.normal.transpose();
  const auto xi =
      (tangent * (c.history + dt * relative)).cast<long double>().eval();
  const auto dxi = (dt * tangent * direction).cast<long double>().eval();
  const long double radius = static_cast<long double>(c.mu) * c.kn / c.kt;
  const long double n = radius * compression, dn = radius * dcompression;
  const long double a = dxi.squaredNorm() - dn * dn;
  const long double b = 2.L * (xi.dot(dxi) - n * dn);
  const long double cc = xi.squaredNorm() - n * n;
  bool has_yield = false;
  auto add_root = [&](long double root) {
    const double value = static_cast<double>(root);
    if (std::isfinite(value) && value > lower && value < upper &&
        compression + value * dcompression >= 0.) {
      cuts.push_back(value);
      has_yield = true;
    }
  };
  const long double eps = 64.L * std::numeric_limits<long double>::epsilon();
  if (std::abs(a) <= eps * (std::abs(b) + std::abs(cc))) {
    if (b != 0.)
      add_root(-cc / b);
  } else {
    // Algebraically equivalent to b*b-4*a*c, but it retains the tiny
    // separation of collinear entry/exit roots when ||xi|| >> cap/kt.
    const long double first = (n * dxi - dn * xi).squaredNorm();
    const long double second = xi.cross(dxi).squaredNorm();
    long double discriminant = 4.L * (first - second);
    const long double roundoff = 4.L * eps * (first + second);
    if (discriminant >= -roundoff) {
      discriminant = std::max(0.L, discriminant);
      const long double q =
          -.5L * (b + std::copysign(std::sqrt(discriminant), b));
      if (q != 0.) {
        add_root(q / a);
        add_root(cc / q);
      } else
        add_root(-b / (2.L * a));
    }
  }
  if (!has_yield)
    return intervals;
  std::sort(cuts.begin(), cuts.end());
  cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
  for (std::size_t k = 1; k < cuts.size(); ++k) {
    const long double midpoint = .5L * (cuts[k - 1] + cuts[k]);
    const long double load = n + midpoint * dn;
    if (load > 0. && (xi + midpoint * dxi).squaredNorm() <= load * load)
      intervals.emplace_back(cuts[k - 1], cuts[k]);
  }
  return intervals;
}
inline void validate(const Problem &p, const Options &o) {
  const int n = p.count;
  if (n <= 0 || !p.mobility || int(p.self_mobility.size()) != n ||
      p.prescribed.size() != 11 * n || p.external.size() != 6 * n ||
      p.near_resistance.rows() != 11 * n ||
      p.near_resistance.cols() != 11 * n || !(p.dt > 0.) ||
      !std::isfinite(p.dt) || !p.prescribed.allFinite() ||
      !p.external.allFinite())
    throw std::invalid_argument("bridge solver: invalid dimensions or input");
  if (!std::isfinite(o.relative_tolerance) ||
      !std::isfinite(o.absolute_tolerance) ||
      !std::isfinite(o.linear_relative_tolerance) ||
      !std::isfinite(o.max_linear_forcing) || o.max_linear_forcing < 0. ||
      o.max_linear_forcing >= 1. ||
      (o.max_linear_forcing > 0. &&
       o.max_linear_forcing < o.linear_relative_tolerance) ||
      o.relative_tolerance <= 0. || o.absolute_tolerance < 0. ||
      o.linear_relative_tolerance <= 0. || o.linear_relative_tolerance >= 1. ||
      o.maximum_newton_iterations <= 0 || o.maximum_linear_iterations <= 0 ||
      o.gmres_restart <= 0 || o.maximum_line_search_iterations <= 0 ||
      o.max_breakpoint_intervals < 0)
    throw std::invalid_argument("bridge solver: invalid solver options");
  for (const auto &self : p.self_mobility)
    if (!self.allFinite())
      throw std::invalid_argument("bridge solver: nonfinite self mobility");
  for (int outer = 0; outer < p.near_resistance.outerSize(); ++outer)
    for (Sparse::InnerIterator it(p.near_resistance, outer); it; ++it)
      if (!std::isfinite(it.value()))
        throw std::invalid_argument("bridge solver: nonfinite near resistance");
  for (const auto &c : p.contacts)
    if (!c.normal.allFinite() || c.i < 0 || c.j < 0 || c.i >= n || c.j >= n ||
        std::abs(c.normal.norm() - 1.) > 1e-8 || c.kn < 0. || c.kt < 0. ||
        c.mu < 0. || !std::isfinite(c.kn + c.kt + c.mu + c.gap + c.onset) ||
        !c.Bi.allFinite() || !c.Bj.allFinite() || !c.affine.allFinite() ||
        !c.history.allFinite())
      throw std::invalid_argument("bridge solver: invalid contact");
}
} // namespace detail
inline Result solve_direct(const Problem &p, const Options &opt,
                           const Result *warm = nullptr) {
  detail::validate(p, opt);
  const int n = p.count, nm = 11 * n, nu = 6 * n, dimension = nm + nu;
  Result out;
  const auto solve_start = std::chrono::steady_clock::now();
  // The convergence metric is fixed independently of the warm start; the
  // Krylov preconditioner below is refreshed at each Newton iterate.
  detail::Preconditioner monitor(p, Vector::Zero(nu), opt.workspace);
  Vector x = Vector::Zero(dimension);
  if (warm && warm->mid_moment.size() == nm && warm->velocity.size() == nu &&
      warm->mid_moment.allFinite() && warm->velocity.allFinite()) {
    x.head(nm) = warm->mid_moment;
    x.tail(nu) = warm->velocity;
  }
  auto mob = [&](const Vector &f) -> Vector {
    if (f.isZero(0.)) return Vector::Zero(f.size());
    ++out.mobility_applications;
    return p.mobility(f);
  };
  auto evaluate = [&](const Vector &value, std::vector<ContactResponse> *state,
                      const Vector *known_linear = nullptr) -> Vector {
    const Vector u = value.tail(nu), q = detail::embed(u, n) + p.prescribed;
    Vector r(dimension);
    if (known_linear)
      r = *known_linear;
    else {
      r.head(nm) = mob(value.head(nm)) - q;
      r.tail(nu) = -detail::select(value.head(nm) + p.near_resistance * q, n) +
                   p.external;
    }
    if (state)
      state->clear();
    for (const auto &c : p.contacts) {
      const auto response = contact_response(c, u, p.dt);
      r.segment<6>(nm + 6 * c.i).noalias() -=
          c.Bi.transpose() * response.traction;
      r.segment<6>(nm + 6 * c.j).noalias() -=
          c.Bj.transpose() * response.traction;
      if (state)
        state->push_back(response);
    }
    if (!r.allFinite())
      throw std::runtime_error("bridge solver: nonfinite residual");
    return r;
  };
  // A fixed scale makes convergence independent of the supplied warm start.
  const Vector zero_residual = evaluate(Vector::Zero(dimension), nullptr);
  const double rhs_norm = monitor.left(zero_residual).norm();
  const double tolerance =
      opt.absolute_tolerance + opt.relative_tolerance * rhs_norm;
  Vector raw = evaluate(x, &out.contacts), residual = monitor.left(raw);
  double norm = residual.norm(), previous_norm = norm;
  if (!std::isfinite(rhs_norm) || !std::isfinite(tolerance) ||
      !std::isfinite(norm))
    throw std::runtime_error(
        "bridge solver: nonfinite weighted residual or tolerance");
  for (int iteration = 0;
       norm > tolerance && iteration < opt.maximum_newton_iterations;
       ++iteration) {
    auto state = out.contacts;
    IterationTrace trace;
    trace.iteration = iteration;
    trace.residual = norm;
    trace.merit_residual = norm;
    trace.tolerance = tolerance;
    auto emit = [&](const char *phase) {
      if (!opt.trace_callback)
        return;
      trace.phase = phase;
      trace.total_linear_iterations = out.linear_iterations;
      trace.total_newton_iterations = out.newton_iterations;
      trace.kink_reselections = out.kink_reselections;
      trace.elapsed_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        solve_start)
              .count();
      trace.active_contacts = 0;
      trace.sliding_contacts = 0;
      trace.minimum_yield_distance = 1.;
      for (std::size_t k = 0; k < state.size(); ++k)
        if (state[k].active) {
          ++trace.active_contacts;
          trace.sliding_contacts += state[k].sliding;
          const double trial = p.contacts[k].kt * state[k].trial_history.norm(),
                       cap = p.contacts[k].mu * state[k].normal_load;
          if (trial + cap > 0.)
            trace.minimum_yield_distance =
                std::min(trace.minimum_yield_distance,
                         std::abs(trial - cap) / std::max(trial, cap));
        }
      opt.trace_callback(trace);
    };
    emit("begin");
    auto raw_jacobian = [&](const Vector &dx,
                            bool transpose = false) -> Vector {
      const Vector du = dx.tail(nu), dq = detail::embed(du, n);
      Vector product(dimension);
      product.head(nm) = mob(dx.head(nm)) - dq;
      const Vector near_product =
          transpose ? Vector(p.near_resistance.transpose() * dq)
                    : Vector(p.near_resistance * dq);
      product.tail(nu) = -detail::select(dx.head(nm) + near_product, n);
      for (std::size_t k = 0; k < p.contacts.size(); ++k) {
        const auto &c = p.contacts[k];
        const V3 dv =
            c.Bi * du.segment<6>(6 * c.i) + c.Bj * du.segment<6>(6 * c.j);
        const V3 dc = transpose ? V3(state[k].derivative.transpose() * dv)
                                : V3(state[k].derivative * dv);
        product.segment<6>(nm + 6 * c.i).noalias() -= c.Bi.transpose() * dc;
        product.segment<6>(nm + 6 * c.j).noalias() -= c.Bj.transpose() * dc;
      }
      return product;
    };
    auto jacobian = [&](const Vector &dx) -> Vector {
      return monitor.left(raw_jacobian(dx));
    };
    // Linear solves minimize the same fixed norm as the nonlinear merit.
    double forcing = opt.linear_relative_tolerance;
    if (opt.max_linear_forcing > 0.) {
      const double adaptive = iteration == 0
                                  ? opt.max_linear_forcing
                                  : .9 * std::pow(norm / previous_norm, 1.5);
      forcing = std::max(forcing, std::min(opt.max_linear_forcing, adaptive));
    }
    const double linear_target = forcing * norm;
    double achieved_linear_residual = std::numeric_limits<double>::quiet_NaN();
    Vector dx;
    for (int reselection = 0; reselection < 8; ++reselection) {
      detail::CoupledPreconditioner pre(p, state, opt.workspace);
      auto inverse = [&](const Vector &value) -> Vector {
        return pre.apply(monitor.inverse_left(value));
      };
      const auto linear =
          detail::fgmres(jacobian, inverse, -residual, linear_target, opt);
      out.linear_iterations += linear.iterations;
      trace.linear_iterations += linear.iterations;
      achieved_linear_residual = linear.residual;
      dx = linear.x;
      bool changed = false;
      // At a unilateral/yield kink, select the generalized derivative from
      // the proposed direction before backtracking. Residuals always use
      // the exact return map; this probe never changes contact history.
      const double probe = 1e-7 / std::max(1., dx.tail(nu).norm());
      const Vector trial_u = x.tail(nu) + probe * dx.tail(nu);
      for (std::size_t k = 0; k < p.contacts.size(); ++k) {
        const auto &c = p.contacts[k];
        const auto &actual = out.contacts[k];
        const double gap_scale =
            std::max({1., std::abs(c.gap), std::abs(c.onset),
                      std::abs(actual.predicted_gap)});
        const double yield =
            c.kt * actual.trial_history.norm() - c.mu * actual.normal_load;
        const double force_scale =
            std::max({1., c.kt * actual.trial_history.norm(),
                      c.mu * actual.normal_load});
        const bool normal_kink =
            std::abs((c.onset - c.gap) -
                     p.dt * c.normal.dot(actual.relative)) <=
            128. * std::numeric_limits<double>::epsilon() * gap_scale;
        const bool yield_kink =
            actual.active && c.kt > 0. && c.mu > 0. &&
            std::abs(yield) <=
                128. * std::numeric_limits<double>::epsilon() * force_scale;
        if (!normal_kink && !yield_kink)
          continue;
        const auto trial = contact_response(c, trial_u, p.dt);
        if (trial.active != state[k].active ||
            trial.sliding != state[k].sliding) {
          state[k].derivative = trial.derivative;
          state[k].active = trial.active;
          state[k].sliding = trial.sliding;
          changed = true;
          ++out.kink_reselections;
        }
      }
      if (!changed)
        break;
      if (reselection == 7)
        throw NonlinearFailure(
            "bridge solver: inconsistent contact kink directions", out, norm,
            tolerance);
    }
    const double initial_norm = norm, newton_step_norm = dx.norm();
    const Vector newton_raw_directional = raw_jacobian(dx);
    const Vector newton_directional = monitor.left(newton_raw_directional);
    const double newton_slope = residual.dot(newton_directional);
    const double monitor_linear_defect = (residual + newton_directional).norm();
    trace.linear_target = linear_target;
    trace.linear_residual = achieved_linear_residual;
    trace.linear_defect = monitor_linear_defect;
    emit("linear");
    double last_trial_norm = norm, best_trial_norm = norm, last_step = 0.,
           fallback_slope = 0.;
    bool used_breakpoint = false;
    auto line_search = [&](const Vector &direction, double slope,
                           const Vector &raw_directional) -> bool {
      if (!(slope < 0.) || !direction.allFinite())
        return false;
      // Cache the affine hydrodynamic part once per direction. Its mobility
      // head is already available from F and Jd; compute the sparse near part
      // directly to avoid canceling potentially huge contact derivatives.
      // Individual trials need no M or near-matrix product.
      const Vector du = direction.tail(nu);
      const Vector q = detail::embed(x.tail(nu), n) + p.prescribed;
      const Vector dq = detail::embed(du, n);
      Vector base_linear(dimension), direction_linear(dimension);
      base_linear.head(nm) = raw.head(nm);
      base_linear.tail(nu) =
          -detail::select(x.head(nm) + p.near_resistance * q, n) + p.external;
      direction_linear.head(nm) = raw_directional.head(nm);
      direction_linear.tail(nu) =
          -detail::select(direction.head(nm) + p.near_resistance * dq, n);
      double best_step = 0.,
             best_norm = std::numeric_limits<double>::infinity();
      Vector best_raw, best_weighted;
      auto trial_at = [&](double step) -> Vector {
        const Vector candidate = x + step * direction;
        const Vector linear = base_linear + step * direction_linear;
        Vector trial_raw = evaluate(candidate, nullptr, &linear);
        Vector trial = monitor.left(trial_raw);
        last_trial_norm = trial.norm();
        if (!std::isfinite(last_trial_norm))
          throw std::runtime_error(
              "bridge solver: nonfinite weighted trial residual");
        best_trial_norm = std::min(best_trial_norm, last_trial_norm);
        last_step = step;
        const bool acceptable =
            (last_trial_norm < norm &&
             trial.squaredNorm() <= norm * norm + 2e-4 * step * slope) ||
            last_trial_norm <= tolerance;
        if (acceptable && last_trial_norm < best_norm) {
          best_step = step;
          best_norm = last_trial_norm;
          best_raw = std::move(trial_raw);
          best_weighted = trial;
        }
        return trial;
      };
      double step = 1.;
      for (int search = 0; search < opt.maximum_line_search_iterations;
           ++search, step *= .5) {
        trial_at(step);
        if (best_step > 0.)
          break;
      }
      if (!(best_step > 0.))
        return false;
      const double baseline_step = best_step;
      if (baseline_step < 1. && opt.max_breakpoint_intervals > 0) {
        struct Interval {
          double left, right, score;
        };
        std::vector<Interval> intervals;
        const Vector u = x.tail(nu), sensitivity = monitor.right(residual);
        const double upper = std::min(1., 2. * baseline_step);
        for (const auto &c : p.contacts) {
          const auto ranges = detail::contact_line_intervals(
              c, u, du, p.dt, baseline_step, upper);
          const V3 beta = c.Bi * sensitivity.segment<6>(nm + 6 * c.i) +
                          c.Bj * sensitivity.segment<6>(nm + 6 * c.j);
          for (const auto &range : ranges) {
            const V3 delta =
                contact_response(c, u + range.second * du, p.dt).traction -
                contact_response(c, u + range.first * du, p.dt).traction;
            intervals.push_back(
                {range.first, range.second, beta.norm() * delta.norm()});
          }
        }
        std::stable_sort(intervals.begin(), intervals.end(),
                         [](const Interval &a, const Interval &b) {
                           return a.score > b.score;
                         });
        const int count =
            std::min(opt.max_breakpoint_intervals, int(intervals.size()));
        for (int k = 0; k < count; ++k) {
          const auto &interval = intervals[k];
          const Vector left = trial_at(interval.left),
                       right = trial_at(interval.right);
          const Vector change = right - left;
          if (change.squaredNorm() > 0.) {
            const double fraction =
                std::clamp(-left.dot(change) / change.squaredNorm(), 0., 1.);
            if (fraction > 0. && fraction < 1.)
              trial_at(interval.left +
                       fraction * (interval.right - interval.left));
          }
        }
        used_breakpoint = best_step != baseline_step;
      }
      x += best_step * direction;
      raw = std::move(best_raw);
      residual = std::move(best_weighted);
      norm = best_norm;
      last_step = best_step;
      return true;
    };
    bool used_descent = false;
    bool accepted = line_search(dx, newton_slope, newton_raw_directional);
    if (!accepted) {
      // Inexact Newton or a switching contact branch need not give descent
      // in this fixed metric. Use the Cauchy step of ||W_monitor F||^2/2,
      // in the currently preconditioned variables. This changes no force
      // law or convergence tolerance. The Coulomb transpose is essential.
      const Vector gradient =
          monitor.left(raw_jacobian(monitor.right(residual), true));
      const Vector direction = monitor.right(gradient);
      const Vector raw_image = raw_jacobian(direction);
      const Vector image = monitor.left(raw_image);
      const double denominator = image.squaredNorm();
      if (gradient.allFinite() && denominator > 0. &&
          std::isfinite(denominator)) {
        const double length = -gradient.squaredNorm() / denominator;
        const Vector cauchy = length * direction;
        const Vector cauchy_raw_directional = length * raw_image;
        fallback_slope = length * residual.dot(image);
        accepted = line_search(cauchy, fallback_slope, cauchy_raw_directional);
        if (accepted) {
          ++out.descent_steps;
          used_descent = true;
        }
      }
    }
    if (!accepted) {
      std::ostringstream message;
      message << std::scientific << std::setprecision(9)
              << "bridge solver: Newton and descent line searches failed; "
                 "iteration="
              << iteration << ", residual=" << initial_norm
              << ", tolerance=" << tolerance
              << ", last_trial=" << last_trial_norm
              << ", best_trial=" << best_trial_norm
              << ", last_step=" << last_step
              << ", newton_step_norm=" << newton_step_norm
              << ", linear_residual=" << achieved_linear_residual
              << ", linear_target=" << linear_target
              << ", monitor_linear_defect=" << monitor_linear_defect
              << ", newton_slope=" << newton_slope
              << ", fallback_slope=" << fallback_slope;
      throw NonlinearFailure(message.str(), out, initial_norm, tolerance);
    }
    ++out.newton_iterations;
    if (used_breakpoint)
      ++out.breakpoint_steps;
    previous_norm = initial_norm;
    raw = evaluate(x, &out.contacts);
    residual = monitor.left(raw);
    norm = residual.norm();
    if (!std::isfinite(norm))
      throw std::runtime_error(
          "bridge solver: nonfinite accepted weighted residual");
    state = out.contacts;
    trace.residual = norm;
    trace.merit_residual = norm;
    trace.step_length = last_step;
    trace.descent = used_descent;
    trace.breakpoint = used_breakpoint;
    emit("accepted");
  }
  if (norm > tolerance) {
    std::ostringstream message;
    message << std::scientific << std::setprecision(9)
            << "bridge solver: nonlinear solve did not converge; residual="
            << norm << ", tolerance=" << tolerance
            << ", Newton iterations=" << out.newton_iterations
            << ", descent steps=" << out.descent_steps
            << ", linear iterations=" << out.linear_iterations;
    throw NonlinearFailure(message.str(), out, norm, tolerance);
  }
  out.velocity = x.tail(nu);
  out.mid_moment = x.head(nm);
  out.q = detail::embed(out.velocity, n) + p.prescribed;
  out.total_moment = out.mid_moment + p.near_resistance * out.q;
  out.residual_norm = norm;
  out.relative_residual = norm / std::max(rhs_norm, opt.absolute_tolerance);
  out.mobility_residual_norm = raw.head(nm).norm();
  out.force_residual_norm = raw.tail(nu).norm();
  out.hydro_power = out.q.dot(out.total_moment);
  for (const auto &c : out.contacts)
    out.contact_power += c.traction.dot(c.relative);
  return out;
}
inline Result solve(const Problem &p, const Options &opt = Options{},
                    const Result *warm = nullptr) {
  const bool valid_warm = warm && warm->velocity.size() == 6 * p.count &&
                          warm->mid_moment.size() == 11 * p.count &&
                          warm->velocity.allFinite() &&
                          warm->mid_moment.allFinite();
  const bool has_friction =
      std::any_of(p.contacts.begin(), p.contacts.end(),
                  [](const Contact &c) { return c.mu > 0. && c.kt > 0.; });
  if (!opt.cold_start_continuation || !has_friction)
    return solve_direct(p, opt, warm);
  detail::validate(p, opt);
  // Freeze geometry, loading, dt and the COMMITTED spring histories. The
  // intermediate solves supply guesses only; their returned spring states
  // are never inserted into the next Problem or exposed as a physical step.
  Problem trial = p;
  Result work, seed;
  int successful_stages = 0, retries = 0, attempt = 0;
  const auto begun = std::chrono::steady_clock::now();
  auto accumulate = [&](const auto &value) {
    work.linear_iterations += value.linear_iterations;
    work.newton_iterations += value.newton_iterations;
    work.mobility_applications += value.mobility_applications;
    work.kink_reselections += value.kink_reselections;
    work.descent_steps += value.descent_steps;
    work.breakpoint_steps += value.breakpoint_steps;
  };
  if (valid_warm) {
    // A previous physical step may be far from the current root during a
    // stiff initial transient. Try it directly, then rescue only a typed
    // nonlinear failure with the same frozen current problem.
    Options direct = opt;
    direct.trace_callback = [&](const IterationTrace &input) {
      if (!opt.trace_callback)
        return;
      IterationTrace value = input;
      value.continuation_stage = -1;
      value.elapsed_seconds = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - begun)
                                  .count();
      opt.trace_callback(value);
    };
    try {
      return solve_direct(p, direct, warm);
    } catch (const NonlinearFailure &failure) {
      accumulate(failure);
      ++retries;
      if (opt.trace_callback) {
        IterationTrace value;
        value.phase = "warm_retry";
        value.continuation_stage = -1;
        value.residual = failure.residual;
        value.merit_residual = failure.residual;
        value.tolerance = failure.tolerance;
        value.total_linear_iterations = work.linear_iterations;
        value.total_newton_iterations = work.newton_iterations;
        value.elapsed_seconds = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - begun)
                                    .count();
        opt.trace_callback(value);
      }
    }
  }
  auto stage = [&](double scale, const Result *initial) -> Result {
    for (std::size_t k = 0; k < trial.contacts.size(); ++k)
      trial.contacts[k].kt = scale * p.contacts[k].kt;
    Options local = opt;
    local.cold_start_continuation = false;
    local.trace_callback = [&](const IterationTrace &input) {
      if (!opt.trace_callback)
        return;
      IterationTrace value = input;
      value.continuation_stage = attempt;
      value.stiffness_scale = scale;
      value.total_linear_iterations += work.linear_iterations;
      value.total_newton_iterations += work.newton_iterations;
      value.kink_reselections += work.kink_reselections;
      value.elapsed_seconds = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - begun)
                                  .count();
      opt.trace_callback(value);
    };
    // Emit parameter entry even when the supplied seed already solves this
    // stage. Residual fields are unknown until solve_direct evaluates them.
    if (opt.trace_callback) {
      IterationTrace entry;
      entry.phase = "stage";
      entry.residual = entry.tolerance =
          std::numeric_limits<double>::quiet_NaN();
      local.trace_callback(entry);
    }
    return solve_direct(trial, local, initial);
  };
  // Start with normal equilibrium, then widen the sticking region by reducing
  // kt rather than mu. Reducing mu would narrow the sticking region and leave
  // almost every cold contact sliding with an unestablished direction.
  seed = stage(0., nullptr);
  accumulate(seed);
  ++successful_stages;

  // Bound the stiffness of all candidate springs in isolated-particle drag
  // coordinates. If V_i=((D_i^-1)_UU)^-1 and a_i=||B_i V_i^1/2||_F, a pair
  // contributes at most dt*kt*a_i*(a_i+a_j) to its block row. The maximum row
  // sum bounds the symmetric spring operator's spectral radius. This includes
  // currently inactive candidates, so later activation cannot invalidate it.
  std::vector<Matrix6> free_mobility(p.count);
  for (int i = 0; i < p.count; ++i) {
    Eigen::LLT<Matrix11> self(p.self_mobility[i]);
    if (self.info() != Eigen::Success)
      throw std::runtime_error(
          "bridge solver: continuation self mobility is not positive definite");
    const Matrix6 resistance =
        self.solve(Matrix11::Identity()).template topLeftCorner<6, 6>();
    Eigen::LLT<Matrix6> factor(resistance);
    if (factor.info() != Eigen::Success)
      throw std::runtime_error("bridge solver: continuation self resistance is "
                               "not positive definite");
    free_mobility[i] = factor.solve(Matrix6::Identity());
  }
  std::vector<double> row_bound(p.count, 0.);
  auto norm_map = [&](const Map36 &b, int i) {
    return std::sqrt(
        std::max(0., (b * free_mobility[i] * b.transpose()).trace()));
  };
  for (const auto &c : p.contacts) {
    if (c.kt <= 0. || c.mu <= 0.)
      continue;
    const double stiffness = p.dt * c.kt;
    if (c.i == c.j) {
      const double a = norm_map(c.Bi + c.Bj, c.i);
      row_bound[c.i] += stiffness * a * a;
    } else {
      const double a = norm_map(c.Bi, c.i), b = norm_map(c.Bj, c.j);
      row_bound[c.i] += stiffness * a * (a + b);
      row_bound[c.j] += stiffness * b * (a + b);
    }
  }
  const double bound = *std::max_element(row_bound.begin(), row_bound.end());
  if (!std::isfinite(bound))
    throw std::runtime_error(
        "bridge solver: nonfinite continuation stiffness bound");
  const double first_scale = bound > 1. ? 1. / bound : 1.;
  double scale = 0., target = first_scale, growth = 4.;
  constexpr double minimum_relative_increment = 1. / 1024.;
  constexpr int maximum_attempts = 64;
  while (scale < 1.) {
    if (++attempt > maximum_attempts)
      throw std::runtime_error("bridge solver: stiffness continuation exceeded "
                               "its bounded stage budget");
    try {
      Result candidate = stage(target, &seed);
      const int effort = candidate.newton_iterations;
      accumulate(candidate);
      seed = std::move(candidate);
      scale = target;
      ++successful_stages;
      // Adapt the next multiplicative kt increment to observed difficulty.
      // Every stage uses the original mu and committed spring history.
      if (effort <= 6)
        growth = std::min(4., growth * 1.5);
      else if (effort > 16)
        growth = std::max(1.25, std::sqrt(growth));
      target = std::min(1., scale * growth);
    } catch (const NonlinearFailure &failure) {
      accumulate(failure);
      ++retries;
      if (opt.trace_callback) {
        IterationTrace value;
        value.phase = "retry";
        value.continuation_stage = attempt;
        value.stiffness_scale = target;
        value.residual = failure.residual;
        value.merit_residual = failure.residual;
        value.tolerance = failure.tolerance;
        value.total_linear_iterations = work.linear_iterations;
        value.total_newton_iterations = work.newton_iterations;
        value.elapsed_seconds = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - begun)
                                    .count();
        opt.trace_callback(value);
      }
      target = scale > 0. ? std::sqrt(scale * target) : .5 * target;
      if ((scale > 0. && target / scale - 1. < minimum_relative_increment) ||
          (scale == 0. && target < first_scale * minimum_relative_increment)) {
        std::ostringstream message;
        message << "bridge solver: stiffness continuation cannot advance "
                   "beyond scale="
                << scale << "; original target was not accepted; "
                << failure.what();
        throw NonlinearFailure(message.str(), work, failure.residual,
                               failure.tolerance);
      }
      if (scale > 0.)
        growth = target / scale;
    }
  }
  // scale==1 used exactly the user's coefficients and original stopping rule.
  // Replace only work counters; the final physical response and residuals are
  // those of that unmodified target problem.
  seed.linear_iterations = work.linear_iterations;
  seed.newton_iterations = work.newton_iterations;
  seed.mobility_applications = work.mobility_applications;
  seed.kink_reselections = work.kink_reselections;
  seed.descent_steps = work.descent_steps;
  seed.breakpoint_steps = work.breakpoint_steps;
  seed.continuation_stages = successful_stages;
  seed.continuation_retries = retries;
  return seed;
}
} // namespace SlurryBridgeSolver
#endif
