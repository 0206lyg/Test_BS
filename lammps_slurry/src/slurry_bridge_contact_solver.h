#ifndef SLURRY_BRIDGE_CONTACT_SOLVER_H
#define SLURRY_BRIDGE_CONTACT_SOLVER_H

#include "slurry_bridge_solver.h"

// Force-variable globalization of the same elastic/Coulomb law.
// Retained contact tractions avoid eliminating the narrow sticking region into
// velocity space. The returned state passes the original physical residual
// gate.
namespace SlurryBridgeContactSolver {
using namespace SlurryBridgeSolver;
struct Scale {
  double normal = 1., tangent = 1., rolling = 1., twisting = 1.;
};
struct ProximalResponse {
  V3 residual = V3::Zero();
  Eigen::Matrix3d velocity_derivative = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d traction_derivative = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d traction_inverse = Eigen::Matrix3d::Identity();
};
inline ProximalResponse proximal_response(const Contact &c, const V3 &relative,
                                          const V3 &traction, double dt,
                                          const Scale &rho) {
  ProximalResponse out;
  if (c.kn <= 0.) {
    out.residual = traction;
    return out;
  }
  const auto nn = (c.normal * c.normal.transpose()).eval();
  const auto tangent = (Eigen::Matrix3d::Identity() - nn).eval();
  const double kn = c.kn * dt, kt = c.kt * dt;
  const double normal = -c.normal.dot(traction);
  const double normal_trial =
      normal - rho.normal * (c.normal.dot(relative) + (c.gap - c.onset) / dt +
                             normal / kn);
  const bool active = normal_trial >= 0.;
  // Evaluate active residuals directly rather than subtracting nearly equal
  // contact forces; this preserves small corrections near a stiff root.
  const double normal_residual =
      active ? rho.normal * (c.normal.dot(relative) + (c.gap - c.onset) / dt +
                             normal / kn)
             : normal;
  const double cn = active ? rho.normal / kn : 1.;
  out.residual = -normal_residual * c.normal;
  out.velocity_derivative =
      active ? Eigen::Matrix3d(-rho.normal * nn) : Eigen::Matrix3d::Zero();
  Eigen::Matrix3d tangent_jacobian = tangent, tangent_inverse = tangent;
  V3 cross = V3::Zero();
  const V3 tangential_force = tangent * traction;
  if (c.kt > 0. && c.mu > 0.) {
    const double ratio = rho.tangent / kt;
    const V3 y = (1. - ratio) * tangential_force +
                 rho.tangent * (tangent * relative + tangent * c.history / dt);
    const double length = y.norm(), cap = c.mu * std::max(normal, 0.);
    if (length <= cap && cap > 0.) {
      out.residual +=
          rho.tangent * (tangential_force / kt - tangent * relative -
                         tangent * c.history / dt);
      out.velocity_derivative -= rho.tangent * tangent;
      tangent_jacobian = ratio * tangent;
      tangent_inverse = tangent / ratio;
    } else if (length > 0.) {
      const V3 d = y / length;
      const auto dd = (d * d.transpose()).eval();
      const double fraction = cap / length;
      const auto orthogonal = (tangent - dd).eval();
      out.residual += tangential_force - cap * d;
      out.velocity_derivative -= rho.tangent * fraction * orthogonal;
      const double eigenvalue = (1. - fraction) + ratio * fraction;
      tangent_jacobian = dd + eigenvalue * orthogonal;
      tangent_inverse = dd + orthogonal / eigenvalue;
      // At N=0 use the derivative approached from the inactive N<0 side.
      if (normal > 0.)
        cross = c.mu * d;
    } else {
      // Consistent zero-radius, zero-vector B-subgradient. No unit direction
      // is invented at this degenerate projection.
      out.residual += tangential_force;
    }
  } else
    out.residual += tangential_force;
  out.traction_derivative =
      cn * nn + tangent_jacobian + cross * c.normal.transpose();
  // Analytic block-triangular inverse avoids a generic ill-conditioned3x3
  // factorization when rho/(k dt) is small in the sticking branch.
  out.traction_inverse = nn / cn + tangent_inverse -
                         (tangent_inverse * cross) * c.normal.transpose() / cn;
  if (!out.residual.allFinite() || !out.velocity_derivative.allFinite() ||
      !out.traction_derivative.allFinite() || !out.traction_inverse.allFinite())
    throw std::runtime_error(
        "bridge proximal solver: nonfinite contact response");
  return out;
}
// Six-dimensional retained wrench is used only when angular contact laws
// are enabled. The force-only API and three-dimensional path remain intact.
struct GeneralizedProximalResponse {
  V6 residual = V6::Zero();
  Matrix6 velocity_derivative = Matrix6::Zero();
  Matrix6 traction_derivative = Matrix6::Identity();
  Matrix6 traction_inverse = Matrix6::Identity();
};
inline GeneralizedProximalResponse proximal_response(
    const Contact &c, const V6 &relative, const V6 &traction,
    double dt, const Scale &rho) {
  GeneralizedProximalResponse out;
  const auto force = proximal_response(c, V3(relative.head<3>()), V3(traction.head<3>()), dt, rho);
  out.residual.head<3>() = force.residual;
  out.velocity_derivative.topLeftCorner<3,3>() = force.velocity_derivative;
  out.traction_derivative.topLeftCorner<3,3>() = force.traction_derivative;
  out.traction_inverse.topLeftCorner<3,3>() = force.traction_inverse;
  const Eigen::Matrix3d nn = c.normal * c.normal.transpose(),
                        tangent = Eigen::Matrix3d::Identity() - nn;
  Eigen::Matrix3d angular_jacobian = Eigen::Matrix3d::Zero(),
                  angular_inverse = Eigen::Matrix3d::Zero();
  V3 cross = V3::Zero();
  const double normal = -c.normal.dot(traction.head<3>());
  auto branch = [&](const Eigen::Matrix3d &projector, const V3 &history,
                    double stiffness, double coefficient, double scale) {
    const V3 moment = projector * traction.tail<3>();
    if (!(c.kn > 0. && stiffness > 0. && coefficient > 0.)) {
      out.residual.tail<3>() += moment;
      angular_jacobian += projector;
      angular_inverse += projector;
      return;
    }
    const double ratio = scale / (stiffness * dt);
    const V3 y = (1. - ratio) * moment +
                 scale * (projector * relative.tail<3>() + history / dt);
    const double length = y.norm(), cap = coefficient * std::max(normal, 0.);
    if (length <= cap && cap > 0.) {
      out.residual.tail<3>() += scale * (moment / (stiffness * dt) -
                                  projector * relative.tail<3>() - history / dt);
      out.velocity_derivative.bottomRightCorner<3,3>() -= scale * projector;
      angular_jacobian += ratio * projector;
      angular_inverse += projector / ratio;
    } else if (length > 0.) {
      const V3 d = y / length;
      const Eigen::Matrix3d dd = d*d.transpose(), orthogonal = projector - dd;
      const double fraction = cap / length,
                   eigenvalue = (1. - fraction) + ratio * fraction;
      out.residual.tail<3>() += moment - cap * d;
      out.velocity_derivative.bottomRightCorner<3,3>() -= scale * fraction * orthogonal;
      angular_jacobian += dd + eigenvalue * orthogonal;
      angular_inverse += dd + orthogonal / eigenvalue;
      if (normal > 0.) cross += coefficient * d;
    } else {
      out.residual.tail<3>() += moment;
      angular_jacobian += projector;
      angular_inverse += projector;
    }
  };
  branch(tangent, tangent * c.rolling_history, c.kr,
         c.mu_roll * c.roll_length, rho.rolling);
  branch(nn, c.twisting_history * c.normal, c.kw,
         c.mu_twist * c.twist_length, rho.twisting);
  out.traction_derivative.bottomRightCorner<3,3>() = angular_jacobian;
  out.traction_derivative.bottomLeftCorner<3,3>() = cross * c.normal.transpose();
  out.traction_inverse.bottomRightCorner<3,3>() = angular_inverse;
  out.traction_inverse.bottomLeftCorner<3,3>() =
      -angular_inverse * out.traction_derivative.bottomLeftCorner<3,3>() * force.traction_inverse;
  if (!out.residual.allFinite() || !out.velocity_derivative.allFinite() ||
      !out.traction_derivative.allFinite() || !out.traction_inverse.allFinite())
    throw std::runtime_error("bridge proximal solver: nonfinite angular response");
  return out;
}
template<int K> struct ContactTraits;
template<> struct ContactTraits<3> { using Response = ProximalResponse; };
template<> struct ContactTraits<6> { using Response = GeneralizedProximalResponse; };
template<int K> using LocalVector = Eigen::Matrix<double,K,1>;
template<int K> using LocalMap = Eigen::Matrix<double,K,6>;
template<int K> inline LocalMap<K> local_contact_map(const Contact &c, bool first) {
  return generalized_contact_map(c, first).template topRows<K>();
}
template<int K> inline LocalVector<K> local_affine(const Contact &c) {
  LocalVector<K> out = LocalVector<K>::Zero(); out.template head<3>() = c.affine; return out;
}
template<int K> inline LocalVector<K> local_wrench(const ContactResponse &r) {
  return contact_wrench(r).template head<K>();
}
template<int K> inline void set_condensed_contact(ContactResponse &r,
    const Eigen::Matrix<double,K,K> &a) {
  Matrix6 generalized = Matrix6::Zero(); generalized.template topLeftCorner<K,K>() = a;
  set_contact_jacobian(r, generalized);
}
inline std::vector<Scale> contact_scales(const Problem &p) {
  std::vector<Matrix6> free_mobility(p.count);
  for (int i = 0; i < p.count; ++i) {
    Eigen::LLT<Matrix11> self(p.self_mobility[i]);
    if (self.info() != Eigen::Success)
      throw std::runtime_error("bridge proximal solver: invalid self mobility");
    const Matrix6 resistance =
        self.solve(Matrix11::Identity()).template topLeftCorner<6, 6>();
    Eigen::LLT<Matrix6> drag(resistance);
    if (drag.info() != Eigen::Success)
      throw std::runtime_error("bridge proximal solver: invalid self drag");
    free_mobility[i] = drag.solve(Matrix6::Identity());
  }
  std::vector<Scale> scales(p.contacts.size());
  for (std::size_t k = 0; k < p.contacts.size(); ++k) {
    const auto &c = p.contacts[k];
    Eigen::Matrix3d mobility;
    if (c.i == c.j) {
      const Map36 b = c.Bi + c.Bj;
      mobility = b * free_mobility[c.i] * b.transpose();
    } else
      mobility = c.Bi * free_mobility[c.i] * c.Bi.transpose() +
                 c.Bj * free_mobility[c.j] * c.Bj.transpose();
    const double normal = std::max(0., c.normal.dot(mobility * c.normal));
    const double tangent = .5 * std::max(0., mobility.trace() - normal);
    if (c.kn > 0.)
      scales[k].normal = 1. / (normal + 1. / (c.kn * p.dt));
    if (c.kt > 0. && c.mu > 0.)
      scales[k].tangent = 1. / (tangent + 1. / (c.kt * p.dt));
    if (rotation_enabled(c)) {
      const Map36 ci = generalized_contact_map(c,true).bottomRows<3>(),
                  cj = generalized_contact_map(c,false).bottomRows<3>();
      const Eigen::Matrix3d angular = c.i == c.j ? Eigen::Matrix3d::Zero() :
          Eigen::Matrix3d(ci * free_mobility[c.i] * ci.transpose() +
                          cj * free_mobility[c.j] * cj.transpose());
      const double spin = std::max(0., c.normal.dot(angular * c.normal));
      const double roll = .5 * std::max(0., angular.trace() - spin);
      if (c.kr > 0. && c.mu_roll > 0. && c.roll_length > 0.)
        scales[k].rolling = 1. / (roll + 1. / (c.kr * p.dt));
      if (c.kw > 0. && c.mu_twist > 0. && c.twist_length > 0.)
        scales[k].twisting = 1. / (spin + 1. / (c.kw * p.dt));
    }
    if (!(scales[k].rolling > 0.) || !(scales[k].twisting > 0.) ||
        !std::isfinite(scales[k].rolling + scales[k].twisting))
      throw std::runtime_error("bridge proximal solver: invalid angular scale");
    if (!(scales[k].normal > 0.) || !(scales[k].tangent > 0.) ||
        !std::isfinite(scales[k].normal + scales[k].tangent))
      throw std::runtime_error("bridge proximal solver: invalid contact scale");
  }
  return scales;
}

// Exact elimination of contact INCREMENTS in a frozen Newton Jacobian.
// The nonlinear state and merit still retain all contact tractions. With
// J=[A,B;C,D], B=-contact_map^T and D block diagonal, the Krylov system is
// (A-B D^-1 C) dx = b_core-B D^-1 b_contact. This reduces its dimension to
// 17N, even when the broad-phase contact candidate list is much larger.
template<int K> struct ContactLinearizationT {
  using Local = LocalVector<K>;
  using Proximal = typename ContactTraits<K>::Response;
  const Problem &problem;
  const std::vector<Proximal> &state;
  int nm, nu, core;
  std::vector<ContactResponse> condensed;
  ContactLinearizationT(const Problem &p,
                       const std::vector<Proximal> &responses)
      : problem(p), state(responses), nm(11 * p.count), nu(6 * p.count),
        core(nm + nu), condensed(p.contacts.size()) {
    if (state.size() != p.contacts.size())
      throw std::invalid_argument("bridge contact linearization size mismatch");
    for (std::size_t k = 0; k < state.size(); ++k)
      set_condensed_contact<K>(condensed[k],
          Eigen::Matrix<double,K,K>(-state[k].traction_inverse * state[k].velocity_derivative));
  }
  Vector reduce_rhs(const Vector &full) const {
    Vector out = full.head(core);
    for (std::size_t k = 0; k < state.size(); ++k) {
      const auto &c = problem.contacts[k];
      const Local load = state[k].traction_inverse * full.segment<K>(core + K * k);
      out.segment<6>(nm + 6 * c.i).noalias() += local_contact_map<K>(c, true).transpose() * load;
      out.segment<6>(nm + 6 * c.j).noalias() += local_contact_map<K>(c, false).transpose() * load;
    }
    return out;
  }
  Vector recover(const Vector &increment, const Vector &full_rhs) const {
    Vector out(full_rhs.size());
    out.head(core) = increment;
    for (std::size_t k = 0; k < state.size(); ++k) {
      const auto &c = problem.contacts[k];
      const Local velocity = local_contact_map<K>(c, true) * increment.segment<6>(nm + 6 * c.i) +
                          local_contact_map<K>(c, false) * increment.segment<6>(nm + 6 * c.j);
      out.segment<K>(core + K * k) = state[k].traction_inverse *
          (full_rhs.segment<K>(core + K * k) - state[k].velocity_derivative * velocity);
    }
    return out;
  }
  Vector apply(const Vector &increment, const Apply &mobility) const {
    const Vector du = increment.tail(nu),
                 dq = SlurryBridgeSolver::detail::embed(du, problem.count);
    Vector out(core);
    out.head(nm) = mobility(increment.head(nm)) - dq;
    out.tail(nu) = -SlurryBridgeSolver::detail::select(
        increment.head(nm) + problem.near_resistance * dq, problem.count);
    for (std::size_t k = 0; k < state.size(); ++k) {
      const auto &c = problem.contacts[k];
      const Local velocity = local_contact_map<K>(c, true) * du.segment<6>(6 * c.i) + local_contact_map<K>(c, false) * du.segment<6>(6 * c.j),
               load = contact_jacobian(condensed[k]).template topLeftCorner<K,K>() * velocity;
      out.segment<6>(nm + 6 * c.i).noalias() -= local_contact_map<K>(c, true).transpose() * load;
      out.segment<6>(nm + 6 * c.j).noalias() -= local_contact_map<K>(c, false).transpose() * load;
    }
    return out;
  }
};

using ContactLinearization = ContactLinearizationT<3>;
using GeneralizedContactLinearization = ContactLinearizationT<6>;

template<int K> inline Result solve_proximal_impl(const Problem &p, const Options &opt = Options{},
                             const Result *warm = nullptr) {
  using Local = LocalVector<K>;
  using Proximal = typename ContactTraits<K>::Response;
  SlurryBridgeSolver::detail::validate(p, opt);
  if (p.contacts.empty())
    return SlurryBridgeSolver::solve_direct(p, opt, warm);
  const int n = p.count, nm = 11 * n, nu = 6 * n, core = nm + nu,
            nc = int(p.contacts.size());
  const int dimension = core + K * nc;
  const auto started = std::chrono::steady_clock::now();
  const auto scales = contact_scales(p);
  SlurryBridgeSolver::detail::Preconditioner monitor(p, Vector::Zero(nu), opt.workspace);
  auto contact_scale = [&](const Vector &value, bool inverse) -> Vector {
    Vector out = value;
    for (int k = 0; k < nc; ++k) {
      const auto &c = p.contacts[k];
      for (int block = 0; block < K / 3; ++block) {
        const V3 local = value.segment<3>(core + K * k + 3 * block);
        const V3 normal = c.normal * c.normal.dot(local), tangent = local - normal;
        const double sn = std::sqrt(block ? scales[k].twisting : scales[k].normal),
                     st = std::sqrt(block ? scales[k].rolling : scales[k].tangent);
        out.segment<3>(core + K * k + 3 * block) = inverse ? V3(sn*normal + st*tangent)
                                                         : V3(normal/sn + tangent/st);
      }
    }
    return out;
  };
  auto left = [&](const Vector &value) -> Vector {
    Vector out = contact_scale(value, false);
    out.head(core) = monitor.left(value.head(core));
    return out;
  };
  auto right = [&](const Vector &value) -> Vector {
    Vector out = contact_scale(value, false);
    out.head(core) = monitor.right(value.head(core));
    return out;
  };
  Result work;
  Vector x = Vector::Zero(dimension);
  if (warm && warm->velocity.size() == nu && warm->mid_moment.size() == nm &&
      warm->velocity.allFinite() && warm->mid_moment.allFinite()) {
    x.head(nm) = warm->mid_moment;
    x.segment(nm, nu) = warm->velocity;
  }
  for (int k = 0; k < nc; ++k)
    x.segment<K>(core + K * k) =
        local_wrench<K>(contact_response(p.contacts[k], x.segment(nm, nu), p.dt));
  auto mob = [&](const Vector &value) -> Vector {
    if (value.isZero(0.))
      return Vector::Zero(value.size());
    ++work.mobility_applications;
    return p.mobility(value);
  };
  std::vector<Proximal> states(nc);
  auto evaluate = [&](const Vector &value, std::vector<Proximal> *state,
                      const Vector *cached_core = nullptr) -> Vector {
    const Vector u = value.segment(nm, nu),
                 q = SlurryBridgeSolver::detail::embed(u, n) + p.prescribed;
    Vector residual(dimension);
    if (cached_core)
      residual.head(core) = *cached_core;
    else {
      residual.head(nm) = mob(value.head(nm)) - q;
      residual.segment(nm, nu) =
          -SlurryBridgeSolver::detail::select(
              value.head(nm) + p.near_resistance * q, n) +
          p.external;
      for (int k = 0; k < nc; ++k) {
        const auto &c = p.contacts[k];
        const Local force = value.segment<K>(core + K * k);
        residual.segment<6>(nm + 6 * c.i) -= local_contact_map<K>(c, true).transpose() * force;
        residual.segment<6>(nm + 6 * c.j) -= local_contact_map<K>(c, false).transpose() * force;
      }
    }
    for (int k = 0; k < nc; ++k) {
      const auto &c = p.contacts[k];
      const Local velocity = local_contact_map<K>(c, true) * u.segment<6>(6 * c.i) +
                          local_contact_map<K>(c, false) * u.segment<6>(6 * c.j) + local_affine<K>(c);
      const auto response = proximal_response(
          c, velocity, Local(value.segment<K>(core + K * k)), p.dt, scales[k]);
      residual.segment<K>(core + K * k) = response.residual;
      if (state)
        (*state)[k] = response;
    }
    if (!residual.allFinite())
      throw std::runtime_error("bridge proximal solver: nonfinite residual");
    return residual;
  };
  auto original_residual = [&](const Vector &value,
                               const Vector &retained) -> Vector {
    Vector physical = retained.head(core);
    const Vector u = value.segment(nm, nu);
    for (int k = 0; k < nc; ++k) {
      const auto &c = p.contacts[k];
      const Local difference = value.segment<K>(core + K * k) -
                            local_wrench<K>(contact_response(c, u, p.dt));
      physical.segment<6>(nm + 6 * c.i) += local_contact_map<K>(c, true).transpose() * difference;
      physical.segment<6>(nm + 6 * c.j) += local_contact_map<K>(c, false).transpose() * difference;
    }
    return physical;
  };
  const Vector zero = Vector::Zero(dimension);
  const Vector zero_raw = evaluate(zero, nullptr);
  const double physical_rhs =
      monitor.left(original_residual(zero, zero_raw)).norm();
  const double tolerance =
      opt.absolute_tolerance + opt.relative_tolerance * physical_rhs;
  const double proxy_rhs = left(zero_raw).norm();
  Vector raw = evaluate(x, &states), residual = left(raw);
  double norm = residual.norm(), previous_norm = norm;
  double physical_norm = monitor.left(original_residual(x, raw)).norm();
  if (!std::isfinite(norm + physical_norm + physical_rhs + proxy_rhs +
                     tolerance))
    throw std::runtime_error("bridge proximal solver: nonfinite weighted norm");
  auto trace = [&](const char *phase, int iteration, int linear, double alpha,
                   double target, double linear_residual,
                   bool descent = false) {
    if (!opt.trace_callback)
      return;
    IterationTrace event;
    event.phase = phase;
    event.iteration = iteration;
    event.continuation_stage = -2;
    event.total_newton_iterations = work.newton_iterations;
    event.linear_iterations = linear;
    event.total_linear_iterations = work.linear_iterations;
    event.residual = physical_norm;
    event.merit_residual = norm;
    event.tolerance = tolerance;
    event.step_length = alpha;
    event.linear_target = target;
    event.linear_residual = linear_residual;
    event.descent = descent;
    event.elapsed_seconds = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - started)
                                .count();
    for (const auto &c : p.contacts) {
      const auto response = contact_response(c, x.segment(nm, nu), p.dt);
      event.active_contacts += response.active;
      event.sliding_contacts += response.sliding;
    }
    opt.trace_callback(event);
  };
  bool polish = false;
  for (int iteration = 0;
       physical_norm > tolerance && iteration < opt.maximum_newton_iterations;
       ++iteration) {
    trace("prox_begin", iteration, 0, 0., 0., 0.);
    // At roundoff-level proxy residual use the original formulation for a
    // final polish; its gate is always authoritative, never the proxy norm.
    if (norm <= std::max(opt.absolute_tolerance,
                         128. * std::numeric_limits<double>::epsilon() *
                             std::max(1., proxy_rhs))) {
      polish = true;
      break;
    }
    const ContactLinearizationT<K> reduced(p, states);
    SlurryBridgeSolver::detail::CoupledPreconditioner factor(p, reduced.condensed, opt.workspace);
    auto inverse = [&](const Vector &weighted) -> Vector {
      return factor.apply(monitor.inverse_left(weighted));
    };
    auto jacobian = [&](const Vector &direction,
                        bool transpose = false,
                        const Vector *known_mobility = nullptr) -> Vector {
      const Vector du = direction.segment(nm, nu),
                   dq = SlurryBridgeSolver::detail::embed(du, n);
      Vector out(dimension);
      out.head(nm) = (known_mobility ? *known_mobility : mob(direction.head(nm))) - dq;
      const Vector near = transpose ? Vector(p.near_resistance.transpose() * dq)
                                    : Vector(p.near_resistance * dq);
      out.segment(nm, nu) =
          -SlurryBridgeSolver::detail::select(direction.head(nm) + near, n);
      for (int k = 0; k < nc; ++k) {
        const auto &c = p.contacts[k];
        const Local dt = direction.segment<K>(core + K * k);
        const Local velocity =
            local_contact_map<K>(c, true) * du.segment<6>(6 * c.i) + local_contact_map<K>(c, false) * du.segment<6>(6 * c.j);
        const Local load =
            transpose ? Local(-states[k].velocity_derivative.transpose() * dt)
                      : dt;
        out.segment<6>(nm + 6 * c.i) -= local_contact_map<K>(c, true).transpose() * load;
        out.segment<6>(nm + 6 * c.j) -= local_contact_map<K>(c, false).transpose() * load;
        out.segment<K>(core + K * k) =
            transpose
                ? Local(-velocity + states[k].traction_derivative.transpose() * dt)
                : Local(states[k].velocity_derivative * velocity +
                     states[k].traction_derivative * dt);
      }
      return out;
    };
    double forcing = opt.linear_relative_tolerance;
    if (opt.max_linear_forcing > 0.)
      forcing = std::max(
          forcing,
          std::min(opt.max_linear_forcing,
                   iteration == 0 ? opt.max_linear_forcing
                                  : .9 * std::pow(norm / previous_norm, 1.5)));
    const double target = forcing * norm;
    const Vector full_rhs = -raw;
    const Vector condensed_rhs = monitor.left(reduced.reduce_rhs(full_rhs));
    Vector last_core_direction, last_mobility;
    const auto linear = SlurryBridgeSolver::detail::fgmres(
        [&](const Vector &v) -> Vector {
          last_core_direction = v;
          return monitor.left(reduced.apply(v, [&](const Vector &moment) -> Vector {
            last_mobility = mob(moment);
            return last_mobility;
          }));
        }, inverse, condensed_rhs, target, opt);
    work.linear_iterations += linear.iterations;
    const Vector direction = reduced.recover(linear.x, full_rhs);
    // FGMRES recomputes its true residual at the returned core increment.
    // Reuse that mobility image; the force/traction rows are evaluated afresh.
    const bool cached_image = last_core_direction.size() == core &&
        (last_core_direction.array() == linear.x.array()).all();
    const Vector raw_image = jacobian(direction, false,
        cached_image ? &last_mobility : nullptr);
    trace("prox_linear", iteration, linear.iterations, 0., target,
          linear.residual);
    double chosen_step = 0.;
    auto search = [&](const Vector &d, const Vector &jd) -> bool {
      const double slope = residual.dot(left(jd));
      if (!(slope < 0.))
        return false;
      double alpha = 1.;
      for (int k = 0; k < opt.maximum_line_search_iterations;
           ++k, alpha *= .5) {
        const Vector candidate = x + alpha * d,
                     core_residual = raw.head(core) + alpha * jd.head(core);
        Vector candidate_raw = evaluate(candidate, nullptr, &core_residual),
               weighted = left(candidate_raw);
        const double candidate_norm = weighted.norm();
        if (!std::isfinite(candidate_norm))
          throw std::runtime_error(
              "bridge proximal solver: nonfinite trial norm");
        if (candidate_norm < norm &&
            weighted.squaredNorm() <= norm * norm + 2e-4 * alpha * slope) {
          x = candidate;
          raw = candidate_raw;
          residual = weighted;
          norm = candidate_norm;
          chosen_step = alpha;
          return true;
        }
      }
      return false;
    };
    previous_norm = norm;
    bool accepted = search(direction, raw_image), descent = false;
    if (!accepted) {
      const Vector gradient = left(jacobian(right(residual), true)),
                   d = right(gradient);
      const Vector jd = jacobian(d), weighted = left(jd);
      if (weighted.squaredNorm() > 0.) {
        const double step = -gradient.squaredNorm() / weighted.squaredNorm();
        accepted = search(step * d, step * jd);
        if (accepted) {
          ++work.descent_steps;
          descent = true;
        }
      }
    }
    if (!accepted)
      throw NonlinearFailure(
          "bridge proximal solver: Newton/descent line search failed", work,
          physical_norm, tolerance);
    ++work.newton_iterations;
    // The accepted core residual is affine in the retained variables and
    // was already evaluated by the line search. Only contact derivatives
    // need refreshing here. A fresh physical residual is checked at return.
    const Vector accepted_core = raw.head(core);
    raw = evaluate(x, &states, &accepted_core);
    residual = left(raw);
    norm = residual.norm();
    physical_norm = monitor.left(original_residual(x, raw)).norm();
    if (!std::isfinite(norm + physical_norm))
      throw std::runtime_error(
          "bridge proximal solver: nonfinite accepted norm");
    trace("prox_accepted", iteration, linear.iterations, chosen_step, target,
          linear.residual, descent);
  }
  if (physical_norm > tolerance && !polish)
    throw NonlinearFailure("bridge proximal solver: nonlinear iteration limit",
                           work, physical_norm, tolerance);
  if (physical_norm <= tolerance) {
    // The original law and a freshly evaluated mobility residual remain the
    // acceptance gate. A root that already passes needs no second nonlinear
    // solver, monitor factorization, or zero-state evaluation for 'polish'.
    const Vector final_velocity = x.segment(nm, nu),
                 final_q = SlurryBridgeSolver::detail::embed(final_velocity, n) + p.prescribed,
                 final_moment = x.head(nm) + p.near_resistance * final_q;
    Vector physical(core);
    physical.head(nm) = mob(x.head(nm)) - final_q;
    physical.tail(nu) = -SlurryBridgeSolver::detail::select(final_moment, n) + p.external;
    std::vector<ContactResponse> physical_contacts;
    physical_contacts.reserve(p.contacts.size());
    for (const auto &c : p.contacts) {
      physical_contacts.push_back(contact_response(c, final_velocity, p.dt));
      physical.segment<6>(nm + 6 * c.i).noalias() -=
          local_contact_map<K>(c, true).transpose() * local_wrench<K>(physical_contacts.back());
      physical.segment<6>(nm + 6 * c.j).noalias() -=
          local_contact_map<K>(c, false).transpose() * local_wrench<K>(physical_contacts.back());
    }
    const double final_norm = monitor.left(physical).norm();
    if (std::isfinite(final_norm) && final_norm <= tolerance) {
      work.velocity = final_velocity;
      work.mid_moment = x.head(nm);
      work.q = final_q;
      work.total_moment = final_moment;
      work.residual_norm = final_norm;
      work.relative_residual = final_norm / std::max(physical_rhs, opt.absolute_tolerance);
      work.mobility_residual_norm = physical.head(nm).norm();
      work.force_residual_norm = physical.tail(nu).norm();
      work.hydro_power = work.q.dot(work.total_moment);
      work.contacts = std::move(physical_contacts);
      for (const auto &c : work.contacts)
        work.contact_power += contact_power(c);
      return work;
    }
  }
  Result initial;
  initial.velocity = x.segment(nm, nu);
  initial.mid_moment = x.head(nm);
  Options final_options = opt;
  final_options.cold_start_continuation = false;
  final_options.trace_callback = [&](const IterationTrace &input) {
    if (!opt.trace_callback)
      return;
    IterationTrace event = input;
    event.continuation_stage = -3;
    const std::string phase = input.phase;
    if (phase == "begin")
      event.phase = "polish_begin";
    else if (phase == "linear")
      event.phase = "polish_linear";
    else if (phase == "accepted")
      event.phase = "polish_accepted";
    event.total_newton_iterations += work.newton_iterations;
    event.total_linear_iterations += work.linear_iterations;
    event.elapsed_seconds = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - started)
                                .count();
    opt.trace_callback(event);
  };
  Result result;
  try {
    result = SlurryBridgeSolver::solve_direct(p, final_options, &initial);
  } catch (const NonlinearFailure &error) {
    work.newton_iterations += error.newton_iterations;
    work.linear_iterations += error.linear_iterations;
    work.mobility_applications += error.mobility_applications;
    work.descent_steps += error.descent_steps;
    work.breakpoint_steps += error.breakpoint_steps;
    work.kink_reselections += error.kink_reselections;
    throw NonlinearFailure(error.what(), work, error.residual, error.tolerance);
  }
  result.newton_iterations += work.newton_iterations;
  result.linear_iterations += work.linear_iterations;
  result.mobility_applications += work.mobility_applications;
  result.descent_steps += work.descent_steps;
  return result;
}
inline Result solve_proximal(const Problem &p, const Options &opt = Options{},
                             const Result *warm = nullptr) {
  const bool angular = std::any_of(p.contacts.begin(), p.contacts.end(), rotation_enabled);
  return angular ? solve_proximal_impl<6>(p, opt, warm) : solve_proximal_impl<3>(p, opt, warm);
}
} // namespace SlurryBridgeContactSolver
#endif
