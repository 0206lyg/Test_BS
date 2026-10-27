#ifndef SLURRY_BRIDGE_STEP_H
#define SLURRY_BRIDGE_STEP_H

// Shared endpoint geometry and reference integrators. Production uses the
// explicit Heun stages in fix_slurry_bridge.cpp. The one-stage explicit and
// implicit routines below remain numerical comparison utilities. No path
// projects particle positions.
#include "slurry_bridge_contact_solver.h"
#include "slurry_bridge_linear_solver.h"
#include "slurry_bridge_images.h"
#include "../vendor/gr_re2/ellipsoidGap.h"
#include "slurry_bridge_shape.h"
#include <Eigen/Geometry>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>

namespace SlurryBridgeStep {
namespace bs = SlurryBridgeSolver;
namespace im = SlurryBridgeImages;
namespace gr = slurry::gr_re2::graphite;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Image = Eigen::Vector3i;
inline Vec3 vector(const gr::Vec3 &v) { return {v[0], v[1], v[2]}; }
inline gr::Vec3 array(const Vec3 &v) { return {v[0], v[1], v[2]}; }

// Shared by the trial geometry and the actual LAMMPS advance.
inline Eigen::Quaterniond rotate(const Eigen::Quaterniond &q, const Vec3 &omega,
                                double dt) {
  const double speed = omega.norm(), half = .5 * dt * speed;
  const double factor = speed > 0. ? std::sin(half) / speed : .5 * dt;
  return (Eigen::Quaterniond(std::cos(half), factor * omega[0],
                            factor * omega[1], factor * omega[2]) * q).normalized();
}

struct Endpoint {
  std::vector<gr::Body> bodies;
  Mat3 lattice;
  std::vector<double> gaps;
};
struct Geometry {
  std::vector<gr::Body> bodies;
  // Preserve the original quaternion, so trial and actual integration use
  // identical arithmetic, rather than converting a rotation matrix twice.
  std::vector<Eigen::Quaterniond> quaternions;
  Mat3 lattice = Mat3::Identity(), kappa = Mat3::Zero();
  Vec3 ambient_rotation = Vec3::Zero();
  double contact_onset = 0.;
  bool flake = false;
  double gap_tolerance = 1e-10;
  bool enforce_work_balance = false;
  std::vector<Image> images;

  Endpoint advance(const bs::Problem &p, const bs::Vector &velocity,
                   bool check_candidates = true) const {
    if (int(bodies.size()) != p.count || images.size() != p.contacts.size() ||
        quaternions.size() != bodies.size())
      throw std::invalid_argument("bridge endpoint geometry size mismatch");
    Endpoint out;
    out.bodies = bodies;
    // fix deform 0 updates xz after INITIAL_INTEGRATE. In the supported
    // constant-volume simple shear this is exact. A later integer tilt flip
    // changes the lattice basis, not these physical image separations.
    out.lattice = lattice + p.dt * kappa * lattice;
    for (int i = 0; i < p.count; ++i) {
      out.bodies[i].position = array(vector(bodies[i].position) + p.dt *
          (velocity.segment<3>(6 * i) + kappa * vector(bodies[i].position)));
      const Mat3 rotation = rotate(quaternions[i],
          velocity.segment<3>(6 * i + 3) + ambient_rotation, p.dt).toRotationMatrix();
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          out.bodies[i].rotation[3 * a + b] = rotation(a, b);
    }
    using Key = std::tuple<int, int, int, int, int>;
    std::set<Key> candidates;
    out.gaps.reserve(p.contacts.size());
    for (std::size_t k = 0; k < p.contacts.size(); ++k) {
      const auto &c = p.contacts[k];
      auto second = out.bodies[c.j];
      second.position = array(vector(second.position) + out.lattice * images[k].cast<double>());
      out.gaps.push_back(SlurryBridgeShape::closest_gap(out.bodies[c.i],second,flake,gap_tolerance).gap);
      if (check_candidates)
        candidates.emplace(c.i, c.j, images[k][0], images[k][1], images[k][2]);
    }
    if (!check_candidates)
      return out;
    double onset = contact_onset;
    for (const auto &c : p.contacts)
      onset = std::max(onset, c.onset);
    // Independent endpoint broad phase: a new contact must never silently
    // bypass the force solve just because it was outside the old near list.
    for (int i = 0; i < p.count; ++i)
      for (int j = i; j < p.count; ++j) {
        const double ai=SlurryBridgeShape::bounding_radius(bodies[i],flake);
        const double aj=SlurryBridgeShape::bounding_radius(bodies[j],flake);
        const Vec3 d = vector(out.bodies[j].position) - vector(out.bodies[i].position);
        for (const auto &image : im::images_within(d, out.lattice, ai + aj + onset)) {
          if (i == j && !im::positive_image(image.image))
            continue;
          if (candidates.count(Key(i, j, image.image[0], image.image[1], image.image[2])))
            continue;
          auto second = out.bodies[j];
          second.position = array(vector(out.bodies[i].position) + image.displacement);
          if (SlurryBridgeShape::closest_gap(out.bodies[i],second,flake,gap_tolerance).gap < onset)
            throw std::runtime_error("bridge endpoint gained an unlisted contact; reduce timestep");
        }
      }
    return out;
  }
};

struct Result {
  bs::Result solution;
  // Explicit-history path only: these histories belong to the next state;
  // solution.contacts always contains the CURRENT force-balance loads.
  std::vector<bs::ContactResponse> next_contacts;
  Endpoint endpoint;
  int geometry_iterations = 0;
  double maximum_gap_error = 0., maximum_linear_gap_defect = 0.;
};

inline Result solve_linear(const bs::Problem &problem, const Geometry &geometry,
                           const bs::Options &options = bs::Options{},
                           const bs::Result *warm = nullptr) {
  Result out;
  out.solution = SlurryBridgeLinearSolver::solve(problem, options, warm);
  // A single kinematic evaluation supplies the next normal load for the
  // history cap. It does not correct/re-solve the current force balance.
  // Newly encountered contacts are processed at the next velocity stage.
  out.endpoint = geometry.advance(problem, out.solution.velocity, false);
  out.next_contacts = SlurryBridgeLinearSolver::advance_contacts(
      problem, out.solution, out.endpoint.gaps);
  for (std::size_t k = 0; k < problem.contacts.size(); ++k) {
    const double predicted = problem.contacts[k].gap + problem.dt *
        problem.contacts[k].normal.dot(out.solution.contacts[k].relative);
    out.maximum_linear_gap_defect = std::max(out.maximum_linear_gap_defect,
        std::abs(out.endpoint.gaps[k] - predicted));
  }
  return out;
}

inline Result solve(const bs::Problem &original, const Geometry &geometry,
                    const bs::Options &options = bs::Options{},
                    const bs::Result *warm = nullptr) {
  bs::Problem corrected = original;
  bs::Result previous;
  Result out;
  const int nm = 11 * original.count, nu = 6 * original.count;
  auto workspace = options.workspace ? options.workspace
      : std::make_shared<bs::detail::SolverWorkspace>();
  bs::detail::Preconditioner monitor(original, bs::Vector::Zero(nu), workspace);
  auto raw_residual = [&](const bs::Result &state,
                          const std::vector<bs::ContactResponse> &contacts,
                          bool zero_moment = false) {
    bs::Vector raw(nm + nu);
    if (zero_moment)
      raw.head(nm) = -state.q;
    else
      raw.head(nm) = original.mobility(state.mid_moment) - state.q;
    raw.tail(nu) = -bs::detail::select(state.total_moment, original.count) + original.external;
    for (std::size_t k = 0; k < contacts.size(); ++k) {
      const auto &c = original.contacts[k];
      raw.segment<6>(nm + 6 * c.i) -= bs::generalized_contact_map(c, true).transpose() * bs::contact_wrench(contacts[k]);
      raw.segment<6>(nm + 6 * c.j) -= bs::generalized_contact_map(c, false).transpose() * bs::contact_wrench(contacts[k]);
    }
    return raw;
  };
  // Scale the acceptance test from the same fixed zero state as the inner
  // solver. It does not depend on an artificially excellent warm start.
  bs::Result zero;
  zero.velocity = bs::Vector::Zero(nu);
  zero.mid_moment = bs::Vector::Zero(nm);
  zero.q = original.prescribed;
  zero.total_moment = original.near_resistance * zero.q;
  std::vector<bs::ContactResponse> zero_contacts;
  for (const auto &c : original.contacts)
    zero_contacts.push_back(bs::contact_response(c, zero.velocity, original.dt));
  const double rhs = monitor.left(raw_residual(zero, zero_contacts, true)).norm();
  const double tolerance = options.absolute_tolerance + options.relative_tolerance * rhs;
  bs::Options inner = options;
  inner.workspace = workspace;
  inner.relative_tolerance *= .1;
  inner.absolute_tolerance *= .1;
  // These are nonlinear geometry corrections of ONE physical time step.
  // Every solve starts with the original committed tangential history.
  constexpr int geometry_iteration_block = 24;
  // Angular caps couple endpoint compression back to rotation and hence to
  // the gap. A contracting correction may need more than the legacy fixed
  // budget. Extend only after measured progress, without relaxing acceptance
  // or committing any intermediate position/history.
  const bool angular_contacts = std::any_of(original.contacts.begin(), original.contacts.end(), bs::rotation_enabled);
  const int maximum_geometry_iterations = angular_contacts ? 144 : geometry_iteration_block;
  int geometry_iteration_limit = geometry_iteration_block;
  double checkpoint_error = std::numeric_limits<double>::infinity();
  int linear = 0, newton = 0, applications = 0;
  int stages = 0, retries = 0, descent = 0, breakpoints = 0, reselections = 0;
  for (int iteration = 0; iteration < geometry_iteration_limit; ++iteration) {
    auto state = SlurryBridgeContactSolver::solve(
        corrected, inner, iteration ? &previous : warm);
    linear += state.linear_iterations;
    newton += state.newton_iterations;
    applications += state.mobility_applications;
    stages += state.continuation_stages;
    retries += state.continuation_retries;
    descent += state.descent_steps;
    breakpoints += state.breakpoint_steps;
    reselections += state.kink_reselections;
    auto endpoint = geometry.advance(original, state.velocity);
    std::vector<bs::ContactResponse> exact;
    exact.reserve(original.contacts.size());
    bool constitutive_ok = true;
    double gap_error = 0., linear_defect = 0., maximum_force_error = 0.,
           constitutive_error_ratio = 0.;
    for (std::size_t k = 0; k < original.contacts.size(); ++k) {
      const auto &c = original.contacts[k];
      const auto uncorrected = bs::contact_response(c, state.velocity, original.dt);
      const double defect = endpoint.gaps[k] - uncorrected.predicted_gap;
      corrected.contacts[k].gap = c.gap + defect;
      auto response = bs::contact_response(c, state.velocity, original.dt, endpoint.gaps[k]);
      // Compute compression directly from the endpoint to avoid cancelling
      // h + defect + dt*vn when reporting a very stiff, shallow contact.
      const double force_error = (bs::contact_wrench(response) - bs::contact_wrench(state.contacts[k])).norm();
      maximum_force_error=std::max(maximum_force_error,force_error);
      const double force_tolerance = options.absolute_tolerance +
          options.relative_tolerance * std::max(1., bs::contact_wrench(response).norm());
      constitutive_ok = constitutive_ok && force_error <= force_tolerance;
      constitutive_error_ratio = std::max(constitutive_error_ratio, force_error / force_tolerance);
      if (response.active || state.contacts[k].active)
        gap_error = std::max(gap_error, std::abs(endpoint.gaps[k] - state.contacts[k].predicted_gap));
      linear_defect = std::max(linear_defect, std::abs(defect));
      exact.push_back(response);
    }
    const bs::Vector raw = raw_residual(state, exact);
    ++applications;
    const double residual = monitor.left(raw).norm();
    if (!raw.allFinite() || !std::isfinite(residual))
      throw std::runtime_error("bridge endpoint correction produced a nonfinite residual");
    bool work_ok=true, inner_work_ok=true;
    double work_error_ratio=0.;
    if (geometry.enforce_work_balance) {
      // A contact preload can dominate the norm used to scale the residual.
      // Check the unscaled force residual's physical work as in the validated
      // cylinder pilot, so a small norm alone cannot mask an erroneous stress.
      const double medium=geometry.lattice.determinant()*geometry.kappa.squaredNorm();
      double contact_power=0., affine_contact_power=0.;
      for (std::size_t k=0;k<exact.size();++k) {
        contact_power+=bs::contact_power(exact[k]);
        affine_contact_power+=exact[k].traction.dot(original.contacts[k].affine);
      }
      const double input=medium+original.prescribed.dot(state.total_moment)+affine_contact_power;
      const double work_scale=std::max({std::abs(input),
          std::abs(medium+state.hydro_power)+std::abs(contact_power),1e-300});
      const double work_error=std::abs(state.velocity.dot(raw.tail(nu)));
      const double work_tolerance=std::max(1e-8,options.relative_tolerance)*work_scale;
      work_ok=work_error<=work_tolerance;
      work_error_ratio=work_error/work_tolerance;
      // Separate inner algebraic error from the changed endpoint-gap load.
      // Tightening Krylov/Newton cannot cure the latter; it belongs to the
      // outer geometry iteration and otherwise drives valid inner solves
      // below floating-point accuracy.
      // Only the force block is needed; skip a redundant mobility application.
      const bs::Vector inner_raw=raw_residual(state,state.contacts,true);
      const double inner_work_error=std::abs(state.velocity.dot(inner_raw.tail(nu)));
      inner_work_ok=inner_work_error<=work_tolerance;
      const char *trace=std::getenv("SLURRY_BRIDGE_TRACE");
      if (trace && std::strcmp(trace,"0")) {
        std::fprintf(stderr,"bridge endpoint iteration=%d residual=%.9e tolerance=%.9e work_error=%.9e inner_work_error=%.9e work_tolerance=%.9e force_error=%.9e gap_error=%.9e gap_certificate_tol=%.9e constitutive=%d\n",
          iteration+1,residual,tolerance,work_error,inner_work_error,work_tolerance,
          maximum_force_error,gap_error,geometry.gap_tolerance,int(constitutive_ok));
        std::fflush(stderr);
      }
    }
    if (constitutive_ok && residual <= tolerance && work_ok) {
      state.contacts = std::move(exact);
      state.residual_norm = residual;
      state.relative_residual = residual / std::max(rhs, options.absolute_tolerance);
      state.force_residual_norm = raw.tail(nu).norm();
      state.mobility_residual_norm = raw.head(nm).norm();
      state.contact_power = 0.;
      for (const auto &c : state.contacts)
        state.contact_power += bs::contact_power(c);
      state.linear_iterations = linear;
      state.newton_iterations = newton;
      state.mobility_applications = applications;
      state.continuation_stages = stages;
      state.continuation_retries = retries;
      state.descent_steps = descent;
      state.breakpoint_steps = breakpoints;
      state.kink_reselections = reselections;
      out.solution = std::move(state);
      out.endpoint = std::move(endpoint);
      out.geometry_iterations = iteration + 1;
      out.maximum_gap_error = gap_error;
      out.maximum_linear_gap_defect = linear_defect;
      return out;
    }
    bool refine_algebra = residual<=tolerance && !work_ok && !inner_work_ok;
    if (constitutive_ok && residual > tolerance) {
      // Corrected-gap preloads change the inner solver's relative residual
      // scale. It can accept an unchanged warm root even after the geometry
      // has converged, while that root misses the ORIGINAL outer tolerance.
      // Refine only a measured algebraic residual, not a remaining gap defect.
      bs::Vector inner_raw = raw_residual(state,state.contacts,true);
      inner_raw.head(nm) = raw.head(nm);
      refine_algebra = refine_algebra || monitor.left(inner_raw).norm() > .1*tolerance;
    }
    if (refine_algebra) {
      inner.relative_tolerance*=.1;
      inner.absolute_tolerance*=.1;
      inner.linear_relative_tolerance*=.1;
    }
    const double error_ratio = std::max({
        residual / std::max(tolerance, std::numeric_limits<double>::min()),
        constitutive_error_ratio, work_error_ratio});
    if (iteration == 0) checkpoint_error = error_ratio;
    if (iteration + 1 == geometry_iteration_limit &&
        geometry_iteration_limit < maximum_geometry_iterations &&
        error_ratio < .75 * checkpoint_error) {
      geometry_iteration_limit = std::min(maximum_geometry_iterations,
          geometry_iteration_limit + geometry_iteration_block);
      checkpoint_error = error_ratio;
    }
    previous = std::move(state);
  }
  throw std::runtime_error("bridge endpoint force/torque correction did not converge; reduce timestep (no position/history advance accepted)");
}
} // namespace SlurryBridgeStep
#endif
