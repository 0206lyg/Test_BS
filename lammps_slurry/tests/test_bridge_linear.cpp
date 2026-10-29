#include "slurry_bridge_step.h"
#include <Eigen/LU>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace bs = SlurryBridgeSolver;
namespace ls = SlurryBridgeLinearSolver;
namespace st = SlurryBridgeStep;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
bs::Problem simple(int n = 2) {
  bs::Problem p;
  p.count = n;
  p.dt = .01;
  p.prescribed = bs::Vector::Zero(11 * n);
  p.external = bs::Vector::Zero(6 * n);
  p.near_resistance.resize(11 * n, 11 * n);
  p.self_mobility.assign(n, bs::Matrix11::Identity());
  p.mobility = [](const bs::Vector &x) { return x; };
  return p;
}
bs::Contact pair() {
  bs::Contact c;
  c.i = 0; c.j = 1;
  c.Bi.leftCols<3>() = -Mat3::Identity();
  c.Bj.leftCols<3>() = Mat3::Identity();
  c.normal = Vec3::UnitZ();
  c.gap = -.04; c.kn = 100.; c.kt = 50.; c.mu = .4;
  c.history = Vec3(.1, .06, 0.);
  c.kr = 30.; c.mu_roll = .2; c.roll_length = .7;
  c.rolling_history = Vec3(.08, .05, 0.);
  return c;
}
bs::Options accuracy() {
  bs::Options o;
  o.relative_tolerance = 1e-11;
  o.absolute_tolerance = 1e-12;
  o.linear_relative_tolerance = 1e-12;
  return o;
}

void dense_reference() {
  auto p = simple(3);
  const int nm = 33, nu = 18;
  std::mt19937 generator(5041);
  std::normal_distribution<double> random;
  Eigen::MatrixXd a(nm, nm), b(nm, nm);
  for (int k = 0; k < a.size(); ++k) {
    a.data()[k] = random(generator); b.data()[k] = random(generator);
  }
  const Eigen::MatrixXd m = Eigen::MatrixXd::Identity(nm, nm) + .02 * a * a.transpose();
  const Eigen::MatrixXd l = .03 * b * b.transpose();
  p.mobility = [&](const bs::Vector &x) -> bs::Vector { return m * x; };
  p.near_resistance = l.sparseView();
  for (int i = 0; i < 3; ++i) {
    p.self_mobility[i] = m.block<11, 11>(11 * i, 11 * i);
    for (int j = 6; j < 11; ++j) p.prescribed[11 * i + j] = random(generator);
  }
  for (int i = 0; i < nu; ++i) p.external[i] = .1 * random(generator);
  auto c = pair();
  c.Bi(0, 4) = .8; c.Bj(1, 3) = -.4;
  c.affine = Vec3(100., -40., -50.);
  p.contacts.push_back(c);
  // A closing separated candidate must not create an implicit endpoint load.
  auto separated = c;
  separated.i = 1; separated.j = 2;
  separated.gap = .001; separated.affine[2] = -10000.;
  p.contacts.push_back(separated);

  // Independently assemble the current load, including lever torques and
  // saturated rolling. No implementation contact evaluator is used here.
  const double normal = -c.kn * c.gap;
  const Vec3 tangent = c.mu * normal * c.history.normalized();
  const Vec3 torque = c.mu_roll * normal * c.roll_length * c.rolling_history.normalized();
  const Vec3 traction = tangent - normal * Vec3::UnitZ();
  bs::Vector force = p.external;
  force.segment<6>(0) -= c.Bi.transpose() * traction;
  force.segment<6>(6) -= c.Bj.transpose() * traction;
  force.segment<3>(3) += torque;
  force.segment<3>(9) -= torque;
  Eigen::MatrixXd projection = Eigen::MatrixXd::Zero(nm, nu);
  for (int i = 0; i < 3; ++i) projection.block<6, 6>(11 * i, 6 * i).setIdentity();
  const Eigen::MatrixXd resistance = m.inverse() + l;
  const bs::Vector reference = (projection.transpose() * resistance * projection)
      .ldlt().solve(force - projection.transpose() * resistance * p.prescribed);
  const auto result = ls::solve(p, accuracy());
  require((result.velocity - reference).norm() < 2e-9,
          "current-load linear solve differs from independent dense resistance");
  require((result.contacts[0].traction - traction).norm() < 1e-12 &&
          (result.contacts[0].torque - torque).norm() < 1e-12,
          "linear solve changed known contact loads");
  require(!result.contacts[1].active, "closing separated pair gained an implicit load");
  require(result.newton_iterations == 0 && result.continuation_stages == 0 &&
          result.continuation_retries == 0 && result.kink_reselections == 0,
          "linear path entered nonlinear contact machinery");
  const auto warm = ls::solve(p, accuracy(), &result);
  require(warm.linear_iterations == 0, "already converged linear warm root was re-solved");
  p.dt = 1.5;
  const auto changed_dt = ls::solve(p, accuracy());
  require((changed_dt.velocity - result.velocity).norm() < 1e-11 &&
          (changed_dt.contacts[0].traction - result.contacts[0].traction).norm() == 0.,
          "time step changed the known-state force balance");
}

void history_phases() {
  auto p = simple();
  auto c = pair();
  c.history = Vec3(.001, .002, 0.);
  c.rolling_history = Vec3(.002, .001, 0.);
  c.affine = Vec3(.1, -.2, 0.);
  p.contacts.push_back(c);
  auto now = ls::solve(p, accuracy());
  const auto snapshot = now.contacts[0];
  const auto next = ls::advance_contacts(p, now, {-.02});
  Vec3 trial = snapshot.tangent_history + p.dt * snapshot.relative;
  trial[2] = 0.;
  const double cap = c.mu * c.kn * .02 / c.kt;
  if (trial.norm() > cap) trial *= cap / trial.norm();
  require((next[0].tangent_history - trial).norm() < 1e-13,
          "explicit history did not advance current velocity with endpoint cap");
  require((now.contacts[0].traction - snapshot.traction).norm() == 0.,
          "history update overwrote current force-balance loads");
  require(std::abs(now.contacts[0].normal_load - 4.) < 1e-14 &&
          std::abs(next[0].normal_load - 2.) < 1e-14,
          "current and next normal load phases were mixed");
  const auto opened = ls::advance_contacts(p, now, {.01});
  require(!opened[0].active && opened[0].tangent_history.isZero(0.) &&
          opened[0].rolling_history.isZero(0.), "opening contact kept a spring history");

  p.contacts[0].gap = .01;
  p.contacts[0].affine = Vec3(100., 20., -10.);
  now = ls::solve(p, accuracy());
  const auto created = ls::advance_contacts(p, now, {-.02});
  require(!now.contacts[0].active && created[0].active &&
          created[0].tangent_history.isZero(0.) && created[0].rolling_history.isZero(0.),
          "new contact charged spring displacement before contact existed");
}

double relax(double dt) {
  auto p = simple(); p.dt = dt;
  auto c = pair();
  c.kn = 1.; c.kt = 0.; c.kr = 0.; c.mu = 0.;
  c.gap = -.1;
  p.contacts.push_back(c);
  for (int step = 0; step < std::lround(.2 / dt); ++step) {
    const auto current = ls::solve(p, accuracy());
    p.contacts[0].gap += dt * (current.velocity[8] - current.velocity[2]);
  }
  return -p.contacts[0].gap;
}
void convergence_and_geometry() {
  const double exact = .1 * std::exp(-.4);
  const double coarse = std::abs(relax(.02) - exact);
  const double fine = std::abs(relax(.01) - exact);
  require(coarse / fine > 1.95 && coarse / fine < 2.1,
          "explicit normal relaxation does not converge to its known ODE");

  auto p = simple();
  p.contacts.push_back(pair());
  st::Geometry g;
  g.lattice = 100. * Mat3::Identity();
  g.bodies.resize(2);
  g.quaternions.assign(2, Eigen::Quaterniond::Identity());
  g.images.emplace_back(0, 0, 0);
  for (auto &b : g.bodies) {
    b.axes = {2., 2., 1.};
    b.rotation = {1,0,0,0,1,0,0,0,1};
  }
  g.bodies[0].position = {0.,0.,0.};
  g.bodies[1].position = {0.,0.,1.96};
  const auto step = st::solve_linear(p, g, accuracy());
  require(step.geometry_iterations == 0 && step.solution.newton_iterations == 0 &&
          step.next_contacts.size() == 1 && step.endpoint.gaps.size() == 1,
          "spheroid linear step performed a nonlinear endpoint correction");
  require(std::abs(step.solution.contacts[0].normal_load - 4.) < 1e-14,
          "spheroid endpoint changed the current normal load");
}

void explicit_stability_is_not_hidden() {
  auto energy_ratio = [](double dt) {
    auto p = simple(); p.dt = dt;
    auto c = pair();
    c.kn = 1.; c.kt = 10.; c.kr = 0.; c.mu = 1e9;
    c.history = Vec3(.01, 0., 0.);
    c.rolling_history.setZero();
    p.contacts.push_back(c);
    // Hold the normal compression with external loads, leaving exactly
    // d(xi)/dt=-2*kt*xi in the tangential coordinate for identity mobility.
    p.external[2] = -c.kn * c.gap;
    p.external[8] = c.kn * c.gap;
    const auto current = ls::solve(p, accuracy());
    const auto next = ls::advance_contacts(p, current, {c.gap});
    const double expected = std::pow(1. - 2. * c.kt * dt, 2);
    const double actual = next[0].tangent_energy / current.contacts[0].tangent_energy;
    require(std::abs(actual - expected) < 1e-11,
            "explicit spring update hid the time-discretization error");
    return actual;
  };
  require(energy_ratio(.01) < 1., "stable explicit spring step did not dissipate energy");
  require(energy_ratio(.11) > 1.,
          "unstable explicit step was silently replaced by implicit stabilization");
}

int main() {
  dense_reference();
  history_phases();
  convergence_and_geometry();
  explicit_stability_is_not_hidden();
  std::cout << "linear force balance, phase separation and timestep convergence passed\n";
}
