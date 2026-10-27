#include "slurry_bridge_contact_solver.h"
#include <iostream>
#include <stdexcept>
#include <string>

using namespace SlurryBridgeSolver;
namespace pc = SlurryBridgeContactSolver;

static void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

static Eigen::Matrix3d cross(const V3 &v) {
  Eigen::Matrix3d out;
  out << 0., -v.z(), v.y(), v.z(), 0., -v.x(), -v.y(), v.x(), 0.;
  return out;
}

// Frozen algebraic contact fixtures from a deterministic two-body scan.
// Constants replace random generation so the regression is independent of
// the standard library's normal-distribution implementation and argument
// evaluation order. The isolated mobility is deliberately simple and SPD;
// this test isolates nonlinear contact globalization from FFT/geometry.
// Both cases use the production dimensionless stiffness and mu_s=mu_r=10.
// On the original monotone proximal solver, case 0 fails its line search and
// case 1 exhausts 40 Newton iterations. No physical parameter is softened in
// the equations whose roots are checked below.
static Problem fixture(int which) {
  Problem p;
  p.count = 2;
  p.dt = 1e-4;
  p.prescribed = Vector::Zero(22);
  p.external = Vector::Zero(12);
  p.near_resistance.resize(22, 22);
  p.self_mobility.assign(2, Matrix11::Identity());
  p.mobility = [](const Vector &x) -> Vector { return x; };
  Contact c;
  c.i = 0; c.j = 1;
  c.kn = 25e6;
  c.kt = c.kr = 20e6;
  c.mu = c.mu_roll = 10.;
  c.roll_length = 1.;
  V3 lever;
  if (which == 0) {
    c.normal = V3(1.7582471788833465, .51440203782084204, -1.0022761951571899).normalized();
    lever = V3(.25402763520501148, -.32568699840362186, -1.0616724017181867);
    c.gap = .00017240001688011286;
    c.affine = V3(.91834918977153579, .11026373160290613, -.31696959266516561);
    c.history = 1e-4 * V3(-1.2835701897346656, -.74313585036033425, -.018380462477013929);
    c.rolling_history = 1e-4 * V3(1.3080179476867038, -1.9095746173495254, .28462345411467249);
    p.external << 2.0928620281360315, 1.2686978686484163, .32680582368008099,
        .55095581847331421, .18702757311200968, .68356334228805826,
        -2.0093939488018084, 1.6615427120880084, -.21098072901723672,
        -.4242612995126025, -.097464196052029148, 2.4343426594908939;
  } else {
    c.normal = V3(-.63205494059539069, .90769114483207325, -.46282841866590785).normalized();
    lever = V3(-.138909391699124, 1.6940312757439282, 1.5623878722551627);
    c.gap = 6.5616037262114973e-05;
    c.affine = V3(1.2073541465107722, 1.7971625462931042, 1.673465554309943);
    c.history = 1e-4 * V3(1.3752077971344994, -.00089027707037017982, 1.025755466357082);
    c.rolling_history = 1e-4 * V3(-1.0540401988266388, -.12684448129350329, 1.1346089287455552);
    p.external << 1.3541202683134272, -.66634757985639126, -1.5632166539512979,
        -.23394849396706813, 1.4725439324009477, .35703005478182381,
        -.48936048584706693, -1.007986226681874, -.49550924681211383,
        .87195640833085497, .13491461173590025, -.16024464002370375;
  }
  const V3 second_lever = lever - 2. * c.normal;
  c.Bi.leftCols<3>() = -Eigen::Matrix3d::Identity();
  c.Bj.leftCols<3>() = Eigen::Matrix3d::Identity();
  c.Bi.rightCols<3>() = cross(lever);
  c.Bj.rightCols<3>() = -cross(second_lever);
  p.contacts.push_back(c);
  return p;
}

// Reevaluate force/torque balance using the ORIGINAL elastic return map,
// independently of the solver's retained wrench or continuation residual.
static Vector physical_residual(const Problem &p, const Vector &moment,
                                const Vector &velocity) {
  const int nm = 11 * p.count, nu = 6 * p.count;
  Vector q = p.prescribed;
  for (int i = 0; i < p.count; ++i) q.segment<6>(11*i) += velocity.segment<6>(6*i);
  const Vector total = moment + p.near_resistance * q;
  Vector residual(nm + nu);
  residual.head(nm) = p.mobility(moment) - q;
  residual.tail(nu) = p.external;
  for (int i = 0; i < p.count; ++i)
    residual.segment<6>(nm+6*i) -= total.segment<6>(11*i);
  for (const auto &c : p.contacts) {
    const auto force = contact_response(c, velocity, p.dt);
    residual.segment<6>(nm+6*c.i) -= generalized_contact_map(c, true).transpose() * contact_wrench(force);
    residual.segment<6>(nm+6*c.j) -= generalized_contact_map(c, false).transpose() * contact_wrench(force);
  }
  return residual;
}

static void unchanged(const Problem &p, const Problem &original) {
  require(p.dt == original.dt && p.count == original.count &&
          (p.prescribed-original.prescribed).isZero(0.) &&
          (p.external-original.external).isZero(0.) &&
          (p.near_resistance-original.near_resistance).norm() == 0.,
          "globalization changed the frozen physical problem");
  require(p.contacts.size() == original.contacts.size(), "globalization changed contact candidates");
  for (std::size_t k = 0; k < p.contacts.size(); ++k) {
    const auto &a = p.contacts[k], &b = original.contacts[k];
    require(a.kn == b.kn && a.kt == b.kt && a.kr == b.kr && a.kw == b.kw &&
            a.mu == b.mu && a.mu_roll == b.mu_roll && a.mu_twist == b.mu_twist &&
            a.gap == b.gap && a.onset == b.onset && a.roll_length == b.roll_length &&
            a.twist_length == b.twist_length && a.twisting_history == b.twisting_history &&
            (a.history-b.history).isZero(0.) &&
            (a.rolling_history-b.rolling_history).isZero(0.) &&
            (a.normal-b.normal).isZero(0.) && (a.affine-b.affine).isZero(0.) &&
            (a.Bi-b.Bi).isZero(0.) && (a.Bj-b.Bj).isZero(0.),
            "globalization committed a trial spring history or coefficient");
  }
}

int main() {
  Options opt;
  opt.relative_tolerance = 1e-7;
  opt.absolute_tolerance = 1e-12;
  opt.linear_relative_tolerance = 1e-8;
  opt.max_linear_forcing = .05;
  for (int index = 0; index < 2; ++index) {
    auto p = fixture(index);
    const Problem original = p;
    const auto result = pc::solve(p, opt);
    unchanged(p, original);
    detail::Preconditioner monitor(original, Vector::Zero(12));
    const double rhs = monitor.left(physical_residual(original, Vector::Zero(22), Vector::Zero(12))).norm();
    const double tolerance = opt.absolute_tolerance + opt.relative_tolerance * rhs;
    const double residual = monitor.left(physical_residual(original, result.mid_moment, result.velocity)).norm();
    require(std::isfinite(residual) && residual <= tolerance,
            "robust solve missed the original force/torque residual tolerance");
    require(result.contacts.size() == original.contacts.size(), "missing final contact state");
    const auto exact = contact_response(original.contacts[0], result.velocity, original.dt);
    require((contact_wrench(result.contacts[0])-contact_wrench(exact)).norm() <= 1e-12,
            "returned contact uses an intermediate continuation coefficient");
    require((result.contacts[0].tangent_history-exact.tangent_history).norm() <= 1e-14 &&
            (result.contacts[0].rolling_history-exact.rolling_history).norm() <= 1e-14,
            "returned spring history differs from original constitutive law");
    const auto warm = pc::solve(p, opt, &result);
    require(warm.linear_iterations == 0 && warm.newton_iterations == 0 &&
            warm.continuation_stages == 0 && warm.continuation_retries == 0,
            "already converged warm state ran extra nonlinear/continuation work");
    require((warm.velocity-result.velocity).isZero(0.), "warm root changed");
    unchanged(p, original);
    std::cout << "globalization fixture " << index << " residual=" << residual
              << " tolerance=" << tolerance << " stages=" << result.continuation_stages
              << " retries=" << result.continuation_retries << '\n';
  }

  // A fatal hydrodynamic/operator error is not a nonlinear convergence
  // failure. It must propagate once, rather than trigger a softened retry.
  auto bad = fixture(0);
  int operator_calls = 0;
  bad.mobility = [&](const Vector &) -> Vector {
    ++operator_calls;
    throw std::runtime_error("sentinel hydrodynamic operator failure");
  };
  Result initial;
  initial.mid_moment = Vector::Ones(22);
  initial.velocity = Vector::Zero(12);
  bool propagated = false;
  try { (void)pc::solve(bad, opt, &initial); }
  catch (const std::runtime_error &error) {
    propagated = std::string(error.what()) == "sentinel hydrodynamic operator failure";
  }
  require(propagated && operator_calls == 1, "fatal operator failure was swallowed or retried");
  std::cout << "bridge globalization: hard contact roots, original law, warm root, fatal error propagation passed\n";
}
