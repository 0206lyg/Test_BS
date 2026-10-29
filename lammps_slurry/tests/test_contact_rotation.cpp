#include "slurry_bridge_step.h"
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace SlurryBridgeSolver;
namespace pc = SlurryBridgeContactSolver;
using Six = Eigen::Matrix<double, 6, 1>;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

static Contact contact() {
  Contact c;
  c.i = 0; c.j = 1;
  c.Bi.leftCols<3>() = -Eigen::Matrix3d::Identity();
  c.Bj.leftCols<3>() = Eigen::Matrix3d::Identity();
  c.normal = V3::UnitZ();
  c.gap = .1; c.onset = .2; c.kn = 10.;
  c.kt = 7.; c.mu = .3;
  c.kr = 4.; c.kw = 3.;
  c.mu_roll = .4; c.roll_length = .2;
  c.mu_twist = .3; c.twist_length = .2;
  return c;
}

static Vector pair_velocity(const V3 &translation, const V3 &rotation) {
  Vector u = Vector::Zero(12);
  u.segment<3>(6) = translation;
  u.segment<3>(9) = rotation;
  return u;
}

static void check_jacobian(const Contact &c, const Vector &u, double dt) {
  const auto state = contact_response(c, u, dt);
  const Matrix6 analytic = contact_jacobian(state);
  Matrix6 difference;
  const double eps = 1e-6;
  for (int k = 0; k < 6; ++k) {
    Vector plus = u, minus = u;
    plus[6+k] += eps; minus[6+k] -= eps;
    difference.col(k) =
        (contact_wrench(contact_response(c, plus, dt)) -
         contact_wrench(contact_response(c, minus, dt))) / (2.*eps);
  }
  require((analytic-difference).norm() < 2e-8*(1.+analytic.norm()),
          "contact six-dimensional Jacobian fails central finite differences");
  require((contact_velocity(state) -
           generalized_contact_map(c, true)*u.head<6>() -
           generalized_contact_map(c, false)*u.tail<6>() -
           (Six() << c.affine, V3::Zero()).finished()).norm() < 1e-13,
          "contact velocity and generalized maps disagree");
}

static Problem problem() {
  Problem p;
  p.count = 2; p.dt = .005;
  p.self_mobility.assign(2, Matrix11::Identity());
  p.near_resistance.resize(22, 22);
  p.prescribed = Vector::Zero(22); p.external = Vector::Zero(12);
  p.mobility = [](const Vector &v) -> Vector { return v; };
  return p;
}

static pc::GeneralizedProximalResponse check_proximal(
    const Contact &c, const Six &velocity, const Six &wrench, double dt) {
  const pc::Scale scale{.7, .4, .3, .2};
  const auto state = pc::proximal_response(c, velocity, wrench, dt, scale);
  require((state.traction_inverse*state.traction_derivative-Matrix6::Identity()).norm() < 1e-11,
          "retained-wrench Jacobian inverse is incorrect");
  Matrix6 dv, df;
  const double eps = 1e-6;
  for (int axis = 0; axis < 6; ++axis) {
    const Six delta = eps*Six::Unit(axis);
    dv.col(axis) = (pc::proximal_response(c, Six(velocity+delta), wrench, dt, scale).residual-
                   pc::proximal_response(c, Six(velocity-delta), wrench, dt, scale).residual)/(2.*eps);
    df.col(axis) = (pc::proximal_response(c, velocity, Six(wrench+delta), dt, scale).residual-
                   pc::proximal_response(c, velocity, Six(wrench-delta), dt, scale).residual)/(2.*eps);
  }
  require((dv-state.velocity_derivative).norm() < 2e-8*(1.+dv.norm()) &&
              (df-state.traction_derivative).norm() < 2e-8*(1.+df.norm()),
          "retained-wrench Jacobians fail independent finite differences");
  return state;
}

static void check_condensation() {
  auto p = problem(); p.dt = .1;
  p.contacts = {contact(), contact(), contact(), contact()};
  p.contacts[2].gap = 10.;
  p.contacts[3].i = p.contacts[3].j = 1;
  p.contacts[3].Bi(0,4) = .3; p.contacts[3].Bj(1,3) = -.2;
  const int nm = 22, nu = 12, core = nm+nu, nc = 4, full = core+6*nc;
  const Six velocity[nc] = {
    (Six() << .01,.02,-.01,.01,.02,.01).finished(),
    (Six() << 3.,1.,-.01,2.,-1.,3.).finished(),
    (Six() << .1,0.,.1,.2,.1,.3).finished(),
    (Six() << .05,.02,-.01,0.,0.,0.).finished()};
  const Six wrench[nc] = {
    (Six() << .01,.02,-1.,.001,.002,.003).finished(),
    (Six() << .2,.1,-2.,.01,.02,.03).finished(),
    (Six() << .03,.02,.1,.01,.01,.01).finished(),
    (Six() << .1,.02,-1.,.02,.03,.04).finished()};
  Eigen::MatrixXd projection = Eigen::MatrixXd::Zero(nm,nu);
  for (int i=0; i<2; ++i) projection.block<6,6>(11*i,6*i).setIdentity();
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(full,full);
  jacobian.topLeftCorner(nm,nm).setIdentity();
  jacobian.block(0,nm,nm,nu) = -projection;
  jacobian.block(nm,0,nu,nm) = -projection.transpose();
  std::vector<pc::GeneralizedProximalResponse> states;
  for (int k=0; k<nc; ++k) {
    const auto &c = p.contacts[k];
    states.push_back(check_proximal(c,velocity[k],wrench[k],p.dt));
    Eigen::MatrixXd map = Eigen::MatrixXd::Zero(6,nu);
    map.block<6,6>(0,6*c.i) += generalized_contact_map(c,true);
    map.block<6,6>(0,6*c.j) += generalized_contact_map(c,false);
    jacobian.block(nm,core+6*k,nu,6) = -map.transpose();
    jacobian.block(core+6*k,nm,6,nu) = states[k].velocity_derivative*map;
    jacobian.block<6,6>(core+6*k,core+6*k) = states[k].traction_derivative;
  }
  pc::GeneralizedContactLinearization reduced(p,states);
  Eigen::MatrixXd condensed(core,core);
  for (int j=0; j<core; ++j) {
    Vector unit = Vector::Zero(core); unit[j] = 1.;
    condensed.col(j) = reduced.apply(unit,p.mobility);
  }
  const Eigen::MatrixXd d = jacobian.bottomRightCorner(6*nc,6*nc);
  const Eigen::MatrixXd reference = jacobian.topLeftCorner(core,core)-
      jacobian.topRightCorner(core,6*nc)*d.fullPivLu().solve(jacobian.bottomLeftCorner(6*nc,core));
  require((condensed-reference).norm() < 2e-12*(1.+reference.norm()),
          "six-dimensional contact condensation changed the full Newton operator");
  Vector rhs(full);
  for (int j=0; j<full; ++j) rhs[j] = std::sin(.37*j+.3);
  const Vector direction = reduced.recover(condensed.fullPivLu().solve(reduced.reduce_rhs(rhs)),rhs);
  const Vector exact = jacobian.fullPivLu().solve(rhs);
  require((direction-exact).norm() < 2e-10*(1.+exact.norm()) &&
              (jacobian*direction-rhs).norm() < 2e-10*(1.+rhs.norm()),
          "six-dimensional wrench recovery differs from the full Newton solve");
}

static void check_endpoint(const Options &options, double dt) {
  namespace st = SlurryBridgeStep;
  auto p = problem(); p.dt = dt;
  p.external[4] = .2; p.external[10] = -.2;
  p.external[5] = .3; p.external[11] = -.3;
  const double gap = .02+(dt > .05 ? .001 : .00001);
  const double separation = 2.+gap;
  auto c = contact();
  c.gap = gap; c.onset = .02; c.kn = 100.; c.kt = c.mu = 0.;
  c.kr = c.kw = 10.;
  c.mu_roll = .5; c.mu_twist = .25;
  c.roll_length = c.twist_length = .2;
  Eigen::Matrix3d cross;
  cross << 0.,-separation/2.,0.,separation/2.,0.,0.,0.,0.,0.;
  c.Bi.rightCols<3>() = cross; c.Bj.rightCols<3>() = cross;
  p.contacts.push_back(c);
  st::Geometry geometry;
  geometry.lattice = 100.*Eigen::Matrix3d::Identity();
  geometry.bodies.resize(2);
  geometry.quaternions.assign(2,Eigen::Quaterniond::Identity());
  for (auto &body : geometry.bodies) {
    body.axes = {20.,20.,1.};
    body.rotation = {1.,0.,0.,0.,1.,0.,0.,0.,1.};
  }
  geometry.bodies[0].position = {0.,0.,0.};
  geometry.bodies[1].position = {0.,0.,separation};
  geometry.images.emplace_back(0,0,0);
  geometry.contact_onset = c.onset;
  geometry.enforce_work_balance = true;
  const auto frozen = pc::solve_proximal(p,options);
  require(!frozen.contacts[0].active,
          "endpoint fixture must be open under the frozen linear predictor");
  const auto corrected = st::solve(p,geometry,options);
  const auto actual = contact_response(c,corrected.solution.velocity,p.dt,corrected.endpoint.gaps[0]);
  require(actual.active && actual.torque.norm() > 0. && corrected.geometry_iterations >= 2,
          "nonlinear endpoint geometry failed to activate angular contact");
  require((contact_wrench(actual)-contact_wrench(corrected.solution.contacts[0])).norm() < 2e-9,
          "endpoint angular caps use stale normal compression");
  require(corrected.solution.relative_residual <= options.relative_tolerance*1.01,
          "endpoint solve accepted an inconsistent force/torque balance");
  require(std::abs(corrected.solution.contact_power-contact_wrench(actual).dot(contact_velocity(actual))) < 2e-11,
          "endpoint contact power omits or duplicates angular work");
  require(p.contacts[0].rolling_history.isZero() && p.contacts[0].twisting_history == 0.,
          "endpoint trial changed committed angular history");
}

int main() {
  const double dt = .1;
  auto c = contact();
  // These fixtures stay away from normal activation and Coulomb branch kinks.
  const Vector sticking = pair_velocity(V3(.01, -.02, -.01), V3(.01, -.02, .01));
  check_jacobian(c, sticking, dt);
  const auto stick = contact_response(c, sticking, dt);
  require(stick.derivative_rot_velocity.norm() == 0.,
          "unsaturated angular springs acquired a normal-load derivative");
  require(stick.torque.norm() > 0., "angular spring fixture is inactive");

  const Vector saturated = pair_velocity(V3(3., 1., -.01), V3(2., -1., 3.));
  check_jacobian(c, saturated, dt);
  const auto slip = contact_response(c, saturated, dt);
  require(slip.derivative_rot_velocity.norm() > .01,
          "saturated angular torque omitted normal-load coupling");
  check_jacobian(c, pair_velocity(V3(.01, -.02, -.01), V3(2., -1., .001)), dt);
  check_jacobian(c, pair_velocity(V3(3., 1., -.01), V3(.001, -.002, 3.)), dt);
  for (const auto &response : {stick, slip}) {
    const auto retained = check_proximal(c,contact_velocity(response),contact_wrench(response),dt);
    require(retained.residual.norm() < 1e-12 &&
                (-retained.traction_inverse*retained.velocity_derivative-
                 contact_jacobian(response)).norm() < 1e-11,
            "retained-wrench root or its condensed derivative changes the contact law");
  }
  check_condensation();

  // Angular friction must not depend on whether sliding friction is enabled.
  auto angular_only = c;
  angular_only.kt = 0.; angular_only.mu = 0.;
  check_jacobian(angular_only, saturated, dt);
  require(contact_response(angular_only, saturated, dt).torque.norm() > 0.,
          "disabling sliding friction bypassed rolling/twisting contact");

  const auto tangent = (Eigen::Matrix3d::Identity()-c.normal*c.normal.transpose()).eval();
  const double rolling_cap = c.mu_roll*c.roll_length*slip.normal_load;
  const double twisting_cap = c.mu_twist*c.twist_length*slip.normal_load;
  require(std::abs((tangent*slip.torque).norm()-rolling_cap) < 1e-13,
          "rolling torque fails the load-times-length cap");
  require(std::abs(std::abs(c.normal.dot(slip.torque))-twisting_cap) < 1e-13,
          "twisting torque fails the load-times-length cap");
  require(slip.rolling_work > 0. && slip.twisting_work > 0.,
          "saturated angular springs did not record plastic work");
  require(std::abs(slip.rolling_energy-.5*c.kr*slip.rolling_history.squaredNorm()) < 1e-13 &&
              std::abs(slip.twisting_energy-.5*c.kw*slip.twisting_history*slip.twisting_history) < 1e-13,
          "angular stored energy disagrees with returned elastic histories");
  const V3 trial_roll = tangent*(c.rolling_history+dt*slip.relative_rotation);
  const double trial_twist = c.twisting_history+dt*c.normal.dot(slip.relative_rotation);
  require(std::abs(slip.rolling_work-(tangent*slip.torque).dot(trial_roll-slip.rolling_history)) < 1e-13 &&
              std::abs(slip.twisting_work-c.normal.dot(slip.torque)*(trial_twist-slip.twisting_history)) < 1e-13,
          "angular plastic work disagrees with the return-map displacement");

  // Stored energy can be released: check the discrete energy inequality, not
  // the incorrect assertion that elastic contact power is always positive.
  for (const V3 &omega : {V3(2., -1., 3.), V3(-.15, -.04, -.1), V3::Zero().eval()}) {
    auto old = angular_only;
    old.rolling_history = V3(.02, .01, 0.); old.twisting_history = .01;
    const auto state = contact_response(old, pair_velocity(V3::Zero(), omega), dt);
    const double before = .5*old.kr*old.rolling_history.squaredNorm() +
                          .5*old.kw*old.twisting_history*old.twisting_history;
    const double after = state.rolling_energy+state.twisting_energy;
    require(dt*state.torque.dot(omega) >= after-before-1e-13,
            "angular update creates energy beyond mechanical work");
  }

  auto inactive = c;
  inactive.gap = 1.;
  inactive.rolling_history = V3(.1, .2, 0.); inactive.twisting_history = .1;
  const auto open = contact_response(inactive, saturated, dt);
  require(!open.active && open.torque.norm() == 0. && open.rolling_energy == 0. &&
              open.twisting_energy == 0. && open.derivative_rot_velocity.norm() == 0. &&
              open.derivative_rotation.norm() == 0.,
          "an open contact retained angular torque, energy, or stiffness");
  Vector common = Vector::Zero(12);
  common.segment<3>(3) = V3(.7, -.3, .9);
  common.segment<3>(9) = common.segment<3>(3);
  require(contact_response(c, common, dt).torque.norm() == 0.,
          "angular contact resists common rotation rather than relative rotation");

  // Objectivity and label exchange are independent of solver convergence.
  c.history = V3(.02, -.01, 0.);
  c.rolling_history = V3(.03, .01, 0.); c.twisting_history = -.02;
  const Vector motion = pair_velocity(V3(.3, -.1, -.01), V3(.6, -.3, .8));
  const auto initial = contact_response(c, motion, dt);
  const Eigen::Matrix3d q = Eigen::AngleAxisd(.7, V3(1., 2., 3.).normalized()).toRotationMatrix();
  auto rotated = c;
  rotated.normal = q*c.normal; rotated.history = q*c.history;
  rotated.rolling_history = q*c.rolling_history; rotated.affine = q*c.affine;
  const auto objective = contact_response(rotated,
      pair_velocity(q*motion.segment<3>(6), q*motion.segment<3>(9)), dt);
  require((objective.traction-q*initial.traction).norm() < 1e-12 &&
              (objective.torque-q*initial.torque).norm() < 1e-12 &&
              (objective.rolling_history-q*initial.rolling_history).norm() < 1e-12 &&
              std::abs(objective.twisting_history-initial.twisting_history) < 1e-12,
          "angular contact is not objective under a rigid rotation of the fixture");
  require(std::abs(objective.rolling_work-initial.rolling_work) < 1e-12 &&
              std::abs(objective.twisting_work-initial.twisting_work) < 1e-12,
          "angular dissipation depends on laboratory axes");
  auto exchanged = c;
  exchanged.i = 1; exchanged.j = 0;
  exchanged.normal = -c.normal; exchanged.history = -c.history;
  exchanged.rolling_history = -c.rolling_history; exchanged.affine = -c.affine;
  // Both n and Omega_j-Omega_i reverse: the scalar twist history does not.
  const auto reverse = contact_response(exchanged, motion, dt);
  require((reverse.traction+initial.traction).norm() < 1e-12 &&
              (reverse.torque+initial.torque).norm() < 1e-12 &&
              std::abs(contact_wrench(reverse).dot(contact_velocity(reverse))-
                       contact_wrench(initial).dot(contact_velocity(initial))) < 1e-12,
          "pair-label exchange changes the physical angular contact");

  // Identity mobility admits exact two-particle normal and angular roots.
  // Compare both nonlinear solvers with that independent solution; comparing
  // only the solvers to each other would miss a shared torque/sign error.
  Options options;
  options.relative_tolerance = 1e-10; options.absolute_tolerance = 1e-12;
  options.linear_relative_tolerance = 1e-11;
  check_endpoint(options,.01);
  // This deliberately larger step has slowly converging cap/geometry
  // feedback. A decreasing physical residual must not hit the old 24-step
  // ceiling before the unchanged force/torque acceptance test is met.
  check_endpoint(options,.1);
  for (bool production_stiffness : {false,true}) {
    const std::vector<std::pair<double,double>> cases = production_stiffness ?
        std::vector<std::pair<double,double>>{{.0001,.0001},{.05,.05},{.05,.0001}} :
        std::vector<std::pair<double,double>>{{.05,.05},{2.,2.},{2.,.01}};
    for (const auto &loads : cases) {
      auto p = problem();
      auto cc = contact();
      cc.gap = .1; cc.onset = .11; cc.kn = 1e3; cc.kt = cc.mu = 0.;
      cc.kr = 40.; cc.kw = 25.;
      cc.mu_roll = .4; cc.roll_length = .3;
      cc.mu_twist = .3; cc.twist_length = .2;
      if (production_stiffness) {
        p.dt = 1e-4;
        cc.gap = 0.; cc.onset = 1e-5; cc.kn = 6.7e8;
        cc.kr = cc.kw = cc.kn*.02*.02;
        cc.mu_roll = cc.mu_twist = .1;
        cc.roll_length = cc.twist_length = .02;
      }
      p.contacts.push_back(cc);
      p.external[2] = 1.; p.external[8] = -1.;
      p.external[3] = loads.first; p.external[9] = -loads.first;
      p.external[5] = loads.second; p.external[11] = -loads.second;
      const double normal = cc.kn*((cc.onset-cc.gap)+2.*p.dt)/(1.+2.*cc.kn*p.dt);
      const double roll = std::min(2.*cc.kr*p.dt*loads.first/(1.+2.*cc.kr*p.dt),
                                   cc.mu_roll*cc.roll_length*normal);
      const double twist = std::min(2.*cc.kw*p.dt*loads.second/(1.+2.*cc.kw*p.dt),
                                    cc.mu_twist*cc.twist_length*normal);
      const auto condensed = pc::solve_proximal(p, options);
      const auto eliminated = solve_direct(p, options);
      require(std::abs(condensed.contacts[0].normal_load-normal) < 2e-9 &&
                  std::abs(condensed.contacts[0].torque[0]+roll) < 2e-9 &&
                  std::abs(condensed.contacts[0].torque[2]+twist) < 2e-9,
              "condensed contact solve disagrees with exact normal/angular loads");
      require(std::abs(condensed.velocity[3]-(loads.first-roll)) < 2e-9 &&
                  std::abs(condensed.velocity[5]-(loads.second-twist)) < 2e-9,
              "angular torque scatter has the wrong sign or magnitude");
      require((condensed.velocity-eliminated.velocity).norm() < 2e-8,
              "retained-wrench and eliminated-contact solvers give different roots");
      require(std::abs(p.external.dot(condensed.velocity)-condensed.hydro_power-
                       condensed.contact_power) < 2e-8,
              "full-system work balance omits contact torque power");
      const auto warm = pc::solve_proximal(p, options, &condensed);
      require(warm.linear_iterations == 0,
              "an unchanged angular contact root triggered a new Krylov solve");
    }
  }
  std::cout << "contact rotation: six-dimensional Jacobians, caps/work, objectivity, "
               "pair exchange, stiff analytic roots, endpoint caps and power balance passed\n";
}
