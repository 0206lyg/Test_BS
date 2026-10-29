#include "slurry_bridge_contact_solver.h"
#include <Eigen/LU>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace SlurryBridgeSolver;
namespace pc = SlurryBridgeContactSolver;
static void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
static Problem base(int n) {
  Problem p;
  p.count = n; p.dt = .1;
  p.self_mobility.assign(n, Matrix11::Identity());
  p.near_resistance.resize(11*n, 11*n);
  p.prescribed = Vector::Zero(11*n); p.external = Vector::Zero(6*n);
  p.mobility = [](const Vector &v) -> Vector { return v; };
  return p;
}
static Contact contact(int i, int j) {
  Contact c; c.i = i; c.j = j; c.normal = V3::UnitZ();
  c.Bi.leftCols<3>() = -Eigen::Matrix3d::Identity();
  c.Bj.leftCols<3>() = Eigen::Matrix3d::Identity();
  c.Bi(0,4) = .3; c.Bj(1,3) = -.2;
  c.kn = 100.; c.kt = 50.; c.mu = .3;
  return c;
}
int main() {
  // Independent dense assembly of the RETAINED-force Jacobian. The four
  // contacts exercise sticking, sliding, open, and stiff self-image branches
  // at off-root tractions; the reduction must be exact before convergence.
  auto p = base(4);
  const int nm = 44, nu = 24, core = nm+nu, nc = 4, full = core+3*nc;
  std::mt19937 generator(1948);
  std::normal_distribution<double> random;
  Eigen::MatrixXd a(nm,nm);
  for (int k = 0; k < a.size(); ++k) a.data()[k] = random(generator);
  const Eigen::MatrixXd mobility = Eigen::MatrixXd::Identity(nm,nm)+.005*a*a.transpose();
  const Eigen::MatrixXd near = .03*Eigen::MatrixXd::Identity(nm,nm)+.002*a.transpose()*a;
  p.near_resistance = near.sparseView();
  p.mobility = [&](const Vector &v) -> Vector { return mobility*v; };
  p.contacts = {contact(0,1),contact(1,2),contact(2,3),contact(3,3)};
  p.contacts[1].mu = .1;
  p.contacts[2].gap = 10.;
  p.contacts[3].kn = 1e6; p.contacts[3].kt = 4e5; p.contacts[3].gap = -.01;
  p.contacts[3].Bj(0,4) = .8;
  const V3 relative[nc] = {V3(.05,.02,-.2),V3(3.,1.,-.2),V3(.1,0.,.1),V3(.01,.02,-.1)};
  const V3 traction[nc] = {V3(.01,.02,-1.),V3(.2,.1,-2.),V3(.03,.02,.1),V3(.1,.02,-1.)};
  std::vector<pc::ProximalResponse> state;
  Eigen::MatrixXd projection = Eigen::MatrixXd::Zero(nm,nu);
  for (int i = 0; i < p.count; ++i) projection.block<6,6>(11*i,6*i).setIdentity();
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(full,full);
  jacobian.topLeftCorner(nm,nm) = mobility;
  jacobian.block(0,nm,nm,nu) = -projection;
  jacobian.block(nm,0,nu,nm) = -projection.transpose();
  jacobian.block(nm,nm,nu,nu) = -projection.transpose()*near*projection;
  for (int k = 0; k < nc; ++k) {
    const auto &c = p.contacts[k];
    state.push_back(pc::proximal_response(c,relative[k],traction[k],p.dt,{.7,.4}));
    Eigen::MatrixXd map = Eigen::MatrixXd::Zero(3,nu);
    map.block<3,6>(0,6*c.i) += c.Bi; map.block<3,6>(0,6*c.j) += c.Bj;
    jacobian.block(nm,core+3*k,nu,3) = -map.transpose();
    jacobian.block(core+3*k,nm,3,nu) = state[k].velocity_derivative*map;
    jacobian.block<3,3>(core+3*k,core+3*k) = state[k].traction_derivative;
  }
  require(state[0].velocity_derivative.trace() < -1., "sticking fixture missing");
  require((state[1].traction_derivative-state[1].traction_derivative.transpose()).norm()>.01,
          "sliding Coulomb coupling missing");
  require(state[2].velocity_derivative.norm()==0., "open fixture missing");
  require(state[3].traction_inverse.norm()>1e4, "stiff fixture missing");
  pc::ContactLinearization reduced(p,state);
  Eigen::MatrixXd condensed(core,core);
  for (int j=0;j<core;++j) {
    Vector e=Vector::Zero(core);e[j]=1.;
    condensed.col(j)=reduced.apply(e,p.mobility);
  }
  const Eigen::MatrixXd d=jacobian.bottomRightCorner(3*nc,3*nc);
  const Eigen::MatrixXd reference=jacobian.topLeftCorner(core,core)-
      jacobian.topRightCorner(core,3*nc)*d.fullPivLu().solve(jacobian.bottomLeftCorner(3*nc,core));
  require((condensed-reference).norm()/reference.norm()<2e-13,
          "condensed operator differs from independent full-Jacobian Schur complement");
  Vector rhs(full);
  for(int j=0;j<full;++j)rhs[j]=random(generator);
  const Vector direction=reduced.recover(condensed.fullPivLu().solve(reduced.reduce_rhs(rhs)),rhs);
  const Vector exact=jacobian.fullPivLu().solve(rhs);
  require((direction-exact).norm()/exact.norm()<2e-10,
          "contact recovery differs from full nonsymmetric Newton solve");
  require((jacobian*direction-rhs).norm()/rhs.norm()<2e-9,
          "recovered increment fails retained-force linear residual");

  // Analytic two-body roots check actual normal force, stick/slip traction,
  // and the ORIGINAL physical residual, including a near-inactive swarm.
  Options opt; opt.relative_tolerance=1e-10;opt.absolute_tolerance=1e-12;
  opt.linear_relative_tolerance=1e-11;opt.max_linear_forcing=.05;
  for(double tangential : {.01,2.}) {
    auto q=base(2);q.dt=.001;
    auto c=contact(0,1);c.Bi.rightCols<3>().setZero();c.Bj.rightCols<3>().setZero();
    c.gap=.002;c.onset=.003;c.kn=1e6;c.kt=8e5;c.mu=.1;
    q.contacts.push_back(c);
    q.external[0]=tangential;q.external[6]=-tangential;
    q.external[2]=1.;q.external[8]=-1.;
    const auto r=pc::solve_proximal(q,opt);
    const double normal=c.kn*((c.onset-c.gap)+2*q.dt)/(1+2*c.kn*q.dt);
    const double tangent_load=std::min(2*c.kt*q.dt*tangential/(1+2*c.kt*q.dt),c.mu*normal);
    require(std::abs(r.contacts[0].normal_load-normal)<1e-8,"stiff normal root changed");
    require(std::abs(r.contacts[0].traction[0]+tangent_load)<1e-8,"stick/slip traction changed");
    require(std::abs(r.velocity[0]-(tangential-tangent_load))<1e-8,"tangential velocity changed");
    require(r.relative_residual<=opt.relative_tolerance*1.01,"physical residual gate changed");
    require(r.contacts[0].sliding==(tangential>1.),"wrong contact branch");
    auto open=c;open.gap=10.;
    for(int k=0;k<800;++k)q.contacts.push_back(open);
    const auto with_candidates=pc::solve_proximal(q,opt);
    require((with_candidates.velocity-r.velocity).norm()<1e-8,
            "inactive candidates changed the physical root");
    require(with_candidates.linear_iterations==r.linear_iterations,
            "inactive candidates changed condensed Krylov work");
    const auto warm=pc::solve_proximal(q,opt,&with_candidates);
    require(warm.linear_iterations==0,"unchanged warm root caused extra linear iterations");
  }
  std::cout<<"contact condensation: full Jacobian, stiff stick/slip, inactive candidates, warm root passed\n";
}
