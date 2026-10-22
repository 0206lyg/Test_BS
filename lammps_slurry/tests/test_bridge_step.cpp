#include "slurry_bridge_step.h"
#include <iostream>
#include <stdexcept>

namespace st = SlurryBridgeStep;
namespace bs = SlurryBridgeSolver;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
void require(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
bs::Problem problem() {
  bs::Problem p;
  p.count = 2;
  p.dt = .1;
  p.prescribed = bs::Vector::Zero(22);
  p.external = bs::Vector::Zero(12);
  p.near_resistance.resize(22,22);
  p.self_mobility.assign(2, bs::Matrix11::Identity());
  p.mobility = [](const bs::Vector &x) { return x; };
  return p;
}
st::Geometry geometry(double a, double c, double separation) {
  st::Geometry g;
  g.lattice = 100. * Mat3::Identity();
  g.bodies.resize(2);
  g.quaternions.assign(2, Eigen::Quaterniond::Identity());
  for (auto &b : g.bodies) {
    b.axes = {a,a,c};
    b.rotation = {1,0,0,0,1,0,0,0,1};
  }
  g.bodies[0].position = {0,0,0};
  g.bodies[1].position = {0,0,separation};
  g.images.emplace_back(0,0,0);
  return g;
}
bs::Contact contact(double gap, double onset, double stiffness, double separation) {
  bs::Contact c;
  c.i=0; c.j=1; c.gap=gap; c.onset=onset; c.kn=stiffness;
  c.normal = Vec3::UnitZ();
  c.Bi.leftCols<3>() = -Mat3::Identity();
  c.Bj.leftCols<3>() = Mat3::Identity();
  // Contact midpoint levers +/- separation/2 along z.
  Mat3 cross;
  cross << 0,-separation/2,0,separation/2,0,0,0,0,0;
  c.Bi.rightCols<3>() = cross;
  c.Bj.rightCols<3>() = cross;
  return c;
}
int main() {
  bs::Options options;
  options.relative_tolerance = 1e-10;
  options.absolute_tolerance = 1e-12;
  options.linear_relative_tolerance = 1e-11;
  // A symmetric counter-rotation fixture has an exact endpoint gap:
  // h = d + 2*dt*N - 2*sqrt(a^2 sin^2(theta)+c^2 cos^2(theta)).
  // At theta=0 the first derivative of rotational gap closure is ZERO.
  // The old linear predictor therefore misses the newly active contact.
  auto p=problem();
  const double a=20., c=1., onset=.02, separation=2*c+onset+.001;
  p.external[4]=.2;
  p.external[10]=-.2;
  p.contacts.push_back(contact(separation-2*c,onset,100.,separation));
  auto g=geometry(a,c,separation);
  g.contact_onset=onset;
  const auto frozen=SlurryBridgeContactSolver::solve_proximal(p,options);
  require(!frozen.contacts[0].active,"fixture must be inactive under linear gap prediction");
  require(g.advance(p,frozen.velocity).gaps[0]<0.,"fixture must expose rotation-induced penetration");
  const auto corrected=st::solve(p,g,options);
  const double theta=p.dt*.2;
  const double reach=std::sqrt(a*a*std::sin(theta)*std::sin(theta)+c*c*std::cos(theta)*std::cos(theta));
  const double load=100.*(onset-separation+2*reach)/(1.+2*p.dt*100.);
  const double exact_gap=separation+2*p.dt*load-2*reach;
  require(corrected.geometry_iterations>=2,"nonlinear geometry was not corrected");
  require(std::abs(corrected.solution.contacts[0].normal_load-load)<2e-9,"rotation contact load differs from analytic endpoint solution");
  require(std::abs(corrected.endpoint.gaps[0]-exact_gap)<2e-11,"rotation endpoint gap differs from analytic solution");
  require(std::abs(corrected.solution.contacts[0].normal_load-100.*(onset-corrected.endpoint.gaps[0]))<1e-13,
          "normal force must use actual endpoint compression");
  require(corrected.solution.relative_residual<options.relative_tolerance,"actual endpoint force residual failed");
  require(p.contacts[0].gap==separation-2*c && p.contacts[0].history.isZero(),"trial solve changed committed geometry/history");
  const auto repeated=st::solve(p,g,options,&corrected.solution);
  require((repeated.solution.velocity-corrected.solution.velocity).norm()<2e-8,"run0-style repeated solve changed the physical state");

  // Production stiffness scale: kn/(eta*c*rate) for 100 N/m, .01 Pa s,
  // c=150 nm and shear=100/s, with physical dt=1 microsecond.
  p=problem(); p.dt=1e-4;
  p.external[4]=.5; p.external[10]=-.5;
  const double stiff=100./(.01*150e-9*100.);
  p.contacts.push_back(contact(.02,.02,stiff,2.02));
  g=geometry(20.,1.,2.02); g.contact_onset=.02;
  auto production_options=options; production_options.relative_tolerance=1e-6;
  const auto production=st::solve(p,g,production_options);
  require(production.solution.relative_residual<=1e-6,"production stiffness endpoint residual failed");
  require(std::abs(production.solution.contacts[0].normal_load-stiff*(.02-production.endpoint.gaps[0]))<1e-12,
          "production stiffness normal load is inconsistent with actual geometry");

  // Pure translations reduce exactly to the original implicit spring law.
  p=problem();
  p.external[2]=2.; p.external[8]=-2.;
  p.contacts.push_back(contact(.01,.02,100.,2.01));
  g=geometry(1.,1.,2.01);
  auto expected=SlurryBridgeContactSolver::solve_proximal(p,options);
  auto translation=st::solve(p,g,options);
  require((translation.solution.velocity-expected.velocity).norm()<2e-10,"endpoint correction changed exact translational law");

  // The endpoint image must include the shear of the periodic lattice.
  p=problem();
  p.contacts.push_back(contact(.5,.02,100.,2.5));
  g=geometry(1.,1.,2.5);
  g.lattice=10.*Mat3::Identity();
  g.bodies[0].position={0.,0.,8.};
  g.bodies[1].position={.3,0.,.5};
  g.images[0]=st::Image(0,0,1);
  g.kappa(0,2)=.7;
  bs::Vector zero=bs::Vector::Zero(12);
  auto endpoint=g.advance(p,zero,false);
  const Vec3 initial(.3,0.,2.5), actual=initial+p.dt*g.kappa*initial;
  require(std::abs(endpoint.gaps[0]-(actual.norm()-2.))<1e-12,"periodic shear image was advanced inconsistently");

  // New endpoint contacts outside the original candidate set fail closed.
  p=problem();
  g=geometry(1.,1.,3.);
  g.images.clear(); g.contact_onset=.02;
  bs::Vector crossing=bs::Vector::Zero(12);
  crossing[2]=6.; crossing[8]=-6.;
  bool rejected=false;
  try { g.advance(p,crossing); }
  catch(const std::runtime_error &e) { rejected=std::string(e.what()).find("unlisted contact")!=std::string::npos; }
  require(rejected,"unlisted endpoint contact was silently accepted");
  std::cout << "bridge endpoint geometry/contact consistency passed\n";
}
