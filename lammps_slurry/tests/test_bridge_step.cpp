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

// Construct pairs from analytic ellipsoid support points, independently of
// the gap optimizer. Small negative offsets test signed penetration depth;
// the intersecting surfaces themselves have zero unsigned distance.
void check_signed_spheroid_gap() {
  namespace gr = slurry::gr_re2::graphite;
  const double radius = 4e-6; // Fixed 8 micrometre volume-equivalent diameter.
  for (double ar : {1., 2., 2.09, 3., 5.}) {
    const double a = radius * std::cbrt(ar), c = a / ar;
    for (int orientation = 0; orientation < 12; ++orientation) {
      gr::Body first, second;
      first.axes = second.axes = {a, a, c};
      first.rotation = gr::rotationIncrement({.11*orientation, -.17, .23});
      second.rotation = gr::rotationIncrement({-.27, .07*orientation, -.31});
      const auto n = gr::normalized(gr::Vec3{std::sin(.7*orientation+.2),
                                            std::cos(.9*orientation+.3), .4});
      const auto qi = gr::rotatedDiagonal(first.rotation, {a*a,a*a,c*c});
      const auto qj = gr::rotatedDiagonal(second.rotation, {a*a,a*a,c*c});
      const auto qin = gr::mul(qi,n), qjn = gr::mul(qj,n);
      const auto pi = gr::scale(qin,1./std::sqrt(gr::dot(n,qin)));
      const auto pj = gr::scale(qjn,-1./std::sqrt(gr::dot(n,qjn)));
      for (double gap : {8e-9, 0., -1e-10, -4e-9}) {
        second.position = gr::add(gr::sub(pi,pj),gr::scale(n,gap));
        const auto actual = gr::closestEllipsoidGap(first,second);
        require(std::abs(actual.gap-gap)<5e-12*a,
                "AR 1-5 signed spheroid gap differs from exact support construction");
        require(gr::norm(gr::sub(actual.normal,n))<2e-9,
                "AR 1-5 shallow-overlap normal differs from exact support construction");
        require(gr::norm(gr::sub(gr::sub(second.position,actual.leverI),
                                gr::sub(gr::scale(actual.normal,actual.gap),actual.leverJ)))<1e-10*a,
                "signed-gap support points do not close along the returned normal");
        // The same answer must survive a change of length units.
        auto scaled_first = first, scaled_second = second;
        for (int k=0;k<3;++k) {
          scaled_first.axes[k] /= radius;
          scaled_second.axes[k] /= radius;
          scaled_second.position[k] /= radius;
        }
        const auto scaled = gr::closestEllipsoidGap(scaled_first,scaled_second);
        require(std::abs(scaled.gap*radius-actual.gap)<5e-12*a,
                "signed gap depends on metre versus reference-radius units");
      }
    }
  }
}

void check_surface_zero_contact() {
  const double aij=4e-6, kr_force=80., dt=1e-6, overlap=1e-9;
  auto c=contact(-overlap,0.,100.,8e-6-overlap);
  c.kt=80.; c.mu=.7; c.mu_roll=.4; c.roll_length=aij;
  c.kr=kr_force*aij*aij;
  c.history=Vec3(1e-9,0.,0.);
  c.rolling_history=Vec3(0.,1e-4,0.);
  auto zero=bs::Vector::Zero(12).eval();
  for (double gap : {8e-9, 1e-15, 0.}) {
    const auto response=bs::contact_response(c,zero,dt,gap);
    require(!response.active && response.normal_load==0. &&
            response.traction.isZero() && response.torque.isZero() &&
            response.tangent_history.isZero() && response.rolling_history.isZero(),
            "surface-zero law retained force, torque or spring history without overlap");
  }
  c.history.setZero(); c.rolling_history.setZero();
  const auto normal=bs::contact_response(c,zero,dt,-overlap);
  require(std::abs(normal.normal_load-100.*overlap)<1e-22,
          "surface-zero normal force is not kn times the actual overlap");
  require(normal.active && normal.traction.dot(c.normal)<0.,
          "negative gap must produce a repulsive normal penalty");

  // Singh's quasi-force spring has displacement units and torque aij*n x F.
  // The solver instead stores angular history, so its stiffness is kr*aij^2.
  bs::Vector rolling=zero;
  rolling.segment<3>(9)=Vec3(.7,-.2,.6);
  const auto elastic=bs::contact_response(c,rolling,dt,-overlap);
  const Vec3 psi=aij*dt*elastic.relative_rotation.cross(c.normal);
  const Vec3 expected=aij*c.normal.cross(kr_force*psi);
  require(!elastic.rolling && (elastic.torque-expected).norm()<1e-28,
          "angular rolling spring differs from Singh's quasi-force formulation");
  require(std::abs(elastic.rolling_energy-.5*kr_force*psi.squaredNorm())<1e-33,
          "rolling spring energy changed under angular-history normalization");

  rolling[6]=1.; rolling[9]=1e5;
  const auto yielded=bs::contact_response(c,rolling,dt,-overlap);
  require(yielded.sliding && yielded.rolling,
          "large sliding and rolling trial motions failed to reach Coulomb caps");
  require(std::abs(yielded.tangent_force.norm()-c.mu*yielded.normal_load)<1e-22 &&
          std::abs(yielded.torque.norm()-c.mu_roll*aij*yielded.normal_load)<1e-28,
          "sliding or rolling cap differs from mu_s*Fn or mu_r*aij*Fn");
  const bs::V6 force_i=-bs::generalized_contact_map(c,true).transpose()*bs::contact_wrench(yielded);
  const bs::V6 force_j=-bs::generalized_contact_map(c,false).transpose()*bs::contact_wrench(yielded);
  require((force_i.head<3>()+force_j.head<3>()).norm()<1e-25 &&
          yielded.torque.dot(yielded.relative_rotation)>0.,
          "contact signs do not give opposite forces and resist relative rolling");

  bs::Vector common=zero;
  common.segment<3>(3)=common.segment<3>(9)=Vec3(.7,-.2,.6);
  require(bs::contact_response(c,common,dt,-overlap).torque.isZero(),
          "common rigid rotation incorrectly charged the rolling spring");
  bs::Vector twist=zero; twist[11]=100.;
  require(bs::contact_response(c,twist,dt,-overlap).torque.isZero(),
          "rolling-only model incorrectly resists twisting about the normal");

}

int main() {
  check_signed_spheroid_gap();
  check_surface_zero_contact();
  bs::Options options;
  options.relative_tolerance = 1e-10;
  options.absolute_tolerance = 1e-12;
  options.linear_relative_tolerance = 1e-11;
  // Exact implicit endpoint penalty at zero surface onset, including real
  // negative gaps, for the entire supported AR range. Identity self mobility
  // makes the compressed pair an independently solvable scalar problem.
  for (double ar : {1.,2.,2.09,3.,5.}) {
    auto pair=problem();
    const double a=std::cbrt(ar), minor=a/ar, initial_gap=1e-3;
    const double separation=2.*minor+initial_gap, load=2., stiffness=100.;
    pair.external[2]=load; pair.external[8]=-load;
    pair.contacts.push_back(contact(initial_gap,0.,stiffness,separation));
    const auto shape=geometry(a,minor,separation);
    const auto state=st::solve(pair,shape,options);
    const double expected_gap=(initial_gap-2.*pair.dt*load)/(1.+2.*pair.dt*stiffness);
    require(expected_gap<0. && std::abs(state.endpoint.gaps[0]-expected_gap)<2e-12,
            "surface-zero endpoint solve rejected or changed the finite overlap");
    require(std::abs(state.solution.contacts[0].normal_load+stiffness*expected_gap)<2e-10,
            "surface-zero endpoint normal load differs from analytic implicit penalty");
    require(std::abs(state.solution.contacts[0].normal_load+
                     stiffness*state.endpoint.gaps[0])<1e-13,
            "surface-zero contact force is inconsistent with accepted endpoint geometry");
  }
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
