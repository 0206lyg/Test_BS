#include "slurry_implicit_math.h"
#include "slurry_pair_hydro_math.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace si = SlurryImplicit;

namespace {
void close(double value, double expected, double relative = 2.0e-9,
           double absolute = 1.0e-16)
{
  if (!std::isfinite(value) ||
      std::abs(value-expected) > absolute + relative*std::abs(expected))
    throw std::runtime_error("Implicit analytical comparison failed: got " +
                             std::to_string(value) + ", expected " + std::to_string(expected));
}

si::Particle particle(double drag)
{
  si::Particle p;
  for (int k = 0; k < 6; ++k) p.resistance[7*k] = drag;
  return p;
}

void isolated_drag()
{
  auto p = particle(4.0e-8);
  p.target = {0.012,-0.04,0.003,0.2,0.05,-0.01};
  // A symmetric anisotropic block, including off-diagonal entries.
  p.resistance[0] = 8.0e-8;
  p.resistance[1] = p.resistance[6] = 1.0e-8;
  p.resistance[7] = 5.0e-8;
  const si::Vec6 expected{0.016,-0.02,0.001,-0.1,0.09,0.03};
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 6; ++c)
      p.external[r] += p.resistance[6*r+c]*(expected[c]-p.target[c]);
  for (double dt : {1.0e-12,1.0e-6,1.0e-3}) {
    const auto result = si::solve({p},{},dt,3.0e-7);
    for (int k = 0; k < 6; ++k) close(result.velocity[0][k],expected[k]);
    if (result.relative_residual > 1.0e-10)
      throw std::runtime_error("Single-particle residual did not converge");
  }
}

void pair_drag_and_images()
{
  constexpr double zeta = 4.0e-8, zn = 3.0e-6, speed = 1.0e-3;
  auto a = particle(zeta), b = particle(zeta);
  a.target[0] = speed;
  b.target[0] = -speed;
  si::Pair pair;
  pair.i = 0;
  pair.j = 1;
  pair.normal_resistance = zn;
  pair.tangent_resistance = 1.0e-7;
  pair.gap = 3.0e-9;
  pair.lever_i = {3.0e-7,0,0};
  pair.lever_j = {-3.0e-7,0,0};
  auto result = si::solve({a,b},{pair},1.0e-6,3.0e-7);
  close(result.velocity[0][0],zeta*speed/(zeta+2*zn));
  close(result.velocity[1][0],-result.velocity[0][0]);
  a.target = b.target = {};
  pair.image_velocity = {speed,0,0};
  result = si::solve({a,b},{pair},1.0e-6,3.0e-7);
  close(result.velocity[0][0],zn*speed/(zeta+2*zn));
  close(result.velocity[1][0],-result.velocity[0][0]);
  const auto slip = si::pair_velocity(pair,result.velocity,3.0e-7);
  close(slip[0],zeta*speed/(zeta+2*zn));
}

void contact_at_microsecond()
{
  constexpr double zeta = 4.0e-8, stiffness = 1, dt = 1.0e-6, overlap = 2.0e-9;
  auto a = particle(zeta), b = particle(zeta);
  si::Pair pair;
  pair.i = 0;
  pair.j = 1;
  pair.stiffness = stiffness;
  pair.gap = -overlap;
  auto result = si::solve({a,b},{pair},dt,3.0e-7);
  const double expected_i = -stiffness*overlap/(zeta+2*stiffness*dt);
  close(result.velocity[0][0],expected_i);
  close(result.velocity[1][0],-expected_i);
  close(result.predicted_gap[0],-overlap*zeta/(zeta+2*stiffness*dt),2.0e-9,1.0e-22);
  close(result.contact_load[0],-stiffness*result.predicted_gap[0],2.0e-9,1.0e-22);
  // Strong, finite lubrication changes relaxation, and never requires a ps step.
  pair.normal_resistance = 3.0e-6;
  result = si::solve({a,b},{pair},dt,3.0e-7);
  close(result.velocity[0][0],-stiffness*overlap/
        (zeta+2*pair.normal_resistance+2*stiffness*dt));
  // Contact activation within the step from a non-overlapping configuration.
  pair.normal_resistance = 0;
  pair.gap = 1.0e-9;
  a.target[0] = 1.0e-3;
  b.target[0] = -1.0e-3;
  result = si::solve({a,b},{pair},dt,3.0e-7);
  close(result.velocity[0][0],(zeta*a.target[0]+stiffness*pair.gap)/
        (zeta+2*stiffness*dt));
  if (result.contacts != 1 || !(result.predicted_gap[0] < 0))
    throw std::runtime_error("Closing pair did not activate implicit contact");
  // A sufficiently separated pair has no artificial contact drag.
  pair.gap = 1.0e-7;
  result = si::solve({a,b},{pair},dt,3.0e-7);
  close(result.velocity[0][0],a.target[0]);
  if (result.contacts != 0) throw std::runtime_error("False contact activation");
  // An initially overlapping pair separating faster than necessary releases.
  pair.gap = -1.0e-9;
  a.target[0] = -1.0e-2;
  b.target[0] = 1.0e-2;
  result = si::solve({a,b},{pair},dt,3.0e-7);
  close(result.velocity[0][0],a.target[0]);
  if (result.contacts != 0) throw std::runtime_error("Opening pair did not release contact");
  // Multiple 1 us steps relax an overlap monotonically without force blow-up.
  pair.gap = -overlap;
  pair.normal_resistance = 3.0e-6;
  a.target = b.target = {};
  for (int step = 0; step < 100; ++step) {
    result = si::solve({a,b},{pair},dt,3.0e-7);
    if (result.predicted_gap[0] < pair.gap-1.0e-22 || result.predicted_gap[0] > 1.0e-22)
      throw std::runtime_error("Implicit contact relaxation is not monotone");
    pair.gap = result.predicted_gap[0];
  }
  if (std::abs(pair.gap) > 1.0e-18)
    throw std::runtime_error("Implicit overlap did not relax");
}

// Independent explicit B construction for a dense algebra reference.
std::vector<double> pair_B(const si::Pair &p, std::size_t n, double length)
{
  const int ndof = static_cast<int>(6*n);
  std::vector<double> B(3*ndof,0);
  for (int side = 0; side < 2; ++side) {
    const auto id = side == 0 ? p.i : p.j;
    const auto r = side == 0 ? p.lever_i : p.lever_j;
    const double sign = side == 0 ? -1 : 1;
    for (int k = 0; k < 3; ++k) B[k*ndof+6*id+k] = sign;
    // omega cross r = -[r]_x omega.
    const double c[9] = {0,r[2],-r[1],-r[2],0,r[0],r[1],-r[0],0};
    for (int k = 0; k < 3; ++k)
      for (int l = 0; l < 3; ++l) B[k*ndof+6*id+l+3] = sign*c[3*k+l]/length;
  }
  return B;
}

void coupled_manufactured_solution()
{
  constexpr int n = 7, ndof = 6*n;
  constexpr double length = 3.0e-7, dt = 1.0e-6;
  std::vector<si::Particle> particles;
  std::vector<si::Vec6> desired(n);
  std::vector<si::Pair> pairs;
  for (int i = 0; i < n; ++i) {
    auto p = particle(4.0e-8);
    for (int k = 0; k < 6; ++k) {
      p.resistance[7*k] *= 1+0.2*k;
      p.target[k] = 4.0e-5*std::cos(0.83*i+0.7*k);
      desired[i][k] = 7.0e-5*std::sin(0.57*i+0.9*k);
    }
    p.resistance[1] = p.resistance[6] = 1.0e-8;
    p.resistance[22] = p.resistance[27] = 1.0e-8;
    for (int r = 0; r < 6; ++r)
      for (int c = 0; c < 6; ++c)
        p.external[r] += p.resistance[6*r+c]*(desired[i][c]-p.target[c]);
    particles.push_back(p);
  }
  for (int i = 0; i < n-1; ++i) {
    si::Pair p;
    p.i = i;
    p.j = i+1;
    p.normal = {std::cos(0.5*i),std::sin(0.5*i),0};
    p.lever_i = {2.0e-7,1.0e-7,0.7e-7};
    p.lever_j = {-1.0e-7,0.5e-7,-0.9e-7};
    p.image_velocity = {1.0e-5*i,-2.0e-5,0.3e-5};
    p.normal_resistance = (i+1)*1.0e-6;
    p.tangent_resistance = 3.0e-7;
    p.stiffness = 1;
    p.gap = i%2 == 0 ? -1.0e-9 : 2.0e-9;
    pairs.push_back(p);
  }
  double dissipation = 0;
  for (const auto &p : pairs) {
    const auto B = pair_B(p,n,length);
    si::Vec3 relative = p.image_velocity;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < ndof; ++c) relative[r] += B[r*ndof+c]*desired[c/6][c%6];
    double normal = 0;
    for (int k = 0; k < 3; ++k) normal += p.normal[k]*relative[k];
    const double next_gap = p.gap+dt*normal;
    si::Vec3 force{};
    for (int k = 0; k < 3; ++k) {
      force[k] = p.tangent_resistance*relative[k] +
          (p.normal_resistance-p.tangent_resistance)*normal*p.normal[k];
      dissipation += force[k]*relative[k];
      force[k] += p.stiffness*std::min(next_gap,0.0)*p.normal[k];
    }
    for (int c = 0; c < ndof; ++c)
      for (int k = 0; k < 3; ++k) particles[c/6].external[c%6] += B[k*ndof+c]*force[k];
  }
  if (!(dissipation > 0)) throw std::runtime_error("Lubrication did not dissipate energy");
  const auto result = si::solve(particles,pairs,dt,length);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 6; ++k) close(result.velocity[i][k],desired[i][k],3.0e-8,1.0e-13);
  if (result.contacts != 3 || result.relative_residual > 1.0e-10)
    throw std::runtime_error("Coupled implicit contact equilibrium failed");
  si::Options restricted;
  restricted.maximum_cg_iterations = 1;
  restricted.preconditioner_backend = si::PreconditionerBackend::Nested;
  bool rejected = false;
  try { si::solve(particles,pairs,dt,length,restricted); }
  catch (const std::runtime_error &) { rejected = true; }
  if (!rejected) throw std::runtime_error("Unconverged solve was silently accepted");
}

void angular_balance()
{
  constexpr double length = 3.0e-7;
  auto a = particle(4.0e-8), b = particle(4.0e-8);
  a.target = {0,1.0e-4,0,0,0,3.0e-5};
  b.target = {0,-1.0e-4,0,0,0,2.0e-5};
  si::Pair p;
  p.i = 0;
  p.j = 1;
  p.lever_i = {length,0,0};
  p.lever_j = {-length,0,0};
  p.gap = 3.0e-9;
  p.normal_resistance = 5.0e-6;
  p.tangent_resistance = 3.0e-7;
  const auto result = si::solve({a,b},{p},1.0e-6,length);
  const auto relative = si::pair_velocity(p,result.velocity,length);
  const si::Vec3 force_i{p.normal_resistance*relative[0],
                         p.tangent_resistance*relative[1],
                         p.tangent_resistance*relative[2]};
  const si::Vec3 ti = si::cross3(p.lever_i,force_i);
  si::Vec3 negative_force{};
  for (int k = 0; k < 3; ++k) negative_force[k] = -force_i[k];
  const si::Vec3 tj = si::cross3(p.lever_j,negative_force);
  const si::Vec3 xi{-length,0,0}, xj{length,0,0};
  const auto oi = si::cross3(xi,force_i), oj = si::cross3(xj,negative_force);
  for (int k = 0; k < 3; ++k) {
    close(ti[k]+tj[k]+oi[k]+oj[k],0,0,1.0e-28);
    close(4.0e-8*(a.target[k]-result.velocity[0][k])+force_i[k],0,0,1.0e-22);
    close(4.0e-8*(a.target[k+3]-result.velocity[0][k+3])+ti[k]/length,0,0,1.0e-22);
  }
}

void roughness_onset_and_coupled_friction()
{
  constexpr double drag=4.0e-8, dt=1.0e-6, kn=1, kt=80, mu=0.1;
  auto a=particle(drag), b=particle(drag);
  si::Pair p;
  p.i=0; p.j=1;
  p.contact_onset=3.0e-9;
  p.gap=2.0e-9; // positive geometric gap, already touching the roughness shell
  p.stiffness=kn;
  auto normal_only=si::solve({a,b},{p},dt,3.0e-7);
  const double compression=p.contact_onset-p.gap;
  const double un=-kn*compression/(drag+2*dt*kn);
  const double load=kn*compression*drag/(drag+2*dt*kn);
  close(normal_only.velocity[0][0],un);
  close(normal_only.contact_load[0],load,1.0e-9,1.0e-24);
  if (!(normal_only.predicted_gap[0]>0) || normal_only.contacts!=1)
    throw std::runtime_error("Rough contact onset did not act at a positive geometric gap");
  p.friction_coefficient=mu;
  p.tangent_stiffness=kt;
  // Sticking with a stiff tangential spring at a 1 us step.
  a.target[1]=1.0e-5; b.target[1]=-a.target[1];
  auto result=si::solve({a,b},{p},dt,3.0e-7);
  const double ut=drag*a.target[1]/(drag+2*dt*kt);
  close(result.velocity[0][0],un);
  close(result.velocity[0][1],ut,1.0e-8,1.0e-16);
  close(result.tangent_force[0][1],-2*dt*kt*ut,1.0e-8,1.0e-24);
  close(result.contact_load[0],load,1.0e-9,1.0e-24);
  if (result.sliding[0] || result.gmres_iterations==0)
    throw std::runtime_error("Implicit sticking branch failed");
  // Sliding: normal force and Coulomb cap must converge together, not lagged.
  a.target[1]=3.0e-4; b.target[1]=-a.target[1];
  result=si::solve({a,b},{p},dt,3.0e-7);
  close(result.velocity[0][1],a.target[1]-mu*load/drag,1.0e-8,1.0e-16);
  close(result.tangent_force[0][1],-mu*load,1.0e-8,1.0e-24);
  close(result.tangent_history[0][1],-mu*load/kt,1.0e-8,1.0e-26);
  if (!result.sliding[0] || !(result.sliding_dissipation[0]>0))
    throw std::runtime_error("Implicit Coulomb sliding branch failed");
  if (result.relative_residual>1.0e-10)
    throw std::runtime_error("Coupled friction force balance did not converge");
  // Core evaluation and repeated setup-like solves do not commit spring history.
  const auto repeated=si::solve({a,b},{p},dt,3.0e-7);
  for (int k=0;k<3;++k) {
    close(p.tangent_history[k],0,0,0);
    close(repeated.tangent_history[0][k],result.tangent_history[0][k],1.0e-9,1.0e-27);
  }
}

void friction_return_map_energy_and_reversal()
{
  constexpr double dt=1.0e-6;
  si::Pair p;
  p.contact_onset=3.0e-9; p.gap=2.0e-9; p.stiffness=1;
  p.friction_coefficient=0.1; p.tangent_stiffness=80;
  const double limit=1.0e-10;
  auto response=si::contact_response(p,{0,5.0e-7,0},dt);
  close(response.tangent_force[1],4.0e-11,1.0e-12,1.0e-25);
  if (response.sliding) throw std::runtime_error("Small tangential displacement should stick");
  response=si::contact_response(p,{0,5.0e-6,0},dt);
  close(response.tangent_force[1],limit,1.0e-12,1.0e-25);
  if (!response.sliding) throw std::runtime_error("Return map did not cap Coulomb force");
  p.tangent_history=response.tangent_history;
  // Reverse while the existing spring still pulls in its old direction.
  response=si::contact_response(p,{0,-5.0e-7,0},dt);
  close(response.tangent_force[1],6.0e-11,1.0e-12,1.0e-25);
  if (response.sliding) throw std::runtime_error("Elastic unloading was misclassified as sliding");
  // Further reversal crosses zero and slides in the opposite direction.
  const si::Vec3 relative{0,-5.0e-6,0};
  response=si::contact_response(p,relative,dt);
  close(response.tangent_force[1],-limit,1.0e-12,1.0e-25);
  if (!response.sliding) throw std::runtime_error("Reverse Coulomb slip failed");
  const double old_energy=0.5*p.tangent_stiffness*si::dot3(p.tangent_history,p.tangent_history);
  si::Vec3 elastic_increment{};
  for (int k=0;k<3;++k) elastic_increment[k]=response.tangent_history[k]-p.tangent_history[k];
  const double numerical_dissipation=0.5*p.tangent_stiffness*
                                    si::dot3(elastic_increment,elastic_increment);
  const double work=dt*si::dot3(response.tangent_force,relative);
  close(work,response.tangential_elastic_energy-old_energy+numerical_dissipation+
             response.sliding_dissipation,1.0e-11,1.0e-32);
  // Separation drops both force and spring; recontact starts without an old preload.
  p.gap=4.0e-9;
  response=si::contact_response(p,{0,5.0e-6,0},dt);
  for (int k=0;k<3;++k) {
    close(response.tangent_force[k],0,0,0);
    close(response.tangent_history[k],0,0,0);
  }
  if (response.active || response.sliding) throw std::runtime_error("Separated contact retained friction");
  p.tangent_history=response.tangent_history; p.gap=2.0e-9;
  response=si::contact_response(p,{0,0,0},dt);
  close(response.tangential_elastic_energy,0,0,0);
}

si::Vec3 rotate_test(const si::Vec3 &v)
{
  // Rotation by 0.73 about (1,2,3), used for frame covariance including twist.
  const double length=std::sqrt(14.0),c=std::cos(0.73),s=std::sin(0.73);
  const si::Vec3 axis{1/length,2/length,3/length};
  const auto cross=si::cross3(axis,v);
  const double along=si::dot3(axis,v);
  si::Vec3 result{};
  for (int k=0;k<3;++k) result[k]=c*v[k]+s*cross[k]+(1-c)*along*axis[k];
  return result;
}

void friction_covariance_and_exact_jacobian()
{
  constexpr double dt=1.0e-6;
  si::Pair p;
  p.contact_onset=3.0e-9; p.gap=2.0e-9; p.stiffness=1;
  p.friction_coefficient=0.1; p.tangent_stiffness=80;
  p.tangent_history={0,2.0e-13,-1.0e-13};
  const si::Vec3 relative{1.0e-4,3.0e-4,-2.0e-4};
  const auto response=si::contact_response(p,relative,dt);
  auto rotated=p;
  rotated.normal=rotate_test(p.normal);
  rotated.tangent_history=rotate_test(p.tangent_history);
  const auto covariant=si::contact_response(rotated,rotate_test(relative),dt);
  const auto force=rotate_test(response.tangent_force),history=rotate_test(response.tangent_history);
  for (int k=0;k<3;++k) {
    close(covariant.tangent_force[k],force[k],1.0e-11,1.0e-24);
    close(covariant.tangent_history[k],history[k],1.0e-11,1.0e-26);
  }
  close(covariant.sliding_dissipation,response.sliding_dissipation,1.0e-11,1.0e-30);
  // Independent finite-difference check includes changing normal load in sliding.
  const si::Vec3 increment{3.0e-4,-2.0e-4,1.0e-4};
  for (bool sliding : {false,true}) {
    p.tangent_history={};
    const si::Vec3 r=sliding?relative:si::Vec3{1.0e-4,1.0e-7,-2.0e-7};
    const auto state=si::contact_response(p,r,dt);
    if (state.sliding!=sliding) throw std::runtime_error("Bad friction Jacobian test branch");
    auto derivative=si::detail::tangent_derivative(p,state,increment,dt,true);
    for (int k=0;k<3;++k) derivative[k]+=dt*p.stiffness*si::dot3(p.normal,increment)*p.normal[k];
    constexpr double epsilon=1.0e-5;
    si::Vec3 plus=r,minus=r;
    for (int k=0;k<3;++k) {
      plus[k]+=epsilon*increment[k]; minus[k]-=epsilon*increment[k];
    }
    const auto cp=si::contact_response(p,plus,dt),cm=si::contact_response(p,minus,dt);
    for (int k=0;k<3;++k) {
      const double fd=(cp.tangent_force[k]-cm.tangent_force[k]-
                      (cp.normal_load-cm.normal_load)*p.normal[k])/(2*epsilon);
      close(derivative[k],fd,2.0e-7,1.0e-20);
    }
  }
}

void coupled_friction_network()
{
  constexpr int n=5, ndof=6*n;
  constexpr double length=3.0e-7,dt=1.0e-6;
  std::vector<si::Particle> particles;
  std::vector<si::Vec6> desired(n);
  std::vector<si::Pair> pairs;
  for (int i=0;i<n;++i) {
    auto p=particle(4.0e-8);
    for (int k=0;k<6;++k) {
      p.resistance[7*k]*=1+0.2*k;
      p.target[k]=4.0e-5*std::cos(0.83*i+0.7*k);
      desired[i][k]=7.0e-5*std::sin(0.57*i+0.9*k);
      p.external[k]=p.resistance[7*k]*(desired[i][k]-p.target[k]);
    }
    particles.push_back(p);
  }
  int expected_sticks=0,expected_slides=0;
  for (int i=0;i<n-1;++i) {
    si::Pair p;
    p.i=i; p.j=i+1;
    p.normal={std::cos(0.5*i),std::sin(0.5*i),0};
    p.lever_i={2.0e-7,1.0e-7,0.7e-7};
    p.lever_j={-1.0e-7,0.5e-7,-0.9e-7};
    p.image_velocity={1.0e-5*i,-2.0e-5,0.3e-5};
    p.normal_resistance=(i+1)*1.0e-6;
    p.tangent_resistance=3.0e-7;
    p.contact_onset=3.0e-9; p.gap=2.0e-9;
    p.stiffness=1; p.friction_coefficient=0.1; p.tangent_stiffness=80;
    const auto B=pair_B(p,n,length);
    si::Vec3 relative=p.image_velocity;
    for (int r=0;r<3;++r)
      for (int c=0;c<ndof;++c) relative[r]+=B[r*ndof+c]*desired[c/6][c%6];
    const double vn=si::dot3(p.normal,relative);
    si::Vec3 tangent{};
    for (int k=0;k<3;++k) tangent[k]=relative[k]-vn*p.normal[k];
    const double normal_load=p.stiffness*(p.contact_onset-p.gap-dt*vn);
    if (i%2) {
      // A spring preload that makes the manufactured endpoint stick.
      const double vt=std::sqrt(si::dot3(tangent,tangent));
      for (int k=0;k<3;++k)
        p.tangent_history[k]=-dt*tangent[k]+0.2*p.friction_coefficient*normal_load/
                              p.tangent_stiffness*tangent[k]/vt;
    }
    si::Vec3 trial{},contact{};
    for (int k=0;k<3;++k) trial[k]=p.tangent_history[k]+dt*tangent[k];
    const double trialnorm=std::sqrt(si::dot3(trial,trial));
    const double cap=p.friction_coefficient*normal_load;
    const bool sliding=p.tangent_stiffness*trialnorm>cap;
    if(sliding)++expected_slides;else ++expected_sticks;
    for (int k=0;k<3;++k) {
      const double ft=std::min(p.tangent_stiffness,cap/trialnorm)*trial[k];
      contact[k]=p.tangent_resistance*relative[k]+
                  (p.normal_resistance-p.tangent_resistance)*vn*p.normal[k]+
                  ft-normal_load*p.normal[k];
    }
    for (int c=0;c<ndof;++c)
      for (int k=0;k<3;++k) particles[c/6].external[c%6]+=B[k*ndof+c]*contact[k];
    pairs.push_back(p);
  }
  const auto result=si::solve(particles,pairs,dt,length);
  for (int i=0;i<n;++i)
    for (int k=0;k<6;++k) close(result.velocity[i][k],desired[i][k],3.0e-7,1.0e-12);
  int slides=0;
  for (const auto value:result.sliding) slides+=value;
  if (slides!=expected_slides || result.contacts-slides!=expected_sticks)
    throw std::runtime_error("Coupled friction network chose the wrong stick/slip set");
  si::Options restricted;
  restricted.maximum_gmres_iterations=1;
  bool failed=false;
  try { si::solve(particles,pairs,dt,length,restricted); }
  catch(const std::runtime_error &) { failed=true; }
  if (!failed) throw std::runtime_error("Unconverged friction GMRES was silently accepted");
}
void directional_coulomb_kink()
{
  si::Pair p;
  p.i=0; p.j=1; p.stiffness=1; p.tangent_stiffness=80;
  p.friction_coefficient=0.1; p.gap=-1e-9;
  p.tangent_history={0,0.1e-9/80,0};
  const auto physical=si::contact_response(p,{0,0,0},1e-6);
  std::vector<si::ContactResponse> state{physical}, linear=state;
  si::detail::Vector direction(2);
  linear[0].sliding=true;
  direction[1][1]=-1;
  if (si::detail::reselect_friction_kinks({p},state,direction,1,linear)!=1 || linear[0].sliding)
    throw std::runtime_error("Coulomb kink derivative did not select inward stick branch");
  direction[1][1]=1;
  if (si::detail::reselect_friction_kinks({p},state,direction,1,linear)!=1 || !linear[0].sliding)
    throw std::runtime_error("Coulomb kink derivative did not select outward sliding branch");
  // Normal load dependence can reverse the branch selected from tangential motion alone.
  direction[1][0]=-1000;
  if (si::detail::reselect_friction_kinks({p},state,direction,1,linear)!=1 || linear[0].sliding)
    throw std::runtime_error("Coulomb kink derivative omitted changing normal load");
  if (state[0].tangent_force!=physical.tangent_force || p.tangent_history[1]!=0.1e-9/80)
    throw std::runtime_error("Derivative safeguard modified physical force or history");
}

// An independent modal reference exercises every translation/rotation block,
// including couplings absent from a common-point normal/tangent drag. Affine
// mode offsets represent imposed flow/strain and periodic-image contributions.
struct GeneralizedFixture {
  si::Pair pair;
  std::array<si::Vec12,5> modes{};
  std::array<double,5> weights{}, offsets{};
};

GeneralizedFixture generalized_fixture(int index, int i, int j)
{
  GeneralizedFixture f;
  auto &p = f.pair;
  p.i=i; p.j=j;
  p.generalized_hydro=true;
  p.normal={std::cos(0.43*index),std::sin(0.43*index),0};
  p.lever_i={2e-7,0.7e-7,-0.4e-7};
  p.lever_j={-2.4e-7,0.3e-7,0.8e-7};
  p.image_velocity={2e-6*index,-0.7e-6,0.2e-6};
  for (int mode=0;mode<5;++mode) {
    f.weights[mode]=(mode==0 ? 1e-4 : 2e-6*(mode+1))*(1+0.1*index);
    f.offsets[mode]=3e-5*std::sin(0.9*index+0.8*mode);
    for (int k=0;k<3;++k) {
      f.modes[mode][k]=std::cos(0.73*mode+0.53*k+0.3*index);
      f.modes[mode][k+6]=-f.modes[mode][k];
      f.modes[mode][k+3]=std::sin(0.41*mode+0.67*k+0.2*index);
      f.modes[mode][k+9]=std::cos(0.83*mode+0.29*k+0.4*index);
    }
    for (int r=0;r<12;++r) {
      p.hydro_affine[r]+=f.weights[mode]*f.modes[mode][r]*f.offsets[mode];
      for (int c=0;c<12;++c)
        p.hydro_resistance[12*r+c]+=f.weights[mode]*f.modes[mode][r]*f.modes[mode][c];
    }
    p.hydro_energy_constant+=0.5*f.weights[mode]*f.offsets[mode]*f.offsets[mode];
  }
  return f;
}

void generalized_hydro_manufactured(bool friction)
{
  constexpr int n=5,ndof=6*n;
  constexpr double dt=1e-6,length=3e-7;
  std::vector<si::Particle> particles;
  std::vector<si::Vec6> desired(n);
  std::vector<GeneralizedFixture> fixtures;
  std::vector<si::Pair> pairs;
  for (int i=0;i<n;++i) {
    auto p=particle(4e-8);
    for(int k=0;k<6;++k) {
      desired[i][k]=2e-5*std::sin(0.71*i+0.6*k);
      p.target[k]=3e-5*std::cos(0.37*i-0.8*k);
      p.resistance[7*k]*=1+0.2*k;
      p.external[k]=p.resistance[7*k]*(desired[i][k]-p.target[k]);
    }
    particles.push_back(p);
  }
  for (int index=0;index<n;++index) {
    auto f=generalized_fixture(index,index,(index+1)%n);
    auto &p=f.pair;
    p.stiffness=1;
    p.contact_onset=3e-9;
    p.gap=index==n-1 ? 8e-9 : 2.5e-9;
    p.friction_coefficient=friction ? 0.1 : 0;
    p.tangent_stiffness=friction ? 80 : 0;
    const auto B=pair_B(p,n,length);
    si::Vec3 relative=p.image_velocity;
    for(int r=0;r<3;++r)
      for(int c=0;c<ndof;++c) relative[r]+=B[r*ndof+c]*desired[c/6][c%6];
    const double vn=si::dot3(p.normal,relative);
    const double load=p.stiffness*std::max(p.contact_onset-p.gap-dt*vn,0.0);
    si::Vec3 trial{},force{};
    for(int k=0;k<3;++k) {
      // Alternate a cancelling preload (stick) and an unloaded spring (slip).
      p.tangent_history[k]=index%2 ? -dt*(relative[k]-vn*p.normal[k]) : 0;
      trial[k]=p.tangent_history[k]+dt*(relative[k]-vn*p.normal[k]);
    }
    const double trial_length=std::sqrt(si::dot3(trial,trial));
    const double tangential_coefficient=trial_length==0 ? 0 :
        std::min(p.tangent_stiffness,p.friction_coefficient*load/trial_length);
    for(int k=0;k<3;++k) force[k]=-load*p.normal[k]+tangential_coefficient*trial[k];
    for(int c=0;c<ndof;++c)
      for(int k=0;k<3;++k) particles[c/6].external[c%6]+=B[k*ndof+c]*force[k];
    // Assemble the manufactured external load from modes directly, rather than
    // using the dense matrix or the implementation's pair_hydro_gradient().
    si::Vec12 expected_gradient{};
    for(int mode=0;mode<5;++mode) {
      double speed=f.offsets[mode];
      for(int c=0;c<12;++c) speed+=f.modes[mode][c]*desired[c<6 ? p.i : p.j][c%6];
      for(int r=0;r<12;++r) {
        const double value=f.weights[mode]*f.modes[mode][r]*speed;
        particles[r<6 ? p.i : p.j].external[r%6]+=value;
        expected_gradient[r]+=value;
      }
    }
    const auto actual=si::pair_hydro_gradient(p,desired);
    for(int k=0;k<12;++k) close(actual[k],expected_gradient[k],2e-12,1e-23);
    pairs.push_back(p);
    fixtures.push_back(f);
  }
  const auto result=si::solve(particles,pairs,dt,length);
  for(int i=0;i<n;++i)
    for(int k=0;k<6;++k) close(result.velocity[i][k],desired[i][k],1e-6,2e-11);
  if(result.contacts!=n-1 || result.relative_residual>1e-10)
    throw std::runtime_error("Generalized hydro/contact equilibrium failed");

  std::vector<si::ContactResponse> state;
  si::detail::Vector gradient;
  si::detail::friction_residual(particles,pairs,dt,length,desired,gradient,state);
  const auto blocks=si::detail::friction_blocks(particles,pairs,state,dt,length);
  const auto diagonal=si::detail::friction_diagonal(particles,pairs,state,dt,length);
  for(int i=0;i<n;++i)
    for(int c=0;c<6;++c) {
      si::detail::Vector basis(n),column;
      basis[i][c]=1;
      si::detail::friction_jacobian(particles,pairs,state,dt,length,basis,column,false);
      for(int r=0;r<6;++r) close(blocks[i][6*r+c],column[i][r],2e-12,1e-19);
      close(diagonal[i][c],column[i][c],2e-12,1e-19);
    }
  if(!friction) {
    // Energy must contain the affine term with the same sign as the residual.
    long double expected=0;
    for(int i=0;i<n;++i)
      for(int k=0;k<6;++k) {
        const double delta=desired[i][k]-particles[i].target[k];
        expected+=0.5L*particles[i].resistance[7*k]*delta*delta-
                  static_cast<long double>(particles[i].external[k])*desired[i][k];
      }
    for(const auto &f:fixtures) {
      for(int mode=0;mode<5;++mode) {
        long double speed=f.offsets[mode];
        for(int k=0;k<12;++k) speed+=f.modes[mode][k]*desired[k<6 ? f.pair.i : f.pair.j][k%6];
        expected+=0.5L*f.weights[mode]*speed*speed;
      }
      const auto relative=si::pair_velocity(f.pair,desired,length);
      const double overlap=std::min(f.pair.gap-f.pair.contact_onset+dt*si::dot3(f.pair.normal,relative),0.0);
      expected+=0.5L*f.pair.stiffness/dt*overlap*overlap;
    }
    close(si::detail::energy(particles,pairs,dt,length,desired),expected,2e-12,1e-24);
  }
}

void generalized_hydro_validation()
{
  auto f=generalized_fixture(1,0,1);
  const std::vector<si::Particle> particles{particle(4e-8),particle(4e-8)};
  // Rigid-body null modes are legal: this is a rank-five 12x12 matrix.
  si::detail::validate(particles,{f.pair},1e-6,3e-7,si::Options{});
  for(int failure=0;failure<3;++failure) {
    auto p=f.pair;
    if(failure==0) p.hydro_resistance[1]+=1e-5; // nonsymmetric
    if(failure==1) p.hydro_resistance[0]=-1e-5; // indefinite
    if(failure==2) p.hydro_affine[5]=std::numeric_limits<double>::infinity();
    bool rejected=false;
    try { si::solve(particles,{p},1e-6,3e-7); }
    catch(const std::invalid_argument &) { rejected=true; }
    if(!rejected) throw std::runtime_error("Invalid generalized pair hydro input was accepted");
  }
}

void mari_pair_coupled_balance_and_rate_scaling()
{
  namespace ph=SlurryPairHydro;
  constexpr double ai=3e-7,aj=4.2e-7,eta=0.01,length=ai,gap=1.5e-9;
  const double pi=std::acos(-1.0);
  const si::Vec3 normal{std::sqrt(0.5),std::sqrt(0.5),0};
  const auto hydro=ph::make_sphere_pair(ai,aj,normal,gap,eta,3e-10,1.5e-7);
  std::vector<si::Vec6> reference;
  double reference_stress=0,reference_power=0;
  for(double rate:{1.0,100.0}) {
    std::vector<si::Particle> particles(2);
    ph::Vec12 background{};
    const double separation=ai+aj+gap;
    for(int i=0;i<2;++i) {
      const double a=i==0 ? ai : aj;
      const double y=(i==0 ? -0.5 : 0.5)*separation*normal[1];
      for(int k=0;k<3;++k) {
        particles[i].resistance[7*k]=6*pi*eta*a;
        particles[i].resistance[7*(k+3)]=8*pi*eta*a*a*a/(length*length);
      }
      particles[i].target={rate*y,0,0,0,0,-0.5*length*rate};
      for(int k=0;k<6;++k) background[6*i+k]=particles[i].target[k];
    }
    ph::Mat3 strain{};
    strain[1]=strain[3]=0.5*rate;
    const auto assembly=ph::assemble(hydro,background,{},strain,length);
    si::Pair pair;
    pair.i=0;pair.j=1;pair.normal=normal;pair.gap=gap;
    pair.generalized_hydro=true;
    pair.hydro_resistance=assembly.resistance;
    pair.hydro_affine=assembly.affine;
    const auto result=si::solve(particles,{pair},1e-6,length);
    ph::Vec12 q{};
    double one_body_power=0;
    for(int i=0;i<2;++i)
      for(int k=0;k<6;++k) {
        q[6*i+k]=result.velocity[i][k];
        const double delta=result.velocity[i][k]-particles[i].target[k];
        one_body_power+=particles[i].resistance[7*k]*delta*delta;
      }
    const auto evaluation=ph::evaluate(hydro,q,background,{},strain,length);
    for(int i=0;i<2;++i)
      for(int k=0;k<6;++k) {
        const double one_body_gradient=particles[i].resistance[7*k]*
            (result.velocity[i][k]-particles[i].target[k]);
        close(one_body_gradient+evaluation.gradient[6*i+k],0,0,2e-23*rate);
      }
    const double input_power=evaluation.stress[1]*rate;
    close(one_body_power+evaluation.power,input_power,1e-10,1e-28);
    if(!(evaluation.power_pump>0) || !(input_power>0))
      throw std::runtime_error("Unequal spheres did not exercise positive pump dissipation");
    if(rate==1) {
      reference=result.velocity;
      reference_stress=evaluation.stress[1];
      reference_power=input_power;
    } else {
      for(int i=0;i<2;++i)
        for(int k=0;k<6;++k) close(result.velocity[i][k],rate*reference[i][k],2e-10,1e-18);
      close(evaluation.stress[1],rate*reference_stress,2e-10,1e-28);
      close(input_power,rate*rate*reference_power,2e-10,1e-26);
    }
  }
}
} // namespace

int main()
{
  isolated_drag();
  pair_drag_and_images();
  contact_at_microsecond();
  coupled_manufactured_solution();
  angular_balance();
  roughness_onset_and_coupled_friction();
  friction_return_map_energy_and_reversal();
  friction_covariance_and_exact_jacobian();
  coupled_friction_network();
  directional_coulomb_kink();
  generalized_hydro_manufactured(false);
  generalized_hydro_manufactured(true);
  generalized_hydro_validation();
  mari_pair_coupled_balance_and_rate_scaling();
  std::puts("PASS: implicit anisotropic drag, lubrication/image velocities, 1 us unilateral contact, "
            "coupled rotational/contact equilibrium, angular balance, rough contact onset, "
            "Coulomb stick/slip/reversal/release, energy balance, frame covariance, "
            "exact load-dependent Jacobian, coupled friction network, full pair hydro/affine modes, "
            "6x6 block preconditioning, Mari pair force/power balance and rate scaling, "
            "and failure reporting");
}
