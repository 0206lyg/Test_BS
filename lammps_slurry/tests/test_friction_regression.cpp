#include "slurry_implicit_math.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

// Captured at the rejected step of N108, phi=.60, AR2, dt=.5 us,
// hc=3 nm, kn=1 N/m, kt=80 N/m, mu=.1, shear=100/s. The input includes
// committed spring history and the iterate exactly at a stick/slip kink.
int main(int argc, char **argv)
{
  namespace si = SlurryImplicit;
  if (argc < 2 || argc > 4)
    throw std::runtime_error("Pass the dense friction fixture path and optional lu/ilu backend");
  std::ifstream in(argv[1]);
  std::size_t count=0, pair_count=0;
  double dt=0, length=0;
  in >> count >> pair_count >> dt >> length;
  if (!in || count!=108 || (pair_count!=388 && pair_count!=387))
    throw std::runtime_error("Invalid dense friction fixture header");
  std::vector<si::Particle> particles(count);
  std::vector<si::Pair> pairs(pair_count);
  std::vector<si::Vec6> initial(count);
  for (auto &p : particles) {
    for (auto &x : p.resistance) in >> x;
    for (auto &x : p.target) in >> x;
    for (auto &x : p.external) in >> x;
  }
  for (auto &p : pairs) {
    in >> p.i >> p.j;
    for (auto *v : {&p.normal,&p.lever_i,&p.lever_j,&p.image_velocity})
      for (auto &x : *v) in >> x;
    in >> p.normal_resistance >> p.tangent_resistance >> p.gap >> p.stiffness
       >> p.contact_onset >> p.friction_coefficient >> p.tangent_stiffness;
    for (auto &x : p.tangent_history) in >> x;
  }
  for (auto &q : initial) for (auto &x : q) in >> x;
  if (!in) throw std::runtime_error("Truncated dense friction fixture");
  const auto committed = pairs;
  si::Options options;
  if (argc >= 3) {
    if (std::string(argv[2]) == "lu") options.preconditioner_backend = si::PreconditionerBackend::FullLU;
    else if (std::string(argv[2]) == "ilu") options.preconditioner_backend = si::PreconditionerBackend::IncompleteLU;
    else throw std::runtime_error("Unknown regression backend");
  }
  if (argc == 4) {
    if (std::string(argv[3]) != "adaptive") throw std::runtime_error("Unknown forcing option");
    options.adaptive_forcing = true;
  }
  options.linear_relative_tolerance=1e-3;
  const auto result=si::solve(particles,pairs,dt,length,options,initial);
  if (!(result.relative_residual <= options.relative_tolerance) || result.kink_reselections==0)
    throw std::runtime_error("Dense Coulomb kink safeguard did not converge or was not exercised");
  for (std::size_t k=0;k<pairs.size();++k)
    if (pairs[k].tangent_history!=committed[k].tangent_history)
      throw std::runtime_error("Nonlinear trial modified committed contact history");
  si::detail::Vector residual;
  std::vector<si::ContactResponse> state;
  const double scale=si::detail::friction_residual(particles,pairs,dt,length,
                                                result.velocity,residual,state);
  if (si::detail::norm(residual)>options.absolute_tolerance+options.relative_tolerance*scale)
    throw std::runtime_error("Dense friction result failed independent exact residual check");
  std::cout << "PASS dense Coulomb kink: relative residual=" << result.relative_residual
            << ", Newton=" << result.newton_iterations
            << ", directional reselections=" << result.kink_reselections
            << ", LU factors=" << result.lu_factorizations
            << ", LU fallback=" << result.lu_fallbacks << '\n';
}
