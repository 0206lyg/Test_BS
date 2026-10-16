#include "slurry_implicit_math.h"
#include <iostream>
#include <stdexcept>

#ifndef SLURRY_HAVE_EIGEN
#error This test requires the Eigen backend
#endif
namespace si = SlurryImplicit;
si::PreconditionerBackend tested_backend = si::PreconditionerBackend::FullLU;
bool adaptive_forcing = false;
bool exercise_cache = false;
void require(bool ok, const char *message)
{ if (!ok) throw std::runtime_error(message); }

void independent_scaled_lu()
{
  constexpr int n = 11;
  std::vector<SlurryEigen::Entry> entries;
  std::vector<double> scale(n), expected(n), rhs(n,0), actual;
  for (int i = 0; i < n; ++i) {
    scale[i] = std::pow(10.0,(i%7)-3);
    expected[i] = std::cos(0.7*i)/scale[i];
  }
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c) {
      const double value = ((r == c ? 9.0 : 0.0)+std::sin(0.7*r+0.3*c))*scale[r]*scale[c];
      entries.push_back({r,c,0.3*value}); entries.push_back({r,c,0.7*value});
      rhs[r] += value*expected[c];
    }
  SlurryEigen::FullLU lu;
  lu.factor(n,entries); lu.solve(rhs,actual);
  for (int i = 0; i < n; ++i)
    require(std::abs((actual[i]-expected[i])*scale[i]) < 2e-13,
            "Scaled nonsymmetric LU solve differs from independent known solution");
  // Same pattern, different coefficients exercises symbolic reuse.
  for (auto &e : entries) e.value *= 1.25;
  for (double &v : rhs) v *= 1.25;
  lu.factor(n,entries); lu.solve(rhs,actual);
  for (int i = 0; i < n; ++i)
    require(std::abs((actual[i]-expected[i])*scale[i]) < 2e-13,"LU symbolic reuse changed solution");
  bool failed = false;
  try { lu.factor(2,{{0,0,0},{1,1,0}}); }
  catch (const std::runtime_error &) { failed = true; }
  require(failed,"Singular full LU factor did not fail");
  failed = false;
  try { lu.solve(rhs,actual); }
  catch (const std::exception &) { failed = true; }
  require(failed,"Failed LU factorization left a stale factor available for solve");
}

void exact_assembly_and_nonlinear_solution()
{
  constexpr int count = 5, ndof = 30;
  constexpr double dt = 1e-6, length = 3e-7;
  std::vector<si::Particle> particles(count);
  std::vector<si::Pair> pairs;
  si::detail::Vector desired(count), initial(count), direction(count);
  for (int i = 0; i < count; ++i)
    for (int k = 0; k < 6; ++k) {
      particles[i].resistance[7*k] = 3e-8*(1+0.1*k);
      particles[i].target[k] = 1e-5*std::cos(0.3*i+0.5*k);
      desired[i][k] = 1e-5*std::sin(0.7*i+0.9*k);
      initial[i][k] = desired[i][k]+1e-8*std::cos(0.2*i+0.3*k);
      direction[i][k] = 1e-5*std::cos(0.6*i+0.1*k);
    }
  for (int i = 0; i < count; ++i) {
    si::Pair p;
    p.i = (i+1)%count; p.j = i; // Reverse index ordering is intentional.
    p.normal = {1+0.2*i,0.3,-0.2};
    const double norm = std::sqrt(si::dot3(p.normal,p.normal));
    for (double &x : p.normal) x /= norm;
    p.lever_i = {1.1e-7,0.2e-7,-0.3e-7};
    p.lever_j = {-1.3e-7,0.4e-7,0.1e-7};
    p.image_velocity = {1e-6,-2e-6,3e-6};
    p.normal_resistance = 7e-7;
    p.tangent_resistance = 1e-7;
    p.gap = i == 4 ? 7e-9 : 2.9e-9;
    p.contact_onset = 3e-9; p.stiffness = 1;
    p.friction_coefficient = 0.1; p.tangent_stiffness = 80;
    p.generalized_hydro = true;
    for (int r = 0; r < 12; ++r)
      for (int c = 0; c < 12; ++c)
        p.hydro_resistance[12*r+c] = 2e-7*std::sin(0.7*r)*std::sin(0.7*c);
    if (i == 1 || i == 3) {
      const auto v = si::pair_velocity(p,desired,length);
      const double vn = si::dot3(p.normal,v);
      for (int d = 0; d < 3; ++d) p.tangent_history[d] = -dt*(v[d]-vn*p.normal[d]);
    }
    pairs.push_back(p);
  }
  si::detail::Vector force;
  std::vector<si::ContactResponse> state;
  si::detail::friction_residual(particles,pairs,dt,length,desired,force,state);
  for (int i = 0; i < count; ++i) particles[i].external = force[i];
  const auto entries = si::detail::exact_sparse_entries(particles,pairs,state,dt,length);
  std::vector<double> matrix(ndof*ndof,0);
  for (const auto &e : entries) matrix[ndof*e.row+e.column] += e.value;
  double nonsymmetry = 0;
  for (int c = 0; c < ndof; ++c) {
    si::detail::Vector basis(count), column;
    basis[c/6][c%6] = 1;
    si::detail::friction_jacobian(particles,pairs,state,dt,length,basis,column,true);
    for (int r = 0; r < ndof; ++r) {
      require(std::abs(matrix[ndof*r+c]-column[r/6][r%6]) < 3e-19,
              "Full sparse Jacobian column differs from exact matrix-free operator");
      nonsymmetry = std::max(nonsymmetry,std::abs(matrix[ndof*r+c]-matrix[ndof*c+r]));
    }
  }
  require(nonsymmetry > 1e-9,"Sliding Jacobian was incorrectly symmetrized");
  auto plus = desired, minus = desired;
  constexpr double step = 1e-5;
  for (int i = 0; i < count; ++i)
    for (int k = 0; k < 6; ++k) {
      plus[i][k] += step*direction[i][k]; minus[i][k] -= step*direction[i][k];
    }
  si::detail::Vector fp,fm,jv;
  std::vector<si::ContactResponse> temporary;
  si::detail::friction_residual(particles,pairs,dt,length,plus,fp,temporary);
  si::detail::friction_residual(particles,pairs,dt,length,minus,fm,temporary);
  si::detail::friction_jacobian(particles,pairs,state,dt,length,direction,jv,true);
  for (int i = 0; i < count; ++i)
    for (int k = 0; k < 6; ++k)
      fp[i][k] = (fp[i][k]-fm[i][k])/(2*step)-jv[i][k];
  require(si::detail::norm(fp) < 2e-8*si::detail::norm(jv),
          "Full Jacobian fails independent residual finite difference");
  si::Options lu_options, nested_options;
  lu_options.preconditioner_backend = tested_backend;
  lu_options.adaptive_forcing = adaptive_forcing;
  nested_options.preconditioner_backend = si::PreconditionerBackend::Nested;
  const auto lu = si::solve(particles,pairs,dt,length,lu_options,initial);
  const auto nested = si::solve(particles,pairs,dt,length,nested_options,initial);
  require(lu.lu_factorizations > 0 && lu.lu_solves > 0 && lu.lu_fallbacks == 0,
          "Full LU backend was not exercised successfully");
  require(lu.gmres_iterations > 0 && lu.cg_iterations == 0,"Full LU preconditioned FGMRES was not exercised");
  require(lu.relative_residual <= 1e-10 && nested.relative_residual <= 1e-10,
          "Full LU relaxed the final nonlinear convergence tolerance");
  for (int i = 0; i < count; ++i)
    for (int k = 0; k < 6; ++k)
      require(std::abs(lu.velocity[i][k]-nested.velocity[i][k]) < 2e-12,
              "LU and Nested produce different converged velocities");
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    require(lu.sliding[k] == nested.sliding[k],"LU changed the converged contact branch");
    require(std::abs(lu.contact_load[k]-nested.contact_load[k]) < 2e-18,"LU changed contact normal load");
    for (int d = 0; d < 3; ++d)
      require(std::abs(lu.tangent_history[k][d]-nested.tangent_history[k][d]) < 2e-18,
              "LU changed the committed friction history");
  }
  // Preparation runs may disable friction; explicit LU must still select the
  // requested backend and preserve the normal-contact solution.
  for (auto &p : pairs) { p.friction_coefficient = 0; p.tangent_stiffness = 0; }
  for (auto &p : particles) p.external = {};
  si::detail::friction_residual(particles,pairs,dt,length,desired,force,state);
  for (int i = 0; i < count; ++i) particles[i].external = force[i];
  const auto normal_lu = si::solve(particles,pairs,dt,length,lu_options,initial);
  const auto normal_old = si::solve(particles,pairs,dt,length,nested_options,initial);
  require(normal_lu.lu_factorizations > 0 && normal_lu.cg_iterations == 0,
          "Frictionless preparation did not use the explicit LU backend");
  for (int i = 0; i < count; ++i)
    for (int k = 0; k < 6; ++k)
      require(std::abs(normal_lu.velocity[i][k]-normal_old.velocity[i][k]) < 2e-12,
              "Frictionless LU and old solver solutions differ");
  if (exercise_cache) {
    si::SolverWorkspace workspace;
    lu_options.preconditioner_backend = si::PreconditionerBackend::IncompleteLU;
    lu_options.adaptive_forcing = false;
    const auto first = si::solve(particles,pairs,dt,length,lu_options,initial,&workspace);
    require(first.lu_factorizations > 0 && workspace.ready,"ILUT cache was not populated");
    // Reorder pair storage without changing the physical pair identities.
    // A positional cache of contact branches would be invalid here.
    std::reverse(pairs.begin(),pairs.end());
    const auto cached = si::solve(particles,pairs,dt,length,lu_options,initial,&workspace);
    require(cached.lu_factorizations == 0 && cached.lu_solves > 0,
            "ILUT factor was not reused across successive solves");
    for (int i = 0; i < count; ++i)
      for (int k = 0; k < 6; ++k)
        require(std::abs(first.velocity[i][k]-cached.velocity[i][k]) < 2e-12,
                "Reordered pair cache changed the true solution");
    // Changing numerical/physical scaling must force a fresh factor.
    const auto changed_dt = si::solve(particles,pairs,0.9*dt,length,lu_options,initial,&workspace);
    require(changed_dt.lu_factorizations > 0,"Timestep change did not invalidate ILUT cache");
    const auto old_without_cache = si::solve(particles,pairs,0.9*dt,length,lu_options,initial);
    for (int i = 0; i < count; ++i)
      for (int k = 0; k < 6; ++k)
        require(std::abs(changed_dt.velocity[i][k]-old_without_cache.velocity[i][k]) < 2e-12,
                "Cached solve after timestep change differs from uncached solve");
    lu_options.ilu_drop_tolerance *= 2;
    const auto changed_drop = si::solve(particles,pairs,0.9*dt,length,lu_options,initial,&workspace);
    require(changed_drop.lu_factorizations > 0,"ILUT configuration change did not invalidate cache");
    workspace.age = 10;
    const auto expired = si::solve(particles,pairs,0.9*dt,length,lu_options,initial,&workspace);
    require(expired.lu_factorizations > 0,"ILUT cache age limit was ignored");
    workspace.reset();
    require(!workspace.ready && !workspace.factor && workspace.branches.empty(),
            "Explicit workspace reset retained stale numerical state");
  }
}

int main(int argc, char **argv)
{
  if ((argc == 2 || argc == 3) && std::string(argv[1]) == "ilu")
    tested_backend = si::PreconditionerBackend::IncompleteLU;
  else if (argc != 1) throw std::runtime_error("Optional argument: ilu");
  if (argc == 3) {
    if (std::string(argv[2]) == "adaptive") adaptive_forcing = true;
    else if (std::string(argv[2]) == "cache") exercise_cache = true;
    else throw std::runtime_error("Optional second argument: adaptive or cache");
  }
  independent_scaled_lu();
  exact_assembly_and_nonlinear_solution();
  std::cout << "PASS: established Eigen full nonsymmetric LU, scaling, symbolic reuse, "
               "independent Jacobian differences, and unchanged converged contact solutions\n";
}
