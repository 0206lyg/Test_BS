#include "slurry_implicit_math.h"
#include "slurry_pair_hydro_math.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace si = SlurryImplicit;
namespace ss = SlurrySparse;

void require(bool condition, const char *message)
{ if (!condition) throw std::runtime_error(message); }

void independent_sparse_factor()
{
  constexpr int n = 37;
  std::vector<double> matrix(n*n,0), expected(n), scale(n);
  for (int i = 0; i < n; ++i) {
    matrix[n*i+i] = 2+0.1*i;
    scale[i] = std::pow(10.0,((i*7)%13-6)/2.0);
    expected[i] = std::cos(0.3*i)/scale[i];
  }
  // Independent SPD matrix with nontrivial fill and highly unequal units.
  for (int k = 0; k < 60; ++k) {
    std::vector<double> v(n,0);
    for (int j = 0; j < 5; ++j) v[(k*7+j*11)%n] = std::sin(k+0.9*j);
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < n; ++c) matrix[n*r+c] += v[r]*v[c];
  }
  std::vector<ss::Entry> entries;
  std::vector<double> rhs(n,0), actual;
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c) {
      matrix[n*r+c] *= scale[r]*scale[c];
      rhs[r] += matrix[n*r+c]*expected[c];
      if (r <= c && matrix[n*r+c] != 0) {
        // Duplicate entries and both triangle conventions must be accepted.
        entries.push_back({r,c,0.3*matrix[n*r+c]});
        entries.push_back({c,r,0.7*matrix[n*r+c]});
      }
    }
  std::vector<int> order(n);
  for (int k = 0; k < n; ++k) order[k] = k*13%n;
  ss::LDLT factor;
  factor.factor(n,entries,order,10000);
  factor.solve(rhs,actual);
  for (int i = 0; i < n; ++i)
    require(std::abs((actual[i]-expected[i])*scale[i]) < 2e-13,
            "Scaled, permuted sparse solve differs from independently constructed solution");
  bool failed = false;
  try { factor.factor(2,{{0,0,1},{1,1,1},{0,1,2}},{0,1},100); }
  catch (const std::runtime_error &) { failed = true; }
  require(failed,"Indefinite matrix accepted by sparse SPD factor");
  failed = false;
  try { factor.factor(n,entries,order,2); }
  catch (const std::runtime_error &) { failed = true; }
  require(failed,"Sparse fill budget was ignored");
}

struct Fixture {
  std::vector<si::Particle> particles;
  std::vector<si::Pair> pairs;
  std::vector<si::Vec6> desired, initial;
  double dt = 1e-6, length = 3e-7;
};

Fixture fixture(int side)
{
  Fixture f;
  const int count = side*side*side;
  f.particles.resize(count);
  f.desired.resize(count);
  for (int i = 0; i < count; ++i)
    for (int k = 0; k < 6; ++k) {
      f.particles[i].resistance[7*k] = 6*std::acos(-1.0)*0.01*1.5e-7*(1+0.13*k);
      f.particles[i].target[k] = 1e-5*std::cos(0.3*i+0.5*k);
      f.desired[i][k] = 1e-5*std::sin(0.7*i+0.9*k);
    }
  for (int z = 0; z < side; ++z)
    for (int y = 0; y < side; ++y)
      for (int x = 0; x < side; ++x)
        for (int direction = 0; direction < 3; ++direction) {
          const int i = x+side*(y+side*z);
          int coords[3] = {x,y,z};
          coords[direction] = (coords[direction]+1)%side;
          const int j = coords[0]+side*(coords[1]+side*coords[2]);
          si::Pair p;
          p.i = i; p.j = j;
          p.normal = {0,0,0}; p.normal[direction] = 1;
          for (int k = 0; k < 3; ++k) {
            p.lever_i[k] = 1.5e-7*p.normal[k];
            p.lever_j[k] = -p.lever_i[k];
          }
          p.gap = 2.9e-9;
          p.contact_onset = 3e-9;
          p.stiffness = 1;
          p.tangent_stiffness = 80;
          p.friction_coefficient = 0.1;
          const auto hydro = SlurryPairHydro::make_sphere_pair(1.5e-7,1.5e-7,p.normal,
              p.gap,0.01,0.15e-9,75e-9);
          const auto assembly = SlurryPairHydro::assemble(hydro,{}, {}, {},f.length);
          p.generalized_hydro = true;
          p.hydro_resistance = assembly.resistance;
          if ((i+direction)%3 == 0) {
            const auto relative = si::linear_pair_velocity(p,f.desired,f.length);
            const double normal = si::dot3(p.normal,relative);
            for (int k = 0; k < 3; ++k)
              p.tangent_history[k] = -0.99*f.dt*(relative[k]-normal*p.normal[k]);
          }
          f.pairs.push_back(p);
        }
  // Manufactured forcing fixes the nonlinear solution. Assembly verification
  // below separately checks every column against the matrix-free operator.
  si::detail::Vector force;
  std::vector<si::ContactResponse> state;
  si::detail::friction_residual(f.particles,f.pairs,f.dt,f.length,f.desired,force,state);
  for (int i = 0; i < count; ++i) f.particles[i].external = force[i];
  f.initial = f.desired;
  for (int i = 0; i < count; ++i)
    for (int k = 0; k < 6; ++k) f.initial[i][k] += 1e-8*std::cos(0.8*i+0.2*k);
  return f;
}

void assembly_and_fallback()
{
  const auto f = fixture(2);
  si::detail::Vector residual;
  std::vector<si::ContactResponse> state;
  si::detail::friction_residual(f.particles,f.pairs,f.dt,f.length,f.desired,residual,state);
  const auto entries = si::detail::symmetric_sparse_entries(f.particles,f.pairs,state,f.dt,f.length);
  const int n = 6*f.particles.size();
  std::vector<double> dense(n*n,0);
  for (const auto &e : entries) {
    dense[n*e.row+e.column] += e.value;
    if (e.row != e.column) dense[n*e.column+e.row] += e.value;
  }
  for (int c = 0; c < n; ++c) {
    si::detail::Vector basis(f.particles.size()), column;
    basis[c/6][c%6] = 1;
    si::detail::friction_jacobian(f.particles,f.pairs,state,f.dt,f.length,basis,column,false);
    for (int r = 0; r < n; ++r)
      require(std::abs(dense[n*r+c]-column[r/6][r%6]) < 2e-18,
              "Sparse friction proxy assembly differs from matrix-free operator");
  }
  si::Options options;
  options.preconditioner_backend = si::PreconditionerBackend::Sparse;
  options.sparse_max_factor_nonzeros = 1;
  const auto result = si::solve(f.particles,f.pairs,f.dt,f.length,options,f.initial);
  require(result.sparse_fallbacks > 0 && result.cg_iterations > 0,
          "Sparse budget fallback was not reported or did not use nested CG");
  require(result.relative_residual <= 1e-10,"Sparse fallback relaxed final force tolerance");
}

void compare_backends(int side)
{
  const auto f = fixture(side);
  si::Options nested_options, sparse_options;
  nested_options.preconditioner_backend = si::PreconditionerBackend::Nested;
  sparse_options.preconditioner_backend = si::PreconditionerBackend::Sparse;
  auto start = std::chrono::steady_clock::now();
  const auto nested = si::solve(f.particles,f.pairs,f.dt,f.length,nested_options,f.initial);
  const double nested_time = si::detail::elapsed_seconds(start);
  start = std::chrono::steady_clock::now();
  const auto sparse = si::solve(f.particles,f.pairs,f.dt,f.length,sparse_options,f.initial);
  const double sparse_time = si::detail::elapsed_seconds(start);
  require(sparse.cg_iterations == 0 && sparse.sparse_factorizations > 0 && sparse.sparse_fallbacks == 0,
          "Sparse preconditioner was not used successfully");
  require(nested.relative_residual <= 1e-10 && sparse.relative_residual <= 1e-10,
          "Backend comparison did not meet original final tolerance");
  double difference = 0;
  for (std::size_t i = 0; i < f.particles.size(); ++i)
    for (int k = 0; k < 6; ++k) {
      difference = std::max(difference,std::abs(sparse.velocity[i][k]-nested.velocity[i][k]));
      require(std::abs(sparse.velocity[i][k]-f.desired[i][k]) < 2e-12,
              "Sparse result differs from manufactured nonlinear solution");
    }
  require(difference < 2e-12,"Sparse and nested velocity solutions differ");
  for (std::size_t k = 0; k < f.pairs.size(); ++k) {
    require(sparse.sliding[k] == nested.sliding[k],"Sparse and nested stick/slip states differ");
    require(std::abs(sparse.contact_load[k]-nested.contact_load[k]) < 2e-18,
            "Sparse and nested contact loads differ");
    for (int d = 0; d < 3; ++d)
      require(std::abs(sparse.tangent_force[k][d]-nested.tangent_force[k][d]) < 2e-18,
              "Sparse and nested tangential loads differ");
  }
  std::cout << std::setprecision(12)
      << "{\"particles\":" << f.particles.size() << ",\"pairs\":" << f.pairs.size()
      << ",\"nested_seconds\":" << nested_time << ",\"sparse_seconds\":" << sparse_time
      << ",\"speedup\":" << nested_time/sparse_time
      << ",\"nested_cg_iterations\":" << nested.cg_iterations
      << ",\"nested_gmres_iterations\":" << nested.gmres_iterations
      << ",\"sparse_gmres_iterations\":" << sparse.gmres_iterations
      << ",\"sparse_factorizations\":" << sparse.sparse_factorizations
      << ",\"sparse_factor_nonzeros\":" << sparse.max_sparse_factor_nonzeros
      << ",\"sparse_setup_seconds\":" << sparse.preconditioner_setup_seconds
      << ",\"sparse_solve_seconds\":" << sparse.preconditioner_solve_seconds
      << ",\"nested_residual\":" << nested.relative_residual
      << ",\"sparse_residual\":" << sparse.relative_residual
      << ",\"maximum_velocity_difference\":" << difference << "}\n";
}

int main(int argc, char **argv)
{
  independent_sparse_factor();
  assembly_and_fallback();
  compare_backends(argc > 1 ? std::atoi(argv[1]) : 3);
  std::cout << "PASS: sparse SPD factor, equilibration/permutation, assembly, explicit fallback, "
               "and unchanged coupled friction solutions\n";
}
