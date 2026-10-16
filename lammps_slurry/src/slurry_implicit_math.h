// Matrix-free overdamped resistance solve, with backward-Euler rough contact
// and a fully coupled elastic/Coulomb tangential return map.
// Geometry and resistance coefficients are held fixed during one time step.
// This removes explicit drag/contact stiffness restrictions, not the need to
// resolve configuration changes or to check timestep convergence.
#ifndef SLURRY_IMPLICIT_MATH_H
#define SLURRY_IMPLICIT_MATH_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "slurry_sparse_math.h"
#include "slurry_eigen_lu_math.h"
#include "slurry_eigen_ilu_math.h"
#include "slurry_solver_workspace.h"

namespace SlurryImplicit {
using Vec3 = std::array<double, 3>;
using Vec6 = std::array<double, 6>;
using Mat6 = std::array<double, 36>;
using Vec12 = std::array<double, 12>;
using Mat12 = std::array<double, 144>;

// q=(vx,vy,vz,L*wx,L*wy,L*wz); conjugate force=(F,T/L).
// resistance is an SPD, row-major matrix in these scaled coordinates.
struct Particle {
  Mat6 resistance{};
  Vec6 target{};
  Vec6 external{};
};

struct Pair {
  std::size_t i = 0, j = 0;
  Vec3 normal{1, 0, 0};           // points from particle i towards j
  Vec3 lever_i{}, lever_j{};     // vectors to ONE common force application point
  Vec3 hydro_lever_i{}, hydro_lever_j{}; // actual closest surface locations, metadata
  Vec3 image_velocity{};         // additive periodic-image velocity of j minus i
  double normal_resistance = 0;
  double tangent_resistance = 0;
  // Optional generalized pair resistance, in [q_i,q_j] ordering with the SAME
  // rotational scaling q=(U,L*Omega) as Particle. Its resistance gradient is
  // R_pair*q_pair + hydro_affine; the physical generalized force is its negative.
  // The caller includes prescribed strain, ambient flow and periodic images in
  // hydro_affine. In particular the hydro angular reference is fluid vorticity,
  // which need not equal the one-body Jeffery target for a nonspherical particle.
  // The scalar normal/tangent channel above is independently additive (normally
  // zero when this generalized channel is enabled). Contact geometry below is
  // always the actual common-point relative velocity, never the hydro offset.
  bool generalized_hydro = false;
  Mat12 hydro_resistance{};
  Vec12 hydro_affine{};
  double hydro_energy_constant = 0; // optional q-independent term; power units
  double gap = 0;                // signed current surface gap, metres
  double stiffness = 0;          // normal penalty stiffness, N/m
  double contact_onset = 0;      // rough surfaces touch at this signed gap, metres
  double friction_coefficient = 0;
  double tangent_stiffness = 0;  // elastic sliding spring, N/m
  Vec3 tangent_history{};        // committed elastic extension in CURRENT contact plane
};

enum class PreconditionerBackend { Auto, Sparse, Nested, FullLU, IncompleteLU };

struct Options {
  double relative_tolerance = 1.0e-10;
  double absolute_tolerance = 1.0e-28; // scaled generalized force, N
  double linear_relative_tolerance = 1.0e-12;
  int maximum_cg_iterations = 2000;
  int maximum_newton_iterations = 80;
  int maximum_line_search_iterations = 50;
  int maximum_gmres_iterations = 2000;
  int gmres_restart = 50;
  PreconditionerBackend preconditioner_backend = PreconditionerBackend::Auto;
  // Bounds prevent a dense contact graph from exhausting memory. Auto uses
  // nested CG above this size; explicit Sparse still respects the fill budget.
  std::size_t sparse_max_dofs = 6000;
  std::size_t sparse_max_factor_nonzeros = 4000000;
  double ilu_drop_tolerance = 1e-4;
  int ilu_fill_factor = 20;
  int ilu_refresh_iterations = 40;
  bool adaptive_forcing = false;
};

struct Result {
  std::vector<Vec6> velocity;
  std::vector<double> predicted_gap;
  std::vector<double> contact_load; // nonnegative N; force on i is -load*normal
  std::vector<Vec3> tangent_force; // physical tangential contact force on particle i
  std::vector<Vec3> tangent_history; // returned spring state; caller commits once per step
  std::vector<unsigned char> sliding;
  std::vector<double> normal_elastic_energy;     // Joules, at the predicted end of step
  std::vector<double> tangential_elastic_energy; // Joules, after the return map
  std::vector<double> sliding_dissipation;       // Joules over this step, T . plastic increment
  double residual_norm = 0;
  double residual_scale = 0; // raw force scale in atol + rtol*scale acceptance
  double relative_residual = 0;
  int cg_iterations = 0;
  int gmres_iterations = 0;
  int newton_iterations = 0;
  int contacts = 0;
  int kink_reselections = 0;
  int sparse_factorizations = 0;
  int sparse_fallbacks = 0;
  int sparse_preconditioner_solves = 0;
  int sparse_preconditioner_reuses = 0;
  int lu_factorizations = 0;
  int lu_solves = 0;
  int lu_fallbacks = 0;
  int forcing_retries = 0;
  std::size_t max_sparse_factor_nonzeros = 0;
  double preconditioner_setup_seconds = 0;
  double preconditioner_solve_seconds = 0;
  double linear_apply_seconds = 0;
};

inline double dot3(const Vec3 &a, const Vec3 &b)
{ return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

inline Vec3 cross3(const Vec3 &a, const Vec3 &b)
{ return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]}; }

struct ContactResponse {
  double predicted_gap = 0, normal_load = 0;
  Vec3 tangent_force{}, tangent_history{}, trial_history{};
  double trial_length = 0;
  bool active = false, sliding = false;
  double normal_elastic_energy = 0, tangential_elastic_energy = 0;
  double sliding_dissipation = 0;
};

// Local fully implicit return map. No committed input state is modified.
// The contact frame is frozen during the nonlinear solve and is transported
// objectively by the caller when the physical configuration is advanced.
inline ContactResponse contact_response(const Pair &pair, const Vec3 &relative, double dt)
{
  ContactResponse out;
  const double normal_speed = dot3(pair.normal,relative);
  out.predicted_gap = pair.gap + dt*normal_speed;
  const double compression = std::max(-(pair.gap-pair.contact_onset+dt*normal_speed),0.0);
  out.normal_load = pair.stiffness*compression;
  out.active = out.normal_load > 0;
  out.normal_elastic_energy = 0.5*pair.stiffness*compression*compression;
  if (!out.active || pair.friction_coefficient == 0 || pair.tangent_stiffness == 0)
    return out;
  const double old_normal = dot3(pair.normal,pair.tangent_history);
  for (int k = 0; k < 3; ++k)
    out.trial_history[k] = pair.tangent_history[k] - old_normal*pair.normal[k] +
                          dt*(relative[k]-normal_speed*pair.normal[k]);
  out.trial_length = std::sqrt(dot3(out.trial_history,out.trial_history));
  const double limit = pair.friction_coefficient*out.normal_load;
  out.sliding = pair.tangent_stiffness*out.trial_length > limit;
  const double factor = out.sliding ? limit/out.trial_length : pair.tangent_stiffness;
  for (int k = 0; k < 3; ++k) {
    out.tangent_force[k] = factor*out.trial_history[k];
    out.tangent_history[k] = out.tangent_force[k]/pair.tangent_stiffness;
    out.sliding_dissipation += out.tangent_force[k]*
                              (out.trial_history[k]-out.tangent_history[k]);
  }
  out.sliding_dissipation = out.sliding ? std::max(out.sliding_dissipation,0.0) : 0.0;
  out.tangential_elastic_energy = 0.5*pair.tangent_stiffness*
                                 dot3(out.tangent_history,out.tangent_history);
  return out;
}

// Linear relative velocity Bq, before the periodic-image offset is added.
inline Vec3 linear_pair_velocity(const Pair &pair, const std::vector<Vec6> &q,
                                double reference_length)
{
  const auto &a = q[pair.i];
  const auto &b = q[pair.j];
  const Vec3 wa{a[3]/reference_length, a[4]/reference_length, a[5]/reference_length};
  const Vec3 wb{b[3]/reference_length, b[4]/reference_length, b[5]/reference_length};
  const Vec3 ca = cross3(wa, pair.lever_i), cb = cross3(wb, pair.lever_j);
  return {b[0]-a[0]+cb[0]-ca[0], b[1]-a[1]+cb[1]-ca[1],
          b[2]-a[2]+cb[2]-ca[2]};
}

inline Vec3 pair_velocity(const Pair &pair, const std::vector<Vec6> &q,
                         double reference_length)
{
  Vec3 value = linear_pair_velocity(pair, q, reference_length);
  for (int k = 0; k < 3; ++k) value[k] += pair.image_velocity[k];
  return value;
}

// Generalized resistance gradient only; excludes the legacy scalar channel
// and contact. This public helper also supports independent force/torque audits.
inline Vec12 pair_hydro_gradient(const Pair &pair, const std::vector<Vec6> &q)
{
  Vec12 out{};
  if (!pair.generalized_hydro) return out;
  for (int r = 0; r < 12; ++r) {
    long double value = pair.hydro_affine[r];
    for (int c = 0; c < 12; ++c)
      value += static_cast<long double>(pair.hydro_resistance[12*r+c]) *
               q[c < 6 ? pair.i : pair.j][c%6];
    out[r] = static_cast<double>(value);
  }
  return out;
}

namespace detail {
using Vector = std::vector<Vec6>;
using Blocks = std::vector<Mat6>;
using Clock = std::chrono::steady_clock;
inline double elapsed_seconds(const Clock::time_point &start)
{ return std::chrono::duration<double>(Clock::now()-start).count(); }

inline Vec6 multiply(const Mat6 &matrix, const Vec6 &x)
{
  Vec6 result{};
  for (int r = 0; r < 6; ++r)
    for (int c = 0; c < 6; ++c) result[r] += matrix[6*r+c]*x[c];
  return result;
}

inline double dot(const Vector &a, const Vector &b)
{
  long double result = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    for (int k = 0; k < 6; ++k)
      result += static_cast<long double>(a[i][k])*b[i][k];
  return static_cast<double>(result);
}

inline double norm(const Vector &a) { return std::sqrt(std::max(dot(a,a), 0.0)); }

inline void add_hydro(const Pair &pair, const Vector &q, Vector &out,
                      bool include_affine)
{
  if (!pair.generalized_hydro) return;
  for (int r = 0; r < 12; ++r) {
    long double value = include_affine ? pair.hydro_affine[r] : 0;
    for (int c = 0; c < 12; ++c)
      value += static_cast<long double>(pair.hydro_resistance[12*r+c]) *
               q[c < 6 ? pair.i : pair.j][c%6];
    out[r < 6 ? pair.i : pair.j][r%6] += static_cast<double>(value);
  }
}

inline Vec3 resistance(const Pair &pair, const Vec3 &relative,
                       double extra_normal = 0)
{
  const double normal = (pair.normal_resistance + extra_normal -
                         pair.tangent_resistance)*dot3(pair.normal, relative);
  Vec3 result{};
  for (int k = 0; k < 3; ++k)
    result[k] = pair.tangent_resistance*relative[k] + normal*pair.normal[k];
  return result;
}

// Add B^T f. This is a resistance gradient; the physical force is its negative.
inline void add_transpose(const Pair &pair, const Vec3 &f, double length, Vector &out)
{
  const Vec3 ti = cross3(pair.lever_i, f), tj = cross3(pair.lever_j, f);
  for (int k = 0; k < 3; ++k) {
    out[pair.i][k] -= f[k];
    out[pair.j][k] += f[k];
    out[pair.i][k+3] -= ti[k]/length;
    out[pair.j][k+3] += tj[k]/length;
  }
}

// A pair matrix is semidefinite (rigid translation/rotation may be null modes),
// so ordinary positive-definite Cholesky is not a valid input check. Pivot on
// the largest remaining diagonal and stop only if the residual block is zero
// to rounding accuracy. This catches a finite, symmetric but indefinite input
// before either PCG or the Coulomb preconditioner sees it.
inline void validate_pair_hydro(const Pair &pair)
{
  if (!pair.generalized_hydro) return;
  double scale = 0;
  for (double x : pair.hydro_resistance) {
    if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite generalized pair resistance");
    scale = std::max(scale,std::abs(x));
  }
  for (double x : pair.hydro_affine)
    if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite generalized pair affine load");
  if (!std::isfinite(pair.hydro_energy_constant))
    throw std::invalid_argument("Nonfinite generalized pair energy constant");
  const long double tolerance = 1.0e-12L*scale;
  std::array<long double,144> residual{};
  for (int r = 0; r < 12; ++r)
    for (int c = 0; c < 12; ++c) {
      if (std::abs(pair.hydro_resistance[12*r+c]-pair.hydro_resistance[12*c+r]) > tolerance)
        throw std::invalid_argument("Generalized pair resistance is not symmetric");
      residual[12*r+c] = pair.hydro_resistance[12*r+c];
    }
  for (int k = 0; k < 12; ++k) {
    int pivot = k;
    for (int r = k+1; r < 12; ++r)
      if (residual[13*r] > residual[13*pivot]) pivot = r;
    if (residual[13*pivot] <= tolerance) {
      for (int r = k; r < 12; ++r)
        for (int c = k; c < 12; ++c)
          if (std::abs(residual[12*r+c]) > 4*tolerance)
            throw std::invalid_argument("Generalized pair resistance is not positive semidefinite");
      return;
    }
    if (pivot != k) {
      for (int r = 0; r < 12; ++r) std::swap(residual[12*r+k],residual[12*r+pivot]);
      for (int c = 0; c < 12; ++c) std::swap(residual[12*k+c],residual[12*pivot+c]);
    }
    const long double diagonal = residual[13*k];
    for (int r = k+1; r < 12; ++r)
      for (int c = k+1; c < 12; ++c)
        residual[12*r+c] -= residual[12*r+k]*residual[12*k+c]/diagonal;
  }
}

inline void validate(const std::vector<Particle> &particles,
                     const std::vector<Pair> &pairs, double dt, double length,
                     const Options &options)
{
  if (particles.empty() || !(dt > 0) || !std::isfinite(dt) ||
      !(length > 0) || !std::isfinite(length))
    throw std::invalid_argument("Implicit solver requires particles and positive finite dt/length");
  if (!(options.relative_tolerance > 0) || !(options.absolute_tolerance > 0) ||
      !(options.linear_relative_tolerance > 0) ||
      !std::isfinite(options.relative_tolerance) ||
      !std::isfinite(options.absolute_tolerance) ||
      !std::isfinite(options.linear_relative_tolerance) ||
      options.maximum_cg_iterations < 1 || options.maximum_newton_iterations < 1 ||
      options.maximum_line_search_iterations < 1 || options.maximum_gmres_iterations < 1 ||
      options.gmres_restart < 1 || !(options.ilu_drop_tolerance > 0) ||
      !std::isfinite(options.ilu_drop_tolerance) || options.ilu_fill_factor < 1 ||
      options.ilu_refresh_iterations < 1)
    throw std::invalid_argument("Invalid implicit solver tolerances/iteration limits");
  for (const auto &p : particles) {
    double scale = 0;
    for (double x : p.resistance) {
      if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite one-body resistance");
      scale = std::max(scale, std::abs(x));
    }
    Mat6 cholesky{};
    for (int r = 0; r < 6; ++r) {
      if (!std::isfinite(p.target[r]) || !std::isfinite(p.external[r]))
        throw std::invalid_argument("Nonfinite implicit target/load");
      for (int c = 0; c <= r; ++c) {
        if (std::abs(p.resistance[6*r+c]-p.resistance[6*c+r]) > 1.0e-12*scale)
          throw std::invalid_argument("One-body resistance is not symmetric");
        double s = p.resistance[6*r+c];
        for (int k = 0; k < c; ++k) s -= cholesky[6*r+k]*cholesky[6*c+k];
        if (r == c) {
          if (!(s > 0)) throw std::invalid_argument("One-body resistance is not positive definite");
          cholesky[6*r+c] = std::sqrt(s);
        } else cholesky[6*r+c] = s/cholesky[6*c+c];
      }
    }
  }
  for (const auto &p : pairs) {
    if (p.i >= particles.size() || p.j >= particles.size() || p.i == p.j)
      throw std::invalid_argument("Invalid implicit pair particle indices");
    for (int k = 0; k < 3; ++k)
      if (!std::isfinite(p.normal[k]) || !std::isfinite(p.lever_i[k]) ||
          !std::isfinite(p.lever_j[k]) || !std::isfinite(p.image_velocity[k]) ||
          !std::isfinite(p.tangent_history[k]))
        throw std::invalid_argument("Nonfinite implicit pair geometry");
    if (std::abs(dot3(p.normal,p.normal)-1) > 1.0e-9 ||
        !(p.normal_resistance >= 0) || !std::isfinite(p.normal_resistance) ||
        !(p.tangent_resistance >= 0) || !std::isfinite(p.tangent_resistance) ||
        !(p.stiffness >= 0) || !std::isfinite(p.stiffness) || !std::isfinite(p.gap) ||
        !(p.contact_onset >= 0) || !std::isfinite(p.contact_onset) ||
        !(p.friction_coefficient >= 0) || !std::isfinite(p.friction_coefficient) ||
        !(p.tangent_stiffness >= 0) || !std::isfinite(p.tangent_stiffness))
      throw std::invalid_argument("Invalid implicit pair coefficients/normal");
    if (std::abs(dot3(p.normal,p.tangent_history)) >
        1.0e-24 + 1.0e-8*std::sqrt(dot3(p.tangent_history,p.tangent_history)))
      throw std::invalid_argument("Implicit tangential history is not in the current contact plane");
    validate_pair_hydro(p);
  }
}

inline void apply(const std::vector<Particle> &particles, const std::vector<Pair> &pairs,
                  const std::vector<unsigned char> &active, double dt, double length,
                  const Vector &q, Vector &out)
{
  out.assign(particles.size(), Vec6{});
  for (std::size_t i = 0; i < particles.size(); ++i)
    out[i] = multiply(particles[i].resistance, q[i]);
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    const auto &p = pairs[k];
    add_hydro(p,q,out,false);
    add_transpose(p, resistance(p, linear_pair_velocity(p,q,length),
                               active[k] ? dt*p.stiffness : 0), length, out);
  }
}

// Exact diagonal of the active SPD Hessian, including translation/rotation.
inline Vector diagonal(const std::vector<Particle> &particles,
                       const std::vector<Pair> &pairs,
                       const std::vector<unsigned char> &active,
                       double dt, double length)
{
  Vector d(particles.size());
  for (std::size_t i = 0; i < particles.size(); ++i)
    for (int k = 0; k < 6; ++k) d[i][k] = particles[i].resistance[7*k];
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    const auto &p = pairs[index];
    const double extra = active[index] ? dt*p.stiffness : 0;
    for (int side = 0; side < 2; ++side) {
      const auto &lever = side == 0 ? p.lever_i : p.lever_j;
      const auto id = side == 0 ? p.i : p.j;
      for (int k = 0; k < 6; ++k) {
        Vec3 b{};
        if (k < 3) b[k] = 1;
        else {
          Vec3 e{};
          e[k-3] = 1/length;
          b = cross3(e,lever);
        }
        d[id][k] += dot3(b,resistance(p,b,extra));
        if (p.generalized_hydro) d[id][k] += p.hydro_resistance[13*(6*side+k)];
      }
    }
  }
  return d;
}

inline std::array<Vec3,6> lever_columns(const Vec3 &lever, double length)
{
  std::array<Vec3,6> columns{};
  for (int k = 0; k < 3; ++k) {
    columns[k][k] = 1;
    Vec3 axis{};
    axis[k] = 1/length;
    columns[k+3] = cross3(axis,lever);
  }
  return columns;
}

// Exact 6x6 on-particle blocks of the active SPD operator. Keeping translation
// and rotation coupled is particularly useful when the lubrication modes span
// very different resistance scales. Cross-particle couplings stay in apply().
inline Blocks block_diagonal(const std::vector<Particle> &particles,
                             const std::vector<Pair> &pairs,
                             const std::vector<unsigned char> &active,
                             double dt, double length)
{
  Blocks out;
  out.reserve(particles.size());
  for (const auto &p : particles) out.push_back(p.resistance);
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    const auto &p = pairs[index];
    for (int side = 0; side < 2; ++side) {
      auto &block = out[side == 0 ? p.i : p.j];
      const auto columns = lever_columns(side == 0 ? p.lever_i : p.lever_j,length);
      for (int c = 0; c < 6; ++c) {
        const Vec3 force = resistance(p,columns[c],active[index] ? dt*p.stiffness : 0);
        for (int r = 0; r < 6; ++r) {
          block[6*r+c] += dot3(columns[r],force);
          if (p.generalized_hydro)
            block[6*r+c] += p.hydro_resistance[12*(6*side+r)+6*side+c];
        }
      }
    }
  }
  return out;
}

// Store a lower-triangular Cholesky factor for each block. No physical
// resistance or diagonal shift is changed to obtain the preconditioner.
inline Blocks factor_blocks(const Blocks &blocks)
{
  Blocks factors(blocks.size());
  for (std::size_t i = 0; i < blocks.size(); ++i)
    for (int r = 0; r < 6; ++r)
      for (int c = 0; c <= r; ++c) {
        long double value = blocks[i][6*r+c];
        for (int k = 0; k < c; ++k)
          value -= static_cast<long double>(factors[i][6*r+k])*factors[i][6*c+k];
        if (r == c) {
          if (!(value > 0) || !std::isfinite(value))
            throw std::runtime_error("Implicit 6x6 preconditioner is not positive definite");
          factors[i][6*r+c] = std::sqrt(value);
        } else factors[i][6*r+c] = value/factors[i][6*c+c];
      }
  return factors;
}

inline void block_inverse(const Blocks &factors, const Vector &rhs, Vector &out)
{
  out.resize(rhs.size());
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    Vec6 intermediate{};
    for (int r = 0; r < 6; ++r) {
      long double value = rhs[i][r];
      for (int c = 0; c < r; ++c) value -= factors[i][6*r+c]*intermediate[c];
      intermediate[r] = value/factors[i][7*r];
    }
    for (int r = 5; r >= 0; --r) {
      long double value = intermediate[r];
      for (int c = r+1; c < 6; ++c) value -= factors[i][6*c+r]*out[i][c];
      out[i][r] = value/factors[i][7*r];
    }
  }
}

inline int cg(const std::vector<Particle> &particles, const std::vector<Pair> &pairs,
              const std::vector<unsigned char> &active, double dt, double length,
              const Vector &rhs, const Options &options, Vector &x)
{
  x.assign(particles.size(), Vec6{});
  Vector r = rhs, z(particles.size()), p(particles.size()), ap;
  const Blocks factors = factor_blocks(block_diagonal(particles,pairs,active,dt,length));
  const double threshold = std::max(options.absolute_tolerance*0.01,
                                    options.linear_relative_tolerance*norm(rhs));
  if (norm(r) <= threshold) return 0;
  block_inverse(factors,r,z);
  p = z;
  double rz = dot(r,z);
  for (int iteration = 1; iteration <= options.maximum_cg_iterations; ++iteration) {
    apply(particles,pairs,active,dt,length,p,ap);
    const double pap = dot(p,ap);
    if (!(pap > 0) || !std::isfinite(pap) || !(rz > 0) || !std::isfinite(rz))
      throw std::runtime_error("Implicit PCG lost positive definiteness or finite arithmetic");
    const double alpha = rz/pap;
    for (std::size_t i = 0; i < particles.size(); ++i)
      for (int k = 0; k < 6; ++k) {
        x[i][k] += alpha*p[i][k];
        r[i][k] -= alpha*ap[i][k];
      }
    if (norm(r) <= threshold) {
      // Recompute the residual; recursive CG residuals can drift at high stiffness.
      apply(particles,pairs,active,dt,length,x,ap);
      for (std::size_t i = 0; i < particles.size(); ++i)
        for (int k = 0; k < 6; ++k) r[i][k] = rhs[i][k]-ap[i][k];
      if (norm(r) <= 4*threshold) return iteration;
      // Reliable restart from the current x and true residual.
      block_inverse(factors,r,z);
      p = z;
      rz = dot(r,z);
      continue;
    }
    block_inverse(factors,r,z);
    const double next_rz = dot(r,z), beta = next_rz/rz;
    for (std::size_t i = 0; i < particles.size(); ++i)
      for (int k = 0; k < 6; ++k) p[i][k] = z[i][k]+beta*p[i][k];
    rz = next_rz;
  }
  throw std::runtime_error("Implicit PCG exceeded maximum iterations");
}

inline long double energy(const std::vector<Particle> &particles,
                          const std::vector<Pair> &pairs, double dt, double length,
                          const Vector &q)
{
  long double result = 0;
  for (std::size_t i = 0; i < particles.size(); ++i) {
    Vec6 delta{};
    for (int k = 0; k < 6; ++k) delta[k] = q[i][k]-particles[i].target[k];
    const Vec6 force = multiply(particles[i].resistance,delta);
    for (int k = 0; k < 6; ++k)
      result += 0.5L*delta[k]*force[k] -
                static_cast<long double>(particles[i].external[k])*q[i][k];
  }
  for (const auto &p : pairs) {
    if (p.generalized_hydro) {
      result += p.hydro_energy_constant;
      for (int r = 0; r < 12; ++r) {
        const double qr = q[r < 6 ? p.i : p.j][r%6];
        result += static_cast<long double>(p.hydro_affine[r])*qr;
        for (int c = 0; c < 12; ++c)
          result += 0.5L*qr*p.hydro_resistance[12*r+c]*q[c < 6 ? p.i : p.j][c%6];
      }
    }
    const Vec3 relative = pair_velocity(p,q,length);
    const double closing = std::min(p.gap-p.contact_onset + dt*dot3(p.normal,relative),0.0);
    result += 0.5L*dot3(relative,resistance(p,relative)) +
              0.5L*p.stiffness/dt*closing*closing;
  }
  return result;
}

inline double gradient(const std::vector<Particle> &particles,
                       const std::vector<Pair> &pairs, double dt, double length,
                       const Vector &q, Vector &g, std::vector<unsigned char> &active)
{
  g.assign(particles.size(),Vec6{});
  Vector rhs(particles.size());
  for (std::size_t i = 0; i < particles.size(); ++i) {
    rhs[i] = multiply(particles[i].resistance,particles[i].target);
    g[i] = multiply(particles[i].resistance,q[i]);
    for (int k = 0; k < 6; ++k) {
      rhs[i][k] += particles[i].external[k];
      g[i][k] -= rhs[i][k];
    }
  }
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    const auto &p = pairs[index];
    add_hydro(p,q,g,true);
    if (p.generalized_hydro)
      for (int r = 0; r < 12; ++r) rhs[r < 6 ? p.i : p.j][r%6] -= p.hydro_affine[r];
    const Vec3 relative = pair_velocity(p,q,length);
    const double gap_next = p.gap-p.contact_onset + dt*dot3(p.normal,relative);
    active[index] = p.stiffness > 0 && gap_next < 0;
    Vec3 force = resistance(p,relative);
    Vec3 constant = resistance(p,p.image_velocity);
    if (active[index])
      for (int k = 0; k < 3; ++k) {
        force[k] += p.stiffness*gap_next*p.normal[k];
        constant[k] += p.stiffness*(p.gap-p.contact_onset + dt*dot3(p.normal,p.image_velocity))*p.normal[k];
      }
    add_transpose(p,force,length,g);
    for (double &x : constant) x = -x;
    add_transpose(p,constant,length,rhs);
  }
  return norm(rhs);
}

inline void fill_result(const std::vector<Pair> &pairs, double dt, double length,
                        Result &result)
{
  result.predicted_gap.resize(pairs.size());
  result.contact_load.resize(pairs.size());
  result.tangent_force.resize(pairs.size());
  result.tangent_history.resize(pairs.size());
  result.sliding.resize(pairs.size());
  result.normal_elastic_energy.resize(pairs.size());
  result.tangential_elastic_energy.resize(pairs.size());
  result.sliding_dissipation.resize(pairs.size());
  result.contacts = 0;
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    const auto c = contact_response(pairs[k],pair_velocity(pairs[k],result.velocity,length),dt);
    result.predicted_gap[k] = c.predicted_gap;
    result.contact_load[k] = c.normal_load;
    result.tangent_force[k] = c.tangent_force;
    result.tangent_history[k] = c.tangent_history;
    result.sliding[k] = c.sliding;
    result.normal_elastic_energy[k] = c.normal_elastic_energy;
    result.tangential_elastic_energy[k] = c.tangential_elastic_energy;
    result.sliding_dissipation[k] = c.sliding_dissipation;
    if (c.active) ++result.contacts;
  }
}

inline double friction_residual(const std::vector<Particle> &particles,
                                const std::vector<Pair> &pairs, double dt, double length,
                                const Vector &q, Vector &out,
                                std::vector<ContactResponse> &state)
{
  std::vector<unsigned char> active(pairs.size());
  const double normal_scale = gradient(particles,pairs,dt,length,q,out,active);
  Vector tangent_load(particles.size());
  state.resize(pairs.size());
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    state[k] = contact_response(pairs[k],pair_velocity(pairs[k],q,length),dt);
    add_transpose(pairs[k],state[k].tangent_force,length,tangent_load);
  }
  for (std::size_t i = 0; i < particles.size(); ++i)
    for (int k = 0; k < 6; ++k) out[i][k] += tangent_load[i][k];
  return std::max(normal_scale,norm(tangent_load));
}

// Tangential part of the exact generalized derivative. The last sliding term
// differentiates mu*N(q): it couples normal approach to tangential friction and
// is nonsymmetric. Only the PRECONDITIONER omits that term.
inline Vec3 tangent_derivative(const Pair &p, const ContactResponse &c,
                               const Vec3 &relative_increment, double dt,
                               bool exact_normal_coupling)
{
  Vec3 out{};
  if (!c.active || p.friction_coefficient == 0 || p.tangent_stiffness == 0) return out;
  const double dn = dot3(p.normal,relative_increment);
  Vec3 tangent{};
  for (int k = 0; k < 3; ++k) tangent[k] = relative_increment[k]-dn*p.normal[k];
  if (!c.sliding) {
    for (int k = 0; k < 3; ++k) out[k] = dt*p.tangent_stiffness*tangent[k];
    return out;
  }
  Vec3 direction{};
  for (int k = 0; k < 3; ++k) direction[k] = c.trial_history[k]/c.trial_length;
  const double along = dot3(direction,tangent);
  const double coefficient = dt*p.friction_coefficient*c.normal_load/c.trial_length;
  for (int k = 0; k < 3; ++k) {
    out[k] = coefficient*(tangent[k]-along*direction[k]);
    if (exact_normal_coupling)
      out[k] -= dt*p.friction_coefficient*p.stiffness*dn*direction[k];
  }
  return out;
}

inline void friction_jacobian(const std::vector<Particle> &particles,
                              const std::vector<Pair> &pairs,
                              const std::vector<ContactResponse> &state,
                              double dt, double length, const Vector &q, Vector &out,
                              bool exact_normal_coupling = true)
{
  out.assign(particles.size(),Vec6{});
  for (std::size_t i = 0; i < particles.size(); ++i)
    out[i] = multiply(particles[i].resistance,q[i]);
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    const auto &p = pairs[k];
    add_hydro(p,q,out,false);
    // Generalized lubrication has already been applied. An inactive contact
    // with no legacy scalar resistance contributes exactly zero below.
    if (p.normal_resistance == 0 && p.tangent_resistance == 0 && !state[k].active)
      continue;
    const Vec3 relative = linear_pair_velocity(p,q,length);
    Vec3 force = resistance(p,relative,state[k].active ? dt*p.stiffness : 0);
    const Vec3 tangent = tangent_derivative(p,state[k],relative,dt,exact_normal_coupling);
    for (int l = 0; l < 3; ++l) force[l] += tangent[l];
    add_transpose(p,force,length,out);
  }
}

inline Vector friction_diagonal(const std::vector<Particle> &particles,
                                 const std::vector<Pair> &pairs,
                                 const std::vector<ContactResponse> &state,
                                 double dt, double length)
{
  std::vector<unsigned char> active(pairs.size());
  for (std::size_t k = 0; k < pairs.size(); ++k) active[k] = state[k].active;
  Vector out = diagonal(particles,pairs,active,dt,length);
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    const auto &p = pairs[index];
    if (!state[index].active || p.friction_coefficient == 0 || p.tangent_stiffness == 0)
      continue;
    for (int side = 0; side < 2; ++side) {
      const auto &lever = side == 0 ? p.lever_i : p.lever_j;
      const auto id = side == 0 ? p.i : p.j;
      for (int k = 0; k < 6; ++k) {
        Vec3 b{};
        if (k < 3) b[k] = 1;
        else { Vec3 e{}; e[k-3] = 1/length; b = cross3(e,lever); }
        out[id][k] += dot3(b,tangent_derivative(p,state[index],b,dt,false));
      }
    }
  }
  return out;
}

inline Blocks friction_blocks(const std::vector<Particle> &particles,
                              const std::vector<Pair> &pairs,
                              const std::vector<ContactResponse> &state,
                              double dt, double length)
{
  std::vector<unsigned char> active(pairs.size());
  for (std::size_t k = 0; k < pairs.size(); ++k) active[k] = state[k].active;
  Blocks out = block_diagonal(particles,pairs,active,dt,length);
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    const auto &p = pairs[index];
    if (!state[index].active || p.friction_coefficient == 0 || p.tangent_stiffness == 0)
      continue;
    for (int side = 0; side < 2; ++side) {
      auto &block = out[side == 0 ? p.i : p.j];
      const auto columns = lever_columns(side == 0 ? p.lever_i : p.lever_j,length);
      for (int c = 0; c < 6; ++c) {
        const auto force = tangent_derivative(p,state[index],columns[c],dt,false);
        for (int r = 0; r < 6; ++r) block[6*r+c] += dot3(columns[r],force);
      }
    }
  }
  return out;
}

// Assemble exactly the symmetric friction Jacobian used by the existing
// nested-CG preconditioner. The load-dependent nonsymmetric Coulomb term stays
// in friction_jacobian(...,true), used by the OUTER solve and residual checks.
inline std::vector<SlurrySparse::Entry> symmetric_sparse_entries(
    const std::vector<Particle> &particles, const std::vector<Pair> &pairs,
    const std::vector<ContactResponse> &state, double dt, double length)
{
  std::vector<SlurrySparse::Entry> entries;
  entries.reserve(21*particles.size()+78*pairs.size());
  for (std::size_t i = 0; i < particles.size(); ++i)
    for (int r = 0; r < 6; ++r)
      for (int c = r; c < 6; ++c)
        entries.push_back({static_cast<int>(6*i+r),static_cast<int>(6*i+c),
                           particles[i].resistance[6*r+c]});
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    const auto &p = pairs[k];
    const auto a = lever_columns(p.lever_i,length), b = lever_columns(p.lever_j,length);
    std::array<Vec3,12> columns{};
    for (int c = 0; c < 6; ++c)
      for (int d = 0; d < 3; ++d) {
        columns[c][d] = -a[c][d];
        columns[c+6][d] = b[c][d];
      }
    for (int c = 0; c < 12; ++c) {
      auto force = resistance(p,columns[c],state[k].active ? dt*p.stiffness : 0);
      const auto tangent = tangent_derivative(p,state[k],columns[c],dt,false);
      for (int d = 0; d < 3; ++d) force[d] += tangent[d];
      for (int r = 0; r <= c; ++r) {
        const double value = dot3(columns[r],force)+
                            (p.generalized_hydro ? p.hydro_resistance[12*r+c] : 0);
        if (value != 0)
          entries.push_back({static_cast<int>(6*(r < 6 ? p.i : p.j)+r%6),
                             static_cast<int>(6*(c < 6 ? p.i : p.j)+c%6),value});
      }
    }
  }
  return entries;
}

#ifdef SLURRY_HAVE_EIGEN
// Full nonsymmetric derivative. In particular the derivative of mu*N(q) is
// retained; neither triangle is copied onto the other. Structural zeros are
// retained for reliable reuse of Eigen's symbolic analysis within a timestep.
inline std::vector<SlurryEigen::Entry> exact_sparse_entries(
    const std::vector<Particle> &particles, const std::vector<Pair> &pairs,
    const std::vector<ContactResponse> &state, double dt, double length)
{
  std::vector<SlurryEigen::Entry> entries;
  entries.reserve(36*particles.size()+144*pairs.size());
  for (std::size_t i = 0; i < particles.size(); ++i)
    for (int r = 0; r < 6; ++r)
      for (int c = 0; c < 6; ++c)
        entries.push_back({static_cast<int>(6*i+r),static_cast<int>(6*i+c),
                           particles[i].resistance[6*r+c]});
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    const auto &p = pairs[k];
    const auto a = lever_columns(p.lever_i,length), b = lever_columns(p.lever_j,length);
    std::array<Vec3,12> columns{};
    for (int c = 0; c < 6; ++c)
      for (int d = 0; d < 3; ++d) {
        columns[c][d] = -a[c][d];
        columns[c+6][d] = b[c][d];
      }
    for (int c = 0; c < 12; ++c) {
      auto force = resistance(p,columns[c],state[k].active ? dt*p.stiffness : 0);
      const auto tangent = tangent_derivative(p,state[k],columns[c],dt,true);
      for (int d = 0; d < 3; ++d) force[d] += tangent[d];
      for (int r = 0; r < 12; ++r)
        entries.push_back({static_cast<int>(6*(r < 6 ? p.i : p.j)+r%6),
                           static_cast<int>(6*(c < 6 ? p.i : p.j)+c%6),
                           dot3(columns[r],force)+
                           (p.generalized_hydro ? p.hydro_resistance[12*r+c] : 0)});
    }
  }
  return entries;
}

template<class Factor>
inline void full_lu_inverse(const Factor &factor,
                            const Vector &rhs, Vector &out)
{
  std::vector<double> flat(6*rhs.size()), solution;
  for (std::size_t i = 0; i < rhs.size(); ++i)
    for (int k = 0; k < 6; ++k) flat[6*i+k] = rhs[i][k];
  factor.solve(flat,solution);
  out.resize(rhs.size());
  for (std::size_t i = 0; i < rhs.size(); ++i)
    for (int k = 0; k < 6; ++k) out[i][k] = solution[6*i+k];
}
#endif

inline bool sparse_requested(const Options &options, std::size_t particles)
{
  return options.preconditioner_backend == PreconditionerBackend::Sparse ||
      (options.preconditioner_backend == PreconditionerBackend::Auto &&
       6*particles <= options.sparse_max_dofs);
}

inline std::vector<int> sparse_ordering(std::size_t particles, const std::vector<Pair> &pairs)
{
  std::vector<std::pair<std::size_t,std::size_t>> edges;
  edges.reserve(pairs.size());
  for (const auto &p : pairs) edges.emplace_back(p.i,p.j);
  return SlurrySparse::block_minimum_degree(particles,edges);
}

inline std::vector<int> prepare_sparse_ordering(std::size_t particles,
    const std::vector<Pair> &pairs, const Options &options, Result &result)
{
  if (!sparse_requested(options,particles)) return {};
  try { return sparse_ordering(particles,pairs); }
  catch (const std::bad_alloc &) {
    ++result.sparse_fallbacks;
    return {};
  }
}

inline bool factor_sparse(const std::vector<Particle> &particles,
                          const std::vector<Pair> &pairs,
                          const std::vector<ContactResponse> &state,
                          double dt, double length, const Options &options,
                          const std::vector<int> &ordering,
                          SlurrySparse::LDLT &factor, Result &result)
{
  if (!sparse_requested(options,particles.size()) || ordering.size() != 6*particles.size())
    return false;
  try {
    factor.factor(static_cast<int>(6*particles.size()),
                  symmetric_sparse_entries(particles,pairs,state,dt,length),ordering,
                  options.sparse_max_factor_nonzeros);
    ++result.sparse_factorizations;
    result.max_sparse_factor_nonzeros = std::max(result.max_sparse_factor_nonzeros,factor.nonzeros());
    return true;
  } catch (const std::runtime_error &) {
    // A failed/budget-limited PRECONDITIONER does not alter the physical solve.
    // The diagnostic is exposed to the caller; the established nested backend
    // provides an explicit safe fallback with identical final tolerances.
    ++result.sparse_fallbacks;
    factor = SlurrySparse::LDLT{};
    return false;
  } catch (const std::bad_alloc &) {
    ++result.sparse_fallbacks;
    factor = SlurrySparse::LDLT{};
    return false;
  }
}

inline void sparse_inverse(const SlurrySparse::LDLT &factor, const Vector &rhs, Vector &out)
{
  std::vector<double> flat(6*rhs.size()), solution;
  for (std::size_t i = 0; i < rhs.size(); ++i)
    for (int k = 0; k < 6; ++k) flat[6*i+k] = rhs[i][k];
  factor.solve(flat,solution);
  out.resize(rhs.size());
  for (std::size_t i = 0; i < rhs.size(); ++i)
    for (int k = 0; k < 6; ++k) out[i][k] = solution[6*i+k];
}

// An approximate SPD inverse is a PRECONDITIONER, not an accepted physical
// solution. FGMRES accommodates its variable accuracy; the outer true residual
// is always checked. Stop preconditioning as soon as its 10% target is met.
inline int friction_precondition(const std::vector<Particle> &particles,
                                 const std::vector<Pair> &pairs,
                                 const std::vector<ContactResponse> &state,
                                 const Blocks &factors, double dt, double length,
                                 const Vector &rhs, int maximum_iterations, Vector &x)
{
  x.assign(rhs.size(),Vec6{});
  const double threshold = 0.1*norm(rhs);
  if (threshold == 0) return 0;
  Vector r=rhs, z(rhs.size()), p(rhs.size()), ap;
  block_inverse(factors,r,z);
  p = z;
  double rz = dot(r,z);
  const int budget = std::min(maximum_iterations,100);
  for (int iteration = 1; iteration <= budget; ++iteration) {
    friction_jacobian(particles,pairs,state,dt,length,p,ap,false);
    const double pap = dot(p,ap);
    if (!(pap > 0) || !std::isfinite(pap) || !(rz > 0) || !std::isfinite(rz))
      throw std::runtime_error("Friction SPD preconditioner lost positive definiteness");
    const double alpha = rz/pap;
    for (std::size_t i = 0; i < rhs.size(); ++i)
      for (int k = 0; k < 6; ++k) {
        x[i][k] += alpha*p[i][k];
        r[i][k] -= alpha*ap[i][k];
      }
    if (norm(r) <= threshold) return iteration;
    block_inverse(factors,r,z);
    const double next = dot(r,z), beta=next/rz;
    for (std::size_t i = 0; i < rhs.size(); ++i)
      for (int k = 0; k < 6; ++k) p[i][k] = z[i][k]+beta*p[i][k];
    rz = next;
  }
  return budget;
}

template<class Apply, class Precondition>
inline int fgmres(const Apply &apply, const Precondition &precondition,
                  const Vector &rhs, const Options &options, Vector &x)
{
  x.assign(rhs.size(),Vec6{});
  Vector r=rhs, ax, work;
  const double threshold = std::max(options.absolute_tolerance*0.01,
                                    options.linear_relative_tolerance*norm(rhs));
  if (norm(r) <= threshold) return 0;
  const int restart = std::min(options.gmres_restart,static_cast<int>(6*rhs.size()));
  int iterations = 0;
  while (iterations < options.maximum_gmres_iterations) {
    const int m = std::min(restart,options.maximum_gmres_iterations-iterations);
    std::vector<Vector> v(m+1,Vector(rhs.size())), z(m,Vector(rhs.size()));
    std::vector<std::vector<double>> h(m+1,std::vector<double>(m,0));
    std::vector<double> c(m), s(m), g(m+1,0);
    const double beta = norm(r);
    if (!std::isfinite(beta)) throw std::runtime_error("Nonfinite friction GMRES residual");
    for (std::size_t i = 0; i < rhs.size(); ++i)
      for (int k = 0; k < 6; ++k) v[0][i][k] = r[i][k]/beta;
    g[0] = beta;
    int used = 0;
    for (int j = 0; j < m; ++j) {
      precondition(v[j],z[j]);
      apply(z[j],work);
      // Reorthogonalized MGS is reliable for the large kt/drag ratio.
      for (int pass = 0; pass < 2; ++pass)
        for (int l = 0; l <= j; ++l) {
          const double coefficient = dot(v[l],work);
          h[l][j] += coefficient;
          for (std::size_t i = 0; i < rhs.size(); ++i)
            for (int k = 0; k < 6; ++k) work[i][k] -= coefficient*v[l][i][k];
        }
      h[j+1][j] = norm(work);
      const bool breakdown = h[j+1][j] <= 16*std::numeric_limits<double>::epsilon()*
          std::max(std::abs(h[j][j]),std::numeric_limits<double>::min());
      if (!breakdown)
        for (std::size_t i = 0; i < rhs.size(); ++i)
          for (int k = 0; k < 6; ++k) v[j+1][i][k] = work[i][k]/h[j+1][j];
      for (int l = 0; l < j; ++l) {
        const double next = c[l]*h[l][j]+s[l]*h[l+1][j];
        h[l+1][j] = -s[l]*h[l][j]+c[l]*h[l+1][j];
        h[l][j] = next;
      }
      const double diagonal = std::hypot(h[j][j],h[j+1][j]);
      if (!(diagonal > 0) || !std::isfinite(diagonal))
        throw std::runtime_error("Friction GMRES Arnoldi factorization broke down");
      c[j] = h[j][j]/diagonal;
      s[j] = h[j+1][j]/diagonal;
      h[j][j] = diagonal;
      h[j+1][j] = 0;
      g[j+1] = -s[j]*g[j];
      g[j] *= c[j];
      ++iterations;
      used = j+1;
      if (std::abs(g[j+1]) <= threshold || breakdown) break;
    }
    std::vector<double> y(used);
    for (int k = used-1; k >= 0; --k) {
      double value = g[k];
      for (int j = k+1; j < used; ++j) value -= h[k][j]*y[j];
      y[k] = value/h[k][k];
    }
    for (int j = 0; j < used; ++j)
      for (std::size_t i = 0; i < rhs.size(); ++i)
        for (int k = 0; k < 6; ++k) x[i][k] += y[j]*z[j][i][k];
    apply(x,ax);
    for (std::size_t i = 0; i < rhs.size(); ++i)
      for (int k = 0; k < 6; ++k) r[i][k] = rhs[i][k]-ax[i][k];
    if (norm(r) <= 4*threshold) return iterations;
  }
  throw std::runtime_error("Friction GMRES exceeded maximum iterations without force convergence");
}

// Select the directional member of the generalized Jacobian at a Coulomb
// yield kink before backtracking. An inconsistent one-sided derivative can
// predict descent even though every positive step enters the opposite branch.
// This never changes the evaluated return map, coefficient, or input history.
inline int reselect_friction_kinks(const std::vector<Pair> &pairs,
                                  const std::vector<ContactResponse> &state,
                                  const Vector &direction, double length,
                                  std::vector<ContactResponse> &linear_state)
{
  int changes = 0;
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    const auto &p = pairs[k];
    const auto &c = state[k];
    if (!c.active || c.trial_length == 0 || p.friction_coefficient == 0 ||
        p.tangent_stiffness == 0) continue;
    const double trial_force = p.tangent_stiffness*c.trial_length;
    const double cap = p.friction_coefficient*c.normal_load;
    // Numerical band for DERIVATIVE selection only; the force law is exact.
    if (std::abs(trial_force-cap) > 1e-8*std::max(trial_force,cap)) continue;
    auto dr = linear_pair_velocity(p,direction,length);
    const double dn = dot3(p.normal,dr);
    for (int l = 0; l < 3; ++l) dr[l] -= dn*p.normal[l];
    const double along = dot3(c.trial_history,dr)/c.trial_length;
    const double yield_rate = p.tangent_stiffness*along+
                              p.friction_coefficient*p.stiffness*dn;
    const bool sliding = yield_rate > 0;
    if (linear_state[k].sliding != sliding) {
      linear_state[k].sliding = sliding;
      ++changes;
    }
  }
  return changes;
}

inline Result solve_friction(const std::vector<Particle> &particles,
                             const std::vector<Pair> &pairs, double dt, double length,
                             const Options &options, const Vector &initial,
                             SolverWorkspace *workspace = nullptr)
{
  Result result;
  result.velocity = initial;
  const auto ordering_start = Clock::now();
  const auto ordering = prepare_sparse_ordering(particles.size(),pairs,options,result);
  result.preconditioner_setup_seconds += elapsed_seconds(ordering_start);
  SlurrySparse::LDLT sparse_factor;
#ifdef SLURRY_HAVE_EIGEN
  SlurryEigen::FullLU full_lu;
  SlurryEigen::IncompleteLU local_incomplete_lu(options.ilu_drop_tolerance,options.ilu_fill_factor);
  const bool use_incomplete_lu = options.preconditioner_backend == PreconditionerBackend::IncompleteLU;
  bool lu_ready = false, lu_disabled = false, refresh_lu = false;
  bool cross_step_factor = false;
  std::vector<ContactResponse> lu_factor_state;
  std::size_t removed_active_branches = 0;
  if (workspace && use_incomplete_lu) {
    if (workspace->dofs != 6*particles.size() || workspace->dt != dt || workspace->length != length ||
        workspace->drop_tolerance != options.ilu_drop_tolerance ||
        workspace->fill_factor != options.ilu_fill_factor || workspace->age >= 10)
      workspace->reset();
    if (!workspace->factor)
      workspace->factor = std::make_unique<SlurryEigen::IncompleteLU>(
          options.ilu_drop_tolerance,options.ilu_fill_factor);
    workspace->dofs = 6*particles.size(); workspace->dt = dt; workspace->length = length;
    workspace->drop_tolerance = options.ilu_drop_tolerance;
    workspace->fill_factor = options.ilu_fill_factor;
    lu_ready = workspace->ready;
    refresh_lu = workspace->refresh;
    cross_step_factor = lu_ready;
    if (lu_ready) {
      lu_factor_state.resize(pairs.size());
      auto remaining = workspace->branches;
      for (std::size_t k = 0; k < pairs.size(); ++k) {
        const auto found = remaining.find({pairs[k].i,pairs[k].j});
        if (found != remaining.end()) {
          lu_factor_state[k].active = found->second.first;
          lu_factor_state[k].sliding = found->second.second;
          remaining.erase(found);
        }
      }
      for (const auto &entry : remaining) if (entry.second.first) ++removed_active_branches;
    }
  }
  auto &incomplete_lu = workspace && use_incomplete_lu ? *workspace->factor : local_incomplete_lu;
#else
  if (options.preconditioner_backend == PreconditionerBackend::FullLU ||
      options.preconditioner_backend == PreconditionerBackend::IncompleteLU)
    throw std::invalid_argument("Full LU backend requires a build with Eigen support");
#endif
  bool sparse_ready = false, sparse_disabled = ordering.empty(), refresh_sparse = false;
  std::vector<ContactResponse> factor_state;
  Vector residual, rhs, direction, trial, trial_residual, jdirection;
  std::vector<ContactResponse> state, trial_state;
  double previous_residual = 0;
  for (int iteration = 0; iteration <= options.maximum_newton_iterations; ++iteration) {
    const double scale = friction_residual(particles,pairs,dt,length,result.velocity,residual,state);
    result.residual_norm = norm(residual);
    result.residual_scale = scale;
    result.relative_residual = result.residual_norm/std::max(scale,options.absolute_tolerance);
    result.newton_iterations = iteration;
    if (!std::isfinite(result.residual_norm))
      throw std::runtime_error("Friction solver produced a nonfinite force residual");
    if (result.residual_norm <= options.absolute_tolerance+options.relative_tolerance*scale) {
      fill_result(pairs,dt,length,result);
      if (workspace && workspace->ready) ++workspace->age;
      return result;
    }
    if (iteration == options.maximum_newton_iterations) break;
    rhs = residual;
    for (auto &q : rhs) for (double &value : q) value = -value;
    // At a Coulomb kink, an arbitrary generalized Jacobian can point into the
    // other branch. Backtracking then cannot recover descent. Select the
    // direction-consistent derivative before searching, while keeping
    // the physical force law and the final force-residual tolerance unchanged.
    auto linear_state = state;
    Options linear_options = options;
    if (options.adaptive_forcing) {
      // Conservative Eisenstat-Walker-type forcing. Early corrections may be
      // approximate, but near the root we tighten back towards the original
      // linear tolerance. The FINAL nonlinear force criterion above is fixed.
      const double ratio = previous_residual > 0 ? result.residual_norm/previous_residual : 0;
      const double ew = previous_residual > 0 ? 0.9*std::pow(ratio,1.5) : 0.05;
      const double force_scale = std::max(scale,options.absolute_tolerance/options.relative_tolerance);
      const double near_root = 0.1*std::sqrt(result.residual_norm/force_scale);
      linear_options.linear_relative_tolerance = std::max(options.linear_relative_tolerance,
          std::min({0.1,ew,near_root}));
    }
    previous_residual = result.residual_norm;
    bool strict_linear = linear_options.linear_relative_tolerance <= options.linear_relative_tolerance;
    const auto retry_strict = [&]() {
      if (strict_linear) return false;
      strict_linear = true;
      linear_options.linear_relative_tolerance = options.linear_relative_tolerance;
      linear_state = state;
      refresh_sparse = true;
#ifdef SLURRY_HAVE_EIGEN
      refresh_lu = true;
#endif
      ++result.forcing_retries;
      return true;
    };
    bool accepted = false;
    for (int branch = 0; branch < options.maximum_newton_iterations; ++branch) {
      const auto apply = [&](const Vector &q, Vector &out) {
        const auto start = Clock::now();
        friction_jacobian(particles,pairs,linear_state,dt,length,q,out,true);
        result.linear_apply_seconds += elapsed_seconds(start);
      };
      bool direct_solved = false;
#ifdef SLURRY_HAVE_EIGEN
      if ((options.preconditioner_backend == PreconditionerBackend::FullLU || use_incomplete_lu) &&
          !lu_disabled) {
        std::size_t changed = removed_active_branches;
        if (lu_ready)
          for (std::size_t k = 0; k < pairs.size(); ++k)
            if (lu_factor_state[k].active != linear_state[k].active ||
                lu_factor_state[k].sliding != linear_state[k].sliding) ++changed;
        const bool rebuild = !lu_ready || refresh_lu ||
            changed > std::max<std::size_t>(1,pairs.size()/4);
        if (rebuild) {
          const auto lu_start = Clock::now();
          try {
            const auto entries = exact_sparse_entries(particles,pairs,linear_state,dt,length);
            if (use_incomplete_lu) incomplete_lu.factor(static_cast<int>(6*particles.size()),entries);
            else full_lu.factor(static_cast<int>(6*particles.size()),entries);
            ++result.lu_factorizations;
            result.max_sparse_factor_nonzeros = std::max(result.max_sparse_factor_nonzeros,
                use_incomplete_lu ? incomplete_lu.nonzeros() : full_lu.nonzeros());
            lu_ready = true;
            refresh_lu = false;
            cross_step_factor = false;
            removed_active_branches = 0;
            lu_factor_state = linear_state;
            if (workspace && use_incomplete_lu) {
              workspace->ready = true; workspace->refresh = false; workspace->age = 0;
              workspace->branches.clear();
              for (std::size_t k = 0; k < pairs.size(); ++k)
                workspace->branches[{pairs[k].i,pairs[k].j}] = {linear_state[k].active,linear_state[k].sliding};
            }
          } catch (const std::runtime_error &) {
            ++result.lu_fallbacks; lu_disabled = true;
            if (workspace) workspace->reset();
          } catch (const std::bad_alloc &) {
            ++result.lu_fallbacks; lu_disabled = true;
            if (workspace) workspace->reset();
          }
          result.preconditioner_setup_seconds += elapsed_seconds(lu_start);
        }
        if (lu_ready && !lu_disabled) {
          try {
            const auto inverse = [&](const Vector &q, Vector &out) {
              const auto start = Clock::now();
              if (use_incomplete_lu) full_lu_inverse(incomplete_lu,q,out);
              else full_lu_inverse(full_lu,q,out);
              ++result.lu_solves;
              ++result.gmres_iterations; // Includes a bounded stale-cache attempt.
              result.preconditioner_solve_seconds += elapsed_seconds(start);
            };
            // Reused full/incomplete nonsymmetric factors are RIGHT PRECONDITIONERS only.
            // Every FGMRES multiplication uses the current exact Jacobian and
            // checks its true residual. Stale factors never define an accepted
            // Newton correction or modify the contact law.
            Options lu_options = linear_options;
            if (cross_step_factor)
              lu_options.maximum_gmres_iterations = std::min(lu_options.maximum_gmres_iterations,
                  options.ilu_refresh_iterations > std::numeric_limits<int>::max()/2 ?
                  std::numeric_limits<int>::max() : 2*options.ilu_refresh_iterations);
            int iterations = 0;
            try { iterations = fgmres(apply,inverse,rhs,lu_options,direction); }
            catch (const std::runtime_error &) {
              if (!cross_step_factor) throw;
              // A stale cross-step inverse receives a bounded attempt. Rebuild
              // for this exact current state before falling back to Nested.
              const auto refresh_start = Clock::now();
              incomplete_lu.factor(static_cast<int>(6*particles.size()),
                                   exact_sparse_entries(particles,pairs,linear_state,dt,length));
              result.preconditioner_setup_seconds += elapsed_seconds(refresh_start);
              ++result.lu_factorizations;
              result.max_sparse_factor_nonzeros = std::max(result.max_sparse_factor_nonzeros,incomplete_lu.nonzeros());
              cross_step_factor = false;
              removed_active_branches = 0;
              lu_factor_state = linear_state;
              if (workspace) {
                workspace->ready = true; workspace->refresh = false; workspace->age = 0;
                workspace->branches.clear();
                for (std::size_t k = 0; k < pairs.size(); ++k)
                  workspace->branches[{pairs[k].i,pairs[k].j}] = {linear_state[k].active,linear_state[k].sliding};
              }
              iterations = fgmres(apply,inverse,rhs,linear_options,direction);
            }
            refresh_lu = iterations > (use_incomplete_lu ? options.ilu_refresh_iterations : 40);
            if (workspace && use_incomplete_lu) workspace->refresh = refresh_lu;
            direct_solved = true;
          } catch (const std::runtime_error &) {
            ++result.lu_fallbacks;
            lu_disabled = true;
            if (workspace) workspace->reset();
          } catch (const std::bad_alloc &) {
            ++result.lu_fallbacks;
            lu_disabled = true;
            if (workspace) workspace->reset();
          }
        }
      }
#endif
      if (!direct_solved) {
      const auto setup_start = Clock::now();
      // A preconditioner need not equal the current Jacobian. Reuse one SPD
      // factor within this frozen-geometry timestep; exact forces, the exact
      // nonsymmetric Jacobian and the final residual test remain unchanged.
      // Refresh when many contact branches change or the preceding outer
      // linear solve needed >40 iterations. No factor survives a timestep.
      std::size_t changed = 0;
      if (sparse_ready)
        for (std::size_t k = 0; k < pairs.size(); ++k)
          if (factor_state[k].active != linear_state[k].active ||
              factor_state[k].sliding != linear_state[k].sliding) ++changed;
      const bool rebuild = !sparse_ready || refresh_sparse ||
          changed > std::max<std::size_t>(1,pairs.size()/4);
      if (!sparse_disabled && sparse_requested(options,particles.size()) && rebuild) {
        sparse_ready = factor_sparse(particles,pairs,linear_state,dt,length,options,
                                     ordering,sparse_factor,result);
        sparse_disabled = !sparse_ready;
        refresh_sparse = false;
        if (sparse_ready) factor_state = linear_state;
      } else if (sparse_ready) ++result.sparse_preconditioner_reuses;
      const bool use_sparse = sparse_ready;
      const Blocks factors = use_sparse ? Blocks{} :
          factor_blocks(friction_blocks(particles,pairs,linear_state,dt,length));
      result.preconditioner_setup_seconds += elapsed_seconds(setup_start);
      const auto precondition = [&](const Vector &q, Vector &out) {
        const auto start = Clock::now();
        if (use_sparse) {
          sparse_inverse(sparse_factor,q,out);
          ++result.sparse_preconditioner_solves;
        } else
          result.cg_iterations += friction_precondition(particles,pairs,linear_state,factors,dt,length,q,
                                                         options.maximum_cg_iterations,out);
        result.preconditioner_solve_seconds += elapsed_seconds(start);
      };
      int linear_iterations = 0;
      try { linear_iterations = fgmres(apply,precondition,rhs,linear_options,direction); }
      catch (const std::runtime_error &) {
        if (retry_strict()) continue;
        throw;
      }
      result.gmres_iterations += linear_iterations;
      refresh_sparse = linear_iterations > 40;
      }
      const int directional_changes = reselect_friction_kinks(pairs,state,direction,length,linear_state);
      result.kink_reselections += directional_changes;
      if (directional_changes != 0) continue;
      apply(direction,jdirection);
      const double descent = dot(residual,jdirection);
      if (!(descent < 0) || !std::isfinite(descent)) {
        if (retry_strict()) continue;
        throw std::runtime_error("Coupled friction Newton step is not a residual descent direction");
      }
      const double current_merit = 0.5*dot(residual,residual);
      double alpha = 1;
      for (int search = 0; search < options.maximum_line_search_iterations; ++search) {
        trial = result.velocity;
        for (std::size_t i = 0; i < trial.size(); ++i)
          for (int k = 0; k < 6; ++k) trial[i][k] += alpha*direction[i][k];
        friction_residual(particles,pairs,dt,length,trial,trial_residual,trial_state);
        const double merit = 0.5*dot(trial_residual,trial_residual);
        const double allowance = 32*std::numeric_limits<double>::epsilon()*
            std::max(current_merit,merit);
        // A roundoff-sized unchanged iterate is not progress: accepting it can
        // repeatedly reset the same wrong derivative until the iteration cap.
        if (std::isfinite(merit) && merit < current_merit-allowance &&
            merit <= current_merit+1.0e-4*alpha*descent+allowance) {
          // A loose solve that gives <10% residual reduction is retried at
          // the original linear tolerance before committing an iterate.
          if (!strict_linear && merit > 0.81*current_merit) break;
          result.velocity.swap(trial);
          accepted = true;
          break;
        }
        alpha *= 0.5;
      }
      if (accepted) break;
      if (retry_strict()) continue;
      const int changes = reselect_friction_kinks(pairs,state,direction,length,linear_state);
      result.kink_reselections += changes;
      if (changes == 0) break;
    }
    if (!accepted) {
      throw std::runtime_error("Coupled friction Newton line search failed; committed history unchanged");
    }
  }
  throw std::runtime_error("Coupled friction solver exceeded maximum Newton iterations");
}
} // namespace detail

// Without friction, solves the gradient of the strictly convex objective
//   1/2(q-q_inf)^T R(q-q_inf) + 1/2(Bq+b)^T D(Bq+b)
//   + sum_pairs[1/2 q_pair^T R_pair q_pair + affine_pair.q_pair + constant]
//   + k/(2dt)*min(g-hc+dt*n.(Bq+b),0)^2 - external.q.
// This branch uses an SPD active Hessian/block-PCG and an energy line search.
// Coulomb friction depends on the simultaneously solved N(q), so the coupled
// branch uses the exact nonsymmetric Jacobian/FGMRES and a force-residual line
// search instead of incorrectly assuming a convex global friction potential.
// Failures throw so the caller can reduce the timestep/retry or report the error.
inline Result solve(const std::vector<Particle> &particles,
                    const std::vector<Pair> &pairs, double dt, double reference_length,
                    const Options &options = Options{},
                    const std::vector<Vec6> &initial_guess = {},
                    SolverWorkspace *workspace = nullptr)
{
  detail::validate(particles,pairs,dt,reference_length,options);
  if (workspace && options.preconditioner_backend != PreconditionerBackend::IncompleteLU)
    workspace->reset();
  Result result;
  if (!initial_guess.empty() && initial_guess.size() != particles.size())
    throw std::invalid_argument("Implicit initial guess has incorrect particle count");
  result.velocity = initial_guess;
  if (result.velocity.empty())
    for (const auto &p : particles) result.velocity.push_back(p.target);
  for (const auto &q : result.velocity)
    for (double x : q)
      if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite implicit initial guess");
  if (options.preconditioner_backend == PreconditionerBackend::FullLU ||
      options.preconditioner_backend == PreconditionerBackend::IncompleteLU ||
      std::any_of(pairs.begin(),pairs.end(),[](const Pair &p) {
        return p.friction_coefficient > 0 && p.tangent_stiffness > 0;
      })) {
    try { return detail::solve_friction(particles,pairs,dt,reference_length,options,result.velocity,workspace); }
    catch (...) {
      if (workspace) workspace->reset();
      throw;
    }
  }
  const auto ordering_start = detail::Clock::now();
  const auto ordering = detail::prepare_sparse_ordering(particles.size(),pairs,options,result);
  result.preconditioner_setup_seconds += detail::elapsed_seconds(ordering_start);
  std::vector<unsigned char> active(pairs.size());
  detail::Vector gradient, rhs, direction, trial;
  for (int iteration = 0; iteration <= options.maximum_newton_iterations; ++iteration) {
    const double rhs_norm = detail::gradient(particles,pairs,dt,reference_length,
                                             result.velocity,gradient,active);
    result.residual_norm = detail::norm(gradient);
    result.residual_scale = rhs_norm;
    result.relative_residual = result.residual_norm/std::max(rhs_norm,options.absolute_tolerance);
    result.newton_iterations = iteration;
    if (!std::isfinite(result.residual_norm))
      throw std::runtime_error("Implicit solver produced a nonfinite force residual");
    if (result.residual_norm <= options.absolute_tolerance + options.relative_tolerance*rhs_norm) {
      detail::fill_result(pairs,dt,reference_length,result);
      return result;
    }
    if (iteration == options.maximum_newton_iterations) break;
    rhs = gradient;
    for (auto &q : rhs) for (double &x : q) x = -x;
    std::vector<ContactResponse> linear_state(pairs.size());
    for (std::size_t k = 0; k < pairs.size(); ++k) linear_state[k].active = active[k];
    const auto setup_start = detail::Clock::now();
    SlurrySparse::LDLT sparse_factor;
    const bool use_sparse = detail::factor_sparse(particles,pairs,linear_state,dt,
        reference_length,options,ordering,sparse_factor,result);
    result.preconditioner_setup_seconds += detail::elapsed_seconds(setup_start);
    if (use_sparse) {
      const auto apply = [&](const detail::Vector &q, detail::Vector &out) {
        const auto start = detail::Clock::now();
        detail::apply(particles,pairs,active,dt,reference_length,q,out);
        result.linear_apply_seconds += detail::elapsed_seconds(start);
      };
      const auto precondition = [&](const detail::Vector &q, detail::Vector &out) {
        const auto start = detail::Clock::now();
        detail::sparse_inverse(sparse_factor,q,out);
        ++result.sparse_preconditioner_solves;
        result.preconditioner_solve_seconds += detail::elapsed_seconds(start);
      };
      result.gmres_iterations += detail::fgmres(apply,precondition,rhs,options,direction);
    } else {
      const auto start = detail::Clock::now();
      result.cg_iterations += detail::cg(particles,pairs,active,dt,reference_length,
                                          rhs,options,direction);
      result.preconditioner_solve_seconds += detail::elapsed_seconds(start);
    }
    const double descent = detail::dot(gradient,direction);
    if (!(descent < 0) || !std::isfinite(descent))
      throw std::runtime_error("Implicit Newton direction is not finite descent");
    const long double current_energy = detail::energy(particles,pairs,dt,reference_length,result.velocity);
    double alpha = 1;
    bool accepted = false;
    for (int search = 0; search < options.maximum_line_search_iterations; ++search) {
      trial = result.velocity;
      for (std::size_t i = 0; i < trial.size(); ++i)
        for (int k = 0; k < 6; ++k) trial[i][k] += alpha*direction[i][k];
      const long double e = detail::energy(particles,pairs,dt,reference_length,trial);
      const long double allowance = 32*std::numeric_limits<double>::epsilon()*
          std::max(std::abs(current_energy),std::abs(e));
      if (std::isfinite(e) && e <= current_energy + 1.0e-4L*alpha*descent + allowance) {
        result.velocity.swap(trial);
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted) throw std::runtime_error("Implicit Newton line search failed");
  }
  throw std::runtime_error("Implicit normal-contact solver exceeded maximum Newton iterations");
}
} // namespace SlurryImplicit
#endif
