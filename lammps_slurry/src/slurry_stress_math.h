// Exact isolated-spheroid stresslet in an incompressible linear Stokes flow.
// This is a one-body hydrodynamic term, not a pair or many-body stresslet.
#ifndef SLURRY_STRESS_MATH_H
#define SLURRY_STRESS_MATH_H

#include "slurry_hydro_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace SlurryStress {

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;    // row major, body-to-laboratory rotation

struct Coefficients {
  Vec3 axes{};
  Vec3 mode{};             // torque-free strain eigenvalues, units m^3
  Vec3 unit_rotation{};    // rotational resistance / viscosity, units m^3
  double jeffery_parameter = 0;
};

struct Result {
  Mat3 free{};              // torque-free Jeffery stresslet, units N m
  Mat3 rotational{};        // correction for actual angular velocity
  Mat3 total{};
  Vec3 hydro_torque{};      // fluid torque on particle, units N m
  Vec3 jeffery_omega{};
};

namespace detail {
inline double dot(const Vec3 &a, const Vec3 &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline Vec3 cross(const Vec3 &a, const Vec3 &b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
}    // namespace detail

// Sources and normalization:
// Einarsson et al., Phys. Fluids 27, 063301 (2015), Appendix A, Eq. (63)
// and Table 3; https://doi.org/10.1063/1.4921543
// Open author manuscript: https://arxiv.org/html/1504.02309v1
// The modal coefficients below are the torque-free reduction of that exact
// spheroid solution. The positive angular-integral representation is
// algebraically equivalent to its logarithm/arccos formulas; it avoids their
// cancellation at aspect ratio 1. It is valid for both oblate and prolate
// spheroids. It does NOT implement a general triaxial ellipsoid stresslet.
//
// For l=c/a<1, t=1-l^2, g=sqrt(t)*acos(l), the independent closed forms are
// mode[0]/a^3 = -16*pi*t^3/[9*(-g*(2*l^2+1)+3*l*t)],
// mode[1]/a^3 = -16*pi*l*t^3/[3*(l^2+1)*(3*g*l-t*(l^2+2))],
// mode[2]/a^3 = -32*pi*t^3/[3*(-3*g+l*t*(5-2*l^2))].
// Calling the three inner bracket expressions A,B,C, the positive integrals
// evaluated below satisfy D0=-A/(2*t^3), D1=-(l^2+1)*B/(l*t^3),
// D2=-C/(2*t^3). At l=1 they are 2/15,4/5,4/5 respectively. For l>1
// the closed forms continue with g=-sqrt(l^2-1)*acosh(l).
inline Coefficients unit_stresslet(const Vec3 &axes)
{
  for (double axis : axes)
    if (!(axis > 0) || !std::isfinite(axis))
      throw std::invalid_argument("Stresslet semi-axes must be finite and positive");
  if (std::abs(axes[0] - axes[1]) > 1.0e-12 * std::max(axes[0], axes[1]))
    throw std::invalid_argument("Isolated stresslet requires a spheroid with axes a,a,c");

  Coefficients result{};
  result.axes = axes;
  result.unit_rotation = SlurryHydro::unit_resistance(axes).rotation;
  const long double a = (static_cast<long double>(axes[0]) + axes[1]) * 0.5L;
  const long double l = static_cast<long double>(axes[2]) / a;
  const long double l2 = l * l;
  result.jeffery_parameter = static_cast<double>((l2 - 1) / (l2 + 1));
  constexpr long double pi = 3.141592653589793238462643383279502884L;
  const long double a3 = a * a * a;
  for (int mode = 0; mode < 3; ++mode) {
    const long double integral = SlurryHydro::detail::integral([&](long double u) {
      const long double u2 = u * u;
      const long double one_minus_u2 = (1 - u) * (1 + u);
      const long double w = one_minus_u2 + l2 * u2;
      long double numerator;
      if (mode == 0)
        numerator = u2 * one_minus_u2;
      else if (mode == 1)
        numerator = (2 * u2 - 0.75L) * (2 * u2 - 0.75L) + 0.4375L;
      else
        numerator = one_minus_u2 * (1 + u2);
      return numerator / (w * std::sqrt(w));
    });
    const long double prefactor = mode == 0 ? 8 * pi / 9 : 16 * pi / 3;
    result.mode[mode] = static_cast<double>(a3 * prefactor / integral);
    if (!(result.mode[mode] > 0) || !std::isfinite(result.mode[mode]))
      throw std::runtime_error("Invalid isolated spheroid stresslet coefficient");
  }
  return result;
}

// E must be symmetric and trace-free; omega_fluid is half the fluid vorticity.
// The returned sign is the particle contribution to bulk suspension stress:
// for a sphere, S=(20*pi/3)*viscosity*a^3*E, giving Einstein [eta]=5/2.
// Let J:E=lambda*p x (E*p), omega_J=omega_fluid+J:E, and
// T_H=-R_r*(omega-omega_J). Lorentz reciprocity gives
// S=S_free+J^T*T_H. In particular the cross product below is T_H x p.
// This follows equivalently by differentiating the Rayleigh function
// 1/2*(omega-omega_J).R_r*(omega-omega_J)+1/2*E:C_free:E
// with respect to E at fixed actual omega and omega_fluid.
inline Result stresslet(const Coefficients &coefficients, const Mat3 &body_to_lab,
                        double viscosity, const Mat3 &E, const Vec3 &omega_fluid,
                        const Vec3 &actual_omega)
{
  if (!(viscosity > 0) || !std::isfinite(viscosity))
    throw std::invalid_argument("Stresslet viscosity must be finite and positive");
  Vec3 p{body_to_lab[2], body_to_lab[5], body_to_lab[8]};
  const double pnorm = std::sqrt(detail::dot(p, p));
  if (!(pnorm > 0) || !std::isfinite(pnorm))
    throw std::invalid_argument("Invalid stresslet orientation");
  for (double &x : p) x /= pnorm;
  Vec3 Ep{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) Ep[i] += E[3 * i + j] * p[j];
  const double s = detail::dot(p, Ep);
  Vec3 q{};
  for (int i = 0; i < 3; ++i) q[i] = Ep[i] - s * p[i];
  const double lambda = coefficients.jeffery_parameter;
  const Vec3 p_cross_Ep = detail::cross(p, Ep);
  Result result{};
  Vec3 slip{};
  for (int i = 0; i < 3; ++i) {
    result.jeffery_omega[i] = omega_fluid[i] + lambda * p_cross_Ep[i];
    slip[i] = actual_omega[i] - result.jeffery_omega[i];
  }
  const double axial_slip = detail::dot(p, slip);
  const double rperp = 0.5 * (coefficients.unit_rotation[0] + coefficients.unit_rotation[1]);
  const double rparallel = coefficients.unit_rotation[2];
  for (int i = 0; i < 3; ++i)
    result.hydro_torque[i] =
        -viscosity * (rperp * slip[i] + (rparallel - rperp) * axial_slip * p[i]);
  const Vec3 torque_cross_p = detail::cross(result.hydro_torque, p);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      const int ij = 3 * i + j;
      const double E0 = 1.5 * s * (p[i] * p[j] - (i == j ? 1.0 / 3 : 0));
      const double E1 = p[i] * q[j] + q[i] * p[j];
      const double E2 = E[ij] - E0 - E1;
      result.free[ij] = viscosity * (coefficients.mode[0] * E0 +
          coefficients.mode[1] * E1 + coefficients.mode[2] * E2);
      result.rotational[ij] =
          0.5 * lambda * (torque_cross_p[i] * p[j] + p[i] * torque_cross_p[j]);
      result.total[ij] = result.free[ij] + result.rotational[ij];
    }
  return result;
}

// Simple shear u_inf=(shear_rate*y,0,0), in laboratory coordinates.
inline Result stresslet(const Coefficients &coefficients, const Mat3 &body_to_lab,
                        double viscosity, double shear_rate, const Vec3 &actual_omega)
{
  const Mat3 E{0, 0.5 * shear_rate, 0, 0.5 * shear_rate, 0, 0, 0, 0, 0};
  const Vec3 omega_fluid{0, 0, -0.5 * shear_rate};
  return stresslet(coefficients, body_to_lab, viscosity, E, omega_fluid, actual_omega);
}

}    // namespace SlurryStress
#endif
