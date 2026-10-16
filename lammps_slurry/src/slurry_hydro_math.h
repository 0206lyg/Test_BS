// Exact isolated-ellipsoid Stokes resistance and spheroid Jeffery rotation.
// No pair interactions, stresslets, or far-field many-body interactions.
#ifndef SLURRY_HYDRO_MATH_H
#define SLURRY_HYDRO_MATH_H

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace SlurryHydro {

struct Resistance {
  std::array<double, 3> translation;
  std::array<double, 3> rotation;
};

namespace detail {

// Integrate after s=A^2*(t^-2-1), where A is the largest semi-axis.
// This avoids integrating a dimensional, semi-infinite interval. Long-double
// arithmetic resolves thin spheroids without subtracting nearly equal terms.
template <class Function>
long double adaptive_simpson(const Function &f, long double left, long double right,
                             long double fleft, long double fmid, long double fright,
                             long double coarse, long double tolerance, int remaining)
{
  const long double middle = (left + right) * 0.5L;
  const long double fl = f((left + middle) * 0.5L);
  const long double fr = f((middle + right) * 0.5L);
  const long double lsum = (middle - left) * (fleft + 4 * fl + fmid) / 6;
  const long double rsum = (right - middle) * (fmid + 4 * fr + fright) / 6;
  const long double fine = lsum + rsum;
  const long double delta = fine - coarse;
  const long double roundoff = 64 * std::numeric_limits<long double>::epsilon() *
      std::max(std::abs(fine), std::abs(coarse));
  if (std::abs(delta) <= std::max(15 * tolerance, roundoff)) return fine + delta / 15;
  if (remaining == 0)
    throw std::runtime_error("Spheroid hydrodynamic resistance quadrature did not converge");
  return adaptive_simpson(f, left, middle, fleft, fl, fmid, lsum, tolerance * 0.5L,
                          remaining - 1) +
      adaptive_simpson(f, middle, right, fmid, fr, fright, rsum, tolerance * 0.5L,
                       remaining - 1);
}

template <class Function> long double integral(const Function &f)
{
  const long double f0 = f(0), fm = f(0.5L), f1 = f(1);
  return adaptive_simpson(f, 0, 1, f0, fm, f1, (f0 + 4 * fm + f1) / 6, 1.0e-11L, 36);
}

}    // namespace detail

// Principal-axis resistance coefficients at unit dynamic viscosity. For a
// sphere of radius a these are 6*pi*a and 8*pi*a^3, respectively.
inline Resistance unit_resistance(const std::array<double, 3> &axes)
{
  for (double axis : axes)
    if (!(axis > 0) || !std::isfinite(axis))
      throw std::invalid_argument("Hydrodynamic semi-axes must be finite and positive");
  const long double scale = *std::max_element(axes.begin(), axes.end());
  std::array<long double, 3> ratio2{};
  for (int k = 0; k < 3; ++k) {
    const long double ratio = axes[k] / scale;
    ratio2[k] = ratio * ratio;
  }
  auto denominator = [&ratio2](long double t, std::array<long double, 3> &b) {
    // The form (1-t)*(1+t)+ratio2*t*t is stable near t=1.
    const long double t2 = t * t;
    for (int k = 0; k < 3; ++k) b[k] = (1 - t) * (1 + t) + ratio2[k] * t2;
    return std::sqrt(b[0] * b[1] * b[2]);
  };
  const long double chi = 2 / scale * detail::integral([&](long double t) {
    std::array<long double, 3> b{};
    return 1 / denominator(t, b);
  });
  std::array<long double, 3> alpha{};
  for (int k = 0; k < 3; ++k) {
    alpha[k] = 2 / (scale * scale * scale) * detail::integral([&](long double t) {
      std::array<long double, 3> b{};
      const long double d = denominator(t, b);
      return t * t / (b[k] * d);
    });
  }

  constexpr long double pi = 3.141592653589793238462643383279502884L;
  Resistance result{};
  for (int k = 0; k < 3; ++k) {
    const int j = (k + 1) % 3, l = (k + 2) % 3;
    const long double a2 = static_cast<long double>(axes[k]) * axes[k];
    const long double b2 = static_cast<long double>(axes[j]) * axes[j];
    const long double c2 = static_cast<long double>(axes[l]) * axes[l];
    result.translation[k] = static_cast<double>(16 * pi / (chi + a2 * alpha[k]));
    result.rotation[k] =
        static_cast<double>((16 * pi / 3) * (b2 + c2) / (b2 * alpha[j] + c2 * alpha[l]));
    if (!(result.translation[k] > 0) || !std::isfinite(result.translation[k]) ||
        !(result.rotation[k] > 0) || !std::isfinite(result.rotation[k]))
      throw std::runtime_error("Invalid spheroid hydrodynamic resistance");
  }
  return result;
}

// Force-free spheroid angular velocity in u_inf=(shear_rate*y,0,0).
// p is the space-frame unit vector along the local z symmetry axis.
inline std::array<double, 3> jeffery_omega(const std::array<double, 3> &p,
                                         double equatorial_axis, double polar_axis,
                                         double shear_rate)
{
  // Scale before squaring to avoid overflow/underflow from dimensional lengths.
  const double scale = std::max(equatorial_axis, polar_axis);
  const double a = equatorial_axis / scale, c = polar_axis / scale;
  const double lambda = (c * c - a * a) / (c * c + a * a);
  const double epx = 0.5 * shear_rate * p[1];
  const double epy = 0.5 * shear_rate * p[0];
  return {-lambda * p[2] * epy, lambda * p[2] * epx,
          -0.5 * shear_rate + lambda * (p[0] * epy - p[1] * epx)};
}

}    // namespace SlurryHydro
#endif
