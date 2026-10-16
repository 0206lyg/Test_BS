#include "slurry_stress_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {
using SlurryStress::Mat3;
using SlurryStress::Vec3;
constexpr double pi = 3.141592653589793238462643383279502884;
const Mat3 identity{1, 0, 0, 0, 1, 0, 0, 0, 1};

void close(double actual, double expected, double relative = 2.0e-10,
           double absolute = 0)
{
  if (!std::isfinite(actual) ||
      std::abs(actual - expected) > absolute + relative * std::abs(expected)) {
    std::fprintf(stderr, "actual %.17g, expected %.17g\n", actual, expected);
    throw std::runtime_error("Isolated stresslet comparison failed");
  }
}

template <std::size_t N>
void close_array(const std::array<double, N> &actual, const std::array<double, N> &expected,
                 double tolerance = 2.0e-10)
{
  double scale = 0;
  for (double x : expected) scale = std::max(scale, std::abs(x));
  for (std::size_t i = 0; i < N; ++i) close(actual[i], expected[i], tolerance, tolerance * scale);
}

double contract(const Mat3 &a, const Mat3 &b)
{
  double value = 0;
  for (int i = 0; i < 9; ++i) value += a[i] * b[i];
  return value;
}

double dot(const Vec3 &a, const Vec3 &b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Mat3 multiply(const Mat3 &a, const Mat3 &b)
{
  Mat3 c{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) c[3 * i + j] += a[3 * i + k] * b[3 * k + j];
  return c;
}

Vec3 rotate(const Mat3 &a, const Vec3 &v)
{
  Vec3 result{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) result[i] += a[3 * i + j] * v[j];
  return result;
}

Mat3 rotate(const Mat3 &a, const Mat3 &tensor)
{
  const Mat3 transpose{a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]};
  return multiply(multiply(a, tensor), transpose);
}

Mat3 rotation(Vec3 axis, double angle)
{
  const double length = std::sqrt(dot(axis, axis));
  for (double &x : axis) x /= length;
  const double c = std::cos(angle), s = std::sin(angle);
  const double x = axis[0], y = axis[1], z = axis[2];
  return {c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s,
          y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s,
          z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)};
}

// Independent elementary closed forms of Einarsson et al. (2015), Appendix A.
// Tests deliberately do not duplicate the implementation's quadrature.
Vec3 oblate_closed_form(long double l)
{
  const long double t = 1 - l * l;
  const long double g = std::sqrt(t) * std::acos(l);
  const long double A = -g * (2 * l * l + 1) + 3 * l * t;
  const long double B = 3 * g * l - t * (l * l + 2);
  const long double C = -3 * g + l * t * (5 - 2 * l * l);
  return {static_cast<double>(-16 * pi * t * t * t / (9 * A)),
          static_cast<double>(-16 * pi * l * t * t * t / (3 * (l * l + 1) * B)),
          static_cast<double>(-32 * pi * t * t * t / (3 * C))};
}
}    // namespace

int main()
{
  // The dimensional sphere stresslet must give Einstein's 2.5*phi viscosity.
  for (double a : {1.0, 1.65e-6}) {
    const auto coefficients = SlurryStress::unit_stresslet({a, a, a});
    const double eta = 0.37, shear = 13;
    const auto result = SlurryStress::stresslet(coefficients, identity, eta, shear, {1, 2, 3});
    const double reference = (20 * pi / 3) * a * a * a;
    for (double mode : coefficients.mode) close(mode, reference);
    close(result.total[1], reference * eta * shear * 0.5);
    close(result.total[3], result.total[1]);
    const double volume = 4 * pi * a * a * a / 3;
    close(result.total[1] / (volume * shear), 2.5 * eta);
    for (double x : result.rotational) close(x, 0, 0, 1.0e-50);
  }

  // Oblate analytical benchmarks, including the production aspect ratio 8.25.
  for (double l : {0.001, 1.0 / 8.25, 0.2, 0.5, 0.8}) {
    const auto coefficients = SlurryStress::unit_stresslet({1, 1, l});
    close_array(coefficients.mode, oblate_closed_form(l));
  }
  const auto aspect2 = SlurryStress::unit_stresslet({1, 1, 0.5});
  close_array(aspect2.mode, Vec3{10.011469147671463, 8.642608183478222, 13.682907462098928});
  close(aspect2.unit_rotation[0], 14.220501793939086);
  // A nearly spherical shape tests cancellation rather than a sphere branch.
  const auto almost_sphere = SlurryStress::unit_stresslet({1, 1, std::sqrt(1 - 1.0e-10)});
  for (double mode : almost_sphere.mode) close(mode, 20 * pi / 3, 1.0e-10);
  const auto scaled = SlurryStress::unit_stresslet({2.3e-6, 2.3e-6, 1.15e-6});
  for (int k = 0; k < 3; ++k) close(scaled.mode[k], aspect2.mode[k] * std::pow(2.3e-6, 3));

  // Mixed strain, polar axis z: prescribed omega=fluid vorticity differs from
  // torque-free Jeffery rotation. Independent fixed-rotation coefficient.
  const Mat3 mixed{0, 0, 0.5, 0, 0, 0, 0.5, 0, 0};
  const Vec3 zero{};
  const auto fixed = SlurryStress::stresslet(aspect2, identity, 1, mixed, zero, zero);
  close(fixed.total[2] / mixed[2], 11.202298506387258);
  close(fixed.rotational[2] / mixed[2],
        aspect2.unit_rotation[0] * 0.6 * 0.6 / 2);
  const auto free = SlurryStress::stresslet(aspect2, identity, 1, mixed, zero, fixed.jeffery_omega);
  close_array(free.total, free.free);
  for (double x : free.hydro_torque) close(x, 0, 0, 1.0e-14);
  // Axial spin changes torque, but cannot change an axisymmetric stresslet.
  Vec3 spin = fixed.jeffery_omega;
  spin[2] += 7;
  const auto spinning = SlurryStress::stresslet(aspect2, identity, 1, mixed, zero, spin);
  close_array(spinning.total, free.total);

  // Objectivity: a rigid rotation of the complete experiment rotates its
  // stress tensor and torque, without changing the dissipation.
  const Mat3 orientation = rotation({1, 2, -0.3}, 0.83);
  const Mat3 observer = rotation({-1, 0.7, 2}, -1.17);
  const Mat3 E{0.3, -0.7, 0.2, -0.7, 0.8, 0.4, 0.2, 0.4, -1.1};
  const Vec3 omega_fluid{0.2, -0.1, -0.8}, omega{0.9, 0.4, -0.2};
  const double eta = 0.61;
  const auto base = SlurryStress::stresslet(aspect2, orientation, eta, E, omega_fluid, omega);
  const auto transformed = SlurryStress::stresslet(aspect2, multiply(observer, orientation), eta,
      rotate(observer, E), rotate(observer, omega_fluid), rotate(observer, omega));
  close_array(transformed.free, rotate(observer, base.free));
  close_array(transformed.rotational, rotate(observer, base.rotational));
  close_array(transformed.total, rotate(observer, base.total));
  close_array(transformed.hydro_torque, rotate(observer, base.hydro_torque));
  close_array(transformed.jeffery_omega, rotate(observer, base.jeffery_omega));
  close(base.total[0] + base.total[4] + base.total[8], 0, 0, 1.0e-12);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) close(base.total[3 * i + j], base.total[3 * j + i]);

  // Positive total dissipation, with the independently measurable torque
  // work included. S:E alone need not be positive for imposed rotation.
  Vec3 relative_omega{}, slip{};
  for (int i = 0; i < 3; ++i) {
    relative_omega[i] = omega[i] - omega_fluid[i];
    slip[i] = omega[i] - base.jeffery_omega[i];
  }
  const double power = contract(base.total, E) - dot(base.hydro_torque, relative_omega);
  const double free_power = contract(base.free, E);
  const double rotation_power = -dot(base.hydro_torque, slip);
  close(power, free_power + rotation_power);
  if (!(free_power > 0 && rotation_power > 0 && power > 0))
    throw std::runtime_error("Isolated spheroid must dissipate positive power");

  // Reciprocity, tested by independent perturbations of angular velocity
  // and strain: dS/domega : dE = -domega . dT_H/dE.
  const Vec3 dw{0.13, -0.31, 0.23};
  const Mat3 dE{0.12, 0.2, -0.3, 0.2, -0.05, 0.4, -0.3, 0.4, -0.07};
  Vec3 perturbed_omega{};
  Mat3 perturbed_E{};
  for (int i = 0; i < 3; ++i) perturbed_omega[i] = omega[i] + dw[i];
  for (int i = 0; i < 9; ++i) perturbed_E[i] = E[i] + dE[i];
  const auto omega_change =
      SlurryStress::stresslet(aspect2, orientation, eta, E, omega_fluid, perturbed_omega);
  const auto strain_change =
      SlurryStress::stresslet(aspect2, orientation, eta, perturbed_E, omega_fluid, omega);
  Mat3 delta_stress{};
  Vec3 delta_torque{};
  for (int i = 0; i < 9; ++i) delta_stress[i] = omega_change.total[i] - base.total[i];
  for (int i = 0; i < 3; ++i) delta_torque[i] = strain_change.hydro_torque[i] - base.hydro_torque[i];
  close(contract(delta_stress, dE), -dot(dw, delta_torque));

  bool rejected_triaxial = false;
  try {
    (void) SlurryStress::unit_stresslet({1, 0.9, 0.5});
  } catch (const std::invalid_argument &) {
    rejected_triaxial = true;
  }
  if (!rejected_triaxial) throw std::runtime_error("Triaxial shape must not use spheroid stresslet");
  std::puts("PASS: exact spheroid stresslet, Einstein limit, rotation coupling, objectivity, power, reciprocity");
}
