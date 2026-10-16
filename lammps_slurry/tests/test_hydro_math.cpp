#include "slurry_hydro_math.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {
void close(double result, double expected, double tolerance)
{
  if (std::abs(result - expected) > tolerance * std::max(std::abs(expected), 1.0e-300))
    throw std::runtime_error("Hydrodynamic analytical comparison failed");
}
}

int main()
{
  constexpr double pi = 3.14159265358979323846;
  // Dimensional sphere limits validate both integrals and normalization.
  for (double radius : {1.0, 1.65e-6}) {
    const auto r = SlurryHydro::unit_resistance({radius, radius, radius});
    for (int k = 0; k < 3; ++k) {
      close(r.translation[k], 6 * pi * radius, 1.0e-10);
      close(r.rotation[k], 8 * pi * radius * radius * radius, 1.0e-10);
    }
  }
  // Disk limits: broadside translation 16*a; edgewise 32*a/3;
  // rotation about any axis approaches 32*a^3/3 at zero thickness.
  const auto disk = SlurryHydro::unit_resistance({1.0, 1.0, 0.001});
  close(disk.translation[2], 16, 2.0e-6);
  close(disk.translation[0], 32.0 / 3, 1.0e-3);
  close(disk.rotation[0], 32.0 / 3, 2.0e-6);
  close(disk.rotation[2], 32.0 / 3, 2.0e-3);
  // Positivity, symmetry, and expected directional drag for oblates/prolates.
  for (double polar : {0.125, 0.5, 1.0, 2.0, 8.0}) {
    const auto r = SlurryHydro::unit_resistance({1.0, 1.0, polar});
    close(r.translation[0], r.translation[1], 1.0e-12);
    close(r.rotation[0], r.rotation[1], 1.0e-12);
    if (polar < 1 && r.translation[2] <= r.translation[0])
      throw std::runtime_error("Oblate broadside drag must exceed edgewise drag");
    if (polar > 1 && r.translation[2] >= r.translation[0])
      throw std::runtime_error("Prolate axial drag must be smaller than transverse drag");
    // Jeffery planar angle velocity at three directions, with independent formula.
    for (double angle : {0.0, pi / 4, pi / 2}) {
      const std::array<double, 3> p{std::cos(angle), std::sin(angle), 0};
      const auto w = SlurryHydro::jeffery_omega(p, 1, polar, 3);
      const double expected = -3 * (p[0] * p[0] + polar * polar * p[1] * p[1]) /
          (1 + polar * polar);
      close(w[2], expected, 1.0e-12);
      if (w[0] != 0 || w[1] != 0)
        throw std::runtime_error("Unexpected Jeffery angular velocity out of planar orbit");
    }
  }
  std::puts("PASS: sphere and thin-disk resistance, oblate/prolate anisotropy, Jeffery rotation");
}
