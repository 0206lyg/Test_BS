#include "slurry_scattering_provider.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// Frozen C++ regression baseline of the repaired source at film grid 161.
// Each fixture passed the 161-versus-321 source-convergence check before the
// intentional baseline update; provenance is recorded beside the fixture.
// This is not independent physical evidence. Separate Stokes data/tolerances
// remain unchanged in test_independent_pair_reference.cpp.
int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::runtime_error("Expected model and golden-input paths");
    SlurryScattering::Provider provider;
    provider.load(argv[1]);
    std::ifstream input(argv[2]);
    int count = 0;
    if (!(input >> count) || count != 3)
      throw std::runtime_error("Invalid hybrid golden input");
    double worst = 0.;
    for (int k = 0; k < count; ++k) {
      std::array<SlurryScattering::Vec3, 2> axes;
      for (auto &axis : axes) for (int j = 0; j < 3; ++j) input >> axis[j];
      double h = 0.; input >> h;
      SlurryScattering::Mat11 expected;
      for (int i = 0; i < 11; ++i) for (int j = 0; j < 11; ++j)
        input >> expected(i, j);
      if (!input || !expected.allFinite()) throw std::runtime_error("Truncated golden input");
      const auto actual = provider.correction(axes, h, 0., 2*k, 2*k+1);
      const double error = (actual-expected).norm()/expected.norm();
      if (!std::isfinite(error) || error > 1e-8)
        throw std::runtime_error("Native hybrid disagrees with frozen regression baseline");
      worst = std::max(worst, error);
    }
    std::cout << std::setprecision(16) << "PASS hybrid golden maximum error " << worst << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
