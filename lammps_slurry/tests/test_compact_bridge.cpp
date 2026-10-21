#include "slurry_compact_bridge.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  using namespace SlurryCompactBridge;
  if (argc != 2) {
    std::cerr << "usage: test_compact_bridge golden.dat\n";
    return 2;
  }
  std::ifstream input(argv[1]);
  if (!input) {
    std::cerr << "cannot read fixture\n";
    return 2;
  }
  int count;
  input >> count;
  double worst = 0, geometry_worst = 0;
  for (int trial = 0; trial < count; ++trial) {
    std::string name;
    double eta, length, floor;
    input >> name >> eta >> length >> floor;
    gr::Body bi, bj;
    for (auto *b : {&bi, &bj}) {
      for (double &x : b->axes)
        input >> x;
      for (double &x : b->rotation)
        input >> x;
      for (double &x : b->position)
        input >> x;
    }
    double gap_ref, elln, ellt, Lp, w, s;
    input >> gap_ref >> elln >> ellt >> Lp >> w >> s;
    Mat22 expected;
    for (int i = 0; i < 22; ++i)
      for (int j = 0; j < 22; ++j)
        input >> expected(i, j);
    if (!input) {
      std::cerr << "invalid fixture " << name << "\n";
      return 2;
    }
    auto got = evaluate(bi, bj, eta, length, floor);
    // Body labels are bookkeeping only, including the prescribed F/T/S basis.
    auto reverse = evaluate(bj, bi, eta, length, floor);
    Mat22 permuted;
    permuted.topLeftCorner<11, 11>() = reverse.resistance.bottomRightCorner<11, 11>();
    permuted.bottomRightCorner<11, 11>() = reverse.resistance.topLeftCorner<11, 11>();
    permuted.topRightCorner<11, 11>() = reverse.resistance.bottomLeftCorner<11, 11>();
    permuted.bottomLeftCorner<11, 11>() = reverse.resistance.topRightCorner<11, 11>();
    const double exchange =
        (permuted - got.resistance).norm() / std::max(got.resistance.norm(), 1e-300);
    if (exchange > 2e-10) {
      std::cerr << "exchange FAIL " << name << " " << exchange << "\n";
      return 1;
    }
    double err = (got.resistance - expected).norm() / std::max(expected.norm(), 1e-300);
    double geom =
        std::max({std::abs(got.gap - gap_ref) / length, std::abs(got.ell_n - elln) / length,
                  std::abs(got.ell_t - ellt) / length, std::abs(got.patch_width - Lp) / length,
                  std::abs(got.weight - w), std::abs(got.normal_scale - s)});
    worst = std::max(worst, err);
    geometry_worst = std::max(geometry_worst, geom);
    // Mixed SI units cannot be eigentested raw: use the model's normalized coordinates.
    Eigen::Matrix<double, 22, 1> d = Eigen::Matrix<double, 22, 1>::Ones();
    d.head<3>().setConstant(length);
    d.segment<3>(11).setConstant(length);
    Mat22 normalized =
        d.asDiagonal() * got.resistance * d.asDiagonal() / (eta * length * length * length);
    Eigen::SelfAdjointEigenSolver<Mat22> es(normalized);
    double negative = es.eigenvalues()[0] / std::max(normalized.norm(), 1e-300);
    std::cout << std::setprecision(12) << name << " matrix=" << err << " geometry=" << geom
              << " min_scaled=" << negative << " w=" << got.weight << "\n";
    if (err > 2e-9 || geom > 2e-10 || negative < -2e-12) {
      std::cerr << "FAIL " << name << "\n";
      return 1;
    }
    // The exact implementation scales q by the same reference length as the model.
    if (got.effective_gap < std::max(got.gap, floor) * (1 - 1e-14))
      return 1;
  }
  std::cout << "PASS " << count << " fixtures, max_matrix_relative=" << worst
            << " max_geometry_scaled=" << geometry_worst << "\n";
  return 0;
}
