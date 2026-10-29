// Adapter/export only: production operators with recorded AR20 defaults.
// This executable is not an independent Stokes solver.
#include "slurry_scattering_provider.h"
#include "slurry_spd_table17.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace sc = SlurryScattering;
constexpr double default_ar = 20.;
constexpr double default_ti = .0049245303592113559;
constexpr double default_tj = .041010932014513654;
constexpr double default_az = 1.7016960206944713;

template<class Matrix> void array(std::ostream &out, const Matrix &matrix) {
  out << '[';
  for (int i = 0; i < matrix.rows(); ++i) {
    if (i) out << ',';
    out << '[';
    for (int j = 0; j < matrix.cols(); ++j) {
      if (j) out << ',';
      out << matrix(i,j);
    }
    out << ']';
  }
  out << ']';
}

int main(int argc, char **argv) {
  try {
    double h = 0., ar = default_ar, c_nm = 150.;
    double ti = default_ti, tj = default_tj, az = default_az;
    std::string total_path, scattering_path;
    for (int k = 1; k < argc; ++k) {
      const std::string option = argv[k];
      if (option == "--gap-over-c" && k+1 < argc) h = std::stod(argv[++k]);
      else if (option == "--aspect-ratio" && k+1 < argc) ar = std::stod(argv[++k]);
      else if (option == "--c-nm" && k+1 < argc) c_nm = std::stod(argv[++k]);
      else if (option == "--theta-i" && k+1 < argc) ti = std::stod(argv[++k]);
      else if (option == "--theta-j" && k+1 < argc) tj = std::stod(argv[++k]);
      else if (option == "--phi" && k+1 < argc) az = std::stod(argv[++k]);
      else if (option == "--total-in" && k+1 < argc) total_path = argv[++k];
      else if (option == "--scattering-model" && k+1 < argc) scattering_path = argv[++k];
      else throw std::runtime_error("Usage: --gap-over-c h [--aspect-ratio AR --c-nm nm --theta-i radians --theta-j radians --phi radians] [--total-in row_major_17x17.txt] [--scattering-model basis_or_hybrid.txt]");
    }
    if (!(h > 0.) || !std::isfinite(h))
      throw std::runtime_error("Positive finite effective gap required");
    if (!(ar >= 1.) || !std::isfinite(ar) || !(c_nm > 0.) || !std::isfinite(c_nm) ||
        !std::isfinite(ti) || !std::isfinite(tj) || !std::isfinite(az) ||
        ti < 0. || ti > std::acos(-1.) || tj < 0. || tj > std::acos(-1.))
      throw std::runtime_error("Finite oblate AR >= 1, positive c_nm, polar angles in [0,pi], finite azimuth required");
    const bool recorded = ar == default_ar && c_nm == 150. && ti == default_ti &&
                          tj == default_tj && az == default_az;
    const std::array<sc::Vec3,2> axes{{
      {std::sin(ti), 0., std::cos(ti)},
      {std::sin(tj)*std::cos(az), std::sin(tj)*std::sin(az), std::cos(tj)}
    }};
    const auto bodies = sc::geometry(axes, h, ar);
    SlurrySPDCoordinates17::Shape shape(ar);
    const sc::Mat17 isolated = shape.self(axes, h);
    const sc::Mat17 leading = SlurryFastSingular::leading17(axes, h, ar);
    sc::Result scattering;
    int scattering_degree = 0;
    if (!scattering_path.empty()) {
      sc::Provider provider;
      provider.load(scattering_path);
      if (provider.basis.aspect_ratio != ar)
        throw std::runtime_error("Scattering basis aspect ratio differs from requested geometry");
      // Export the existing far slot's actual two-body operator, including its
      // raw diagnostics. The assessor decides whether it is usable; neither
      // the legacy finite-gap switch nor a forced reciprocity repair is used.
      scattering = provider.basis.evaluate(bodies);
      scattering_degree = provider.basis.degree;
      if (!scattering.grand.allFinite() || !std::isfinite(scattering.reciprocity) ||
          !std::isfinite(scattering.linear_residual))
        throw std::runtime_error("Nonfinite production scattering operator");
    }
    std::cout << std::setprecision(17)
              << "{\"schema\":\"" << (recorded ? "ar20_production_operator_export_v1"
                                                          : "spheroid_production_operator_export_v1") << "\","
              << "\"independent_reference\":false,"
              << "\"convention_id\":\"slurry_full17_c_eta_1_v1\","
              << "\"aspect_ratio\":" << ar << ",\"c_nm\":" << c_nm
              << ",\"a_nm\":" << ar*c_nm << ",\"h_over_c\":" << h
              << ",\"theta_i\":" << ti << ",\"theta_j\":" << tj << ",\"phi\":" << az
              << ",\"units\":\"c=eta=1; input U-uinf,a(Omega-Omegainf),aE5; output Fsource,Tsource/a,S_intrinsic/a\","
              << "\"isolated17\":";
    array(std::cout, isolated);
    std::cout << ",\"isolated_local11\":";
    array(std::cout, shape.isolated.isolated_local);
    std::cout << ",\"isolated_bodies17\":[";
    for (int k = 0; k < 2; ++k) {
      sc::Mat11 rotation = sc::Mat11::Zero();
      rotation.block<3,3>(0,0) = bodies[k].rotation;
      rotation.block<3,3>(3,3) = bodies[k].rotation;
      rotation.bottomRightCorner<5,5>() = sc::strain_rotation(bodies[k].rotation);
      const sc::Mat11 block = rotation * shape.isolated.isolated_local * rotation.transpose();
      sc::Mat17 one = sc::Mat17::Zero();
      one.block<6,6>(6*k,6*k) = block.topLeftCorner<6,6>();
      one.block<6,5>(6*k,12) = block.topRightCorner<6,5>();
      one.block<5,6>(12,6*k) = block.bottomLeftCorner<5,6>();
      one.bottomRightCorner<5,5>() = block.bottomRightCorner<5,5>();
      if (k) std::cout << ',';
      array(std::cout, one);
    }
    std::cout << "],\"leading17\":";
    array(std::cout, leading);
    std::cout << ",\"near_total17\":";
    array(std::cout, sc::Mat17(isolated+leading));
    std::cout << ",\"near_model\":\"isolated_plus_raw_janoschek_leading_v1\","
                 "\"near_model_scope\":\"candidate total; incomplete finite and subleading terms; no eigenvalue clipping\","
                 "\"far_isolated17\":";
    array(std::cout, isolated);
    std::cout << ",\"far_model\":\"independent_isolated_spheroids_exact_self_v1\"";
    if (!scattering_path.empty()) {
      std::cout << ",\"far_scattering17\":";
      array(std::cout, sc::Mat17(.5*(scattering.grand+scattering.grand.transpose())));
      std::cout << ",\"far_scattering_raw17\":";
      array(std::cout, scattering.grand);
      std::cout << ",\"far_scattering_diagnostics\":{\"degree\":" << scattering_degree
                << ",\"raw_reciprocity\":" << scattering.reciprocity
                << ",\"linear_relative_residual\":" << scattering.linear_residual << '}';
    }
    Eigen::Matrix<double,2,3> centers;
    centers.row(0) = bodies[0].center.transpose();
    centers.row(1) = bodies[1].center.transpose();
    std::cout << ",\"centers\":";
    array(std::cout, centers);

    if (!total_path.empty()) {
      sc::Mat17 total;
      std::ifstream input(total_path);
      for (int i = 0; i < 17; ++i)
        for (int j = 0; j < 17; ++j)
          if (!(input >> total(i,j))) throw std::runtime_error("Expected289 row-major numbers");
      if (!total.allFinite()) throw std::runtime_error("Nonfinite encoding input");
      const double asymmetry = (total-total.transpose()).norm()/std::max(total.norm(),1e-300);
      if (asymmetry > 1e-10)
        throw std::runtime_error("Encoding input must be symmetrized after recording raw reciprocity");
      const sc::Mat17 logarithm = SlurrySPDTable17::encode_total(shape, axes, h, total);
      std::stringstream storage;
      storage << std::setprecision(17);
      int count = 0;
      for (int i = 0; i < 17; ++i)
        for (int j = 0; j <= i; ++j) {
          storage << logarithm(i,j) << ' ';
          ++count;
        }
      sc::Mat17 decoded_log;
      for (int i = 0; i < 17; ++i)
        for (int j = 0; j <= i; ++j) {
          double value;
          if (!(storage >> value)) throw std::runtime_error("153-entry decode failed");
          decoded_log(i,j) = decoded_log(j,i) = value;
        }
      const sc::Mat17 decoded = SlurrySPDTable17::decode_total(shape, axes, h, decoded_log);
      std::cout << ",\"encoded_values\":" << count << ",\"decoded_total17\":";
      array(std::cout, decoded);
    }
    std::cout << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Operator export: " << error.what() << '\n';
    return 1;
  }
}
