#include "slurry_periodic_mobility.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
using namespace SlurryPeriodic;
static void require(bool ok, const char *what) { if (!ok) throw std::runtime_error(what); }
static double seconds(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
static std::vector<Body> particles(int n, double ar, double L) {
  std::mt19937 random(1729); std::uniform_real_distribution<double> u(0., 1.);
  std::vector<Body> out(n);
  for (auto &b : out) {
    b.x = L * Vec3(u(random), u(random), u(random));
    const Mat3 Q = Eigen::AngleAxisd(2 * pi * u(random), Vec3(u(random)-.5, u(random)-.5, u(random)-.5).normalized()).toRotationMatrix();
    b.shape = Q * Vec3(ar * ar, ar * ar, 1.).asDiagonal() * Q.transpose();
  }
  return out;
}
int main(int argc, char **argv) {
  std::cout << std::setprecision(12);
  if (argc > 1 && std::string(argv[1]) == "bench") {
    const int N = argc > 2 ? std::stoi(argv[2]) : 500;
    const double ar = argc > 3 ? std::stod(argv[3]) : 2.;
    const int threads = argc > 4 ? std::stoi(argv[4]) : 4;
    const bool direct = argc > 5 && std::string(argv[5]) == "direct";
    const double L = std::cbrt(N * 4 * pi * ar * ar / (3 * .24));
    Mat3 H = Mat3::Identity() * L; H(0, 2) = .17 * L;
    auto bodies = particles(N, ar, L);
    Settings cfg; cfg.xi = 2 / ar; cfg.threads = threads; cfg.self_order = 128;
    cfg.backend = direct ? Backend::Direct : Backend::Mesh;
    Mobility m; const auto start = std::chrono::steady_clock::now(); m.reset(bodies, H, cfg);
    const double setup = seconds(start);
    const auto plan_builds = m.mesh_fft_plan_builds();
    const auto warm_start = std::chrono::steady_clock::now();
    m.reset(bodies, H, cfg);
    const double warm_setup = seconds(warm_start);
    require(m.mesh_fft_plan_builds() == plan_builds, "Benchmark reuses FFTW plans");
    Eigen::VectorXd g = Eigen::VectorXd::Random(11 * N);
    for (int i = 0; i < N; ++i) g.segment<8>(11*i+3) *= ar;
    const auto apply_start = std::chrono::steady_clock::now();
    Eigen::VectorXd out;
    for (int j = 0; j < 3; ++j) out = m.apply(g);
    const double apply = seconds(apply_start) / 3;
    const auto grid = m.mesh_dimensions();
    std::cout << "backend=" << (direct ? "direct" : "mesh") << " N=" << N << " AR=" << ar
              << " threads=" << threads << " modes=" << m.mode_count() << " grid=" << grid[0] << ',' << grid[1] << ',' << grid[2]
              << " quadrature_order=" << m.mesh_quadrature_order() << " entries=" << m.mesh_stencil_entries()
              << " bytes=" << m.bytes() << " setup_s=" << setup << " warm_setup_s=" << warm_setup << " apply_s=" << apply
              << " fft=" << m.mesh_fft_backend() << " fft_plan_builds=" << m.mesh_fft_plan_builds()
              << " power=" << g.dot(out) << '\n';
    require(out.allFinite() && g.dot(out) > 0, "Benchmark positivity"); return 0;
  }
  for (double ar : {2., 20., 30.}) {
    const int N = 3; const double L = std::cbrt(N * 4 * pi * ar * ar / (3 * .12));
    Mat3 H = Mat3::Identity() * L; H(0, 2) = .37 * L; H(1, 2) = -.13 * L;
    auto bodies = particles(N, ar, L);
    Settings direct_cfg; direct_cfg.xi = 2 / ar; direct_cfg.self_order = 128;
    direct_cfg.threads = 2; direct_cfg.backend = Backend::Direct;
    Settings mesh_cfg = direct_cfg; mesh_cfg.backend = Backend::Mesh;
    Mobility direct, mesh; direct.reset(bodies, H, direct_cfg); mesh.reset(bodies, H, mesh_cfg);
    Eigen::VectorXd scale = Eigen::VectorXd::Ones(11*N);
    for (int i = 0; i < N; ++i) scale.segment<8>(11*i+3).setConstant(ar);
    const Eigen::MatrixXd ref = scale.asDiagonal() * direct.dense() * scale.asDiagonal();
    const Eigen::MatrixXd got = scale.asDiagonal() * mesh.dense() * scale.asDiagonal();
    // All setup changes below reuse one live mobility object as in time
    // integration. Geometry changes must update the stencil without planning
    // new FFTs when array dimensions and thread count remain the same.
    const auto first_builds = mesh.mesh_fft_plan_builds();
    const auto first_dimensions = mesh.mesh_dimensions();
    mesh.reset(bodies, H, mesh_cfg);
    require(mesh.mesh_fft_plan_builds() == first_builds, "FFTW plan reused at identical geometry");
    const Eigen::VectorXd probe = scale.cwiseProduct(Eigen::VectorXd::LinSpaced(11*N, .1, .9));
    const Eigen::VectorXd original_response = mesh.apply(probe);
    require((scale.asDiagonal()*original_response - got*scale.cwiseInverse().asDiagonal()*probe).norm() /
                (scale.asDiagonal()*original_response).norm() < 5e-14, "Mesh repeat reset response");
    require(mesh.apply(Eigen::VectorXd::Zero(11*N)).isZero(0.), "Exact zero mobility");
    auto moved = bodies;
    moved[0].x += H * Vec3(.013, -.017, .009);
    mesh.reset(moved, H, mesh_cfg);
    require(mesh.mesh_dimensions() == first_dimensions &&
            mesh.mesh_fft_plan_builds() == first_builds, "FFTW plan reused after particle motion");
    Mobility fresh;
    fresh.reset(moved, H, mesh_cfg);
    const Eigen::VectorXd moved_response = mesh.apply(probe);
    require((moved_response-fresh.apply(probe)).norm()/moved_response.norm() < 5e-14,
            "Persistent and fresh mesh agree after particle motion");
    Settings serial_cfg = mesh_cfg; serial_cfg.threads = 1;
    mesh.reset(moved, H, serial_cfg);
    require((mesh.apply(probe)-moved_response).norm()/moved_response.norm() < 5e-14,
            "Serial and parallel spread responses agree");
#ifdef SLURRY_HAVE_FFTW
    require(first_builds == 1 && mesh.mesh_fft_plan_builds() == first_builds + 1,
            "FFTW plan rebuilt when thread count changes");
#endif
    // Reuse a larger grid, then switch through the explicit direct backend.
    const Mat3 bigger = 2. * H;
    mesh.reset(moved, bigger, mesh_cfg);
    require(mesh.mesh_dimensions() != first_dimensions, "Grid resize exercised");
    fresh.reset(moved, bigger, mesh_cfg);
    const Eigen::VectorXd enlarged_response = fresh.apply(probe);
    require((mesh.apply(probe)-enlarged_response).norm()/enlarged_response.norm() < 5e-14,
            "FFTW plan follows resized grid storage");
    mesh.reset(bodies, H, direct_cfg);
    mesh.reset(bodies, H, mesh_cfg);
    require((mesh.apply(probe)-original_response).norm()/original_response.norm() < 5e-14,
            "Mesh restored after switching to direct reference");
    const double error = (got-ref).norm()/ref.norm();
    const double symmetry = (got-got.transpose()).norm()/got.norm();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(got);
    require(error < 1e-5, "Mesh/direct scaled mobility accuracy");
    require(symmetry < 3e-13, "Mesh exact adjoint symmetry");
    require(es.eigenvalues().minCoeff() > 0, "Mesh positive mobility eigenvalues");
    double cross_error = 0, column_error = 0;
    for (int j = 0; j < 11*N; ++j)
      column_error = std::max(column_error, (got.col(j)-ref.col(j)).norm()/ref.col(j).norm());
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) if (i != j)
      cross_error = std::max(cross_error, (got.block<11,11>(11*i,11*j)-ref.block<11,11>(11*i,11*j)).norm()/ref.block<11,11>(11*i,11*j).norm());
    require(column_error < 2e-5 && cross_error < 1e-4, "Force torque stresslet and cross-body accuracy");
    auto images = bodies; images[1].x += H * Vec3(3., -2., 4.);
    Mobility periodic; periodic.reset(images, H, mesh_cfg);
    const Eigen::MatrixXd im = scale.asDiagonal() * periodic.dense() * scale.asDiagonal();
    const double image_error = (im-got).norm()/got.norm();
    require(image_error < 2e-12, "Mesh periodic image invariance");
    auto translated = bodies;
    for (auto &b : translated) b.x += H * Vec3(.1237, -.3143, .2719);
    Mobility shifted; shifted.reset(translated, H, mesh_cfg);
    const Eigen::MatrixXd tr = scale.asDiagonal() * shifted.dense() * scale.asDiagonal();
    const double translation_error = (tr-got).norm()/got.norm();
    require(translation_error < 2e-5, "Mesh grid-phase translation accuracy");
    std::cout << "AR=" << ar << " relative=" << error << " column=" << column_error
              << " cross=" << cross_error << " symmetry=" << symmetry
              << " eigmin=" << es.eigenvalues().minCoeff() << " image=" << image_error
              << " translation=" << translation_error << " entries=" << mesh.mesh_stencil_entries() << '\n';
  }
  // Both monopole and derivative moments retain ellipsoid geometry; neither
  // the FFT nor the tests replace an ellipsoid by a point or spherical radius.
  return 0;
}

