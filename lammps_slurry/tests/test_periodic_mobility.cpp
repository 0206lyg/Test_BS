#include "slurry_periodic_mobility.h"
#include <chrono>
#include <iomanip>
#include <iostream>
using namespace SlurryPeriodic;
static void require(bool ok, const char *what) {
  if (!ok)
    throw std::runtime_error(what);
}
int main(int argc, char **argv) {
  std::cout << std::setprecision(16);
  if (argc > 1 && std::string(argv[1]) == "factors") {
    for (double x : {.001, .01, .08, .5, 1., 4., 20., 80.}) {
      auto g = tail_factors(x), h = hard_tail_factors(x);
      std::cout << x << " " << g.first << " " << g.second << " " << h.first << " " << h.second
                << " " << sine_integral(2 * x) << "\n";
    }
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "dump") {
    int N;
    double xi, kmax;
    Mat3 H;
    std::cin >> N >> xi >> kmax;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        std::cin >> H(i, j);
    std::vector<Body> b(N);
    for (auto &x : b) {
      for (int j = 0; j < 3; j++)
        std::cin >> x.x[j];
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          std::cin >> x.shape(i, j);
    }
    Settings s;
    s.backend = Backend::Direct;
    s.xi = xi;
    s.kmax = kmax;
    s.local_high_frequency_tail = xi != 0;
    s.gaussian_split = xi >= 0;
    if (xi < 0)
      s.xi = 0;
    Mobility m;
    m.reset(b, H, s);
    auto M = m.dense();
    std::cout << m.mode_count() << "\n" << M << "\n";
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "bench") {
    int N = 108;
    double ar = 20, phi = .24, L = std::cbrt(N * 4 * pi * ar * ar / (3 * phi));
    Mat3 H = Mat3::Identity() * L;
    std::vector<Body> b(N);
    for (int i = 0; i < N; i++) {
      b[i].x = Vec3((i % 6 + .5) * L / 6, ((i / 6) % 6 + .5) * L / 6, (i / 36 + .5) * L / 3);
      b[i].shape = Vec3(ar * ar, ar * ar, 1).asDiagonal();
    }
    Settings s;
    s.backend = Backend::Direct;
    s.xi = 2 / ar;
    if (argc > 2)
      s.threads = std::stoi(argv[2]);
    Mobility m;
    auto t0 = std::chrono::steady_clock::now();
    m.reset(b, H, s);
    auto t1 = std::chrono::steady_clock::now();
    Eigen::VectorXd g = Eigen::VectorXd::Random(11 * N), out;
    for (int it = 0; it < 5; it++)
      out = m.apply(g);
    auto t2 = std::chrono::steady_clock::now();
    std::cout << "N " << N << " modes " << m.mode_count() << " bytes " << m.bytes() << " setup_s "
              << std::chrono::duration<double>(t1 - t0).count() << " matvec_s "
              << std::chrono::duration<double>(t2 - t1).count() / 5 << " power " << g.dot(out)
              << "\n";
    return 0;
  }
  Settings s;
  s.backend = Backend::Direct;
  s.xi = 1.2;
  s.self_order = 128;
  Mat3 H;
  H << 9, 3, 1, 0, 8, 2, 0, 0, 10;
  std::vector<Body> b(2);
  b[0].x = Vec3(.3, .8, 1.7);
  b[1].x = Vec3(3., 2., 2.5);
  b[0].shape = Vec3(4, 4, 1).asDiagonal();
  Mat3 Q = Eigen::AngleAxisd(.63, Vec3(1, 2, 3).normalized()).toRotationMatrix();
  b[1].shape = Q * b[0].shape * Q.transpose();
  Mobility m;
  m.reset(b, H, s);
  auto M = m.dense();
  double symmetry = (M - M.transpose()).norm() / M.norm();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
  require(symmetry < 1e-13, "Reciprocity");
  require(es.eigenvalues()[0] > 0, "Positive definiteness");
  auto shifted = b;
  shifted[1].x += H * Vec3(2, -3, 1);
  Mobility shifted_m;
  shifted_m.reset(shifted, H, s);
  double image = (M - shifted_m.dense()).norm() / M.norm();
  require(image < 1e-12, "Periodic image invariance");
  Mat3 U;
  U << 1, -2, 0, 0, 1, 1, 0, 0, 1;
  Mobility basis_m;
  basis_m.reset(b, H * U, s);
  double basis = (M - basis_m.dense()).norm() / M.norm();
  require(basis < 1e-12, "Lattice basis invariance");
  Settings parallel_settings = s;
  parallel_settings.threads = 4;
  Mobility parallel_m;
  parallel_m.reset(b, H, parallel_settings);
  double parallel = (M - parallel_m.dense()).norm() / M.norm();
  require(parallel < 1e-12, "Parallel reduction equivalence");
  // Joint rotation includes the lattice and the orthonormal STF coordinates.
  auto rb = b;
  for (auto &v : rb) {
    v.x = Q * v.x;
    v.shape = Q * v.shape * Q.transpose();
  }
  Mobility rotation_m;
  rotation_m.reset(rb, Q * H, s);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(22, 22);
  R.block<11, 11>(0, 0) = rotation11(Q);
  R.block<11, 11>(11, 11) = rotation11(Q);
  double rotation = (rotation_m.dense() - R * M * R.transpose()).norm() / M.norm();
  require(rotation < 1e-11, "Rotation covariance");
  // Full (xi=0) self is the known sphere Galerkin resistance inverse.
  Mat11 sphere = self_tail(Mat3::Identity(), 0, 1, 64), exact = Mat11::Zero();
  exact.diagonal() << 1 / (6 * pi), 1 / (6 * pi), 1 / (6 * pi), 1 / (8 * pi), 1 / (8 * pi),
      1 / (8 * pi), 3 / (20 * pi), 3 / (20 * pi), 3 / (20 * pi), 3 / (20 * pi), 3 / (20 * pi);
  double sphere_err = (sphere - exact).norm() / exact.norm();
  require(sphere_err < 1e-12, "Sphere self normalization");
  std::cout << "symmetry " << symmetry << " min_eigenvalue " << es.eigenvalues()[0] << " image "
            << image << " basis " << basis << " rotation " << rotation << " parallel " << parallel
            << " sphere_self " << sphere_err << " modes " << m.mode_count() << "\n";
}
