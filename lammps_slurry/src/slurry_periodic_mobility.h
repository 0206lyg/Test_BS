// Periodic 11-moment ellipsoid Galerkin mobility.
// Moments act ON THE FLUID. q=(U,Omega,-E), g=(F,T,S_STF).
// The optional local high-frequency closure is an explicitly documented
// approximation: high-frequency cross-body/image blocks are omitted.
#ifndef SLURRY_PERIODIC_MOBILITY_H
#define SLURRY_PERIODIC_MOBILITY_H
#include <Eigen/Dense>
#include "slurry_fft_plans.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace SlurryPeriodic {
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Mat11 = Eigen::Matrix<double, 11, 11>;
constexpr double pi = 3.141592653589793238462643383279502884;
struct Body {
  Vec3 x = Vec3::Zero();
  Mat3 shape = Mat3::Identity();
};
enum class Backend { Mesh, Direct };
struct Settings {
  Backend backend = Backend::Mesh;
  // Mesh accuracy controls are separate from the Fourier truncation tolerance.
  int mesh_window = 10;
  double mesh_alpha = .6;
  int mesh_quadrature_order = 0; // zero chooses from largest kmax*semiaxis
  double mesh_grid_oversampling = 1.;
  std::size_t mesh_max_nodes = 16777216;
  double eta = 1.;
  // xi is inverse length, not an angular or lattice-index cutoff.
  // xi==0 selects the unsmoothed, truncated exact Fourier sum; kmax required.
  double xi = 0.;
  double spectral_tolerance = 1.e-8;
  double kmax = 0.;
  bool local_high_frequency_tail = true;
  bool gaussian_split = true;
  int self_order = 256;
  int threads = 1;
  std::size_t max_modes = 1000000;
};
inline std::array<Mat3, 5> stf_basis() {
  std::array<Mat3, 5> b;
  for (auto &m : b)
    m.setZero();
  b[0](0, 0) = 1 / std::sqrt(2.);
  b[0](1, 1) = -1 / std::sqrt(2.);
  b[1](0, 0) = b[1](1, 1) = 1 / std::sqrt(6.);
  b[1](2, 2) = -2 / std::sqrt(6.);
  b[2](0, 1) = b[2](1, 0) = 1 / std::sqrt(2.);
  b[3](0, 2) = b[3](2, 0) = 1 / std::sqrt(2.);
  b[4](1, 2) = b[4](2, 1) = 1 / std::sqrt(2.);
  return b;
}
inline Mat3 cross(const Vec3 &n) {
  Mat3 m;
  m << 0, -n.z(), n.y(), n.z(), 0, -n.x(), -n.y(), n.x(), 0;
  return m;
}
inline Eigen::Matrix<double, 3, 8> moment_map(const Vec3 &n) {
  Eigen::Matrix<double, 3, 8> b;
  b.leftCols<3>() = -.5 * cross(n);
  auto s = stf_basis();
  for (int j = 0; j < 5; j++)
    b.col(3 + j) = s[j] * n;
  return b;
}
inline std::pair<double, double> tail_factors(double b) {
  if (!(b >= 0) || !std::isfinite(b))
    throw std::invalid_argument("Invalid Ewald tail argument");
  if (b == 0)
    return {1., 1.};
  const double b2 = b * b, invsqrtpi = 1 / std::sqrt(pi);
  // Small-argument series avoids cancellation in the dipole integral.
  if (b < .08) {
    const double f_low =
        invsqrtpi * b * (1 - b2 / 6 + b2 * b2 / 30 - b2 * b2 * b2 / 168 + b2 * b2 * b2 * b2 / 1080);
    const double d_low =
        invsqrtpi * b * b2 * (1. / 6 - b2 / 20 + 3 * b2 * b2 / 280 - b2 * b2 * b2 / 540);
    return {1 - f_low, 1 - d_low};
  }
  const double e = std::exp(-b2), om = -std::expm1(-b2);
  const double f = std::erfc(b) + invsqrtpi * om / b;
  const double d = std::erfc(b) + invsqrtpi * (3 * om / b - 2 * (om - b2 * e) / (b * b2));
  return {f, d};
}
// Sine integral via a small-x convergent series and the E1(ix) continued fraction.
inline double sine_integral(double x) {
  if (x < 0)
    return -sine_integral(-x);
  if (x <= 2) {
    double term = x, sum = x;
    for (int k = 1; k < 80; k++) {
      term *= -x * x / ((2 * k) * (2 * k + 1.));
      double add = term / (2 * k + 1.);
      sum += add;
      if (std::abs(add) < 2e-16 * std::abs(sum))
        break;
    }
    return sum;
  }
  using C = std::complex<double>;
  C b(1, x), c(1e100, 0), d = 1. / b, h = d;
  for (int i = 1; i < 400; i++) {
    double a = -double(i) * i;
    b += 2.;
    d = 1. / (a * d + b);
    c = b + a / c;
    C delta = c * d;
    h *= delta;
    if (std::abs(delta - C(1, 0)) < 4e-16)
      break;
  }
  h *= C(std::cos(x), -std::sin(x));
  return pi / 2 + h.imag();
}
inline std::pair<double, double> hard_tail_factors(double x) {
  if (!(x >= 0) || !std::isfinite(x))
    throw std::invalid_argument("Invalid hard tail argument");
  if (x == 0)
    return {1., 1.};
  double x2 = x * x;
  if (x < .08) {
    double f_low =
        2 / pi * x *
        (1 - x2 / 9 + 2 * x2 * x2 / 225 - x2 * x2 * x2 / 2205 + 2 * x2 * x2 * x2 * x2 / 127575);
    double d_low =
        6 / pi * x * x2 * (1. / 27 - x2 / 225 + x2 * x2 / 3675 - 4 * x2 * x2 * x2 / 382725);
    return {1 - f_low, 1 - d_low};
  }
  const double si = sine_integral(2 * x), co = std::cos(2 * x), sn = std::sin(2 * x);
  const double f = 1 - 2 * si / pi + (1 - co) / (pi * x);
  const double d =
      1 - 2 * si / pi - 2 * sn / (pi * x2) + (3 - co) / (pi * x) + (1 - co) / (pi * x * x2);
  if (f <= 0 || d <= 0)
    throw std::runtime_error("Hard tail integral lost positivity");
  return {f, d};
}
inline std::pair<std::vector<double>, std::vector<double>> gauss_legendre(int n) {
  if (n < 4)
    throw std::invalid_argument("self_order must be >=4");
  std::vector<double> x(n), w(n);
  for (int i = 0; i < (n + 1) / 2; i++) {
    double z = std::cos(pi * (i + .75) / (n + .5)), pp = 0;
    for (int iter = 0; iter < 50; iter++) {
      double p0 = 1, p1 = z;
      for (int j = 2; j <= n; j++) {
        double p = ((2 * j - 1) * z * p1 - (j - 1) * p0) / j;
        p0 = p1;
        p1 = p;
      }
      pp = n * (z * p1 - p0) / (z * z - 1);
      double next = z - p1 / pp;
      if (std::abs(next - z) < 4e-16) {
        z = next;
        break;
      }
      z = next;
    }
    // Re-evaluate derivative at the accepted root.
    double p0 = 1, p1 = z;
    for (int j = 2; j <= n; j++) {
      double p = ((2 * j - 1) * z * p1 - (j - 1) * p0) / j;
      p0 = p1;
      p1 = p;
    }
    pp = n * (z * p1 - p0) / (z * z - 1);
    x[i] = -z;
    x[n - 1 - i] = z;
    w[i] = w[n - 1 - i] = 2 / ((1 - z * z) * pp * pp);
  }
  return {x, w};
}
// Positive angular quadrature, radial integration analytic. General shape;
// callers may cache/rotate body-frame results for repeated identical bodies.
inline Mat11 self_tail(const Mat3 &A, double xi, double eta, int order = 64, bool gaussian = true) {
  Mat11 out = Mat11::Zero();
  auto rule = gauss_legendre(order);
  const bool axisymmetric =
      std::abs(A(0, 0) - A(1, 1)) < 1e-13 * A.norm() &&
      (A - A.diagonal().asDiagonal().toDenseMatrix()).norm() < 1e-13 * A.norm();
  const int nphi = axisymmetric ? 16 : 4 * order;
  for (int i = 0; i < order; i++)
    for (int j = 0; j < nphi; j++) {
      double u = rule.first[i], s = std::sqrt(std::max(0., 1 - u * u)), p = 2 * pi * j / nphi;
      Vec3 n(s * std::cos(p), s * std::sin(p), u);
      double a = std::sqrt(n.dot(A * n));
      auto tail = gaussian ? tail_factors(2 * xi * a) : hard_tail_factors(xi * a);
      Mat3 P = Mat3::Identity() - n * n.transpose();
      auto B = moment_map(n);
      double pref = rule.second[i] * (2 * pi / nphi) / (16 * pi * pi * eta);
      out.topLeftCorner<3, 3>().noalias() += pref * tail.first / a * P;
      out.bottomRightCorner<8, 8>().noalias() +=
          pref * 3 * tail.second / (a * a * a) * (B.transpose() * P * B);
    }
  return out;
}
inline Eigen::Matrix<double, 11, 11> rotation11(const Mat3 &Q) {
  Mat11 R = Mat11::Zero();
  R.topLeftCorner<3, 3>() = Q;
  R.block<3, 3>(3, 3) = Q;
  auto B = stf_basis();
  for (int a = 0; a < 5; a++)
    for (int b = 0; b < 5; b++)
      R(6 + a, 6 + b) = (B[a].array() * (Q * B[b] * Q.transpose()).array()).sum();
  return R;
}

#include "slurry_periodic_mesh.h"

class Mobility {
public:
  struct Mode {
    Vec3 k, n;
    double weight;
    Eigen::Matrix<double, 3, 8> B;
  };
  void reset(const std::vector<Body> &bodies, const Mat3 &cell, const Settings &settings) {
    cfg_ = settings;
    body_ = bodies;
    H_ = cell;
    const double V = cell.determinant();
    if (!(V > 0) || !cell.allFinite() || !(cfg_.eta > 0) || !std::isfinite(cfg_.eta) ||
        cfg_.xi < 0 || !std::isfinite(cfg_.xi))
      throw std::invalid_argument("Invalid periodic cell, eta or xi");
    if (!(cfg_.spectral_tolerance > 0 && cfg_.spectral_tolerance < 1))
      throw std::invalid_argument("spectral_tolerance must lie in (0,1)");
    if (cfg_.xi == 0 && cfg_.gaussian_split && cfg_.local_high_frequency_tail)
      throw std::invalid_argument("Gaussian local tail requires positive xi");
    double kmax =
        cfg_.kmax > 0 ? cfg_.kmax : 2 * cfg_.xi * std::sqrt(-std::log(cfg_.spectral_tolerance));
    if (!(kmax > 0) || !std::isfinite(kmax))
      throw std::invalid_argument("A finite positive reciprocal kmax is required");
    kmax_ = kmax;
    const Mat3 reciprocal = 2 * pi * cell.inverse().transpose();
    Eigen::Vector3i bound;
    for (int j = 0; j < 3; j++) {
      const double r = kmax * cell.col(j).norm() / (2 * pi);
      if (r > 10000)
        throw std::invalid_argument("Reciprocal enumeration bound is too large");
      bound[j] = int(std::ceil(r));
    }
    mode_.clear();
    for (int a = -bound[0]; a <= bound[0]; a++)
      for (int b = -bound[1]; b <= bound[1]; b++)
        for (int c = -bound[2]; c <= bound[2]; c++) {
          // One representative of +/-k. The factor two is in the weight.
          if (a < 0 || (a == 0 && b < 0) || (a == 0 && b == 0 && c <= 0))
            continue;
          Vec3 k = reciprocal * Vec3(a, b, c);
          double k2 = k.squaredNorm();
          if (k2 > kmax * kmax * (1 + 1e-13))
            continue;
          double filter =
              cfg_.gaussian_split && cfg_.xi > 0 ? std::exp(-k2 / (4 * cfg_.xi * cfg_.xi)) : 1.;
          Vec3 n = k / std::sqrt(k2);
          mode_.push_back({k, n, 2 * filter / (cfg_.eta * V * k2), moment_map(n)});
          if (mode_.size() > cfg_.max_modes)
            throw std::runtime_error(
                "Periodic mobility exceeds max_modes; increase explicitly or reduce xi/kmax");
        }
    if (mode_.empty())
      throw std::invalid_argument("Reciprocal cutoff contains no nonzero modes");
    const std::size_t N = body_.size(), K = mode_.size();
    if (cfg_.backend == Backend::Direct)
      coeff_.resize(K * N);
    else
      std::vector<Coeff>().swap(coeff_);
    tails_.resize(N);
    self_.assign(N, Mat11::Zero());
    std::vector<std::pair<Vec3, Mat11>> tail_cache, full_cache;
    for (std::size_t i = 0; i < N; i++) {
      const auto &A = body_[i].shape;
      if (!body_[i].x.allFinite() || !A.allFinite() ||
          (A - A.transpose()).norm() > 1e-12 * std::max(1., A.norm()))
        throw std::invalid_argument("Invalid particle shape/position");
      Eigen::SelfAdjointEigenSolver<Mat3> es(A);
      if (es.info() != Eigen::Success || es.eigenvalues().minCoeff() <= 0)
        throw std::invalid_argument("Particle shape must be SPD");
      // Integrate in principal coordinates to avoid orientation-dependent self quadrature.
      Mat3 Q = es.eigenvectors();
      Vec3 axes2 = es.eigenvalues();
      if (std::abs(axes2[1] - axes2[2]) < 1e-12 * axes2[2]) {
        Mat3 oldQ = Q;
        Q.col(0) = oldQ.col(1);
        Q.col(1) = oldQ.col(2);
        Q.col(2) = oldQ.col(0);
        axes2 = Vec3(es.eigenvalues()[1], es.eigenvalues()[2], es.eigenvalues()[0]);
      }
      if (Q.determinant() < 0)
        Q.col(0) = -Q.col(0);
      Mat11 R = rotation11(Q), body_tail = Mat11::Zero();
      if (cfg_.local_high_frequency_tail) {
        bool found = false;
        for (const auto &entry : tail_cache)
          if ((entry.first - axes2).norm() < 1e-12 * axes2.norm()) {
            body_tail = entry.second;
            found = true;
            break;
          }
        if (!found) {
          body_tail = self_tail(axes2.asDiagonal(), cfg_.gaussian_split ? cfg_.xi : kmax, cfg_.eta,
                                cfg_.self_order, cfg_.gaussian_split);
          tail_cache.push_back({axes2, body_tail});
        }
      }
      tails_[i] = R * body_tail * R.transpose();
      self_[i] = tails_[i];
      if (cfg_.backend == Backend::Mesh) {
        // This is only the isolated continuum preconditioner block. The
        // physical operator is the mesh Gram operator plus the tail below;
        // no analytic/discrete self subtraction is made.
        Mat11 full = Mat11::Zero();
        bool found = false;
        for (const auto &entry : full_cache)
          if ((entry.first - axes2).norm() < 1e-12 * axes2.norm()) {
            full = entry.second; found = true; break;
          }
        if (!found) {
          full = self_tail(axes2.asDiagonal(), 0., cfg_.eta, cfg_.self_order);
          full_cache.push_back({axes2, full});
        }
        self_[i] = R * full * R.transpose();
      }
    }
    if (cfg_.backend == Backend::Mesh) {
      mesh_.reset(body_, H_, cfg_, kmax_, mode_);
      return;
    }
    mesh_.clear();
    int nt = 1;
#ifdef _OPENMP
    nt = std::max(1, cfg_.threads);
#endif
    // Each mode owns a separate coefficient row. Accumulate self blocks per
    // thread, then reduce in thread order, to avoid atomics or shared writes.
    std::vector<std::vector<Mat11>> partial_self;
    if (nt > 1)
      partial_self.assign(nt, std::vector<Mat11>(N, Mat11::Zero()));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (std::int64_t mm = 0; mm < std::int64_t(K); mm++) {
      const std::size_t m = std::size_t(mm);
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      auto &local_self = nt == 1 ? self_ : partial_self[tid];
      const auto &mode = mode_[m];
      const double k = mode.k.norm();
      Mat3 P = Mat3::Identity() - mode.n * mode.n.transpose();
      const Eigen::Matrix<double, 8, 8> BB = mode.B.transpose() * P * mode.B;
      for (std::size_t i = 0; i < N; i++) {
        double a = std::sqrt(mode.n.dot(body_[i].shape * mode.n)), q = k * a, sinc, dip;
        if (std::abs(q) < 1e-3) {
          double q2 = q * q;
          sinc = 1 - q2 / 6 + q2 * q2 / 120 - q2 * q2 * q2 / 5040;
          dip = k * (1 - q2 / 10 + q2 * q2 / 280 - q2 * q2 * q2 / 15120);
        } else {
          sinc = std::sin(q) / q;
          dip = 3 * (std::sin(q) - q * std::cos(q)) / (a * q * q);
        }
        const double phase = mode.k.dot(body_[i].x), co = std::cos(phase), si = std::sin(phase);
        coeff_[m * N + i] = {sinc * co, -sinc * si, -dip * si, -dip * co};
        local_self[i].topLeftCorner<3, 3>().noalias() += mode.weight * sinc * sinc * P;
        local_self[i].bottomRightCorner<8, 8>().noalias() +=
            (mode.weight * dip * dip) * BB;
      }
    }
    for (const auto &local_self : partial_self)
      for (std::size_t i = 0; i < N; i++)
        self_[i] += local_self[i];
  }
  Eigen::VectorXd apply(const Eigen::VectorXd &g) const {
    const std::size_t N = body_.size();
    if (g.size() != Eigen::Index(11 * N))
      throw std::invalid_argument("Mobility vector size mismatch");
    Eigen::VectorXd out = Eigen::VectorXd::Zero(g.size());
    // M is linear. Zero-state residual scaling must not execute a full FFT.
    if (g.isZero(0.)) return out;
    for (std::size_t i = 0; i < N; i++)
      out.segment<11>(11 * i).noalias() = tails_[i] * g.segment<11>(11 * i);
    if (cfg_.backend == Backend::Mesh) {
      out += mesh_.apply(g);
      return out;
    }
    // No global dense matrix; each mode is a rank-at-most-four real Gram term.
    int nt = 1;
#ifdef _OPENMP
    nt = std::max(1, cfg_.threads);
#endif
    std::vector<Eigen::VectorXd> partial(nt, Eigen::VectorXd::Zero(g.size()));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (std::int64_t mm = 0; mm < std::int64_t(mode_.size()); mm++) {
      const std::size_t m = std::size_t(mm);
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      auto &local = partial[tid];
      const auto &mode = mode_[m];
      Vec3 ar = Vec3::Zero(), ai = Vec3::Zero();
      Eigen::Matrix<double, 8, 1> dr = Eigen::Matrix<double, 8, 1>::Zero(),
                                  di = Eigen::Matrix<double, 8, 1>::Zero();
      for (std::size_t i = 0; i < N; i++) {
        const auto &c = coeff_[m * N + i];
        ar += c.fr * g.segment<3>(11 * i);
        ai += c.fi * g.segment<3>(11 * i);
        dr += c.dr * g.segment<8>(11 * i + 3);
        di += c.di * g.segment<8>(11 * i + 3);
      }
      // B depends only on the mode, so apply it after the particle reduction.
      ar.noalias() += mode.B * dr;
      ai.noalias() += mode.B * di;
      ar -= mode.n * mode.n.dot(ar);
      ai -= mode.n * mode.n.dot(ai);
      ar *= mode.weight;
      ai *= mode.weight;
      // Evaluate once: auto would retain a product expression and repeat it
      // for every particle in the transpose accumulation below.
      const Eigen::Matrix<double, 8, 1> br = mode.B.transpose() * ar,
                                        bi = mode.B.transpose() * ai;
      for (std::size_t i = 0; i < N; i++) {
        const auto &c = coeff_[m * N + i];
        local.segment<3>(11 * i) += c.fr * ar + c.fi * ai;
        local.segment<8>(11 * i + 3) += c.dr * br + c.di * bi;
      }
    }
    for (const auto &local : partial)
      out += local;
    return out;
  }
  Eigen::MatrixXd dense() const {
    Eigen::Index n = 11 * body_.size();
    Eigen::MatrixXd M(n, n);
    for (Eigen::Index j = 0; j < n; j++) {
      Eigen::VectorXd e = Eigen::VectorXd::Zero(n);
      e[j] = 1;
      M.col(j) = apply(e);
    }
    return M;
  }
  const std::vector<Mat11> &self_blocks() const { return self_; }
  const std::vector<Mat11> &tail_blocks() const { return tails_; }
  const std::vector<Mode> &modes() const { return mode_; }
  std::size_t mode_count() const { return mode_.size(); }
  std::size_t bytes() const {
    return coeff_.size() * sizeof(Coeff) + mode_.size() * sizeof(Mode) +
           (self_.size() + tails_.size()) * sizeof(Mat11) + mesh_.bytes();
  }
  double cutoff() const { return kmax_; }
  Backend backend() const { return cfg_.backend; }
  const std::array<int, 3> &mesh_dimensions() const { return mesh_.dimensions(); }
  std::size_t mesh_stencil_entries() const { return mesh_.stencil_entries(); }
  int mesh_quadrature_order() const { return mesh_.quadrature_order(); }
  std::size_t mesh_fft_plan_builds() const { return mesh_.fft_plan_builds(); }
  int mesh_threads() const { return mesh_.threads(); }
  const char *mesh_fft_backend() const { return mesh_.fft_backend(); }
  // self_blocks() is exact for Direct; Mesh returns positive continuum
  // isolated blocks for preconditioning only, not physical self corrections.

private:
  struct Coeff {
    double fr, fi, dr, di;
  };
  MeshDetail::Operator mesh_;
  Settings cfg_;
  Mat3 H_;
  double kmax_ = 0;
  std::vector<Body> body_;
  std::vector<Mode> mode_;
  std::vector<Coeff> coeff_;
  std::vector<Mat11> tails_, self_;
};
} // namespace SlurryPeriodic
#endif
