#ifndef SLURRY_COMPACT_BRIDGE_H
#define SLURRY_COMPACT_BRIDGE_H

// Compact lubrication connection with Mari-style additive regularization.
// This is an explicit finite-gap approximation, not an exact Stokes solution.
// The normal-normal subleading logarithm is deliberately omitted.
// q_i = (U_i, Omega_i, e_i), e_i = -E_i, with orthonormal STF coordinates.
// The returned matrix maps physical q to ON-FLUID (F,T,strain-conjugate S).
// Geometry uses the actual signed surface gap. Scalar resistance functions use
// max(h,0)+regularization_length; that length is NOT a contact-onset distance.
// The compact support uses max(h,0), so the regularization does not move its
// geometric outer boundary. Shallow penalty overlap uses the contact-limit
// scalar resistance while retaining the actual surface points and normal.
// The Schur completion is evaluated in coordinates nondimensionalized by L;
// changing L changes that finite, explicitly chosen model. The runtime uses the
// fixed nominal radius, independent of AR, as L.

#include "../vendor/gr_re2/ellipsoidGap.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace SlurryCompactBridge {
namespace gr = slurry::gr_re2::graphite;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Mat2 = Eigen::Matrix2d;
using Vec2 = Eigen::Vector2d;
using Mat11 = Eigen::Matrix<double, 11, 11>;
using Mat22 = Eigen::Matrix<double, 22, 22>;
using Mat10 = Eigen::Matrix<double, 10, 10>;
using Vec11 = Eigen::Matrix<double, 11, 1>;
using Vec10 = Eigen::Matrix<double, 10, 1>;
constexpr double pi = 3.141592653589793238462643383279502884;

inline const std::array<Mat3, 5> &strain_basis() {
  static const std::array<Mat3, 5> e = [] {
    std::array<Mat3, 5> a;
    for (auto &x : a)
      x.setZero();
    a[0].diagonal() << 1 / std::sqrt(2.), -1 / std::sqrt(2.), 0;
    a[1].diagonal() << 1 / std::sqrt(6.), 1 / std::sqrt(6.), -2 / std::sqrt(6.);
    a[2](0, 1) = a[2](1, 0) = 1 / std::sqrt(2.);
    a[3](0, 2) = a[3](2, 0) = 1 / std::sqrt(2.);
    a[4](1, 2) = a[4](2, 1) = 1 / std::sqrt(2.);
    return a;
  }();
  return e;
}
inline Mat3 cross_matrix(const Vec3 &v) {
  Mat3 a;
  a << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
  return a;
}
inline Vec3 vector(const gr::Vec3 &v) { return Vec3(v[0], v[1], v[2]); }
inline Mat3 shape(const gr::Body &b, double length) {
  Mat3 q;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      q(i, j) = b.rotation[3 * i + j];
  Vec3 d;
  for (int i = 0; i < 3; ++i) {
    if (!(b.axes[i] > 0) || !std::isfinite(b.axes[i]))
      throw std::invalid_argument("compact bridge needs finite positive axes");
    d[i] = std::pow(b.axes[i] / length, 2);
  }
  return q * d.asDiagonal() * q.transpose();
}
inline double compact_weight(double z) {
  if (z <= .1)
    return 1.;
  if (z >= 1.)
    return 0.;
  const double t = (z - .1) / .9, om = 1. - t;
  return om * om * om * (1. + 3. * t + 6. * t * t);
}

struct Result {
  Mat22 resistance = Mat22::Zero();
  Mat11 local = Mat11::Zero(); // nondimensional local matrix, INCLUDING w; eta excluded
  double gap = 0, effective_gap = 0, ell_n = 0, ell_t = 0, patch_width = 0;
  double weight = 0, normal_scale = 0;
  double schur_min_scaled = std::numeric_limits<double>::quiet_NaN(); // optional diagnostic
  bool active = false;
};

namespace detail {
constexpr std::array<int, 10> complement{{0, 1, 3, 4, 5, 6, 7, 8, 9, 10}};
inline std::array<double, 4> rotate_cubic(const std::array<double, 4> &p, double c, double s) {
  constexpr int choose[4][4] = {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 2, 1, 0}, {1, 3, 3, 1}};
  std::array<double, 4> out{};
  for (int m = 0; m < 4; ++m)
    for (int a = 0; a <= 3 - m; ++a)
      for (int b = 0; b <= m; ++b)
        out[a + b] += p[m] * choose[3 - m][a] * choose[m][b] * std::pow(c, 3 - m - a) *
                      std::pow(s, a) * std::pow(-s, m - b) * std::pow(c, b);
  return out;
}
inline std::array<double, 4> cubic(const Mat3 &inverse, double support, const Vec3 &n,
                                   const Vec3 &e1, const Vec3 &e2, double k11, double k12,
                                   double k22, double sign) {
  const double x = -.5 * sign * support * e1.dot(inverse * n),
               y = -.5 * sign * support * e2.dot(inverse * n);
  return {{x * k11, 2 * x * k12 + y * k11, x * k22 + 2 * y * k12, y * k22}};
}

// Global minimum x^T Lambda x on the projected ellipse boundary. This is
// the equality-constrained 2-D trust-region secular solve, not angular sampling.
inline double ellipse_boundary_minimum(const Mat2 &B, const Vec2 &p, const Mat2 &lambda) {
  Eigen::SelfAdjointEigenSolver<Mat2> eb(B);
  if (eb.info() != Eigen::Success || !(eb.eigenvalues()[0] > 0))
    throw std::runtime_error("non-positive projected ellipse");
  const Mat2 S =
      eb.eigenvectors() * eb.eigenvalues().cwiseSqrt().asDiagonal() * eb.eigenvectors().transpose();
  const double inside = p.dot(B.ldlt().solve(p));
  if (!(inside < 1.))
    throw std::runtime_error("contact outside projected ellipse interior");
  const Mat2 A = S * lambda * S;
  const Vec2 b = S * lambda * p;
  Eigen::SelfAdjointEigenSolver<Mat2> ea(A);
  if (ea.info() != Eigen::Success)
    throw std::runtime_error("ellipse eigensolve failed");
  const double scale = std::max({ea.eigenvalues()[1], b.norm(), 1e-300});
  const Vec2 d = ea.eigenvalues() / scale, beta = ea.eigenvectors().transpose() * b / scale;
  const double gap = d[1] - d[0], tol = 128 * std::numeric_limits<double>::epsilon();
  Vec2 u;
  if (beta.norm() < tol)
    u = Vec2(1, 0);
  else if (gap < tol)
    u = -beta / beta.norm();
  else if (std::abs(beta[0]) < tol && std::abs(beta[1] / gap) <= 1.) {
    const double u1 = -beta[1] / gap;
    u = Vec2(std::sqrt(std::max(0., 1. - u1 * u1)), u1);
  } else {
    // Monotone root on (0,||beta||]. Bisection has no unreliable Newton
    // derivative near the hard case. 200 iterations matches the prototype cap.
    double low = 0., high = beta.norm();
    for (int iter = 0; iter < 200; ++iter) {
      const double t = .5 * (low + high);
      const double f = std::pow(beta[0] / t, 2) + std::pow(beta[1] / (gap + t), 2) - 1.;
      if (f > 0)
        low = t;
      else
        high = t;
      if (high - low <= 8 * std::numeric_limits<double>::epsilon() * high)
        break;
    }
    const double t = .5 * (low + high);
    u = Vec2(-beta[0] / t, -beta[1] / (gap + t));
    u.normalize();
  }
  const Vec2 x = p + S * ea.eigenvectors() * u;
  return x.dot(lambda * x);
}

struct Coefficients {
  Mat3 Q = Mat3::Identity();
  Mat2 lambda = Mat2::Identity();
  Mat11 C = Mat11::Zero();
  Vec11 b = Vec11::Zero();
  double a0 = 0, ell = 0;
};
inline Coefficients coefficients(const Mat3 &Ai, const Mat3 &Aj, const Vec3 &n,
                                 double minor_scale) {
  Coefficients out;
  const Mat3 qi = Ai.inverse(), qj = Aj.inverse();
  const double support_i = std::sqrt(n.dot(Ai * n)), support_j = std::sqrt(n.dot(Aj * n));
  const Mat3 ci = support_i * qi, cj = support_j * qj;
  const Vec3 reference = std::abs(n[0]) < .8 ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
  const Vec3 u = n.cross(reference).normalized(), v = n.cross(u);
  const double angle = .5 * std::atan2(2 * u.dot(ci * v), u.dot(ci * u) - v.dot(ci * v));
  const Vec3 e1 = u * std::cos(angle) + v * std::sin(angle), e2 = n.cross(e1);
  out.Q.col(0) = e1;
  out.Q.col(1) = e2;
  out.Q.col(2) = n;
  const Eigen::Matrix<double, 3, 2> T = out.Q.leftCols<2>();
  Mat2 P = T.transpose() * ci * T, Pp = T.transpose() * cj * T;
  P = .5 * (P + P.transpose()).eval();
  Pp = .5 * (Pp + Pp.transpose()).eval();
  const double ki1 = P(0, 0), ki2 = P(1, 1), kj11 = Pp(0, 0), kj12 = Pp(0, 1), kj22 = Pp(1, 1);
  const double a = .5 * (ki1 + kj11), b = .5 * kj12, d = .5 * (ki2 + kj22);
  const double chi = -.5 * std::atan2(2 * b, a - d), c = std::cos(chi), s = std::sin(chi);
  const double l1 = a * c * c - 2 * b * s * c + d * s * s,
               l2 = a * s * s + 2 * b * s * c + d * c * c;
  if (!(l1 > 0 && l2 > 0))
    throw std::runtime_error("non-positive contact curvature");
  const double root = std::sqrt(l1 * l2), sm = l1 + l2;
  out.lambda = .5 * (P + Pp);
  const Mat2 D = out.lambda * (2 * out.lambda.trace() * Mat2::Identity() + out.lambda);
  Eigen::Matrix<double, 2, 5> V0 = Eigen::Matrix<double, 2, 5>::Zero(), W0 = V0;
  V0.leftCols<2>().setIdentity();
  W0.leftCols<2>() = P - out.lambda;
  W0(0, 3) = -1.;
  W0(1, 2) = 1.;
  const Eigen::Matrix<double, 5, 5> C5 =
      pi / root * (V0.transpose() * V0 + 3 * W0.transpose() * D.ldlt().solve(W0));
  Eigen::Matrix<double, 6, 6> C6 = Eigen::Matrix<double, 6, 6>::Zero();
  constexpr std::array<int, 5> ids{{0, 1, 3, 4, 5}};
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j)
      C6(ids[i], ids[j]) = C5(i, j);
  const auto gi = cubic(qi, support_i, n, e1, e2, ki1, 0, ki2, 1.);
  const auto gj = cubic(qj, support_j, n, e1, e2, kj11, kj12, kj22, -1.);
  std::array<double, 4> gs;
  for (int i = 0; i < 4; ++i)
    gs[i] = gi[i] + gj[i];
  auto kap = rotate_cubic(gs, c, s);
  for (int k = 0; k < 4; ++k)
    kap[k] /= std::pow(l1, .5 * (3 - k)) * std::pow(l2, .5 * k);
  const double d1 = 3 * l1 + 2 * l2, d2 = 2 * l1 + 3 * l2, nfac = 3 * pi / (2 * root * sm);
  const double AA = (7 * l1 + 2 * l2) * kap[0] + (l1 + 2 * l2) * kap[2];
  const double BB = (2 * l1 + l2) * kap[1] + (2 * l1 + 7 * l2) * kap[3];
  const double CC = 3 * kap[0] * l1 + kap[2] * l2, DD = kap[1] * l1 + 3 * kap[3] * l2;
  const double r1 = std::sqrt(l1), r2 = std::sqrt(l2);
  Eigen::Matrix<double, 6, 1> b6 = Eigen::Matrix<double, 6, 1>::Zero();
  b6[0] =
      nfac * ((2 * r1 * CC - 1.5 * ki1 * AA / r1) * c / d1 +
              (2 * r2 * DD - 1.5 * ki1 * BB / r2) * s / d2 + 3 * gi[0] * (c * c / l1 + s * s / l2) +
              2 * gi[1] * s * c * (1 / l2 - 1 / l1) + gi[2] * (s * s / l1 + c * c / l2));
  b6[1] =
      nfac * ((-2 * r1 * CC + 1.5 * ki2 * AA / r1) * s / d1 +
              (2 * r2 * DD - 1.5 * ki2 * BB / r2) * c / d2 + 3 * gi[3] * (s * s / l1 + c * c / l2) +
              2 * gi[2] * s * c * (1 / l2 - 1 / l1) + gi[1] * (c * c / l1 + s * s / l2));
  b6[3] = -1.5 * nfac * (BB * c / (r2 * d2) - AA * s / (r1 * d1));
  b6[4] = 1.5 * nfac * (AA * c / (r1 * d1) + BB * s / (r2 * d2));
  b6[5] = nfac * s * c * (ki1 - ki2) * (1 / l1 - 1 / l2);
  out.a0 = 3 * pi / (root * sm);
  const Mat2 ML = out.lambda.inverse();
  const double tt = (P * ML).trace();
  Mat3 GS = Mat3::Zero();
  GS.topLeftCorner<2, 2>() = (P * ML + ML * P) / 4. - (tt / 12. + 1. / 3.) * Mat2::Identity();
  GS(2, 2) = 2. / 3. - tt / 3.;
  Eigen::Matrix<double, 6, 11> pn = Eigen::Matrix<double, 6, 11>::Zero();
  pn.leftCols<6>().setIdentity();
  for (int k = 0; k < 5; ++k) {
    pn(3, 6 + k) = strain_basis()[k](1, 2);
    pn(4, 6 + k) = -strain_basis()[k](0, 2);
  }
  out.b = pn.transpose() * b6;
  for (int k = 0; k < 5; ++k)
    out.b[6 + k] += out.a0 * (strain_basis()[k].array() * GS.array()).sum();
  out.C = pn.transpose() * C6 * pn;
  Eigen::SelfAdjointEigenSolver<Mat2> ep(P), ej(Pp);
  out.ell = std::min({minor_scale, 1. / ep.eigenvalues()[1], 1. / ej.eigenvalues()[1]});
  return out;
}
inline Mat11 center_map(const Mat3 &Q, const Vec3 &lever) {
  Mat11 A = Mat11::Zero();
  A.topLeftCorner<3, 3>() = Q.transpose();
  A.block<3, 3>(0, 3) = -Q.transpose() * cross_matrix(lever);
  A.block<3, 3>(3, 3) = Q.transpose();
  for (int k = 0; k < 5; ++k) {
    A.block<3, 1>(0, 6 + k) = Q.transpose() * strain_basis()[k] * lever;
    const Mat3 local = Q.transpose() * strain_basis()[k] * Q;
    for (int j = 0; j < 5; ++j)
      A(6 + j, 6 + k) = (strain_basis()[j].array() * local.array()).sum();
  }
  return A;
}
} // namespace detail

inline Result evaluate(const gr::Body &bi, const gr::Body &bj, const gr::GapResult &gap, double eta,
                       double reference_length, double regularization_length,
                       bool diagnostics = false) {
  if (!(eta > 0 && reference_length > 0 && regularization_length > 0) || !std::isfinite(eta) ||
      !std::isfinite(reference_length) || !std::isfinite(regularization_length) ||
      !std::isfinite(gap.gap))
    throw std::invalid_argument("compact bridge needs finite eta,L,regularization_length > 0 and finite gap");
  const double length = reference_length;
  const Mat3 Ai = shape(bi, length), Aj = shape(bj, length);
  const Vec3 n = vector(gap.normal).normalized(), li = vector(gap.leverI) / length,
             lj = vector(gap.leverJ) / length;
  const double minor = std::min(*std::min_element(bi.axes.begin(), bi.axes.end()),
                                *std::min_element(bj.axes.begin(), bj.axes.end())) /
                       length;
  const auto z = detail::coefficients(Ai, Aj, n, minor);
  const Eigen::Matrix<double, 3, 2> T = z.Q.leftCols<2>();
  const double ell_n = std::min(
      detail::ellipse_boundary_minimum(T.transpose() * Ai * T, -T.transpose() * li, z.lambda),
      detail::ellipse_boundary_minimum(T.transpose() * Aj * T, -T.transpose() * lj, z.lambda));
  Eigen::SelfAdjointEigenSolver<Mat2> el(z.lambda);
  const double Lp = std::sqrt(ell_n / el.eigenvalues()[1]);
  if (!(ell_n > 0 && Lp > 0))
    throw std::runtime_error("non-positive curvature patch");
  const double physical_gap = std::max(gap.gap, 0.) / length;
  const double he = physical_gap + regularization_length / length;
  Result out;
  out.gap = gap.gap;
  out.effective_gap = he * length;
  out.ell_n = ell_n * length;
  out.ell_t = z.ell * length;
  out.patch_width = Lp * length;
  out.weight = compact_weight(physical_gap / Lp);
  out.normal_scale = ell_n / (he + ell_n);
  out.active = out.weight > 0;
  if (!out.active)
    return out;
  const double log = std::log1p(z.ell / he), alpha = z.a0 / he, s = out.normal_scale;
  Vec10 b;
  Mat10 C;
  for (int i = 0; i < 10; ++i) {
    b[i] = z.b[detail::complement[i]];
    for (int j = 0; j < 10; ++j)
      C(i, j) = z.C(detail::complement[i], detail::complement[j]);
  }
  const Mat10 LC = log * C, Ec = log * log / alpha * (b * b.transpose());
  const Mat10 difference = .5 * ((Ec - LC) + (Ec - LC).transpose()).eval();
  Eigen::SelfAdjointEigenSolver<Mat10> es(difference);
  if (es.info() != Eigen::Success)
    throw std::runtime_error("compact Schur eigensolve failed");
  const Mat10 Delta = es.eigenvectors() * es.eigenvalues().cwiseMax(0.).asDiagonal() *
                      es.eigenvectors().transpose();
  const Mat10 completed = LC + Delta;
  out.local(2, 2) = s * s * alpha;
  for (int i = 0; i < 10; ++i) {
    const int k = detail::complement[i];
    out.local(2, k) = out.local(k, 2) = s * log * b[i];
    for (int j = 0; j < 10; ++j)
      out.local(k, detail::complement[j]) = completed(i, j);
  }
  out.local *= out.weight;
  Eigen::Matrix<double, 11, 22> B;
  B.leftCols<11>() = detail::center_map(z.Q, li);
  B.rightCols<11>() = -detail::center_map(z.Q, lj);
  out.resistance = B.transpose() * out.local * B;
  // q_nd=(U/L,Omega,e), P=eta*L^3*q_nd^T R_nd q_nd.
  Eigen::Matrix<double, 22, 1> scale = Eigen::Matrix<double, 22, 1>::Ones();
  scale.head<3>().setConstant(1. / length);
  scale.segment<3>(11).setConstant(1. / length);
  out.resistance =
      eta * length * length * length * (scale.asDiagonal() * out.resistance * scale.asDiagonal());
  out.resistance = .5 * (out.resistance + out.resistance.transpose()).eval();
  if (!out.resistance.allFinite())
    throw std::runtime_error("nonfinite compact bridge resistance");
  // Diagnostic only. The construction itself is the positivity guarantee.
  if (diagnostics) {
    Eigen::SelfAdjointEigenSolver<Mat10> eschur(completed - Ec);
    out.schur_min_scaled = eschur.eigenvalues()[0] / std::max({LC.norm(), Ec.norm(), 1e-300});
  }
  return out;
}
inline Result evaluate(const gr::Body &bi, const gr::Body &bj, double eta, double reference_length,
                       double regularization_length, gr::GapCache *cache = nullptr) {
  return evaluate(bi, bj, gr::closestEllipsoidGap(bi, bj, cache), eta, reference_length,
                  regularization_length);
}

// Guaranteed compact support radius for bodies with maximum semiaxes ai,aj.
inline double support_radius(const gr::Body &bi, const gr::Body &bj) {
  const double ai = *std::max_element(bi.axes.begin(), bi.axes.end());
  const double aj = *std::max_element(bj.axes.begin(), bj.axes.end());
  return ai + aj + std::min(ai, aj);
}
} // namespace SlurryCompactBridge
#endif
