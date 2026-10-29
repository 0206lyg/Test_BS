// Finite real-space image enumeration for compact near resistance/contact.
// H contains lattice vectors in columns; d = x_j - x_i in any common frame.
#ifndef SLURRY_BRIDGE_IMAGES_H
#define SLURRY_BRIDGE_IMAGES_H

#include <Eigen/Core>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace SlurryBridgeImages {
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using ImageIndex = Eigen::Vector3i;

struct Image {
  ImageIndex image;
  Vec3 displacement;
};

// For i == j omit image zero and retain one of n,-n. For i < j retain all n.
inline bool positive_image(const ImageIndex &image) {
  for (int k = 0; k < 3; ++k) {
    if (image[k] > 0)
      return true;
    if (image[k] < 0)
      return false;
  }
  return false;
}

// A frame has one lattice but many pairs. Validate and invert that lattice
// once, rather than repeating a pivoted factorization for every pair.
class ImageEnumerator {
  Mat3 H, inverse;
  Vec3 reciprocal_norms;
public:
  explicit ImageEnumerator(const Mat3 &lattice) : H(lattice) {
    if (!H.allFinite())
      throw std::invalid_argument("Image enumeration requires a finite lattice");
    Eigen::FullPivLU<Mat3> factor(H);
    if (!factor.isInvertible())
      throw std::invalid_argument("Image lattice is singular");
    inverse = factor.inverse();
    if (!inverse.allFinite())
      throw std::invalid_argument("Image lattice inverse is nonfinite");
    for (int k = 0; k < 3; ++k) reciprocal_norms[k] = inverse.row(k).norm();
  }

  std::vector<Image> within(const Vec3 &d, double radius) const {
  if (!d.allFinite() || !std::isfinite(radius) || radius < 0.)
    throw std::invalid_argument(
        "Image enumeration requires finite geometry and nonnegative radius");
  const Vec3 fractional = inverse * d;
  if (!fractional.allFinite())
    throw std::invalid_argument("Image lattice inverse is nonfinite");
  ImageIndex lower, upper;
  constexpr double eps = std::numeric_limits<double>::epsilon();
  for (int k = 0; k < 3; ++k) {
    const double bound = radius * reciprocal_norms[k];
    const double margin = 64. * eps * (1. + std::abs(fractional[k]) + bound);
    const double lo = std::ceil(-fractional[k] - bound - margin);
    const double hi = std::floor(-fractional[k] + bound + margin);
    // Reserve one integer at each end to make the inclusive loops safe.
    if (!std::isfinite(lo + hi) || lo < std::numeric_limits<int>::min() + 1. ||
        hi > std::numeric_limits<int>::max() - 1.)
      throw std::overflow_error("Periodic image index exceeds supported integer range");
    lower[k] = static_cast<int>(lo);
    upper[k] = static_cast<int>(hi);
    if (lower[k] > upper[k])
      return {};
  }
  std::vector<Image> result;
  // The reciprocal-coordinate bound is exact: |s_k+n_k| <= R ||row_k(H^-1)||.
  // The final Euclidean test discards the extra corners. Roundoff enlarges the
  // test only; a near cutoff can independently set the boundary weight to zero.
  for (int ix = lower[0]; ix <= upper[0]; ++ix)
    for (int iy = lower[1]; iy <= upper[1]; ++iy)
      for (int iz = lower[2]; iz <= upper[2]; ++iz) {
        const ImageIndex image(ix, iy, iz);
        const Vec3 shift = H * image.cast<double>();
        const Vec3 dr = d + shift;
        const double tolerance =
            64. * eps *
            std::max({radius, d.norm(), shift.norm(), std::numeric_limits<double>::min()});
        if (dr.norm() <= radius + tolerance)
          result.push_back({image, dr});
      }
  return result;
  }
};

inline std::vector<Image> images_within(const Vec3 &d, const Mat3 &H, double radius) {
  return ImageEnumerator(H).within(d, radius);
}

struct Match {
  bool found = false;
  std::size_t index = 0;
  int sign = 1; // -1 means current_d == -predicted[index] (self-image only).
  double error = std::numeric_limits<double>::infinity();
};

// At timestep commit, store predicted_d = old_d + dt*(U_j,image-U_i).
// This is the actual forward-Euler separation after the pending step, before
// coordinate wrapping or an integer lattice-basis change. Match physical d,
// never the raw image integer, which changes at wrapping/Lees--Edwards flips.
// The caller supplies a tolerance in the SAME LENGTH UNITS as the displacements.
// A missing match starts a new contact history; ambiguity is an implementation
// error rather than a license to choose an unrelated contact's friction spring.
// For a self pair only, allow_reverse=true handles a canonical +/- switch.
// If sign == -1, reverse stored normal and tangential elastic displacement;
// retain the stored common rotation angle. Nonself ordered tags must not reverse.
inline Match match_image(const std::vector<Vec3> &predicted_history_displacements,
                         const Vec3 &current_d, double tolerance, bool allow_reverse = false) {
  if (!current_d.allFinite() || !std::isfinite(tolerance) || tolerance < 0.)
    throw std::invalid_argument("Contact image matching requires finite geometry and tolerance");
  Match result;
  for (std::size_t i = 0; i < predicted_history_displacements.size(); ++i) {
    const Vec3 &predicted = predicted_history_displacements[i];
    if (!predicted.allFinite())
      throw std::invalid_argument("Nonfinite predicted contact displacement");
    const double positive = (current_d - predicted).norm();
    const double negative =
        allow_reverse ? (current_d + predicted).norm() : std::numeric_limits<double>::infinity();
    const int sign = positive <= negative ? 1 : -1;
    const double error = std::min(positive, negative);
    if (error > tolerance)
      continue;
    if (result.found)
      throw std::runtime_error("Ambiguous periodic contact history match");
    result = {true, i, sign, error};
  }
  return result;
}
} // namespace SlurryBridgeImages
#endif
