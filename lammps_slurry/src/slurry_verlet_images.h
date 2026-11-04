#ifndef SLURRY_VERLET_IMAGES_H
#define SLURRY_VERLET_IMAGES_H

// Exact candidate reuse for a deforming periodic lattice. This is only a
// broad phase: callers still evaluate the original geometry, cutoffs and forces
// on every returned candidate. A failed certificate requires a full rebuild.
#include "slurry_bridge_images.h"
#include "../vendor/gr_re2/ellipsoidGap.h"
#include <utility>

namespace SlurryVerletImages {
namespace gr = slurry::gr_re2::graphite;
namespace images = SlurryBridgeImages;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

struct Candidate {
  int i = 0, j = 0;
  Eigen::Vector3i image = Eigen::Vector3i::Zero();
};

class Cache {
  std::vector<Candidate> candidates_;
  std::vector<Vec3> positions_;
  std::vector<gr::Vec3> axes_;
  Mat3 lattice_ = Mat3::Identity(), inverse_ = Mat3::Identity();
  double skin_ = 0., maximum_pair_radius_ = 0.;
  double inverse_residual_bound_ = 0., reference_scale_ = 0.;
  bool initialized_ = false;

  static Vec3 vector(const gr::Vec3 &x) { return {x[0], x[1], x[2]}; }
  static constexpr double eps() { return std::numeric_limits<double>::epsilon(); }

public:
  const std::vector<Candidate> &candidates() const { return candidates_; }
  bool initialized() const { return initialized_; }
  double skin() const { return skin_; }
  double maximum_pair_radius() const { return maximum_pair_radius_; }
  void clear() { *this = Cache{}; }

  // cutoff(i,j) supplies the TOTAL center-distance radius, e.g. bounding_i +
  // bounding_j + physical_gap_cutoff_ij. It must be a fixed rotation-independent
  // bound until the next build. Changing radii, cutoff parameters, particle
  // order or identity requires rebuilding; rotations alone do not. The body
  // axes/count are checked by valid(), but tags and external cutoff parameters
  // are necessarily the caller's responsibility.
  //
  // Build retains all periodic images inside cutoff+skin. Self pairs retain
  // exactly the positive member of +/- image and omit the zero image, matching
  // SlurryBridgeImages. Construction is transactional if validation throws.
  template <class PairCutoff>
  const std::vector<Candidate> &build(const std::vector<gr::Body> &bodies,
                                     const Mat3 &lattice, PairCutoff cutoff,
                                     double skin) {
    if (!(skin >= 0.) || !std::isfinite(skin))
      throw std::invalid_argument("Verlet skin must be finite and nonnegative");
    if (bodies.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("Verlet particle index exceeds supported integer range");
    const images::ImageEnumerator enumerate(lattice); // validates lattice
    Cache next;
    next.lattice_ = lattice;
    next.inverse_ = Eigen::FullPivLU<Mat3>(lattice).inverse();
    next.skin_ = skin;
    // A posteriori bound for inverse roundoff. The additional product term
    // bounds the residual evaluation itself (generously for 3x3 arithmetic).
    const double product_norm = lattice.norm() * next.inverse_.norm();
    next.inverse_residual_bound_ =
        (Mat3::Identity() - lattice * next.inverse_).norm() +
        64. * eps() * (1. + product_norm);
    next.reference_scale_ = std::max(skin, lattice.norm());
    next.positions_.reserve(bodies.size());
    next.axes_.reserve(bodies.size());
    for (const auto &body : bodies) {
      const Vec3 position = vector(body.position);
      if (!position.allFinite())
        throw std::invalid_argument("Verlet cache requires finite particle positions");
      for (double axis : body.axes)
        if (!(axis > 0.) || !std::isfinite(axis))
          throw std::invalid_argument("Verlet cache requires finite positive particle axes");
      next.positions_.push_back(position);
      next.axes_.push_back(body.axes);
      next.reference_scale_ = std::max(next.reference_scale_, position.norm());
    }
    const int count = static_cast<int>(bodies.size());
    for (int i = 0; i < count; ++i)
      for (int j = i; j < count; ++j) {
        const double radius = cutoff(i, j);
        if (!(radius >= 0.) || !std::isfinite(radius) ||
            !std::isfinite(radius + skin))
          throw std::invalid_argument("Verlet pair cutoff must be finite and nonnegative");
        next.maximum_pair_radius_ = std::max(next.maximum_pair_radius_, radius);
        const Vec3 d = next.positions_[j] - next.positions_[i];
        for (const auto &candidate : enumerate.within(d, radius + skin)) {
          if (i == j && !images::positive_image(candidate.image)) continue;
          next.candidates_.push_back({i, j, candidate.image});
        }
      }
    next.reference_scale_ = std::max(next.reference_scale_,
                                    next.maximum_pair_radius_ + skin);
    next.initialized_ = true;
    *this = std::move(next);
    return candidates_;
  }

  // Let A=H H0^-1 and u_i=x_i-A x_i0. Every previously excluded image has
  // |d0|>Rij+skin, and |d| >= s_min |d0| - 2 max_i |u_i|.
  // Using s_min >= max(0,1-||A-I||_F), it remains outside Rij whenever
  // s_min*skin - max(0,1-s_min)*max_ij(Rij) - 2 max_i|u_i| > roundoff.
  // This includes lattice shear, affine particle motion and self images.
  // It does not assume that image indices survive a wrap or basis flip:
  // the same inequality must still certify the supplied coordinates/lattice.
  // The method is O(N), is read-only, and can be called on rejected trial states.
  // reserve_fraction additionally reserves that fraction of the original
  // skin. Rebuild before an interval when valid(...,.5) fails, then require
  // valid(...,0) at its endpoint, to avoid shrinking repeatedly at an almost
  // exhausted cache boundary. With unchanged axes and linear x(s), H(s), the
  // deformation, displacement and roundoff bounds below are convex in s;
  // certification at both endpoints therefore certifies the entire path.
  bool valid(const std::vector<gr::Body> &bodies, const Mat3 &lattice,
             double reserve_fraction = 0.) const {
    if (!(reserve_fraction >= 0. && reserve_fraction < 1.) ||
        !std::isfinite(reserve_fraction)) return false;
    if (!initialized_ || bodies.size() != positions_.size() ||
        !lattice.allFinite() || !(inverse_residual_bound_ < 1.)) return false;
    const Mat3 affine = lattice * inverse_;
    if (!affine.allFinite()) return false;
    // Account for using the computed inverse rather than exact H0^-1 when
    // bounding both the smallest stretch and each nonaffine displacement.
    const double affine_error = lattice.norm() * inverse_.norm() *
        (inverse_residual_bound_ / (1. - inverse_residual_bound_) + 64. * eps());
    const double deformation = (affine - Mat3::Identity()).norm() +
        affine_error + 64. * eps() * (1. + affine.norm());
    const double minimum_stretch = std::max(0., 1. - deformation);
    if (!(minimum_stretch > 0.)) return false;
    double maximum_nonaffine = 0.;
    double scale = std::max(reference_scale_, lattice.norm());
    for (std::size_t i = 0; i < bodies.size(); ++i) {
      if (bodies[i].axes != axes_[i]) return false;
      const Vec3 position = vector(bodies[i].position);
      if (!position.allFinite()) return false;
      const Vec3 residual = position - affine * positions_[i];
      const double arithmetic_error = 64. * eps() *
          (position.norm() + affine.norm() * positions_[i].norm());
      maximum_nonaffine = std::max(maximum_nonaffine,
          residual.norm() + affine_error * positions_[i].norm() + arithmetic_error);
      scale = std::max(scale, position.norm());
    }
    const double margin = minimum_stretch * skin_ -
        std::max(0., 1. - minimum_stretch) * maximum_pair_radius_ -
        2. * maximum_nonaffine;
    const double roundoff = 128. * eps() *
        std::max(scale, std::numeric_limits<double>::min());
    return std::isfinite(margin) && margin > roundoff + reserve_fraction*skin_;
  }
};
} // namespace SlurryVerletImages
#endif
