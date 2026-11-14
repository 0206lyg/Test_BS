#ifndef SLURRY_RHEOLOGY_DIAGNOSTICS_H
#define SLURRY_RHEOLOGY_DIAGNOSTICS_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace SlurryRheologyDiagnostics {

// Imposed flow is u_x = shear_rate * z: x is flow, z is gradient, y is
// vorticity. Inputs are physical total stresses in Pa, not internal scaled
// tensor components. Keep signed N2; a plotted -N2 is a plotting convention.
inline std::array<double, 2> normalized_normal_stress(
    double sigma_xx, double sigma_yy, double sigma_zz,
    double eta, double shear_rate) {
  if (shear_rate == 0.) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan};
  }
  const double denominator = eta * shear_rate;
  return {(sigma_xx - sigma_zz) / denominator,
          (sigma_zz - sigma_yy) / denominator};
}

// Measurement-only contact graph. Each physical contact is inserted once;
// distinct periodic images are distinct edges, and a self-image contributes
// two incidences. Neither contact histories nor solver data are modified.
class ContactNetwork {
public:
  void reset(int count) {
    if (count < 0)
      throw std::invalid_argument("Negative rheology contact-network size");
    head_.assign(static_cast<std::size_t>(count), -1);
    degree_.assign(static_cast<std::size_t>(count), 0);
    neighbor_.clear();
    next_.clear();
    remaining_.resize(static_cast<std::size_t>(count));
    queue_.clear();
    queue_.reserve(static_cast<std::size_t>(count));
  }

  void add_contact(int i, int j) {
    if (i < 0 || j < 0 || static_cast<std::size_t>(i) >= head_.size() ||
        static_cast<std::size_t>(j) >= head_.size())
      throw std::out_of_range("Invalid rheology contact-network endpoint");
    add_incidence(i, j);
    add_incidence(j, i);
  }

  // Singh 2020 cites Seto et al. 2019 for nonrattlers: repeatedly remove
  // particles having fewer than two contacts with the remaining particles.
  // Queue peeling touches each vertex/edge only O(1) times, including a long
  // chain of removals. Scratch buffers preserve the original graph so this
  // query is repeatable. Capacities are reused between accepted substeps.
  double nonrattler_coordination() {
    remaining_ = degree_;
    queue_.clear();
    for (std::size_t i = 0; i < degree_.size(); ++i)
      if (remaining_[i] < 2)
        queue_.push_back(static_cast<int>(i));

    for (std::size_t q = 0; q < queue_.size(); ++q) {
      const int i = queue_[q];
      remaining_[i] = -1;
      for (int edge = head_[i]; edge != -1; edge = next_[edge]) {
        const int j = neighbor_[edge];
        if (remaining_[j] > 0 && --remaining_[j] == 1)
          queue_.push_back(j);
      }
    }

    std::size_t count = 0;
    std::uint64_t incidences = 0;
    for (int degree : remaining_)
      if (degree >= 2) {
        ++count;
        incidences += static_cast<std::uint64_t>(degree);
      }
    return count ? static_cast<double>(incidences) / static_cast<double>(count)
                 : std::numeric_limits<double>::quiet_NaN();
  }

private:
  void add_incidence(int i, int j) {
    const int index = static_cast<int>(neighbor_.size());
    neighbor_.push_back(j);
    next_.push_back(head_[i]);
    head_[i] = index;
    ++degree_[i];
  }

  std::vector<int> head_, degree_, neighbor_, next_, remaining_, queue_;
};

} // namespace SlurryRheologyDiagnostics

#endif
