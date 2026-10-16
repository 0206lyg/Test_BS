// Spheroid lubrication and elastic exclusion for the slurry LAMMPS plugin.
// The particle geometry and near-field kernel are shared with the LB code.
#ifndef LMP_PAIR_SLURRY_SPHEROID_H
#define LMP_PAIR_SLURRY_SPHEROID_H

#include "pair.h"

#include <vector>

namespace LAMMPS_NS {

class PairSlurrySpheroid : public Pair {
 public:
  explicit PairSlurrySpheroid(class LAMMPS *);
  ~PairSlurrySpheroid() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  double memory_usage() override;

 private:
  double viscosity_ = 0.0;
  double cutoff_gap_ = 0.0;
  double minimum_gap_ = 0.0;
  double normal_stiffness_ = 0.0;
  class AtomVecEllipsoid *avec_ = nullptr;
  std::vector<double> maximum_axis_;
  void allocate();
};

}    // namespace LAMMPS_NS
#endif
