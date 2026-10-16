#ifdef FIX_CLASS
// clang-format off
FixStyle(slurry/drag,FixSlurryDrag);
// clang-format on
#else
#ifndef LMP_FIX_SLURRY_DRAG_H
#define LMP_FIX_SLURRY_DRAG_H

#include "fix.h"
#include "slurry_hydro_math.h"

#include <array>
#include <map>

namespace LAMMPS_NS {

class FixSlurryDrag : public Fix {
 public:
  FixSlurryDrag(class LAMMPS *, int, char **);
  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_force(int) override;

 private:
  class AtomVecEllipsoid *avec;
  double viscosity, shear_rate;
  std::map<std::array<double, 3>, SlurryHydro::Resistance> resistance_cache;
  const SlurryHydro::Resistance &resistance(const double *);
};

}    // namespace LAMMPS_NS
#endif
#endif
