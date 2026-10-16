// Per-fix numerical workspace. Only an approximate ILUT inverse and numerical
// branch metadata may persist; forces, geometry, Jacobians, and spring histories
// are always recomputed by the caller's current physical timestep.
#ifndef SLURRY_SOLVER_WORKSPACE_H
#define SLURRY_SOLVER_WORKSPACE_H

#include "slurry_eigen_ilu_math.h"
#include <cstddef>
#include <map>
#include <memory>
#include <utility>

namespace SlurryImplicit {
struct SolverWorkspace {
#ifdef SLURRY_HAVE_EIGEN
  std::unique_ptr<SlurryEigen::IncompleteLU> factor;
#endif
  std::map<std::pair<std::size_t,std::size_t>,std::pair<bool,bool>> branches;
  std::size_t dofs = 0;
  double dt = 0, length = 0, drop_tolerance = 0;
  int fill_factor = 0, age = 0;
  bool ready = false, refresh = false;

  void reset()
  {
#ifdef SLURRY_HAVE_EIGEN
    factor.reset();
#endif
    branches.clear();
    dofs = 0;
    dt = length = drop_tolerance = 0;
    fill_factor = age = 0;
    ready = refresh = false;
  }
};
} // namespace SlurryImplicit
#endif
