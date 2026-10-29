// Adaptive composition of the geometry-only outer and lubrication operators.
#ifndef SLURRY_GENERAL_MODEL_H
#define SLURRY_GENERAL_MODEL_H
#include "slurry_general_types.h"
namespace SlurryGeneral {
struct ModelOptions {
  double tolerance=.005;
  int max_level=6; // Last included zero-based level; 6 means 64 x 256.
  double cutoff_factor=1.;
  bool derived_normal_log=false;
  bool energy_completion=true;
};
struct ModelResult {
  Mat17 grand=Mat17::Zero(),outer=Mat17::Zero(),near=Mat17::Zero(),isolated=Mat17::Zero();
  Mat17 near_completion=Mat17::Zero();
  double outer_relative_change=0.,full_relative_change=0.,linear_residual=0.;
  int nrad=0,nphi=0,levels=0;
  bool far=false;
};
// Resolves integrals at this geometry only; no orientation sampling or fitting.
// Numerical resolution does not certify the retained-moment approximation.
// Throws if SPD, residual or successive-energy-norm checks remain unresolved.
ModelResult evaluate_model(const Geometry&,const ModelOptions& options=ModelOptions{});
}
#endif
