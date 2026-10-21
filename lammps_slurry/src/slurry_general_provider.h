// LAMMPS adapter for the geometry-only general spheroid response.
#ifndef SLURRY_GENERAL_PROVIDER_H
#define SLURRY_GENERAL_PROVIDER_H
#include "slurry_general_model.h"
#include "slurry_finite_gap_math.h"
#include <cstdint>
namespace SlurryGeneral {
using Map6=Eigen::Matrix<double,6,12>;
// Eliminate collective velocities at fixed relative motion. This is a Schur
// reduction, not eigenvalue clipping or interpolation of a pose table.
Mat11 reduced_response(const Mat17& grand,const Map6& relative_map);
class Provider {
 public:
  ModelOptions options;
  double epsilon_m=.15e-9;
  // Runtime closure: isolated collective drag + relative/strain correction.
  // False exposes the diagnostic full-pair additive closure, which can fail
  // the global passivity check even when every isolated pair is positive.
  bool relative_closure=true;
  SlurryFiniteGap::Pair make_pair(const SlurryFiniteGap::gr::Body& bi,
    const SlurryFiniteGap::gr::Body& bj,const SlurryFiniteGap::gr::GapResult& gap,
    double eta,double solver_length,double range,double inner_ratio,
    std::uint64_t id_i=0,std::uint64_t id_j=1) const;
};
}
#endif
