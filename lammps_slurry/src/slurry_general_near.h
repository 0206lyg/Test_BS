// Local lubrication singularities plus the declared compact C1 continuation.
// Default: corrected leading Janoschek entries, normal logarithm omitted.
// No reference data, orientation table, or configuration-specific fit is used.
#ifndef SLURRY_GENERAL_NEAR_H
#define SLURRY_GENERAL_NEAR_H
#include "slurry_general_types.h"
namespace SlurryGeneral {
struct NearResult {
  Mat17 grand=Mat17::Zero(),inverse_gap=Mat17::Zero(),logarithm=Mat17::Zero();
  Mat17 completion=Mat17::Zero();
  Eigen::Matrix<double,6,6> inverse_gap6=Eigen::Matrix<double,6,6>::Zero();
  Eigen::Matrix<double,6,6> logarithm6=Eigen::Matrix<double,6,6>::Zero();
  Eigen::Matrix<double,6,6> completion6=Eigen::Matrix<double,6,6>::Zero();
  Eigen::Matrix<double,6,17> contact_map=Eigen::Matrix<double,6,17>::Zero();
  double reference_length=0.,cutoff=0.,minimum_local_radius=0.;
  double normal_log_coefficient=0.,derived_normal_log_candidate=0.;
  bool energy_completed=false;
};
// Geometry must use the canonical common normal +z. The two closest support
// points are (0,0,0) and (0,0,gap). Rotations/directors may be arbitrary.
// Internal units: c=eta=1. Modes: [Ui,a*wi,Uj,a*wj,a*E5], a=AR.
// The last five outputs represent SUMMED intrinsic stresslets for common E.
// derived_normal_log is experimental and is not source-verified for free pairs.
// energy_completion adds a parameter-free O(h log^2 h) positive-square term.
// It preserves every retained singular coefficient, and is a finite-part model
// choice (see docs/NEAR_ENERGY_COMPLETION_KO.md), not an eigenvalue repair.
// Its C1-cutoff proof is unconditional in leading mode; derived requires the
// positive cutoff coefficient A_nn/hc+B_nn/2 as explained in that document.
// Raw remains the default of this low-level API for coefficient diagnostics;
// the runtime ModelOptions enables energy_completion by default.
NearResult evaluate_near(const Geometry&,double cutoff_factor=1.,
                         bool derived_normal_log=false,bool energy_completion=false);
std::array<double,2> compact_near_factors(double gap,double cutoff);
}
#endif
