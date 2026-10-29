// General oblate force/torque/stresslet closure, Kim (1986).
// No pose tables or fitted reference residuals are used.
#ifndef SLURRY_GENERAL_OUTER_H
#define SLURRY_GENERAL_OUTER_H
#include "slurry_general_types.h"
#include <cstddef>
namespace SlurryGeneral {
class OuterLinearAlgebraError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
struct KernelDiagnostics {
  double minimum_disk_node_distance=0.;
  std::size_t effective_kernel_pairs=0;
  bool axisymmetric_azimuth_reduction=false;
};
struct OuterResult {
  Mat17 grand=Mat17::Zero(),isolated=Mat17::Zero();
  Mat22 mobility=Mat22::Zero(),grand22=Mat22::Zero();
  double linear_residual=0.,minimum_grand_eigenvalue=0.,grand_reciprocity=0.;
  KernelDiagnostics kernel;
};
// Exact isolated spheroid in native (U,omega,+E5) coordinates; c=eta=1.
Mat11 isolated11(double ar,const Mat3& rotation=Mat3::Identity());
Mat11 cross_mobility_disk(const Vec3& r,const Mat3& target_rotation,
                         const Mat3& source_rotation,double ar,int nrad=4,
                         int nphi=16,KernelDiagnostics* diagnostics=nullptr);
// Centre expansion through total derivative order four (R^-5).
Mat11 cross_mobility_far(const Vec3& r,const Mat3& target_shape,
                        const Mat3& source_shape,int max_derivative=4);
OuterResult evaluate_outer(const Geometry& geometry,int nrad=4,int nphi=16,
                           bool far=false);
}
#endif
