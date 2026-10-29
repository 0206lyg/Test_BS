#include "slurry_general_model.h"
#include "slurry_general_near.h"
#include "slurry_general_outer.h"
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
namespace SlurryGeneral {
namespace {
constexpr std::array<std::array<int,2>,7> levels{{{{4,16}},{{8,32}},{{16,64}},{{24,96}},{{32,128}},{{48,192}},{{64,256}}}};
bool positive_factor(const Mat17& matrix,Eigen::LLT<Mat17>& factor) {
  if(!matrix.allFinite())return false;
  factor.compute(.5*(matrix+matrix.transpose()).eval());
  return factor.info()==Eigen::Success;
}
double energy_norm(const Mat17& delta,const Eigen::LLT<Mat17>& factor) {
  const Mat17 left=factor.matrixL().solve(delta);
  const Mat17 scaled=factor.matrixL().solve(left.transpose()).transpose();
  // Keep the tiny antisymmetric numerical part in the same spectral norm used
  // by the Python evaluator. No coefficient or eigenvalue is modified.
  Eigen::JacobiSVD<Mat17> svd(scaled);
  if(svd.info()!=Eigen::Success||!svd.singularValues().allFinite())
    throw std::runtime_error("General model response-norm SVD failed");
  return svd.singularValues()[0];
}
[[noreturn]] void unresolved(const Geometry& g,const ModelResult& r,
                            const ModelOptions& options,const std::string& detail) {
  std::ostringstream message;message<<std::setprecision(17)
    <<"General hydrodynamics numerically unresolved: AR="<<g.ar<<" h_over_c="<<g.gap
    <<" quadrature="<<r.nrad<<'x'<<r.nphi<<" levels="<<r.levels
    <<" outer_relative_change="<<r.outer_relative_change
    <<" full_relative_change="<<r.full_relative_change
    <<" tolerance="<<options.tolerance<<" residual="<<r.linear_residual
    <<"; "<<detail;
  throw std::runtime_error(message.str());
}
}
ModelResult evaluate_model(const Geometry& g,const ModelOptions& options) {
  if(!(options.tolerance>0&&std::isfinite(options.tolerance)))
    throw std::invalid_argument("General model tolerance must be finite and positive");
  if(options.max_level<1||options.max_level>6)
    throw std::invalid_argument("General model max_level must be an integer from 1 to 6");
  if(!(options.cutoff_factor>0&&std::isfinite(options.cutoff_factor)))
    throw std::invalid_argument("General model cutoff factor must be finite and positive");
  if(!(g.ar>=1&&std::isfinite(g.ar)))throw std::invalid_argument("General model requires finite AR>=1");
  const double ratio=(g.centers[0]-g.centers[1]).norm()/g.ar;
  if(!(ratio>0&&std::isfinite(ratio)))throw std::invalid_argument("General model requires finite distinct centres");
  const auto local=evaluate_near(g,options.cutoff_factor,options.derived_normal_log,options.energy_completion);
  ModelResult current;current.near=local.grand;current.near_completion=local.completion;
  current.outer_relative_change=current.full_relative_change=std::numeric_limits<double>::infinity();
  current.linear_residual=std::numeric_limits<double>::infinity();
  const bool blend=ratio>8&&ratio<12;
  OuterResult far;
  if(ratio>8)far=evaluate_outer(g,4,16,true); // Reused at every disk refinement.
  if(ratio>=12) {
    current.outer=far.grand;current.isolated=far.isolated;current.grand=current.outer+current.near;
    current.linear_residual=far.linear_residual;current.far=true;current.levels=1;
    // No disk quadrature is evaluated in this branch. Orders zero mean N/A.
    Eigen::LLT<Mat17> outer_factor,full_factor;
    if(!positive_factor(current.outer,outer_factor)||!positive_factor(current.grand,full_factor)
        ||!(current.linear_residual<1e-8))
      unresolved(g,current,options,"far response failed SPD/residual check");
    current.outer_relative_change=current.full_relative_change=0.;return current;
  }
  const double fraction=blend?(ratio-8)/4:0.;
  const double far_weight=fraction*fraction*fraction*(10-15*fraction+6*fraction*fraction);
  Mat17 previous_outer,previous_grand;bool have_previous=false;
  std::string detail="successive response changes exceeded tolerance";
  for(int level=0;level<=options.max_level;++level) {
    current.nrad=levels[level][0];current.nphi=levels[level][1];current.levels=level+1;
    OuterResult disk;
    try {disk=evaluate_outer(g,current.nrad,current.nphi,false);}
    catch(const OuterLinearAlgebraError& error) {
      // A singular/inaccurate low-order quadrature may resolve on refinement.
      // Other errors (invalid input, singular nodes, shape coefficient errors)
      // are never suppressed by the adaptive controller.
      detail=error.what();have_previous=false;
      current.outer_relative_change=current.full_relative_change=std::numeric_limits<double>::infinity();
      current.linear_residual=std::numeric_limits<double>::infinity();
      if(level==options.max_level)unresolved(g,current,options,detail);
      continue;
    }
    current.outer=blend?((1-far_weight)*disk.grand+far_weight*far.grand).eval():disk.grand;
    current.isolated=disk.isolated;current.grand=current.outer+current.near;
    current.linear_residual=blend?std::max(disk.linear_residual,far.linear_residual):disk.linear_residual;
    current.outer_relative_change=current.full_relative_change=std::numeric_limits<double>::infinity();
    Eigen::LLT<Mat17> outer_factor,full_factor;
    const bool outer_spd=positive_factor(current.outer,outer_factor);
    const bool full_spd=positive_factor(current.grand,full_factor);
    if(have_previous&&outer_spd)current.outer_relative_change=energy_norm(current.outer-previous_outer,outer_factor);
    if(have_previous&&full_spd)current.full_relative_change=energy_norm(current.grand-previous_grand,full_factor);
    if(have_previous&&outer_spd&&full_spd&&current.linear_residual<1e-8
       &&current.outer_relative_change<=options.tolerance&&current.full_relative_change<=options.tolerance)
      return current;
    detail=!outer_spd?"outer response is not SPD":!full_spd?"full response is not SPD":
      !(current.linear_residual<1e-8)?"outer linear residual exceeds limit":
      !have_previous?"a successive valid response is not available":"successive response changes exceeded tolerance";
    previous_outer=current.outer;previous_grand=current.grand;have_previous=true;
  }
  unresolved(g,current,options,detail);
}
}
