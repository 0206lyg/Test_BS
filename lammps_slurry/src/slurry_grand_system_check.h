#ifndef SLURRY_GRAND_SYSTEM_CHECK_H
#define SLURRY_GRAND_SYSTEM_CHECK_H
#include "slurry_implicit_math.h"
#include "slurry_finite_gap_math.h"
#ifdef SLURRY_HAVE_EIGEN
#include <Eigen/SparseCholesky>
#include <Eigen/Dense>
#endif
namespace SlurryGrandSystem {
using Coupling6=std::array<double,30>;
struct PairBlock {
  std::size_t i=0,j=0;std::array<double,60> coupling{};SlurryFiniteGap::Mat5 strain{};
  // POD cache gathered once so the expensive reference surrogate is not
  // recomputed for diagnostics or separately on every MPI rank.
  bool has_finite_model=false,active=false;double cutoff=0;
  SlurryGrandPair::Frame frame{};SlurryGrandPair::Mat17 resistance{};
};
struct Check {double strain_margin=0,min_velocity_pivot=0,solve_residual=0;};
inline Check verify(const std::vector<SlurryImplicit::Particle>& particles,
                    const std::vector<SlurryImplicit::Pair>& pairs,
                    const std::vector<Coupling6>& body_coupling,
                    const SlurryFiniteGap::Mat5& body_and_solvent_strain,
                    const std::vector<PairBlock>& pair_blocks,
                    double viscosity,double length,double volume) {
#ifndef SLURRY_HAVE_EIGEN
  (void)particles;(void)pairs;(void)body_coupling;(void)body_and_solvent_strain;(void)pair_blocks;
  (void)viscosity;(void)length;(void)volume;
  throw std::runtime_error("Experimental finite-gap global passivity check requires bundled Eigen enabled");
#else
  const int n=static_cast<int>(6*particles.size());
  if(body_coupling.size()!=particles.size()||!(viscosity>0&&length>0&&volume>0))throw std::invalid_argument("Invalid global grand check dimensions/scales");
  const double ascale=viscosity*length,cscale=ascale*length,dscale=cscale*length;
  std::vector<Eigen::Triplet<double>> entries;entries.reserve(particles.size()*36+pairs.size()*144);
  Eigen::MatrixXd c=Eigen::MatrixXd::Zero(n,5);SlurryFiniteGap::Mat5 d{};
  for(int i=0;i<25;++i)d[i]=body_and_solvent_strain[i]/dscale;
  for(std::size_t p=0;p<particles.size();++p)for(int i=0;i<6;++i){
    for(int j=0;j<6;++j)entries.emplace_back(static_cast<int>(6*p)+i,static_cast<int>(6*p)+j,particles[p].resistance[6*i+j]/ascale);
    for(int j=0;j<5;++j)c(static_cast<int>(6*p)+i,j)+=body_coupling[p][5*i+j]/cscale;
  }
  for(const auto&p:pairs)if(p.generalized_hydro){
    double scale=0;for(double v:p.hydro_resistance){if(!std::isfinite(v))throw std::runtime_error("Nonfinite global finite-gap velocity resistance");scale=std::max(scale,std::abs(v));}
    for(int i=0;i<12;++i)for(int j=0;j<12;++j){
      if(std::abs(p.hydro_resistance[12*i+j]-p.hydro_resistance[12*j+i])>1e-11*std::max(scale,1e-300))throw std::runtime_error("Nonreciprocal global finite-gap velocity resistance");
      const int row=static_cast<int>(6*(i<6?p.i:p.j))+i%6,col=static_cast<int>(6*(j<6?p.i:p.j))+j%6;
      entries.emplace_back(row,col,p.hydro_resistance[12*i+j]/ascale);
    }
  }
  for(const auto&p:pair_blocks){
    for(int i=0;i<12;++i)for(int j=0;j<5;++j)c(static_cast<int>(6*(i<6?p.i:p.j))+i%6,j)+=p.coupling[5*i+j]/cscale;
    for(int i=0;i<25;++i)d[i]+=p.strain[i]/dscale;
  }
  if(!c.allFinite())throw std::runtime_error("Nonfinite global strain coupling");
  Eigen::SparseMatrix<double> a(n,n);a.setFromTriplets(entries.begin(),entries.end());a.makeCompressed();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor;factor.compute(a);
  if(factor.info()!=Eigen::Success)throw std::runtime_error("Experimental finite-gap global velocity factorization failed");
  Check out;out.min_velocity_pivot=factor.vectorD().minCoeff();
  if(!(out.min_velocity_pivot>0)||!factor.vectorD().allFinite())throw std::runtime_error("Experimental finite-gap GLOBAL velocity resistance is not positive definite; no pair clipping was applied");
  const Eigen::MatrixXd solved=factor.solve(c);
  if(factor.info()!=Eigen::Success||!solved.allFinite())throw std::runtime_error("Experimental finite-gap global strain solve failed");
  out.solve_residual=(a*solved-c).norm()/std::max(c.norm(),1e-300);
  if(out.solve_residual>1e-8)throw std::runtime_error("Experimental finite-gap global strain solve residual is excessive");
  SlurryFiniteGap::Mat5 schur{};
  for(int i=0;i<5;++i)for(int j=0;j<5;++j){long double x=d[5*i+j];for(int k=0;k<n;++k)x-=static_cast<long double>(c(k,i))*solved(k,j);schur[5*i+j]=static_cast<double>(x*length*length*length/volume);}
  for(int i=0;i<5;++i)for(int j=i+1;j<5;++j)schur[5*i+j]=schur[5*j+i]=.5*(schur[5*i+j]+schur[5*j+i]);
  out.strain_margin=SlurryFiniteGap::minimum_eigenvalue(schur);
  if(!(out.strain_margin>1e-12))throw std::runtime_error("Experimental finite-gap GLOBAL grand resistance failed passivity (strain Schur complement includes solvent and exact one-body stress); signed pair terms retained");
  return out;
#endif
}
} // namespace SlurryGrandSystem
#endif
