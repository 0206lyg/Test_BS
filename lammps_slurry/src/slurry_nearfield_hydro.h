// Mari sphere / local-curvature spheroid lubrication in the bridge coordinates.
// This is a pair correction to isolated-body drag, not a full two-body solution.
// No compact bridge, finite-gap scattering response or many-body mobility is used.
#ifndef SLURRY_NEARFIELD_HYDRO_H
#define SLURRY_NEARFIELD_HYDRO_H

#include "slurry_grand_pair_math.h"
#include "slurry_spheroid_hydro_math.h"
#include <Eigen/Core>

namespace SlurryNearfieldHydro {
namespace gr = slurry::gr_re2::graphite;
namespace ph = SlurryPairHydro;
namespace sh = SlurrySpheroidHydro;
using Mat22 = Eigen::Matrix<double,22,22>;

// Size definitions shared by the runtime and the unequal-particle tests.
// For spheroids these are volume-equivalent radii, not bounding radii. Mari's
// sphere gap h=2*gap/(ri+rj) uses the arithmetic mean; Singh's rolling lever
// uses twice the reduced radius. They coincide only for identical particles.
struct PairLengths {
  double mean_radius = 0, epsilon = 0, cutoff = 0, rolling_radius = 0;
};
inline double volume_radius(const gr::Body& body)
{
  for(double axis:body.axes)
    if(!(axis>0.)||!std::isfinite(axis))
      throw std::invalid_argument("Near-field size requires positive finite axes");
  return std::cbrt(body.axes[0]*body.axes[1]*body.axes[2]);
}
inline PairLengths pair_lengths(double ri,double rj,double epsilon_fraction)
{
  if(!(ri>0.)||!(rj>0.)||!(epsilon_fraction>0.)||
     !std::isfinite(ri)||!std::isfinite(rj)||!std::isfinite(epsilon_fraction))
    throw std::invalid_argument("Near-field pair lengths require positive finite radii and epsilon");
  const double mean=.5*(ri+rj);
  return {mean,epsilon_fraction*mean,.5*mean,2.*ri*rj/(ri+rj)};
}

struct Result {
  Mat22 resistance = Mat22::Zero();
  bool active = false, exact_sphere = false;
  double gap = 0, regularized_gap = 0, cutoff = 0;
  double minimum_curvature_radius = 0;
};

// Physical, laboratory-frame coordinates are
//   q=(Ui-Uinf_i, Omegai-Omega_inf, -Ei, Uj-Uinf_j, Omegaj-Omega_inf, -Ej),
// with the same orthonormal STF basis as SlurryGrandPair/SlurryCompactBridge.
// The existing near-field kernels specify a UNIFORM imposed strain. Their
// lift to 22 coordinates uses E=(Ei+Ej)/2. Thus, for the prescribed Ei=Ej,
// forces, torques, summed stress and dissipation exactly match those kernels.
// The two individual strain-conjugate moments each carry half the pair stress;
// this split does not assert independent per-particle strain boundary physics.
// R*q=(on-fluid F,T,-S); physical summed stress is minus the sum of the two
// five-component moments. Do not add a second pair-force virial.
//
// All lengths, including epsilon and cutoff, are physical. reference_length
// only converts the legacy helper's L*omega coordinates; it does not change
// the physical operator. No contact-onset shift is applied: scalar resistance
// uses max(gap,0)+epsilon, independently of the mechanical contact model.
// Spheres use the exact Mari leading squeeze/shear/pump kernel. Nonspheres
// retain SlurrySpheroidHydro's local curvature cutoff and its documented
// normal-square completion, rather than the compact bridge Schur completion.
inline Result evaluate(const gr::Body& bi,const gr::Body& bj,const gr::GapResult& gap,
                       double eta,double reference_length,double epsilon,double cutoff)
{
  if (!(reference_length>0) || !std::isfinite(reference_length))
    throw std::invalid_argument("Near-field adapter requires a finite positive reference length");
  const auto pair=sh::make_pair(bi,bj,gap,eta,epsilon,cutoff);
  Result out;
  out.active=pair.active;out.exact_sphere=pair.exact_sphere;
  out.gap=pair.gap;out.regularized_gap=pair.regularized_gap;
  out.cutoff=pair.cutoff;out.minimum_curvature_radius=pair.minimum_curvature_radius;
  if (!out.active) return out;

  // Each column is one physical grand coordinate. Angular columns become
  // L*omega only on input to the legacy helper; it divides by L internally.
  auto input=[&](int column,ph::Vec12& velocity,ph::Mat3& strain) {
    const int particle=column/11,component=column%11;
    if(component<6) velocity[6*particle+component]=component<3?1.:reference_length;
    else {
      const auto& basis=SlurryGrandPair::strain_basis()[component-6];
      for(int k=0;k<9;++k) strain[k]=-.5*basis[k];
    }
  };
  if(pair.exact_sphere) {
    Eigen::Matrix<double,7,22> map;
    for(int c=0;c<22;++c) {
      ph::Vec12 velocity{};ph::Mat3 strain{};input(c,velocity,strain);
      const auto mode=ph::modes(pair.sphere,velocity,{},strain,reference_length);
      map(0,c)=mode.squeeze;
      for(int k=0;k<3;++k){map(1+k,c)=mode.shear[k];map(4+k,c)=mode.pump[k];}
    }
    Eigen::Matrix<double,7,1> weight;
    weight[0]=pair.sphere.squeeze_resistance;
    weight.segment<3>(1).setConstant(pair.sphere.shear_resistance);
    weight.segment<3>(4).setConstant(pair.sphere.pump_resistance);
    out.resistance.noalias()=map.transpose()*weight.asDiagonal()*map;
  } else {
    Eigen::Matrix<double,6,22> map;
    Eigen::Matrix<double,6,6> resistance;
    for(int r=0;r<6;++r)for(int c=0;c<6;++c)resistance(r,c)=pair.resistance[6*r+c];
    for(int c=0;c<22;++c) {
      ph::Vec12 velocity{};ph::Mat3 strain{};input(c,velocity,strain);
      const auto mode=sh::local_velocity(pair,velocity,{},strain,reference_length);
      for(int r=0;r<6;++r)map(r,c)=mode[r];
    }
    out.resistance.noalias()=map.transpose()*resistance*map;
  }
  // Roundoff symmetry only; no eigenvalue clipping or resistance modification.
  out.resistance=.5*(out.resistance+out.resistance.transpose()).eval();
  if(!out.resistance.allFinite())
    throw std::overflow_error("Nonfinite near-field grand resistance");
  return out;
}
} // namespace SlurryNearfieldHydro
#endif
