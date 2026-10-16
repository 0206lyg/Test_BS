#ifndef SLURRY_GR_RE2_GRAPHITE_NEAR_FIELD_RESISTANCE_H
#define SLURRY_GR_RE2_GRAPHITE_NEAR_FIELD_RESISTANCE_H

#include "particleMath.h"
#include <limits>

// SLURRY SCOPE BEGIN
namespace slurry { namespace gr_re2 {

namespace graphite {

// Sub-grid hydrodynamics used by the quick calculation.  The normal coefficient
// is the leading Reynolds squeeze resistance for two locally quadratic surfaces.
// Tangential resistance is a local Couette-patch approximation, with its induced
// translation/rotation coupling.  This is NOT the complete two-spheroid
// resistance matrix: independent rolling/twisting modes are not included.
struct NearFieldSettings {
  double viscosity=.000890;
  double matchingGap=50.e-9;
  bool enabled=true;
  bool tangential=true;
};

struct NearFieldResult {
  Vec3 forceI{},torqueI{},torqueJ{};
  double normalResistance=0.,tangentialResistance=0.,dissipation=0.;
  bool active=false;
};

inline Mat3 nfCurvature(const Body& b,const Vec3& n) {
  const Vec3 a2{b.axes[0]*b.axes[0],b.axes[1]*b.axes[1],b.axes[2]*b.axes[2]};
  const Mat3 q=rotatedDiagonal(b.rotation,a2);
  const double support=std::sqrt(dot(n,mul(q,n)));
  Mat3 c=rotatedDiagonal(b.rotation,{1./a2[0],1./a2[1],1./a2[2]});
  for(double& v:c)v*=support;
  // Only tangent-plane contractions of c are used below.
  return c;
}

inline NearFieldResult nearFieldResistance(
    const Body& bi,const Body& bj,double gap,const Vec3& normal,
    const Vec3& leverI,const Vec3& leverJ,const NearFieldSettings& settings) {
  NearFieldResult result;
  if(!settings.enabled || gap>=settings.matchingGap)return result;
  if(!(gap>0.) || !std::isfinite(gap))
    throw std::domain_error("Near-field resistance requires a positive gap");
  if(!(settings.viscosity>0.) || !(settings.matchingGap>0.))
    throw std::invalid_argument("Invalid near-field viscosity or matching gap");
  const Vec3 n=normalized(normal);
  const Vec3 e1=normalized(cross(n,std::abs(n[0])<.8?Vec3{1.,0.,0.}:Vec3{0.,1.,0.}));
  const Vec3 e2=cross(n,e1);
  Mat3 c=nfCurvature(bi,n);const Mat3 cj=nfCurvature(bj,n);
  for(int k=0;k<9;++k)c[k]+=cj[k];
  const double c11=dot(e1,mul(c,e1)),c22=dot(e2,mul(c,e2));
  const double c12=dot(e1,mul(c,e2));
  const double rootDet=std::sqrt(std::max(0.,c11*c22-c12*c12));
  if(!(rootDet>0.))throw std::domain_error("Degenerate ellipsoid contact curvature");
  constexpr double pi=3.1415926535897932384626433832795;
  // For equal spheres of radius a, this reduces to
  // (3*pi*eta*a*a/2)*(1/h - 1/h_match).
  result.normalResistance=12.*pi*settings.viscosity/((c11+c22)*rootDet)
      *(1./gap-1./settings.matchingGap);
  result.tangentialResistance=settings.tangential
      ?2.*pi*settings.viscosity/rootDet*std::log(settings.matchingGap/gap):0.;

  // Put the opposing tractions at their common mid-gap point.  This keeps the
  // approximation exactly force/torque conserving, including LE image offsets.
  // The O(h/a) difference from the actual surface application points is explicit.
  const Vec3 point=scale(add(add(bi.position,leverI),add(bj.position,leverJ)),.5);
  const Vec3 li=sub(point,bi.position),lj=sub(point,bj.position);
  const Vec3 vi=add(bi.velocity,cross(bi.omega,li));
  const Vec3 vj=add(bj.velocity,cross(bj.omega,lj));
  const Vec3 slip=sub(vj,vi);
  const double normalSpeed=dot(slip,n);
  const Vec3 tangentSlip=sub(slip,scale(n,normalSpeed));
  result.forceI=add(scale(n,result.normalResistance*normalSpeed),
                    scale(tangentSlip,result.tangentialResistance));
  result.torqueI=cross(li,result.forceI);
  result.torqueJ=scale(cross(lj,result.forceI),-1.);
  result.dissipation=result.normalResistance*normalSpeed*normalSpeed
                     +result.tangentialResistance*dot(tangentSlip,tangentSlip);
  result.active=true;
  return result;
}

} // namespace graphite

} } // SLURRY SCOPE END
#endif
