#ifndef SLURRY_CYLINDER_PACKING_H
#define SLURRY_CYLINDER_PACKING_H

// Geometry-only preparation for continuous cylinders. A supporting plane
// supplies a rigorous LOWER bound on the surface clearance. Maximizing that
// bound gives the separating direction used by the artificial FIRE energy;
// no signed penetration depth is claimed to be a physical contact law.
// Every prepared configuration is independently certified with closest_gap.
#include "slurry_cylinder_geometry.h"
#include <array>

namespace SlurryCylinderPacking {
namespace cg = SlurryCylinderGeometry;
using V3 = Eigen::Vector3d;
struct Plane {
  double gap = -std::numeric_limits<double>::infinity();
  V3 normal = V3::UnitZ(), lever_i = V3::Zero(), lever_j = V3::Zero();
};
inline Plane plane(const cg::Body &a, const cg::Body &b, V3 n) {
  Plane out;
  if (!(n.norm() > 0.)) return out;
  n.normalize();
  out.normal = n;
  out.lever_i = cg::support(a,n)-a.center;
  out.lever_j = cg::support(b,-n)-b.center;
  out.gap = n.dot(b.center-a.center-out.lever_i+out.lever_j);
  return out;
}
inline Plane improve(const cg::Body &a, const cg::Body &b, Plane current,
                     int maximum = 48) {
  const V3 d=b.center-a.center;
  const double scale=std::max({a.radius,a.half_thickness,b.radius,b.half_thickness});
  for(int iteration=0;iteration<maximum;++iteration) {
    const V3 gradient=d-current.lever_i+current.lever_j-current.gap*current.normal;
    const double norm=gradient.norm();
    if(norm<=1e-11*scale)break;
    const V3 tangent=gradient/norm;
    double angle=std::min(.25,norm/scale);
    bool accepted=false;
    for(int line=0;line<32;++line) {
      Plane trial=plane(a,b,std::cos(angle)*current.normal+std::sin(angle)*tangent);
      if(trial.gap>current.gap+1e-4*angle*norm) {
        current=trial;accepted=true;break;
      }
      angle*=.5;
    }
    if(!accepted)break;
  }
  return current;
}
inline Plane separating_plane(const cg::Body &a,const cg::Body &b,
                              const V3 *warm=nullptr) {
  const V3 d=b.center-a.center, na=a.axis.normalized(),nb=b.axis.normalized();
  std::array<V3,7> directions{{d,na,nb,na.cross(nb),na+nb,na-nb,
                              warm?*warm:d}};
  std::array<Plane,7> trials;
  for(std::size_t k=0;k<directions.size();++k) {
    V3 n=directions[k];
    if(n.dot(d)<0.)n=-n;
    trials[k]=plane(a,b,n);
  }
  std::sort(trials.begin(),trials.end(),[](const Plane &x,const Plane &y){return x.gap>y.gap;});
  Plane best=trials.front();
  // Several starts keep a cached face normal from hiding a newly relevant rim.
  // A nonoptimal result stays conservative: its support gap is a lower bound.
  for(int k=0;k<3;++k) {
    if(!std::isfinite(trials[k].gap))continue;
    Plane candidate=improve(a,b,trials[k]);
    if(candidate.gap>best.gap)best=candidate;
  }
  return best;
}
}
#endif
