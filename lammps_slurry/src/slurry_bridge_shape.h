#ifndef SLURRY_BRIDGE_SHAPE_H
#define SLURRY_BRIDGE_SHAPE_H

// LAMMPS ellipsoid atoms store positions, quaternions and three half-extents.
// With shape=flake those extents describe a continuous circular cylinder;
// no spheroid distance or volume is used for its contact geometry.
#include "slurry_cylinder_geometry.h"
#include "../vendor/gr_re2/ellipsoidGap.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace SlurryBridgeShape {
namespace gr = slurry::gr_re2::graphite;
namespace cg = SlurryCylinderGeometry;
inline Eigen::Vector3d vector(const gr::Vec3 &v) { return {v[0],v[1],v[2]}; }
inline gr::Vec3 array(const Eigen::Vector3d &v) { return {v[0],v[1],v[2]}; }
inline cg::Body cylinder(const gr::Body &b) {
  cg::Body out;
  out.center=vector(b.position);
  out.axis={b.rotation[2],b.rotation[5],b.rotation[8]};
  out.radius=b.axes[0]; out.half_thickness=b.axes[2];
  if (std::abs(b.axes[0]-b.axes[1])>1e-10*b.axes[0])
    throw std::invalid_argument("flake requires equal equatorial half-extents");
  cg::validate(out);
  return out;
}
inline double bounding_radius(const gr::Body &b,bool flake) {
  return flake ? std::hypot(b.axes[0],b.axes[2])
               : *std::max_element(b.axes.begin(),b.axes.end());
}
inline cg::Gap cylinder_gap(const cg::Body &a,const cg::Body &b,double tolerance) {
  const auto g=cg::closest_gap(a,b,tolerance);
  if (!g.converged) throw std::runtime_error("flake gap certificate did not converge");
  if (g.overlap) throw std::runtime_error("flake cylinders overlap (no geometry advance accepted)");
  return g;
}
inline gr::GapResult gap_result(const cg::Body &a,const cg::Body &b,const cg::Gap &g) {
  gr::GapResult out;
  out.gap=g.gap; out.normal=array(g.normal);
  out.leverI=array(g.point_i-a.center); out.leverJ=array(g.point_j-b.center);
  out.converged=g.converged; out.iterations=g.iterations;
  return out;
}
inline gr::GapResult closest_gap(const gr::Body &a,const gr::Body &b,
                                 bool flake,double tolerance=1e-10) {
  if (!flake) return gr::closestEllipsoidGap(a,b);
  const auto ca=cylinder(a), cb=cylinder(b);
  return gap_result(ca,cb,cylinder_gap(ca,cb,tolerance));
}
inline std::array<double,3> principal_inertia(double mass,const double *axes,bool flake) {
  if (flake) {
    const double a2=axes[0]*axes[0], c2=axes[2]*axes[2];
    return {mass*(a2/4.+c2/3.),mass*(a2/4.+c2/3.),mass*a2/2.};
  }
  std::array<double,3> result;
  for (int k=0;k<3;++k)
    result[k]=mass/5.*(axes[(k+1)%3]*axes[(k+1)%3]+axes[(k+2)%3]*axes[(k+2)%3]);
  return result;
}
} // namespace SlurryBridgeShape
#endif
