#ifndef SLURRY_CYLINDER_PAIR_H
#define SLURRY_CYLINDER_PAIR_H

// EXPERIMENTAL cylinder near-field model. The face term is finite-domain
// Reynolds lubrication. The rim term below is an explicit positive proxy,
// NOT an analytical sharp-cylinder lubrication solution. The self block is
// deliberately labelled a matched-spheroid approximation until calibrated.
#include "slurry_cylinder_geometry.h"
#include "slurry_cylinder_film.h"
#include "slurry_periodic_mobility.h"
#include <map>
#include <mutex>
#include <utility>

namespace SlurryCylinderPair {
namespace cg = SlurryCylinderGeometry;
namespace cf = SlurryCylinderFilm;
namespace pm = SlurryPeriodic;
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;
using Mat11 = Eigen::Matrix<double, 11, 11>;
using Mat22 = Eigen::Matrix<double, 22, 22>;
struct Options {
  cf::Options film;
  bool face_enabled = true; // diagnostic ablation; retain the same rim weight
  double cutoff_start = .1, cutoff_end = 1.; // actual gap / smaller radius
  double rim_radius_factor = 1.; // proxy radius / smaller half-thickness
  double face_angle_start = .10, face_angle_end = .35; // radians
  double face_area_fraction = .01; // blend only the last 1% of face area
};
// Optional observation of the very same evaluation; never changes its model.
// Raw matrices are copied only when requested, avoiding a second film solve.
struct Diagnostics {
  Mat22 film = Mat22::Zero(), rim = Mat22::Zero();
  double angle = 0., angle_weight = 0., area_weight = 0., cutoff_weight = 0.;
  double face_weight = 0., area = 0., film_min_gap = 0., film_max_gap = 0.;
  double minimum_gap = 0., gap_floor = 0.;
  bool film_evaluated = false, film_active = false, face_enabled = true;
  bool angle_excluded = false;
};
struct Result {
  Mat22 resistance = Mat22::Zero();
  bool active = false, face_active = false;
  double face_weight = 0., cutoff_weight = 0., area = 0., minimum_gap = 0.;
  int pressure_modes = 0;
};
inline M3 cross_matrix(const V3 &v) {
  M3 m; m << 0., -v.z(), v.y(), v.z(), 0., -v.x(), -v.y(), v.x(), 0.;
  return m;
}
inline double smooth_rise(double t) {
  if (t <= 0.) return 0.;
  if (t >= 1.) return 1.;
  return t*t*t*(10.+t*(-15.+6.*t));
}
inline double cutoff_bound(const cg::Body &a, const cg::Body &b,
                           const Options &o = Options{}) {
  return std::hypot(a.radius,a.half_thickness) +
         std::hypot(b.radius,b.half_thickness) +
         o.cutoff_end * std::min(a.radius,b.radius);
}
inline Mat22 rim_resistance(const cg::Body &a, const cg::Body &b,
                           const cg::Gap &g, double floor, double radius_factor) {
  // Equal positive normal/tangential scalar channels at one common point.
  // Using a common application point preserves pair angular momentum and
  // makes a common rigid rotation an exact null mode.
  const double pi = std::acos(-1.), h = std::max(g.gap, floor);
  const double rho = radius_factor * std::min(a.half_thickness,b.half_thickness);
  const double reff = .5 * rho;
  const double zn = 6.*pi*reff*reff/h;
  const double zt = 6.*pi*reff*std::log1p(reff/h);
  const V3 n = g.normal.normalized(), p = .5*(g.point_i+g.point_j);
  const V3 li = p-a.center, lj = p-b.center;
  Eigen::Matrix<double,3,22> map = Eigen::Matrix<double,3,22>::Zero();
  map.block<3,3>(0,0) = -M3::Identity();
  map.block<3,3>(0,3) = cross_matrix(li);
  map.block<3,3>(0,11) = M3::Identity();
  map.block<3,3>(0,14) = -cross_matrix(lj);
  const auto E = pm::stf_basis();
  for (int k=0;k<5;++k) {
    map.col(6+k) = -E[k]*li;
    map.col(17+k) = E[k]*lj;
  }
  const M3 D = zt*M3::Identity()+(zn-zt)*n*n.transpose();
  return map.transpose()*D*map;
}

inline Result evaluate(const cg::Body &a, const cg::Body &b, const cg::Gap &g,
                       const Options &o = Options{}, Diagnostics *diagnostics = nullptr) {
  if (diagnostics) {
    *diagnostics = Diagnostics{}; diagnostics->face_enabled=o.face_enabled;
    diagnostics->minimum_gap=g.gap; diagnostics->gap_floor=o.film.gap_floor;
  }
  if (!g.converged || g.overlap || !std::isfinite(g.gap) || g.gap < 0.)
    throw std::invalid_argument("Cylinder pair requires certified separated geometry");
  if (!(o.cutoff_end > o.cutoff_start && o.cutoff_start >= 0.) ||
      !(o.face_angle_end > o.face_angle_start && o.face_angle_start >= 0.) ||
      !(o.rim_radius_factor > 0. && o.face_area_fraction > 0.) ||
      !(o.film.eta > 0. && o.film.gap_floor > 0.))
    throw std::invalid_argument("Invalid cylinder pair model settings");
  Result r; r.minimum_gap=g.gap;
  const double a0=std::min(a.radius,b.radius), z=g.gap/a0;
  r.cutoff_weight=1.-smooth_rise((z-o.cutoff_start)/(o.cutoff_end-o.cutoff_start));
  if (diagnostics) diagnostics->cutoff_weight=r.cutoff_weight;
  if (r.cutoff_weight == 0.) return r;
  const double angle=std::acos(std::clamp(std::abs(a.axis.normalized().dot(b.axis.normalized())),0.,1.));
  const double angle_weight=1.-smooth_rise((angle-o.face_angle_start)/(o.face_angle_end-o.face_angle_start));
  if (diagnostics) {
    diagnostics->angle=angle; diagnostics->angle_weight=angle_weight;
    diagnostics->angle_excluded=angle>=o.face_angle_end;
  }
  Mat22 film=Mat22::Zero();
  if (angle_weight > 0.) {
    auto options=o.film;
    options.max_face_angle=std::max(options.max_face_angle,o.face_angle_end);
    const auto f=cf::evaluate(a,b,options);
    r.area=f.area;r.pressure_modes=f.modes;
    if (diagnostics) {
      diagnostics->film_evaluated=true; diagnostics->film_active=f.active;
      diagnostics->film=f.resistance; diagnostics->area=f.area;
      diagnostics->film_min_gap=f.min_gap; diagnostics->film_max_gap=f.max_gap;
    }
    if (f.active) {
      r.face_weight=angle_weight*smooth_rise(f.area/(o.face_area_fraction*std::acos(-1.)*a0*a0));
      r.face_active=r.face_weight>0. && o.face_enabled;
      if (diagnostics) {
        diagnostics->area_weight=smooth_rise(f.area/(o.face_area_fraction*std::acos(-1.)*a0*a0));
        diagnostics->face_weight=r.face_weight;
      }
      if (o.face_enabled) film=f.resistance;
    }
  }
  const Mat22 rim=o.film.eta*rim_resistance(a,b,g,o.film.gap_floor,o.rim_radius_factor);
  if (diagnostics) diagnostics->rim=rim;
  r.resistance=r.cutoff_weight*(r.face_weight*film+(1.-r.face_weight)*rim);
  r.resistance=.5*(r.resistance+r.resistance.transpose()).eval();
  r.active=r.resistance.cwiseAbs().maxCoeff()>0.;
  return r;
}

inline Mat11 self_mobility(const cg::Body &b) {
  cg::validate(b);
  // Cache the expensive positive angular integration in body coordinates.
  static std::map<std::pair<double,double>,Mat11> cache;
  static std::mutex mutex;
  Mat11 body;
  {
    std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_pair(b.radius,b.half_thickness);
    auto it = cache.find(key);
    if (it == cache.end()) {
      M3 shape = M3::Zero();
      shape.diagonal() << b.radius*b.radius,b.radius*b.radius,
                          b.half_thickness*b.half_thickness;
      it = cache.emplace(key,pm::self_tail(shape,0.,1.,256)).first;
    }
    body = it->second;
  }
  const M3 Q = Eigen::Quaterniond::FromTwoVectors(V3::UnitZ(),b.axis.normalized()).toRotationMatrix();
  const Mat11 R = pm::rotation11(Q);
  return R*body*R.transpose();
}
} // namespace SlurryCylinderPair
#endif
