#ifndef SLURRY_FLAKE_DIAGNOSTICS_H
#define SLURRY_FLAKE_DIAGNOSTICS_H

// Observations at the accepted, unchanged velocity. These are not a second
// force solve and must not be labelled a counterfactual suspension viscosity.
#include "slurry_cylinder_pair.h"
#include <array>
#include <cstdint>
#include <ostream>
#include <stdexcept>

namespace SlurryFlakeDiagnostics {
namespace cp = SlurryCylinderPair;
namespace cg = SlurryCylinderGeometry;
using V3 = Eigen::Vector3d;
using Vec22 = Eigen::Matrix<double,22,1>;

struct PairRecord {
  int i=0,j=0;
  cp::Diagnostics channels;
  V3 normal=V3::UnitZ(), lever_i=V3::Zero(), lever_j=V3::Zero();
  double gap_over_floor=0., area_fraction=0.;
};
inline PairRecord make_pair(int i,int j,const cg::Body& a,const cg::Body& b,
                            const cg::Gap& gap,const cp::Diagnostics& channels,
                            double gap_floor) {
  if (!(gap_floor>0.)) throw std::invalid_argument("Diagnostic gap floor must be positive");
  PairRecord r; r.i=i;r.j=j;r.channels=channels;
  r.normal=gap.normal.normalized();
  const V3 point=.5*(gap.point_i+gap.point_j);
  r.lever_i=point-a.center;r.lever_j=point-b.center;
  r.gap_over_floor=gap.gap/gap_floor;
  const double radius=std::min(a.radius,b.radius);
  r.area_fraction=channels.area/(std::acos(-1.)*radius*radius);
  return r;
}

struct Stats {
  std::int64_t candidate_pairs=0,cutoff_active_pairs=0,face_active_pairs=0;
  std::int64_t film_evaluated_pairs=0,film_overlap_pairs=0,angle_excluded_pairs=0;
  double face_power=0.,rim_power=0.;
  // Additional face power if only the angle weight were removed while the
  // existing rim remained. Unknown, not zero, beyond the film angle domain.
  double extra_face_frozen_power=0.;
  // Change if the convex face/rim mixture instead set angle_weight=1 at the
  // SAME velocity. The removed rim is accounted for, so this can be negative.
  double replacement_frozen_power_delta=0.;
  double face_shear_moment=0.,rim_shear_moment=0.;
  double overlap_area_fraction_sum=0.;
  double commonpoint_normal_speed2_sum=0.,commonpoint_tangent_speed2_sum=0.;
  void add(const Stats& s) {
    candidate_pairs+=s.candidate_pairs;cutoff_active_pairs+=s.cutoff_active_pairs;
    face_active_pairs+=s.face_active_pairs;film_evaluated_pairs+=s.film_evaluated_pairs;
    film_overlap_pairs+=s.film_overlap_pairs;angle_excluded_pairs+=s.angle_excluded_pairs;
    face_power+=s.face_power;rim_power+=s.rim_power;
    extra_face_frozen_power+=s.extra_face_frozen_power;
    replacement_frozen_power_delta+=s.replacement_frozen_power_delta;
    face_shear_moment+=s.face_shear_moment;rim_shear_moment+=s.rim_shear_moment;
    overlap_area_fraction_sum+=s.overlap_area_fraction_sum;
    commonpoint_normal_speed2_sum+=s.commonpoint_normal_speed2_sum;
    commonpoint_tangent_speed2_sum+=s.commonpoint_tangent_speed2_sum;
  }
};

inline int angle_bin(double angle) {
  const double pi=std::acos(-1.);
  if (angle<.1) return 0;
  if (angle<10.*pi/180.) return 1;
  if (angle<15.*pi/180.) return 2;
  if (angle<.35) return 3;
  return 4;
}
inline int gap_bin(double h_over_floor) {
  if(h_over_floor<1.)return 0;
  if(h_over_floor<3.)return 1;
  if(h_over_floor<10.)return 2;
  if(h_over_floor<30.)return 3;
  if(h_over_floor<100.)return 4;
  return 5;
}
inline const std::array<const char*,5>& angle_labels() {
  static const std::array<const char*,5> v{{"0_to_5.72958_deg","5.72958_to_10_deg",
      "10_to_15_deg","15_to_20.05352_deg","20.05352_to_90_deg"}};return v;
}
inline const std::array<const char*,6>& gap_labels() {
  static const std::array<const char*,6> v{{"below_1_floor","1_to_3_floor","3_to_10_floor",
      "10_to_30_floor","30_to_100_floor","100_plus_floor"}};return v;
}

inline Vec22 gather_q(const PairRecord& r,const Eigen::VectorXd& q) {
  if(r.i<0||r.j<0||11*(std::max(r.i,r.j)+1)>q.size())
    throw std::invalid_argument("Diagnostic pair index exceeds generalized velocity");
  Vec22 local;local.head<11>()=q.segment<11>(11*r.i);
  local.tail<11>()=q.segment<11>(11*r.j);return local;
}
inline double shear_moment(const Vec22& m) {
  // STF index 3 is xz, normalized by 1/sqrt(2), for each particle.
  return -(m[9]+m[20])/std::sqrt(2.);
}
inline double global_shear_moment(const Eigen::VectorXd& m) {
  if(m.size()%11)throw std::invalid_argument("Diagnostic moment must have 11N entries");
  double s=0.;for(Eigen::Index i=9;i<m.size();i+=11)s-=m[i]/std::sqrt(2.);return s;
}

struct Accumulator {
  Stats total;
  std::array<Stats,5> by_angle;
  std::array<Stats,6> by_gap;
  Eigen::VectorXd face_moment,rim_moment;
  explicit Accumulator(int count):face_moment(Eigen::VectorXd::Zero(11*count)),
      rim_moment(Eigen::VectorXd::Zero(11*count)) {}
  Stats add(const PairRecord& r,const Eigen::VectorXd& q) {
    if(q.size()!=face_moment.size())throw std::invalid_argument("Diagnostic system size mismatch");
    const auto& d=r.channels;const Vec22 local=gather_q(r,q);
    const Vec22 raw_face=d.film*local,raw_rim=d.rim*local;
    const double face_coefficient=d.face_enabled?d.cutoff_weight*d.face_weight:0.;
    const double rim_coefficient=d.cutoff_weight*(1.-d.face_weight);
    const Vec22 mf=face_coefficient*raw_face,mr=rim_coefficient*raw_rim;
    // There is no factor 1/2: enumerate i<j/all images and i==j/positive images
    // exactly as the runtime does. A self-image deposits both blocks at i.
    face_moment.segment<11>(11*r.i)+=mf.head<11>();
    face_moment.segment<11>(11*r.j)+=mf.tail<11>();
    rim_moment.segment<11>(11*r.i)+=mr.head<11>();
    rim_moment.segment<11>(11*r.j)+=mr.tail<11>();
    Stats s;s.candidate_pairs=1;s.cutoff_active_pairs=d.cutoff_weight>0.;
    s.face_active_pairs=face_coefficient>0.;s.film_evaluated_pairs=d.film_evaluated;
    s.film_overlap_pairs=d.film_active;s.angle_excluded_pairs=d.angle_excluded;
    s.face_power=local.dot(mf);s.rim_power=local.dot(mr);
    s.face_shear_moment=shear_moment(mf);s.rim_shear_moment=shear_moment(mr);
    if(d.film_active) {
      s.overlap_area_fraction_sum=r.area_fraction;
      const double removed=d.face_enabled?d.cutoff_weight*d.area_weight*(1.-d.angle_weight):0.;
      s.extra_face_frozen_power=removed*local.dot(raw_face);
      s.replacement_frozen_power_delta=removed*local.dot(raw_face-raw_rim);
    }
    if(d.cutoff_weight>0.) {
      // Actual relative rigid-surface velocity at the common force point.
      // q already carries -E and nonaffine velocities; adding K*image again
      // would double count the Lees--Edwards affine image shift.
      V3 slip=local.segment<3>(11)-local.segment<3>(0)+
          local.segment<3>(14).cross(r.lever_j)-local.segment<3>(3).cross(r.lever_i);
      const auto& basis=SlurryCylinderFilm::strain_basis();
      for(int k=0;k<5;++k)
        slip+=local[17+k]*basis[k]*r.lever_j-local[6+k]*basis[k]*r.lever_i;
      const double normal=r.normal.dot(slip);
      s.commonpoint_normal_speed2_sum=normal*normal;
      s.commonpoint_tangent_speed2_sum=(slip-normal*r.normal).squaredNorm();
    }
    total.add(s);by_gap[gap_bin(r.gap_over_floor)].add(s);
    // No meaningful angle was evaluated for pairs outside the compact cutoff.
    if(d.cutoff_weight>0.)by_angle[angle_bin(d.angle)].add(s);
    return s;
  }
};

inline const char* csv_columns() {
  return "candidate_pairs,cutoff_active_pairs,face_active_pairs,film_evaluated_pairs,"
      "film_overlap_pairs,angle_excluded_pairs,face_power_W,rim_power_W,"
      "extra_face_frozen_power_W,replacement_frozen_power_delta_W,"
      "face_sigma_xz_Pa,rim_sigma_xz_Pa,overlap_area_fraction_sum,"
      "commonpoint_normal_speed2_sum_m2_s2,commonpoint_tangent_speed2_sum_m2_s2";
}
inline void write_csv_values(std::ostream& out,const Stats& s,double power_scale,
                              double stress_scale,double velocity_scale) {
  out<<s.candidate_pairs<<','<<s.cutoff_active_pairs<<','<<s.face_active_pairs<<','
     <<s.film_evaluated_pairs<<','<<s.film_overlap_pairs<<','<<s.angle_excluded_pairs<<','
     <<s.face_power*power_scale<<','<<s.rim_power*power_scale<<','
     <<s.extra_face_frozen_power*power_scale<<','<<s.replacement_frozen_power_delta*power_scale<<','
     <<s.face_shear_moment*stress_scale<<','<<s.rim_shear_moment*stress_scale<<','
     <<s.overlap_area_fraction_sum<<','<<s.commonpoint_normal_speed2_sum*velocity_scale*velocity_scale<<','
     <<s.commonpoint_tangent_speed2_sum*velocity_scale*velocity_scale;
}
} // namespace SlurryFlakeDiagnostics
#endif
