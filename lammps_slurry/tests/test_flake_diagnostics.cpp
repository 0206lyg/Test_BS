#include "slurry_flake_diagnostics.h"
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace cp=SlurryCylinderPair;
namespace cg=SlurryCylinderGeometry;
namespace fd=SlurryFlakeDiagnostics;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(double a,double b,const char* message) {
  require(std::abs(a-b)<2e-12*std::max({1.,std::abs(a),std::abs(b)}),message);
}
int main() {
  cg::Body a,b;a.radius=b.radius=2.;a.half_thickness=b.half_thickness=.1;
  cp::Options options;options.film.gap_floor=.001;
  for(double theta:{0.,.2,.349999,.4}) {
    b.axis={std::sin(theta),0.,std::cos(theta)};
    b.center={0.,0.,.1+.1*std::cos(theta)+2.*std::sin(theta)+.02};
    const auto gap=cg::closest_gap(a,b);
    cp::Diagnostics d;
    const auto plain=cp::evaluate(a,b,gap,options);
    const auto observed=cp::evaluate(a,b,gap,options,&d);
    require(std::memcmp(plain.resistance.data(),observed.resistance.data(),22*22*sizeof(double))==0,
            "Enabling diagnostics altered the resistance bits");
    Eigen::VectorXd q=Eigen::VectorXd::LinSpaced(22,-.3,.7);
    fd::Accumulator sum(2);
    const auto r=fd::make_pair(0,1,a,b,gap,d,options.film.gap_floor);
    sum.add(r,q);
    close(sum.total.face_power+sum.total.rim_power,q.dot(plain.resistance*q),"Pair channel power closure failed");
    require((sum.face_moment+sum.rim_moment-plain.resistance*q).norm()<
            2e-12*std::max(1.,(plain.resistance*q).norm()),"Pair channel moment closure failed");
    close(sum.total.face_shear_moment+sum.total.rim_shear_moment,
          fd::global_shear_moment(plain.resistance*q),"Pair shear stress closure failed");
    if(theta==0.)close(sum.total.extra_face_frozen_power,0.,"Parallel pair has spurious angle loss");
    if(theta==.349999) {
      require(d.film_active&&d.angle_weight<1e-9,"Tiny angle-weight fixture not active");
      require(std::isfinite(sum.total.extra_face_frozen_power)&&sum.total.extra_face_frozen_power>=0.,
              "Near cutoff angle diagnostic divides by tiny weight");
    }
    if(theta>=.35) {
      require(sum.total.angle_excluded_pairs==1&&!d.film_evaluated,"Out-of-domain film counted as evaluated");
      require(sum.total.film_overlap_pairs==0,"Unavailable film was reported as known overlap");
    }
    require(sum.by_gap[fd::gap_bin(gap.gap/options.film.gap_floor)].candidate_pairs==1,
            "Gap bin lost pair");
  }

  // Independent synthetic PSD matrices expose self-image assembly factors and
  // the sign of a convex replacement, without relying on one film geometry.
  fd::PairRecord self;self.i=self.j=0;self.gap_over_floor=2.;
  self.channels.film=2.*cp::Mat22::Identity();self.channels.rim=3.*cp::Mat22::Identity();
  self.channels.cutoff_weight=.8;self.channels.area_weight=.75;
  self.channels.angle_weight=2./3.;self.channels.face_weight=.5;
  self.channels.film_active=self.channels.film_evaluated=true;
  Eigen::VectorXd q=Eigen::VectorXd::LinSpaced(11,-.8,1.2);
  const fd::Vec22 local=fd::gather_q(self,q);
  const cp::Mat22 R=.8*(.5*self.channels.film+.5*self.channels.rim);
  const fd::Vec22 moment=R*local;
  fd::Accumulator sum(1);sum.add(self,q);
  const Eigen::VectorXd assembled=moment.head<11>()+moment.tail<11>();
  require((sum.face_moment+sum.rim_moment-assembled).norm()<1e-12,"Self-image blocks were not both deposited");
  close(sum.total.face_power+sum.total.rim_power,q.dot(assembled),"Self-image power has wrong factor");
  close(sum.total.face_shear_moment+sum.total.rim_shear_moment,
        fd::global_shear_moment(assembled),"Self-image stress has wrong factor");
  require(sum.total.extra_face_frozen_power>0.&&sum.total.replacement_frozen_power_delta<0.,
          "Additional face and convex replacement powers were conflated");
  const cp::Mat22 replacement=.8*(.75*self.channels.film+.25*self.channels.rim);
  close(sum.total.replacement_frozen_power_delta,local.dot((replacement-R)*local),
        "Frozen convex replacement delta is incorrect");
  const Eigen::VectorXd middle=.5*q,total=middle+assembled;
  close(q.dot(total),q.dot(middle)+sum.total.face_power+sum.total.rim_power,
        "Middle plus pair power decomposition failed");
  close(fd::global_shear_moment(total),fd::global_shear_moment(middle)+
        sum.total.face_shear_moment+sum.total.rim_shear_moment,"Middle plus pair stress decomposition failed");
  // Common-point velocity uses -E and relative Omega exactly once. In simple
  // shear two nonrotating plates with affine centers slip by gamma*dz.
  fd::PairRecord slip;slip.i=0;slip.j=1;slip.normal=Eigen::Vector3d::UnitZ();
  slip.lever_i={0.,0.,1.};slip.lever_j={0.,0.,-1.};
  slip.channels.cutoff_weight=1.;
  Eigen::VectorXd shear=Eigen::VectorXd::Zero(22);
  shear[4]=shear[15]=-.5; // Omega_particle - Omega_infinity
  shear[9]=shear[20]=-1./std::sqrt(2.); // -E:xz STF
  fd::Accumulator slide(2);slide.add(slip,shear);
  close(slide.total.commonpoint_normal_speed2_sum,0.,"Simple shear created normal slip");
  close(slide.total.commonpoint_tangent_speed2_sum,4.,"Affine image velocity was omitted or counted twice");
  require(fd::angle_bin(.35)==4&&fd::gap_bin(1.)==1&&fd::gap_bin(3.)==2&&
          fd::gap_bin(10.)==3&&fd::gap_bin(30.)==4&&fd::gap_bin(100.)==5,"Bin boundary convention changed");
  std::cout<<"PASS flake diagnostics: bitwise resistance, channel closure, self images, tiny/excluded angles, signed replacement\n";
}
