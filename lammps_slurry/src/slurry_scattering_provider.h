#ifndef SLURRY_SCATTERING_PROVIDER_H
#define SLURRY_SCATTERING_PROVIDER_H
#include "slurry_scattering_math.h"
#include "slurry_finite_gap_math.h"
#include "slurry_stress_math.h"
#include "slurry_objective_film.h"
#include "slurry_anchor_cache.h"
#include <iomanip>
#include <sstream>
namespace SlurryScattering {
using Mat11=Eigen::Matrix<double,11,11>;
using Map6=Eigen::Matrix<double,6,12>;
inline Eigen::Matrix<double,11,11> reduced(const Mat17 &g,const Map6 &b) {
  const Eigen::Matrix<double,12,12> A=g.topLeftCorner<12,12>();
  const Eigen::Matrix<double,12,5> C=g.topRightCorner<12,5>();
  Eigen::LLT<Eigen::Matrix<double,12,12>> factor(A);
  if(factor.info()!=Eigen::Success)throw std::runtime_error("Scattering full pair velocity block is not SPD");
  const Eigen::Matrix<double,6,6> mobility=b*factor.solve(b.transpose());
  Eigen::LLT<Eigen::Matrix<double,6,6>> mf(mobility);
  if(mf.info()!=Eigen::Success)throw std::runtime_error("Scattering reduced mobility is not SPD");
  const Eigen::Matrix<double,6,6> K=mf.solve(Eigen::Matrix<double,6,6>::Identity());
  const Eigen::Matrix<double,12,5> AC=factor.solve(C);
  const Eigen::Matrix<double,6,5> F=b*AC;
  Mat11 H;H.topLeftCorner<6,6>()=K;H.topRightCorner<6,5>()=K*F;
  H.bottomLeftCorner<5,6>()=H.topRightCorner<6,5>().transpose();
  H.bottomRightCorner<5,5>()=g.bottomRightCorner<5,5>()-C.transpose()*AC+F.transpose()*K*F;
  return .5*(H+H.transpose()).eval();
}
// Gap-independent, canonical-frame anchor residuals only. POD storage avoids
// Eigen alignment assumptions inside the cross-step particle-pair cache.
struct AnchorResiduals { std::array<std::array<double,121>,3> values{}; };
inline Mat3 axis_rotation(Vec3 axis) {
  axis.normalize();Vec3 x=Vec3::UnitX();if(std::abs(x.dot(axis))>.9)x=Vec3::UnitY();
  x-=axis*x.dot(axis);x.normalize();Mat3 R;R.col(0)=x;R.col(1)=axis.cross(x);R.col(2)=axis;return R;
}
inline std::array<Body,2> geometry(const std::array<Vec3,2>& axes,double h,double a) {
  std::array<Body,2> b;
  for(int i=0;i<2;++i) {
    const auto f=SlurryObjectiveFilm::body(axes[i],i==0,a);
    b[i].rotation=axis_rotation(axes[i]);b[i].center=f.center;
  }
  b[1].center.z()+=h;return b;
}
inline Map6 midpoint_map(const std::array<Body,2>& b,double h,double a) {
  namespace gp=SlurryGrandPair;const Vec3 middle(0,0,.5*h);
  const Vec3 vi=(middle-b[0].center)/a,vj=(middle-b[1].center)/a;
  const auto p=gp::relative_projection({vi.x(),vi.y(),vi.z()},{vj.x(),vj.y(),vj.z()},gp::SlipConvention::NonaffineMidpoint);
  Map6 B;for(int i=0;i<6;++i)for(int j=0;j<12;++j)B(i,j)=p[17*i+j];return B;
}
class Provider {
 public:
  Basis basis,near_basis;
  bool hybrid=false;
  int hybrid_version=0,anchor_count=0;
  std::array<double,3> anchor_gaps{};
  double minimum_gap_over_c=2.5,reciprocity_tolerance=.01;
  double epsilon_m=.15e-9;
  // AR10 source-repair trial: 161 versus 321 is below 1% at the failed pose.
  // This is not a claim of convergence over every orientation.
  int film_grid=161;
  Mat11 isolated_local;
  mutable SlurryAnchorCache::Cache<AnchorResiduals> anchor_cache;
  void load(const std::string &path) {
    std::ifstream f(path);if(!f)throw std::runtime_error("Cannot open scattering model: "+path);
    std::string magic;f>>magic;f.clear();f.seekg(0);
    hybrid=magic=="SLURRY_HYBRID_V1"||magic=="SLURRY_HYBRID_V2"||magic=="SLURRY_HYBRID_V3";
    hybrid_version=hybrid?(magic=="SLURRY_HYBRID_V3"?3:(magic=="SLURRY_HYBRID_V2"?2:1)):0;
    anchor_count=hybrid?(hybrid_version==2?3:2):0;anchor_gaps={};
    if(hybrid) {
      double ar=0.;int low_degree=0;f>>magic>>ar>>low_degree;
      if(!std::isfinite(ar)||ar<1.||low_degree<1||low_degree>30)
        throw std::runtime_error("Invalid hybrid bundle oblate aspect ratio/order");
      if(hybrid_version>=2) {
        std::string tag;int count=0;f>>tag>>count;
        if(tag!="ANCHORS"||count!=anchor_count)throw std::runtime_error("Hybrid bundle has the wrong number of explicit matching gaps");
        for(int k=0;k<anchor_count;++k)if(!(f>>anchor_gaps[k])||!std::isfinite(anchor_gaps[k])||!(anchor_gaps[k]>0.)||(k&&!(anchor_gaps[k]>anchor_gaps[k-1])))
          throw std::runtime_error("Hybrid matching gaps must be finite, positive and strictly increasing");
        if(hybrid_version==3&&(anchor_gaps[0]!=.5||anchor_gaps[1]!=1.))
          throw std::runtime_error("Hybrid V3 requires anchors 0.5 and 1 for its direct-near Stokes policy");
      }else anchor_gaps={1.,2.5,0.};
      basis.load(f);near_basis.load(f);
      if(basis.aspect_ratio!=ar || near_basis.aspect_ratio!=ar || near_basis.degree!=low_degree)
        throw std::runtime_error("Hybrid bundle basis mismatch");
      minimum_gap_over_c=0.;
    }else {basis.load(f);minimum_gap_over_c=2.5;}
    std::ostringstream revision;revision<<std::setprecision(17)<<path<<"|bounded-log1p-fv-source-v3|bridge="<<hybrid_version<<"|ar="<<basis.aspect_ratio<<"|near="<<(hybrid?near_basis.degree:0)<<"|far="<<basis.degree;
    if(hybrid_version==3)revision<<"|finite-gap-policy=direct-near|far-slot=unused";
    for(int k=0;k<anchor_count;++k)revision<<"|h="<<anchor_gaps[k];
    anchor_cache.configure(0.,revision.str());anchor_cache.clear();
    isolated_local.setZero();const double a=basis.aspect_ratio;
    const auto coeff=SlurryStress::unit_stresslet({a,a,1.});
    const auto drag=SlurryHydro::unit_resistance({a,a,1.});
    for(int k=0;k<3;++k){isolated_local(k,k)=drag.translation[k];isolated_local(k+3,k+3)=drag.rotation[k]/(a*a);}
    const SlurryStress::Mat3 Q{1,0,0,0,1,0,0,0,1};const auto es=strain_basis();
    for(int k=0;k<5;++k) {
      SlurryStress::Mat3 E{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)E[3*i+j]=es[k](i,j)/a;
      const auto s=SlurryStress::stresslet(coeff,Q,1.,E,{0,0,0},{0,0,0});
      for(int i=0;i<3;++i)isolated_local(3+i,6+k)=isolated_local(6+k,3+i)=-s.hydro_torque[i]/a;
      for(int i=0;i<5;++i)for(int r=0;r<3;++r)for(int c=0;c<3;++c)isolated_local(6+i,6+k)+=es[i](r,c)*s.total[3*r+c]/a;
    }
  }
  Mat17 isolated(const std::array<Body,2> &b) const {
    Mat17 g=Mat17::Zero();
    for(int i=0;i<2;++i) {
      Mat11 Q=Mat11::Zero();Q.block<3,3>(0,0)=b[i].rotation;Q.block<3,3>(3,3)=b[i].rotation;
      Q.bottomRightCorner<5,5>()=strain_rotation(b[i].rotation);
      const Mat11 h=Q*isolated_local*Q.transpose();
      g.block<6,6>(6*i,6*i)=h.topLeftCorner<6,6>();
      g.block<6,5>(6*i,12)=h.topRightCorner<6,5>();g.block<5,6>(12,6*i)=h.bottomLeftCorner<5,6>();
      g.bottomRightCorner<5,5>()+=h.bottomRightCorner<5,5>();
    }return g;
  }
  Mat17 checked_grand(const Basis& source,const std::array<Body,2>& bodies) const {
    const auto raw=source.evaluate(bodies);
    if(raw.reciprocity>reciprocity_tolerance)throw std::runtime_error("Scattering approximation exceeds its raw reciprocity tolerance");
    // Numerical symmetrization defines the approximate Rayleigh model; its
    // measured discretization error must not be described as roundoff.
    return .5*(raw.grand+raw.grand.transpose()).eval();
  }
  Mat11 film11(const std::array<Vec3,2>& axes,double h) const {
    if(film_grid<21)throw std::runtime_error("Hybrid finite-film grid must be at least 21");
    const double a=basis.aspect_ratio;
    const auto film=SlurryObjectiveFilm::assemble(axes[0],axes[1],h,film_grid,a,0.);
    if(film.pressure_relative_residual>1e-8)throw std::runtime_error("Finite-film pressure residual exceeds tolerance");
    // The film is evaluated on virtual geometry at h_eff; project with that
    // SAME geometry. Only the final correction is lifted on actual geometry.
    const auto b=geometry(axes,h,a);const Map6 B=midpoint_map(b,h,a);
    const Eigen::Matrix<double,6,6> BB=B*B.transpose();
    const Eigen::Matrix<double,12,6> P=B.transpose()*BB.ldlt().solve(Eigen::Matrix<double,6,6>::Identity());
    Eigen::Matrix<double,17,11> L=Eigen::Matrix<double,17,11>::Zero();
    L.topLeftCorner<12,6>()=P;L.bottomRightCorner<5,5>().setIdentity();
    const Mat11 H=L.transpose()*film.resistance*L;return .5*(H+H.transpose()).eval();
  }
  AnchorResiduals anchors(const std::array<Vec3,2>& axes) const {
    AnchorResiduals out;const double a=basis.aspect_ratio;
    if(!hybrid||anchor_count<2||anchor_count>3)throw std::runtime_error("Matching anchors require a valid hybrid bundle");
    for(int k=0;k<anchor_count;++k) {
      const double h=anchor_gaps[k];const auto b=geometry(axes,h,a);const Map6 B=midpoint_map(b,h,a);
      const Basis& source=(hybrid_version!=3&&k+1==anchor_count)?basis:near_basis;
      const Mat11 R=reduced(checked_grand(source,b),B)-reduced(isolated(b),B)-film11(axes,h);
      auto& values=out.values[k];
      for(int i=0;i<11;++i)for(int j=0;j<11;++j)values[11*i+j]=.5*(R(i,j)+R(j,i));
    }return out;
  }
  // Returns the signed correction in c=eta=1 units. No eigenvalue clipping.
  // Shape and canonical pose are exact cache keys; h is deliberately excluded.
  Mat11 correction(const std::array<Vec3,2>& axes,double h,double epsilon,
                   std::uint64_t id_i=0,std::uint64_t id_j=1) const {
    if(!std::isfinite(h)||!(epsilon>=0)||!std::isfinite(epsilon))throw std::runtime_error("Invalid hybrid fluid gap");
    const double a=basis.aspect_ratio,heff=std::max(h,0.)+epsilon;
    // V3 uses the resolved near Stokes basis at the actual regularized fluid
    // gap throughout the finite-gap range. The serialized far slot is legacy
    // compatibility data only: no far switch or residual interpolation occurs.
    if(hybrid_version==3 && heff>=anchor_gaps[0]) {
      const auto b=geometry(axes,heff,a);const Map6 B=midpoint_map(b,heff,a);
      return reduced(checked_grand(near_basis,b),B)-reduced(isolated(b),B);
    }
    if(!hybrid || (hybrid_version!=3&&heff>anchor_gaps[anchor_count-1])) {
      if(!hybrid && h<minimum_gap_over_c)throw std::runtime_error("Scattering pair is below its checked finite-gap range; load a matched hybrid bundle");
      // Both sides of the match use the same regularized fluid geometry.
      // The returned correction is still lifted with the caller's actual B.
      const double h_eval=hybrid?heff:h;
      const auto b=geometry(axes,h_eval,a);const Map6 B=midpoint_map(b,h_eval,a);
      return reduced(checked_grand(basis,b),B)-reduced(isolated(b),B);
    }
    if(!(heff>0))throw std::runtime_error("Hybrid lubrication requires a positive physical regularization length at contact");
    SlurryAnchorCache::Geometry key;
    key.normalized_axes={a,a,1.,a,a,1.};key.settings={epsilon,anchor_gaps[0],anchor_gaps[1],anchor_gaps[2],double(film_grid),double(hybrid_version),double(near_basis.degree),double(basis.degree)};
    for(int i=0;i<2;++i)for(int j=0;j<3;++j)key.canonical_directors[3*i+j]=axes[i][j];
    const auto pair=SlurryAnchorCache::PairKey{id_i,id_j};
    const AnchorResiduals* stored=anchor_cache.find(pair,key);
    if(!stored){anchor_cache.store(pair,key,anchors(axes));stored=anchor_cache.find(pair,key);}
    int left=0;
    if(hybrid_version==2)while(left+2<anchor_count&&heff>anchor_gaps[left+1])++left;
    // V1 retains the original expression exactly. V2 matches the signed
    // residual at each declared gap. Below the first gap, log1p(h) stays
    // finite as h tends to zero, so extrapolation is bounded without clipping
    // the film resistance, forces, or any matrix eigenvalues. V3 preserves
    // this exact first-segment expression below its direct-Stokes boundary.
    const double t=hybrid_version==1?(std::log1p(heff)-std::log(2.))/(std::log(3.5)-std::log(2.)):
      (std::log1p(heff)-std::log1p(anchor_gaps[left]))/(std::log1p(anchor_gaps[left+1])-std::log1p(anchor_gaps[left]));
    Mat11 H=film11(axes,heff);
    for(int i=0;i<11;++i)for(int j=0;j<11;++j)H(i,j)+=(1-t)*stored->values[left][11*i+j]+t*stored->values[left+1][11*i+j];
    return .5*(H+H.transpose()).eval();
  }
  SlurryFiniteGap::Pair make_pair(const SlurryFiniteGap::gr::Body &bi,
      const SlurryFiniteGap::gr::Body &bj,const SlurryFiniteGap::gr::GapResult &gap,
      double eta,double solver_length,double range,double inner_ratio,
      std::uint64_t id_i=0,std::uint64_t id_j=1) const {
    namespace gp=SlurryGrandPair;namespace gr=SlurryFiniteGap::gr;
    SlurryFiniteGap::Pair p;p.frame=SlurryFiniteGap::canonical_frame(bi,bj,gap.normal,eta,solver_length);
    p.cutoff=gp::ar_scaled_range(bi.axes[0],bj.axes[0],range);
    if(!(inner_ratio>0&&inner_ratio<1))throw std::runtime_error("Invalid finite-gap window");
    p.weight=gp::range_weight(gap.gap,inner_ratio*p.cutoff,p.cutoff);if(p.weight==0)return p;
    const double a=basis.aspect_ratio,c=bi.axes[2],ell=p.frame.model_length;
    for(int k=0;k<3;++k)if(std::abs(bi.axes[k]-bj.axes[k])>1e-10*ell)throw std::runtime_error("Scattering provider requires equal-sized spheroids");
    if(!(a>1.) || std::abs(bi.axes[1]/c-a)>1e-10*a || std::abs(bi.axes[0]/c-a)>1e-10*a)
      throw std::runtime_error("Scattering basis requires equal-sized oblate spheroids matching its aspect ratio; use Mari for AR=1");
    std::array<Vec3,2> axes;const gr::Body *body[]={&bi,&bj};
    for(int i=0;i<2;++i) {
      auto axis=gp::to_local(p.frame,{body[i]->rotation[2],body[i]->rotation[5],body[i]->rotation[8]});
      axes[i]=Vec3(axis[0],axis[1],axis[2]);axes[i].normalize();if(axes[i].z()<0)axes[i]=-axes[i];
    }
    const Mat11 delta=correction(axes,gap.gap/c,epsilon_m/c,id_i,id_j)/a;
    const auto li=gp::to_local(p.frame,gr::scale(gr::add(gap.leverI,gr::scale(gap.normal,.5*gap.gap)),1/ell));
    const auto lj=gp::to_local(p.frame,gr::scale(gr::sub(gap.leverJ,gr::scale(gap.normal,.5*gap.gap)),1/ell));
    const auto projection=gp::relative_projection(li,lj,gp::SlipConvention::NonaffineMidpoint);
    gp::Mat11 d{};for(int i=0;i<11;++i)for(int j=0;j<11;++j)d[11*i+j]=delta(i,j);
    gp::Metadata m;m.kind=gp::MatrixKind::PairCorrection;
    m.model_id=hybrid?(hybrid_version==3?"spheroid_direct_stokes_near_film_v3":(hybrid_version==2?"spheroid_film_bounded_residual_v2":"spheroid_film_bounded_residual_v1")):"isolated_scattering_relative_grand_v1";
    std::ostringstream reference;reference<<std::setprecision(17)<<"spheroid_ar="<<a<<";near_degree="<<(hybrid?near_basis.degree:0)<<";far_degree="<<basis.degree<<";bridge="<<hybrid_version;
    for(int k=0;k<anchor_count;++k)reference<<";anchor="<<anchor_gaps[k];
    if(hybrid_version==3)reference<<";finite_gap_policy=direct_near_basis;far_slot=unused_legacy_data";
    m.reference_id=reference.str();
    p.model=gp::window_correction(gp::lift_reduced(d,projection,m),p.weight);
    gp::validate(p.frame,p.model);p.active=true;return p;
    // Signed pair corrections require the assembled GLOBAL velocity/grand gate.
  }
};
}
#endif
