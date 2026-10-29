#ifndef SLURRY_SPD_TABLE17_H
#define SLURRY_SPD_TABLE17_H

#include "slurry_spd_coordinates17.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

// Interpolates the logarithm of the conditioned TOTAL full17 resistance.
// Stored nodes contain 153 lower-triangular entries, not Fourier coefficients.
// Periodic azimuth interpolation uses all equally spaced samples. No physical
// eigenvalue is clipped; an invalid source matrix must be rejected offline.
// The isolated response is subtracted only after reconstruction, so the
// returned pair correction can be signed. The assembled many-particle system
// still requires its independent velocity/strain passivity check.
namespace SlurrySPDTable17 {
namespace coord=SlurrySPDCoordinates17;
using SlurryScattering::Vec3;
using SlurryScattering::Mat3;
using SlurryScattering::Mat17;
using Axes=std::array<Vec3,2>;
inline constexpr const char* table_magic="SLURRY_SPD_LOGFOURIER_V4";
inline constexpr const char* reference_table_magic="SLURRY_SPD_REFERENCE_SPARSE_V1";
inline constexpr const char* model_revision="JANOSCHEK_MATCHED_HARDFLOOR_FULL17_V4";
inline constexpr const char* dissipative_model_revision="JANOSCHEK_DISSIPATIVE_INCREMENT_FULL17_V1";
inline constexpr double reference_pose_tolerance=1e-10;

struct ReferencePose {
  double theta_i=0.,theta_j=0.,phi=0.,anchor_h_over_c=0.;
  std::string source_json_sha256,solution_sha256;
  Axes axes()const {
    return {Vec3(std::sin(theta_i),0.,std::cos(theta_i)),
      Vec3(std::sin(theta_j)*std::cos(phi),std::sin(theta_j)*std::sin(phi),std::cos(theta_j))};
  }
};

struct Stencil {int size=0;std::array<int,4> index{};std::array<double,4> weight{};};
inline Stencil stencil(const std::vector<double>& nodes,double x,int first,int last) {
  if(first<0||last<first||last>=int(nodes.size())||!std::isfinite(x))
    throw std::runtime_error("Invalid SPD interpolation segment");
  Stencil s;s.size=std::min(4,last-first+1);
  const auto upper=std::upper_bound(nodes.begin()+first,nodes.begin()+last+1,x);
  int begin=int(upper-nodes.begin())-2;
  begin=std::max(first,std::min(begin,last-s.size+1));
  for(int i=0;i<s.size;++i){s.index[i]=begin+i;s.weight[i]=1.;
    for(int j=0;j<s.size;++j)if(i!=j)
      s.weight[i]*=(x-nodes[begin+j])/(nodes[begin+i]-nodes[begin+j]);}
  return s;
}
inline Mat17 rotation17(const Mat3& R) {
  Mat17 Q=Mat17::Zero();const double parity=R.determinant();
  for(int k=0;k<2;++k){Q.block<3,3>(6*k,6*k)=R;Q.block<3,3>(6*k+3,6*k+3)=parity*R;}
  Q.bottomRightCorner<5,5>()=SlurryScattering::strain_rotation(R);return Q;
}
inline Mat3 z_rotation(double angle) {
  Mat3 R;R<<std::cos(angle),-std::sin(angle),0.,std::sin(angle),std::cos(angle),0.,0.,0.,1.;return R;
}
inline Mat17 symmetric_exp(const Mat17& logarithm) {
  if(!logarithm.allFinite())throw std::runtime_error("Nonfinite SPD matrix logarithm");
  const Mat17 symmetric=.5*(logarithm+logarithm.transpose()).eval();
  Eigen::SelfAdjointEigenSolver<Mat17> eig(symmetric);
  if(eig.info()!=Eigen::Success)throw std::runtime_error("SPD matrix exponential eigensolve failed");
  const Eigen::Matrix<double,17,1> values=eig.eigenvalues().array().exp().matrix();
  if(!values.allFinite()||!(values.minCoeff()>0.))
    throw std::runtime_error("SPD matrix exponential overflow or underflow; clipping disabled");
  const Mat17 raw=eig.eigenvectors()*values.asDiagonal()*eig.eigenvectors().transpose();
  const Mat17 result=.5*(raw+raw.transpose()).eval();
  Eigen::LLT<Mat17> check(result);
  if(!result.allFinite()||check.info()!=Eigen::Success)
    throw std::runtime_error("SPD matrix exponential lost positive definiteness");
  return result;
}

inline Mat17 normalized_total(const coord::Shape& shape,const Axes& axes,double h,const Mat17& total) {
  if(!total.allFinite())throw std::runtime_error("Nonfinite full17 source total");
  const Mat17 G=.5*(total+total.transpose()).eval();Eigen::LLT<Mat17> gate(G);
  if(gate.info()!=Eigen::Success)throw std::runtime_error("Full17 source total is not SPD; clipping disabled");
  const Mat17 L=shape.cholesky(axes,h);
  const Mat17 left=L.triangularView<Eigen::Lower>().solve(G);
  const Mat17 C=L.triangularView<Eigen::Lower>().solve(left.transpose()).transpose();
  return .5*(C+C.transpose()).eval();
}
inline Mat17 encode_total(const coord::Shape& shape,const Axes& axes,double h,const Mat17& total) {
  const Mat17 C=normalized_total(shape,axes,h,total);Eigen::SelfAdjointEigenSolver<Mat17> eig(C);
  if(eig.info()!=Eigen::Success||!eig.eigenvalues().allFinite()||!(eig.eigenvalues().minCoeff()>0.))
    throw std::runtime_error("Normalized full17 source total is not SPD; clipping disabled");
  const Mat17 log=eig.eigenvectors()*eig.eigenvalues().array().log().matrix().asDiagonal()*eig.eigenvectors().transpose();
  return .5*(log+log.transpose()).eval();
}
inline Mat17 decode_total(const coord::Shape& shape,const Axes& axes,double h,const Mat17& logarithm) {
  const Mat17 L=shape.cholesky(axes,h),G=L*symmetric_exp(logarithm)*L.transpose();
  const Mat17 total=.5*(G+G.transpose()).eval();Eigen::LLT<Mat17> check(total);
  if(!total.allFinite()||check.info()!=Eigen::Success)
    throw std::runtime_error("Decoded full17 total is not SPD; clipping disabled");
  return total;
}

class Provider {
 public:
  double epsilon_m=1e-9; // Compatibility name: physical hard minimum fluid gap.
  double ar=0.,hmin=0.,hmax=0.;
  int nt=0,np=0,nh=0,format_version=4;
  std::unique_ptr<coord::Shape> shape;
  std::vector<double> theta,phi,gap,angular_coordinate,log_gap,data;
  // Explicit matching joins keep interpolation segments on one source branch.
  std::vector<double> joins;
  // This scope is deliberately distinct from a complete angular grid. It may
  // exercise an imported, provisional reference without certifying its physics.
  bool sparse_reference=false,provisional=false,physical_accuracy_qualified=false;
  std::string reference_scope="complete_angular_grid",source_identity,loaded_model_revision=model_revision;
  double reference_length_m=0.,reference_floor_m=0.;
  std::vector<ReferencePose> reference_poses;
  double aspect_ratio()const{return ar;}
  double min_gap()const{return hmin;}
  double max_gap()const{return hmax;}

  void load(const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error("Cannot open SPD resistance table: "+path);
    std::string magic;f>>magic;
    if(magic==reference_table_magic){load_reference(f);return;}
    if(magic!=table_magic)throw std::runtime_error("SPD table belongs to an old or unsupported physical model; regenerate JANOSCHEK_MATCHED_HARDFLOOR_FULL17_V4 table");
    std::string model_marker,revision;f>>model_marker>>revision;
    if(model_marker!="MODEL"||(revision!=model_revision&&revision!=dissipative_model_revision))
      throw std::runtime_error("SPD table physical model revision mismatch; regenerate table");
    format_version=4;
    f>>ar>>hmin>>hmax>>nt>>np>>nh;
    if(!f||!std::isfinite(ar)||!std::isfinite(hmin)||!std::isfinite(hmax)||
       !(ar>=1.&&hmin>0.&&hmax>hmin)||nt<4||nt>128||np<4||np>256||np%2||nh<4||nh>4096)
      throw std::runtime_error("Invalid SPD resistance table domain");
    joins.clear();
    {
      int count=0;f>>count;
      if(!f||count<0||count>64)throw std::runtime_error("Invalid SPD matching join count");
      joins.resize(count);for(double&x:joins)f>>x;
      for(int k=0;k<count;++k)if(!std::isfinite(joins[k])||joins[k]<hmin||joins[k]>hmax||(k&&joins[k]<=joins[k-1]))
        throw std::runtime_error("Invalid SPD matching joins");
    }
    theta.resize(nt);phi.resize(np);gap.resize(nh);
    for(double&x:theta)f>>x;for(double&x:phi)f>>x;for(double&x:gap)f>>x;
    std::string marker;f>>marker;
    if(!f||marker!="DATA")throw std::runtime_error("Missing SPD table payload");
    f.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    for(const auto* nodes:{&theta,&phi,&gap})for(std::size_t k=0;k<nodes->size();++k)
      if(!std::isfinite((*nodes)[k])||(k&&(*nodes)[k]<=(*nodes)[k-1]))
        throw std::runtime_error("Non-monotone SPD table grid");
    const double pi=std::acos(-1.);
    if(std::abs(theta.front())>1e-12||std::abs(theta.back()-.5*pi)>1e-12||
       std::abs(gap.front()-hmin)>1e-12*hmin||std::abs(gap.back()-hmax)>1e-12*hmax)
      throw std::runtime_error("Incomplete SPD table domain");
    for(double join:joins){
      const auto node=std::lower_bound(gap.begin(),gap.end(),join);
      if(node==gap.end()||std::abs(*node-join)>1e-12*std::max(1.,join))
        throw std::runtime_error("SPD matching join is missing from gap grid");
    }
    for(int k=0;k<np;++k)if(std::abs(phi[k]-(-pi+2*pi*k/np))>1e-12)
      throw std::runtime_error("SPD azimuth samples must be uniform and periodic");
    // Preserve the existing 2^29-double (4 GiB) payload ceiling while checking
    // every multiplication before it can overflow the host's size_t.
    const std::size_t max_count=std::size_t(1)<<29;std::size_t count=1;
    for(int factor:{nt,nt,np,nh,153}){
      if(count>max_count/std::size_t(factor))throw std::runtime_error("SPD table too large");
      count*=std::size_t(factor);
    }
    data.resize(count);
    f.read(reinterpret_cast<char*>(data.data()),std::streamsize(count*sizeof(double)));
    if(!f)throw std::runtime_error("Truncated SPD table payload");
    if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Extra bytes after SPD table payload");
    for(double x:data)if(!std::isfinite(x))throw std::runtime_error("Nonfinite SPD table payload");
    angular_coordinate.resize(nt);log_gap.resize(nh);
    for(int k=0;k<nt;++k)angular_coordinate[k]=std::atan2(ar*std::sin(theta[k]),std::cos(theta[k]));
    for(int k=0;k<nh;++k)log_gap[k]=std::log(gap[k]);
    shape=std::make_unique<coord::Shape>(ar);
    loaded_model_revision=revision;
    sparse_reference=false;provisional=revision==dissipative_model_revision;physical_accuracy_qualified=false;
    reference_scope="complete_angular_grid";source_identity.clear();
    reference_length_m=reference_floor_m=0.;reference_poses.clear();
  }

  // Source coordinates are canonical, but runtime director signs are arbitrary.
  // Match director LINES under a proper rotation about the contact normal. The
  // second candidate handles an equatorial director without inventing a row.
  static bool reference_rotation(const ReferencePose& pose,const Axes& input,Mat3& rotation) {
    const Axes stored=pose.axes();int pivot=-1;
    for(int k=0;k<2;++k)if(stored[k].head<2>().norm()>1e-13){pivot=k;break;}
    const double pi=std::acos(-1.);
    double angle=0.;
    if(pivot>=0) {
      if(input[pivot].head<2>().norm()<1e-13)return false;
      angle=std::atan2(input[pivot].y(),input[pivot].x())-
        std::atan2(stored[pivot].y(),stored[pivot].x());
    }
    for(int branch=0;branch<(pivot<0?1:2);++branch) {
      const Mat3 candidate=z_rotation(angle+branch*pi);bool matches=true;
      for(int k=0;k<2;++k){const Vec3 expected=candidate*stored[k];
        matches=matches&&std::min((input[k]-expected).norm(),(input[k]+expected).norm())<=reference_pose_tolerance;}
      if(matches){rotation=candidate;return true;}
    }
    return false;
  }

  void load_reference(std::istream& f) {
    // Parse into a temporary provider so a malformed sparse file cannot leave
    // a partially loaded table. There is no angular or external-source fallback.
    Provider next;std::string marker,revision,status,identity;int qualified=-1,npose=0;
    f>>marker>>revision;
    if(marker!="MODEL"||(revision!=model_revision&&revision!=dissipative_model_revision))
      throw std::runtime_error("Sparse reference model revision mismatch");
    next.loaded_model_revision=revision;
    f>>marker>>status>>qualified;
    if(marker!="STATUS"||status!="PROVISIONAL"||qualified!=0)
      throw std::runtime_error("Sparse reference tables must explicitly declare PROVISIONAL 0");
    f>>marker>>identity;
    auto hex=[](const std::string& s,std::size_t low,std::size_t high){
      return s.size()>=low&&s.size()<=high&&std::all_of(s.begin(),s.end(),[](char c){
        return (c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F');});};
    if(marker!="SOURCE_FNV1A64"||!hex(identity,1,16))throw std::runtime_error("Invalid sparse source identity");
    f>>marker>>next.reference_length_m>>next.reference_floor_m;
    if(marker!="UNITS"||!std::isfinite(next.reference_length_m)||!(next.reference_length_m>0.)||
       !std::isfinite(next.reference_floor_m)||!(next.reference_floor_m>0.))
      throw std::runtime_error("Invalid sparse reference physical units");
    f>>next.ar>>next.hmin>>next.hmax>>npose>>next.nh;
    if(!f||!std::isfinite(next.ar)||next.ar<1.||!std::isfinite(next.hmin)||!(next.hmin>0.)||
       !std::isfinite(next.hmax)||!(next.hmax>next.hmin)||npose<1||npose>4096||next.nh<2||next.nh>4096)
      throw std::runtime_error("Invalid sparse reference domain");
    const double normalized_floor=next.reference_floor_m/next.reference_length_m;
    if(!std::isfinite(normalized_floor)||std::abs(normalized_floor-next.hmin)>1e-12*next.hmin)
      throw std::runtime_error("Sparse physical floor differs from c_m*hmin");
    f>>marker;if(marker!="POSES")throw std::runtime_error("Missing sparse reference poses");
    next.reference_poses.resize(npose);const double pi=std::acos(-1.);
    for(int k=0;k<npose;++k) {
      auto& p=next.reference_poses[k];
      f>>p.theta_i>>p.theta_j>>p.phi>>p.anchor_h_over_c>>p.source_json_sha256>>p.solution_sha256;
      if(!f||!std::isfinite(p.theta_i)||!std::isfinite(p.theta_j)||!std::isfinite(p.phi)||
         p.theta_i<0.||p.theta_j<0.||p.theta_i>.5*pi||p.theta_j>.5*pi||p.phi< -pi||p.phi>=pi||
         !std::isfinite(p.anchor_h_over_c)||std::abs(p.anchor_h_over_c-next.hmax)>1e-12*next.hmax||
         !hex(p.source_json_sha256,64,64)||(p.solution_sha256!="-"&&!hex(p.solution_sha256,64,64)))
        throw std::runtime_error("Invalid sparse canonical pose, common anchor, or source hash");
      if((p.theta_i==0.||p.theta_j==0.)&&p.phi!=0.)
        throw std::runtime_error("Sparse polar pose must use canonical phi=0");
      for(int j=0;j<k;++j){Mat3 rotation;
        if(reference_rotation(next.reference_poses[j],p.axes(),rotation))
          throw std::runtime_error("Duplicate sparse reference pose");}
    }
    f>>marker;if(marker!="GAPS")throw std::runtime_error("Missing sparse reference gaps");
    next.gap.resize(next.nh);next.log_gap.resize(next.nh);
    for(int k=0;k<next.nh;++k){f>>next.gap[k];
      if(!f||!std::isfinite(next.gap[k])||!(next.gap[k]>0.)||(k&&next.gap[k]<=next.gap[k-1]))
        throw std::runtime_error("Invalid sparse gap grid");
      next.log_gap[k]=std::log(next.gap[k]);}
    if(std::abs(next.gap.front()-next.hmin)>1e-12*next.hmin||
       std::abs(next.gap.back()-next.hmax)>1e-12*next.hmax)
      throw std::runtime_error("Sparse gap grid must include floor and common anchor");
    f>>marker;if(marker!="DATA")throw std::runtime_error("Missing sparse reference payload");
    f.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    const std::size_t count=std::size_t(npose)*next.nh*153;
    if(count>(std::size_t(1)<<29))throw std::runtime_error("Sparse reference table too large");
    next.data.resize(count);f.read(reinterpret_cast<char*>(next.data.data()),std::streamsize(count*sizeof(double)));
    if(!f)throw std::runtime_error("Truncated sparse reference payload");
    if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Extra bytes after sparse reference payload");
    for(double x:next.data)if(!std::isfinite(x))throw std::runtime_error("Nonfinite sparse reference payload");
    next.shape=std::make_unique<coord::Shape>(next.ar);
    next.epsilon_m=next.reference_floor_m;next.sparse_reference=true;next.provisional=true;
    next.reference_scope="exact_poses_only";next.source_identity="fnv1a64:"+identity;
    *this=std::move(next);
  }

  std::size_t reference_match(const Axes& input,Mat3& rotation)const {
    if(!sparse_reference||!shape||reference_poses.empty()||data.empty()||nh<2)
      throw std::runtime_error("Sparse reference table not loaded");
    for(std::size_t k=0;k<reference_poses.size();++k)
      if(reference_rotation(reference_poses[k],input,rotation))return k;
    throw std::runtime_error("Sparse reference table covers exact imported poses only; unsupported orientation");
  }

  Mat17 reference_log(std::size_t pose,double h)const {
    if(pose>=reference_poses.size()||!std::isfinite(h)||h<hmin*(1-1e-12)||h>hmax*(1+1e-12))
      throw std::runtime_error("Gap outside sparse reference domain; extrapolation disabled");
    h=std::max(hmin,std::min(hmax,h));const auto s=gap_stencil(h);
    std::array<double,153> lower{};
    for(int k=0;k<s.size;++k){const double* values=data.data()+153*(pose*nh+s.index[k]);
      for(int q=0;q<153;++q)lower[q]+=s.weight[k]*values[q];}
    Mat17 result;int q=0;
    for(int r=0;r<17;++r)for(int c=0;c<=r;++c)result(r,c)=result(c,r)=lower[q++];
    if(!result.allFinite())throw std::runtime_error("Nonfinite sparse interpolated logarithm");
    return result;
  }

  Stencil gap_stencil(double h)const {
    int first=0,last=nh-1;
    for(double join:joins){
      const auto node=std::lower_bound(gap.begin(),gap.end(),join);
      // The loader requires every explicit join to be an actual grid node.
      const double tolerance=1e-12*std::max(1.,join);
      if(node==gap.end()||std::abs(*node-join)>tolerance)continue;
      const int k=int(node-gap.begin());
      if(h<=join){last=k;break;}else first=k;
    }
    return stencil(log_gap,std::log(h),first,last);
  }

  // Returns the conditioned symmetric matrix LOGARITHM, despite the historical
  // method name shared with the earlier additive-residual prototype.
  Mat17 residual(double ti,double tj,double azimuth,double h)const {
    if(sparse_reference) {
      const double pi=std::acos(-1.);
      if(!std::isfinite(ti)||!std::isfinite(tj)||!std::isfinite(azimuth)||ti<0.||tj<0.||ti>.5*pi||tj>.5*pi)
        throw std::runtime_error("Invalid sparse reference query pose");
      ReferencePose query;query.theta_i=ti;query.theta_j=tj;query.phi=std::remainder(azimuth,2*pi);
      const Axes axes=query.axes();Mat3 R;const auto index=reference_match(axes,R);
      const Mat17 logarithm=reference_log(index,h);
      if((R-Mat3::Identity()).norm()<1e-14)return logarithm;
      // A conditioned logarithm itself does not transform by rotation17: the
      // Cholesky coordinates also change. Transport the TOTAL then re-encode.
      const Mat17 Q=rotation17(R),total=decode_total(*shape,reference_poses[index].axes(),h,logarithm);
      return encode_total(*shape,axes,h,Q*total*Q.transpose());
    }
    if(data.empty()||nt<4||np<4||np%2||nh<4)
      throw std::runtime_error("SPD resistance table not loaded");
    const double pi=std::acos(-1.);
    if(!std::isfinite(ti)||!std::isfinite(tj)||!std::isfinite(azimuth)||!std::isfinite(h)||
       ti< -1e-12||tj< -1e-12||ti>.5*pi+1e-12||tj>.5*pi+1e-12)
      throw std::runtime_error("Invalid SPD query pose");
    if(h<hmin*(1-1e-12)||h>hmax*(1+1e-12))
      throw std::runtime_error("Gap outside SPD table domain; extrapolation disabled");
    h=std::max(hmin,std::min(hmax,h));
    ti=std::max(0.,std::min(.5*pi,ti));tj=std::max(0.,std::min(.5*pi,tj));
    const auto si=stencil(angular_coordinate,std::atan2(ar*std::sin(ti),std::cos(ti)),0,nt-1);
    const auto sj=stencil(angular_coordinate,std::atan2(ar*std::sin(tj),std::cos(tj)),0,nt-1);
    azimuth=std::remainder(azimuth,2*pi);
    std::vector<double> azimuth_weight(np);
    for(int k=0;k<np;++k){const double d=azimuth-phi[k];
      double w=1.+std::cos((np/2)*d);
      for(int m=1;m<np/2;++m)w+=2.*std::cos(m*d);
      azimuth_weight[k]=w/np;
    }
    const auto sh=gap_stencil(h);
    std::array<double,153> lower{};
    for(int i=0;i<si.size;++i)for(int j=0;j<sj.size;++j)for(int p=0;p<np;++p)for(int k=0;k<sh.size;++k){
      const double w=si.weight[i]*sj.weight[j]*azimuth_weight[p]*sh.weight[k];
      const std::size_t node=((std::size_t(si.index[i])*nt+sj.index[j])*np+p)*nh+sh.index[k];
      const double* values=data.data()+153*node;
      for(int q=0;q<153;++q)lower[q]+=w*values[q];
    }
    Mat17 result;int q=0;
    for(int r=0;r<17;++r)for(int c=0;c<=r;++c)result(r,c)=result(c,r)=lower[q++];
    if(!result.allFinite())throw std::runtime_error("Nonfinite interpolated SPD logarithm");
    return result;
  }

  Mat17 correction17(const Axes& input,double h,double epsilon)const {
    if(!shape)throw std::runtime_error("SPD table shape not initialized");
    if(!std::isfinite(h)||!std::isfinite(epsilon)||epsilon<0.)
      throw std::runtime_error("Invalid SPD fluid gap");
    if(sparse_reference) {
      if(epsilon>0.&&std::abs(epsilon-hmin)>1e-10*hmin)
        throw std::runtime_error("Sparse reference query must retain its stored physical gap floor");
      Axes axes=input;for(auto& a:axes){
        if(!a.allFinite()||a.norm()<1e-15)throw std::runtime_error("Invalid sparse reference director");
        a.normalize();if(a.z()<0.)a=-a;}
      Mat3 R;const auto index=reference_match(axes,R);const double heff=std::max(h,hmin);
      const Axes stored=reference_poses[index].axes();
      const Mat17 total=decode_total(*shape,stored,heff,reference_log(index,heff));
      const Mat17 Q=rotation17(R),world=Q*(total-shape->self(stored,heff))*Q.transpose();
      if(!world.allFinite())throw std::runtime_error("Nonfinite sparse reference correction");
      return .5*(world+world.transpose()).eval();
    }
    Axes axes=input;
    for(auto& a:axes){
      if(!a.allFinite()||a.norm()<1e-15)throw std::runtime_error("Invalid SPD director");
      a.normalize();if(a.z()<0.)a=-a;
    }
    double angle=0.;
    if(axes[0].head<2>().norm()>1e-13)angle=std::atan2(axes[0].y(),axes[0].x());
    else if(axes[1].head<2>().norm()>1e-13)angle=std::atan2(axes[1].y(),axes[1].x());
    const Mat3 R=z_rotation(angle);for(auto& a:axes)a=R.transpose()*a;
    const double ti=std::atan2(axes[0].head<2>().norm(),axes[0].z());
    const double tj=std::atan2(axes[1].head<2>().norm(),axes[1].z());
    const double azimuth=std::atan2(axes[1].y(),axes[1].x());
    const double heff=std::max(h,epsilon);
    const Mat17 logarithm=residual(ti,tj,azimuth,heff);
    const Mat17 total=decode_total(*shape,axes,heff,logarithm);
    // Subtract the virtual-gap isolated response; retain all twelve particle
    // motion modes and five strain modes in the actual runtime frame.
    const Mat17 H=total-shape->self(axes,heff);
    const Mat17 Q=rotation17(R),world=Q*H*Q.transpose();
    if(!world.allFinite())throw std::runtime_error("Nonfinite SPD pair correction");
    return .5*(world+world.transpose()).eval();
  }

  SlurryFiniteGap::Pair make_pair(const SlurryFiniteGap::gr::Body& bi,const SlurryFiniteGap::gr::Body& bj,
      const SlurryFiniteGap::gr::GapResult& gap_result,double eta,double solver_length,double range,double inner_ratio,
      std::uint64_t=0,std::uint64_t=1)const {
    namespace gp=SlurryGrandPair;namespace gr=SlurryFiniteGap::gr;
    SlurryFiniteGap::Pair p;p.frame=SlurryFiniteGap::canonical_frame(bi,bj,gap_result.normal,eta,solver_length);
    p.cutoff=gp::ar_scaled_range(bi.axes[0],bj.axes[0],range);
    if(!(inner_ratio>0.&&inner_ratio<1.))throw std::runtime_error("Invalid SPD range window");
    p.weight=gp::range_weight(gap_result.gap,inner_ratio*p.cutoff,p.cutoff);if(p.weight==0.)return p;
    const double c=bi.axes[2],ell=p.frame.model_length;
    for(int k=0;k<3;++k)if(std::abs(bi.axes[k]-bj.axes[k])>1e-10*ell)
      throw std::runtime_error("SPD table requires equal size spheroids");
    if(std::abs(bi.axes[0]/c-ar)>1e-10*ar||std::abs(bi.axes[1]/c-ar)>1e-10*ar)
      throw std::runtime_error("SPD table aspect ratio mismatch");
    if(sparse_reference&&(!std::isfinite(c)||std::abs(c-reference_length_m)>1e-10*reference_length_m||
       !std::isfinite(epsilon_m)||std::abs(epsilon_m-reference_floor_m)>1e-10*reference_floor_m))
      throw std::runtime_error("Sparse reference particle length or physical floor mismatch");
    Axes axes;const gr::Body* body[]={&bi,&bj};
    for(int i=0;i<2;++i){
      const auto d=gp::to_local(p.frame,{body[i]->rotation[2],body[i]->rotation[5],body[i]->rotation[8]});
      axes[i]=Vec3(d[0],d[1],d[2]);
    }
    const Mat17 delta=correction17(axes,gap_result.gap/c,epsilon_m/c)/ar;
    gp::Model model;model.metadata.kind=gp::MatrixKind::PairCorrection;
    model.metadata.model_id=loaded_model_revision;model.metadata.reference_id="janoschek_matched_stokes_full17_offline";
    if(loaded_model_revision==dissipative_model_revision)
      model.metadata.reference_id="provisional;physical_accuracy_qualified=0;complete_angular_grid;janoschek_dissipative_increment_full17_offline";
    if(sparse_reference)model.metadata.reference_id="provisional;physical_accuracy_qualified=0;exact_poses_only;"+source_identity;
    for(int i=0;i<17;++i)for(int j=0;j<17;++j)model.resistance[17*i+j]=delta(i,j);
    p.model=gp::window_correction(model,p.weight);
    gp::validate(p.frame,p.model);p.active=true;return p;
  }
};
}
#endif
