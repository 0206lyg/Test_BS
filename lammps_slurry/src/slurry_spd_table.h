#ifndef SLURRY_SPD_TABLE_H
#define SLURRY_SPD_TABLE_H

#include "slurry_spd_coordinates.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>

// Interpolates the logarithm of the conditioned TOTAL reduced resistance.
// Stored nodes contain 66 lower-triangular entries, not Fourier coefficients.
// Periodic azimuth interpolation uses all equally spaced samples. No physical
// eigenvalue is clipped; an invalid source matrix must be rejected offline.
// The isolated response is subtracted only after reconstruction, so the
// returned pair correction can be signed. The assembled many-particle system
// still requires its independent velocity/strain passivity check.
namespace SlurrySPDTable {
namespace coord=SlurrySPDCoordinates;
using SlurryScattering::Vec3;
using SlurryScattering::Mat3;
using SlurryScattering::Mat11;
using Axes=std::array<Vec3,2>;

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
inline Mat11 rotation11(const Mat3& R) {
  Mat11 Q=Mat11::Zero();Q.topLeftCorner<3,3>()=R;Q.block<3,3>(3,3)=R;
  Q.bottomRightCorner<5,5>()=SlurryScattering::strain_rotation(R);return Q;
}
inline Mat3 z_rotation(double angle) {
  Mat3 R;R<<std::cos(angle),-std::sin(angle),0.,std::sin(angle),std::cos(angle),0.,0.,0.,1.;return R;
}
inline Mat11 symmetric_exp(const Mat11& logarithm) {
  if(!logarithm.allFinite())throw std::runtime_error("Nonfinite SPD matrix logarithm");
  const Mat11 symmetric=.5*(logarithm+logarithm.transpose()).eval();
  Eigen::SelfAdjointEigenSolver<Mat11> eig(symmetric);
  if(eig.info()!=Eigen::Success)throw std::runtime_error("SPD matrix exponential eigensolve failed");
  const Eigen::Matrix<double,11,1> values=eig.eigenvalues().array().exp().matrix();
  if(!values.allFinite()||!(values.minCoeff()>0.))
    throw std::runtime_error("SPD matrix exponential overflow or underflow; clipping disabled");
  const Mat11 raw=eig.eigenvectors()*values.asDiagonal()*eig.eigenvectors().transpose();
  const Mat11 result=.5*(raw+raw.transpose()).eval();
  Eigen::LLT<Mat11> check(result);
  if(!result.allFinite()||check.info()!=Eigen::Success)
    throw std::runtime_error("SPD matrix exponential lost positive definiteness");
  return result;
}

class Provider {
 public:
  double epsilon_m=.15e-9;
  double ar=0.,hmin=0.,hmax=0.;
  int nt=0,np=0,nh=0,format_version=1;
  std::unique_ptr<coord::Shape> shape;
  std::vector<double> theta,phi,gap,angular_coordinate,log_gap,data;
  // V1 has one historical join. V2 records actual source matching joins so
  // the interpolator never bridges a derivative discontinuity implicitly.
  std::vector<double> joins{2.5};
  double aspect_ratio()const{return ar;}
  double min_gap()const{return hmin;}
  double max_gap()const{return hmax;}

  void load(const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error("Cannot open SPD resistance table: "+path);
    std::string magic;f>>magic;
    if(magic!="SLURRY_SPD_LOGFOURIER_V1"&&magic!="SLURRY_SPD_LOGFOURIER_V2")throw std::runtime_error("Invalid SPD resistance table format");
    format_version=magic=="SLURRY_SPD_LOGFOURIER_V2"?2:1;
    f>>ar>>hmin>>hmax>>nt>>np>>nh;
    if(!f||!std::isfinite(ar)||!std::isfinite(hmin)||!std::isfinite(hmax)||
       !(ar>=1.&&hmin>0.&&hmax>hmin)||nt<4||nt>128||np<4||np>256||np%2||nh<4||nh>256)
      throw std::runtime_error("Invalid SPD resistance table domain");
    joins={2.5};
    if(format_version==2){
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
    if(format_version==2)for(double join:joins){
      const auto node=std::lower_bound(gap.begin(),gap.end(),join);
      if(node==gap.end()||std::abs(*node-join)>1e-12*std::max(1.,join))
        throw std::runtime_error("SPD matching join is missing from gap grid");
    }
    for(int k=0;k<np;++k)if(std::abs(phi[k]-(-pi+2*pi*k/np))>1e-12)
      throw std::runtime_error("SPD azimuth samples must be uniform and periodic");
    const std::size_t count=std::size_t(nt)*nt*np*nh*66;
    if(count>(std::size_t(1)<<29))throw std::runtime_error("SPD table too large");
    data.resize(count);
    f.read(reinterpret_cast<char*>(data.data()),std::streamsize(count*sizeof(double)));
    if(!f)throw std::runtime_error("Truncated SPD table payload");
    for(double x:data)if(!std::isfinite(x))throw std::runtime_error("Nonfinite SPD table payload");
    angular_coordinate.resize(nt);log_gap.resize(nh);
    for(int k=0;k<nt;++k)angular_coordinate[k]=std::atan2(ar*std::sin(theta[k]),std::cos(theta[k]));
    for(int k=0;k<nh;++k)log_gap[k]=std::log(gap[k]);
    shape=std::make_unique<coord::Shape>(ar);
  }

  Stencil gap_stencil(double h)const {
    int first=0,last=nh-1;
    for(double join:joins){
      const auto node=std::lower_bound(gap.begin(),gap.end(),join);
      // V1 historically allowed a table without the 2.5 node. V2's loader
      // requires every explicit join to be an actual grid node.
      const double tolerance=format_version==1?1e-12:1e-12*std::max(1.,join);
      if(node==gap.end()||std::abs(*node-join)>tolerance)continue;
      const int k=int(node-gap.begin());
      if(h<=join){last=k;break;}else first=k;
    }
    return stencil(log_gap,std::log(h),first,last);
  }

  // Returns the conditioned symmetric matrix LOGARITHM, despite the historical
  // method name shared with the earlier additive-residual prototype.
  Mat11 residual(double ti,double tj,double azimuth,double h)const {
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
    std::array<double,66> lower{};
    for(int i=0;i<si.size;++i)for(int j=0;j<sj.size;++j)for(int p=0;p<np;++p)for(int k=0;k<sh.size;++k){
      const double w=si.weight[i]*sj.weight[j]*azimuth_weight[p]*sh.weight[k];
      const std::size_t node=((std::size_t(si.index[i])*nt+sj.index[j])*np+p)*nh+sh.index[k];
      const double* values=data.data()+66*node;
      for(int q=0;q<66;++q)lower[q]+=w*values[q];
    }
    Mat11 result;int q=0;
    for(int r=0;r<11;++r)for(int c=0;c<=r;++c)result(r,c)=result(c,r)=lower[q++];
    if(!result.allFinite())throw std::runtime_error("Nonfinite interpolated SPD logarithm");
    return result;
  }

  Mat11 correction(const Axes& input,double h,double epsilon)const {
    if(!shape)throw std::runtime_error("SPD table shape not initialized");
    if(!std::isfinite(h)||!std::isfinite(epsilon)||epsilon<0.)
      throw std::runtime_error("Invalid SPD fluid gap");
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
    const double heff=std::max(h,0.)+epsilon;
    const Mat11 logarithm=residual(ti,tj,azimuth,heff);
    const Mat11 M=shape->contact_map(axes,heff),Mi=coord::inverse_contact_map(M);
    const Mat11 Pc=Mi.transpose()*shape->preconditioner(axes,heff,heff)*Mi;
    if(!Pc.allFinite()||!(Pc(2,2)>0.))throw std::runtime_error("Invalid SPD normal conditioning metric");
    Mat11 D=Mat11::Identity();D(2,2)=std::sqrt(Pc(2,2));
    const Mat11 raw=M.transpose()*D*symmetric_exp(logarithm)*D*M;
    const Mat11 total=.5*(raw+raw.transpose()).eval();
    Eigen::LLT<Mat11> check(total);
    if(!total.allFinite()||check.info()!=Eigen::Success)
      throw std::runtime_error("Reconstructed total pair resistance is not SPD; clipping disabled");
    // Source tables describe virtual h_eff geometry. Subtract its isolated
    // response, then lift the signed correction on actual geometry below.
    const Mat11 H=total-shape->self(axes,heff);
    const Mat11 Q=rotation11(R),world=Q*H*Q.transpose();
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
    Axes axes;const gr::Body* body[]={&bi,&bj};
    for(int i=0;i<2;++i){
      const auto d=gp::to_local(p.frame,{body[i]->rotation[2],body[i]->rotation[5],body[i]->rotation[8]});
      axes[i]=Vec3(d[0],d[1],d[2]);
    }
    const Mat11 delta=correction(axes,gap_result.gap/c,epsilon_m/c)/ar;
    const auto li=gp::to_local(p.frame,gr::scale(gr::add(gap_result.leverI,gr::scale(gap_result.normal,.5*gap_result.gap)),1/ell));
    const auto lj=gp::to_local(p.frame,gr::scale(gr::sub(gap_result.leverJ,gr::scale(gap_result.normal,.5*gap_result.gap)),1/ell));
    const auto projection=gp::relative_projection(li,lj,gp::SlipConvention::NonaffineMidpoint);
    gp::Mat11 d{};for(int i=0;i<11;++i)for(int j=0;j<11;++j)d[11*i+j]=delta(i,j);
    gp::Metadata m;m.kind=gp::MatrixKind::PairCorrection;
    m.model_id="ar10_spd_logfourier_trial_v1";m.reference_id="matched_direct_offline_spd_conditioned";
    p.model=gp::window_correction(gp::lift_reduced(d,projection,m),p.weight);
    gp::validate(p.frame,p.model);p.active=true;return p;
  }
};
}
#endif
