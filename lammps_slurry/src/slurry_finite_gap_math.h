// Experimental finite-gap pair evaluation from explicitly covered reference tables.
// No angular extrapolation and no replacement of unvalidated geometry by another.
#ifndef SLURRY_FINITE_GAP_MATH_H
#define SLURRY_FINITE_GAP_MATH_H
#include "slurry_grand_pair_math.h"
#include "slurry_grand_pair_lookup.h"
#include "../vendor/gr_re2/ellipsoidGap.h"
#include <fstream>
#include <sstream>
#include <limits>
#include <vector>
namespace SlurryFiniteGap {
namespace gp=SlurryGrandPair;
namespace gr=slurry::gr_re2::graphite;
using Mat5=std::array<double,25>;
struct Factor {std::array<double,36> logk{};std::array<double,30> response{};Mat5 remainder{};};
struct Node {double gap=0;gp::Mat11 matrix{};std::string reference;Factor factor{};};
struct Family {
  gp::GeometryKey geometry{};std::string id;
  std::vector<Node> nodes;std::vector<bool> checked;
  bool logarithmic=true,factored=false;double tolerance=1e-8;
};
struct Catalogue {std::vector<Family> families;std::string path;};
struct Pair {
  bool active=false;double cutoff=0,weight=0;
  gp::Frame frame{};gp::Model model{};
};
inline void validate_reduced(const gp::Mat11& h) {
  double scale=0;for(double v:h){if(!std::isfinite(v))throw std::invalid_argument("Nonfinite finite-gap reduced matrix");scale=std::max(scale,std::abs(v));}
  for(int i=0;i<11;++i)for(int j=0;j<11;++j)
    if(std::abs(h[11*i+j]-h[11*j+i])>1e-11*std::max(scale,1e-300))
      throw std::invalid_argument("Nonreciprocal finite-gap reduced matrix");
}
inline Mat5 strain_schur(const gp::Mat11& h) {
  validate_reduced(h);double scale=0;for(double v:h)scale=std::max(scale,std::abs(v));
  std::array<double,36> l{};
  for(int i=0;i<6;++i)for(int j=0;j<=i;++j){
    double x=h[11*i+j];for(int k=0;k<j;++k)x-=l[6*i+k]*l[6*j+k];
    if(i==j){if(!(x>1e-13*std::max(scale,1e-300)))throw std::invalid_argument("Finite-gap velocity block is not positive definite");l[6*i+j]=std::sqrt(x);}
    else l[6*i+j]=x/l[6*j+j];
  }
  std::array<double,30> y{};
  for(int c=0;c<5;++c)for(int i=0;i<6;++i){double x=h[11*i+6+c];for(int k=0;k<i;++k)x-=l[6*i+k]*y[5*k+c];y[5*i+c]=x/l[6*i+i];}
  Mat5 s{};for(int i=0;i<5;++i)for(int j=0;j<5;++j){s[5*i+j]=h[11*(6+i)+6+j];for(int k=0;k<6;++k)s[5*i+j]-=y[5*k+i]*y[5*k+j];}
  return s;
}
inline std::array<double,36> symmetric_function6(std::array<double,36> a,bool logarithm) {
  std::array<double,36> v{};for(int k=0;k<6;++k)v[7*k]=1;
  bool converged=false;
  for(int iter=0;iter<300;++iter){int p=0,q=1;double off=0,scale=0;
    for(int i=0;i<6;++i){scale=std::max(scale,std::abs(a[7*i]));for(int j=i+1;j<6;++j)if(std::abs(a[6*i+j])>off){off=std::abs(a[6*i+j]);p=i;q=j;}}
    if(off<=1e-14*std::max(scale,1e-300)){converged=true;break;}
    const double x=a[7*p],z=a[7*q],y=a[6*p+q];
    const double angle=.5*std::atan2(2*y,z-x),c=std::cos(angle),s=std::sin(angle);
    for(int k=0;k<6;++k)if(k!=p&&k!=q){const double u=a[6*k+p],w=a[6*k+q];a[6*k+p]=a[6*p+k]=c*u-s*w;a[6*k+q]=a[6*q+k]=s*u+c*w;}
    a[7*p]=c*c*x-2*c*s*y+s*s*z;a[7*q]=s*s*x+2*c*s*y+c*c*z;a[6*p+q]=a[6*q+p]=0;
    for(int k=0;k<6;++k){const double u=v[6*k+p],w=v[6*k+q];v[6*k+p]=c*u-s*w;v[6*k+q]=s*u+c*w;}
  }
  if(!converged)throw std::runtime_error("Finite-gap symmetric eigensolver did not converge");
  std::array<double,36> out{};
  for(int k=0;k<6;++k){if(logarithm&&!(a[7*k]>0))throw std::runtime_error("Finite-gap matrix logarithm requires SPD velocity block");
    const double value=logarithm?std::log(a[7*k]):std::exp(a[7*k]);if(!std::isfinite(value))throw std::runtime_error("Finite-gap matrix function overflow");
    for(int i=0;i<6;++i)for(int j=0;j<6;++j)out[6*i+j]+=v[6*i+k]*value*v[6*j+k];
  }
  return out;
}
inline Factor factor_matrix(const gp::Mat11& h) {
  Factor f;f.remainder=strain_schur(h);std::array<double,36> k{},l{};
  for(int i=0;i<6;++i)for(int j=0;j<6;++j)k[6*i+j]=h[11*i+j];
  f.logk=symmetric_function6(k,true);
  for(int i=0;i<6;++i)for(int j=0;j<=i;++j){double x=k[6*i+j];for(int z=0;z<j;++z)x-=l[6*i+z]*l[6*j+z];l[6*i+j]=i==j?std::sqrt(x):x/l[6*j+j];}
  std::array<double,30> y{};
  for(int c=0;c<5;++c){
    for(int i=0;i<6;++i){double x=h[11*i+6+c];for(int k=0;k<i;++k)x-=l[6*i+k]*y[5*k+c];y[5*i+c]=x/l[6*i+i];}
    for(int i=5;i>=0;--i){double x=y[5*i+c];for(int k=i+1;k<6;++k)x-=l[6*k+i]*f.response[5*k+c];f.response[5*i+c]=x/l[6*i+i];}
  }
  return f;
}
inline gp::Mat11 interpolate_factored(const Factor&a,const Factor&b,double t) {
  std::array<double,36> logk{};std::array<double,30> f{},c{};Mat5 s{};
  for(int j=0;j<36;++j)logk[j]=(1-t)*a.logk[j]+t*b.logk[j];
  for(int j=0;j<30;++j)f[j]=(1-t)*a.response[j]+t*b.response[j];
  for(int j=0;j<25;++j)s[j]=(1-t)*a.remainder[j]+t*b.remainder[j];
  const auto k=symmetric_function6(logk,false);gp::Mat11 h{};
  for(int i=0;i<6;++i)for(int j=0;j<6;++j)h[11*i+j]=k[6*i+j];
  for(int i=0;i<6;++i)for(int j=0;j<5;++j){for(int a=0;a<6;++a)c[5*i+j]+=k[6*i+a]*f[5*a+j];h[11*i+6+j]=h[11*(6+j)+i]=c[5*i+j];}
  for(int i=0;i<5;++i)for(int j=0;j<5;++j){double v=s[5*i+j];for(int a=0;a<6;++a)v+=f[5*a+i]*c[5*a+j];h[11*(6+i)+6+j]=v;}
  // Enforce only numerical roundoff reciprocity in the reconstructed block.
  for(int i=0;i<11;++i)for(int j=i+1;j<11;++j)h[11*i+j]=h[11*j+i]=.5*(h[11*i+j]+h[11*j+i]);
  return h;
}
inline std::string token(std::istream& in){std::string s;while(in>>s){if(!s.empty()&&s[0]=='#'){std::getline(in,s);continue;}return s;}throw std::runtime_error("Unexpected end of finite-gap table");}
inline void expect(std::istream& in,const char* s){if(token(in)!=s)throw std::runtime_error(std::string("Expected finite-gap token ")+s);}
inline Catalogue read_catalogue(const std::string& path){
  std::ifstream in(path);if(!in)throw std::runtime_error("Cannot open finite-gap table: "+path);
  Catalogue out;out.path=path;expect(in,"SLURRY_FINITE_GAP_V1");expect(in,"FAMILIES");int count=0;in>>count;
  if(count<1||count>100000)throw std::runtime_error("Invalid finite-gap family count");
  for(int n=0;n<count;++n){
    Family f;expect(in,"FAMILY");f.id=token(in);
    expect(in,"AXES_I");for(double&x:f.geometry.axes_i)in>>x;
    expect(in,"AXES_J");for(double&x:f.geometry.axes_j)in>>x;
    expect(in,"AXIS_I");for(double&x:f.geometry.symmetry_axis_i)in>>x;
    expect(in,"AXIS_J");for(double&x:f.geometry.symmetry_axis_j)in>>x;
    expect(in,"INTERPOLATION");const auto coord=token(in);
    if(coord!="LOG"&&coord!="LINEAR"&&coord!="FACTORED_LOG")throw std::runtime_error("Invalid finite-gap interpolation coordinate");
    f.logarithmic=coord!="LINEAR";f.factored=coord=="FACTORED_LOG";
    expect(in,"TOLERANCE");in>>f.tolerance;
    expect(in,"NODES");int nn=0;in>>nn;if(nn<1||nn>100000)throw std::runtime_error("Invalid finite-gap node count");
    if(!gp::valid_geometry(f.geometry)||!(f.tolerance>0&&f.tolerance<=1e-6))throw std::runtime_error("Invalid finite-gap calibrated geometry/tolerance");
    for(int k=0;k<nn;++k){Node p;expect(in,"NODE");in>>p.gap;p.reference=token(in);for(double&v:p.matrix)in>>v;
      if(!in||!(p.gap>0)||!std::isfinite(p.gap)||(k&&p.gap<=f.nodes.back().gap))throw std::runtime_error("Invalid finite-gap node gap/matrix");
      validate_reduced(p.matrix);if(f.factored)p.factor=factor_matrix(p.matrix);f.nodes.push_back(p);
    }
    expect(in,"CHECKED_INTERVALS");int ni=0;in>>ni;if(ni!=nn-1)throw std::runtime_error("Wrong finite-gap interval count");
    for(int k=0;k<ni;++k){int v=-1;in>>v;if(v!=0&&v!=1)throw std::runtime_error("Invalid finite-gap interval evidence flag");f.checked.push_back(v!=0);}
    expect(in,"END_FAMILY");out.families.push_back(f);
  }
  expect(in,"END");return out;
}
inline gp::Frame canonical_frame(const gr::Body& bi,const gr::Body& bj,const gr::Vec3& normal,double eta,double solver_length){
  gp::Frame f;f.viscosity=eta;f.solver_length=solver_length;f.model_length=.5*(bi.axes[0]+bj.axes[0]);
  auto n=gr::normalized(normal);
  gr::Vec3 ni{bi.rotation[2],bi.rotation[5],bi.rotation[8]},nj{bj.rotation[2],bj.rotation[5],bj.rotation[8]};
  if(gr::dot(ni,n)<0)ni=gr::scale(ni,-1);
  if(gr::dot(nj,n)<0)nj=gr::scale(nj,-1);
  auto x=gr::sub(ni,gr::scale(n,gr::dot(ni,n)));
  if(gr::norm(x)<1e-10)x=gr::sub(nj,gr::scale(n,gr::dot(nj,n)));
  if(gr::norm(x)<1e-10){const gr::Vec3 ref=std::abs(n[0])<.8?gr::Vec3{1,0,0}:gr::Vec3{0,1,0};x=gr::sub(ref,gr::scale(n,gr::dot(ref,n)));}
  // Near a director/normal pole the projection above loses relative accuracy.
  // Build the tangent plane with cross products so that normalizing the small
  // projected vector does not amplify its roundoff component along n.
  x=gr::normalized(x);const auto y=gr::normalized(gr::cross(n,x));
  x=gr::cross(y,n);
  for(int k=0;k<3;++k){f.world_from_local[3*k]=x[k];f.world_from_local[3*k+1]=y[k];f.world_from_local[3*k+2]=n[k];}
  return f;
}
inline Pair make_pair(const Catalogue&cat,const gr::Body&bi,const gr::Body&bj,const gr::GapResult&gap,
                      double eta,double solver_length,double range,double inner_ratio){
  Pair p;p.frame=canonical_frame(bi,bj,gap.normal,eta,solver_length);
  const double ell=p.frame.model_length;
  p.cutoff=gp::ar_scaled_range(bi.axes[0],bj.axes[0],range);
  if(!(inner_ratio>0&&inner_ratio<1))throw std::invalid_argument("Finite-gap inner/outer ratio must be between zero and one");
  p.weight=gp::range_weight(gap.gap,inner_ratio*p.cutoff,p.cutoff);
  if(p.weight==0)return p;
  gp::GeometryKey key;
  for(int k=0;k<3;++k){key.axes_i[k]=bi.axes[k]/ell;key.axes_j[k]=bj.axes[k]/ell;}
  key.symmetry_axis_i=gp::to_local(p.frame,{bi.rotation[2],bi.rotation[5],bi.rotation[8]});
  key.symmetry_axis_j=gp::to_local(p.frame,{bj.rotation[2],bj.rotation[5],bj.rotation[8]});
  const Family* family=nullptr;
  for(const auto&f:cat.families)if(gp::same_geometry(f.geometry,key,f.tolerance)){family=&f;break;}
  if(!family)throw std::runtime_error("Experimental finite-gap table does not cover this particle shape/orientation; angular extrapolation is disabled");
  const double h=gap.gap/ell;const auto&nodes=family->nodes;gp::Mat11 reduced{};std::string ref;
  if(h<nodes.front().gap-1e-12||h>nodes.back().gap+1e-12)throw std::runtime_error("Experimental finite-gap table does not cover this gap; near-gap matching/extrapolation is disabled");
  bool found=false;
  for(std::size_t k=0;k<nodes.size();++k){
    if(std::abs(h-nodes[k].gap)<1e-12){reduced=nodes[k].matrix;ref=nodes[k].reference;found=true;break;}
    if(h<nodes[k].gap&&k){
      if(!family->checked[k-1])throw std::runtime_error("Experimental finite-gap interval lacks an independent off-node check");
      const auto&a=nodes[k-1];const auto&b=nodes[k];
      const double t=family->logarithmic?std::log(h/a.gap)/std::log(b.gap/a.gap):(h-a.gap)/(b.gap-a.gap);
      if(family->factored)reduced=interpolate_factored(a.factor,b.factor,t);
      else for(int j=0;j<121;++j)reduced[j]=(1-t)*a.matrix[j]+t*b.matrix[j];
      ref=a.reference+"|"+b.reference;found=true;break;
    }
  }
  if(!found)throw std::runtime_error("Finite-gap table lookup failed");
  const auto li=gp::to_local(p.frame,gr::scale(gr::add(gap.leverI,gr::scale(gap.normal,.5*gap.gap)),1/ell));
  const auto lj=gp::to_local(p.frame,gr::scale(gr::sub(gap.leverJ,gr::scale(gap.normal,.5*gap.gap)),1/ell));
  gp::Metadata metadata;metadata.kind=gp::MatrixKind::PairCorrection;metadata.model_id=family->id;metadata.reference_id=ref;
  p.model=gp::lift_reduced(reduced,gp::relative_projection(li,lj,gp::SlipConvention::NonaffineMidpoint),metadata);
  p.model=gp::window_correction(p.model,p.weight);gp::validate(p.frame,p.model);
  p.active=true;return p;
}
inline double minimum_eigenvalue(Mat5 matrix){
  for(int iter=0;iter<100;++iter){int p=0,q=1;double off=0,scale=0;
    for(int i=0;i<5;++i){scale=std::max(scale,std::abs(matrix[6*i]));for(int j=i+1;j<5;++j)if(std::abs(matrix[5*i+j])>off){off=std::abs(matrix[5*i+j]);p=i;q=j;}}
    if(off<=1e-14*std::max(scale,1e-300))break;
    const double a=matrix[6*p],d=matrix[6*q],b=matrix[5*p+q];
    const double angle=.5*std::atan2(2*b,d-a),c=std::cos(angle),s=std::sin(angle);
    for(int k=0;k<5;++k)if(k!=p&&k!=q){const double x=matrix[5*k+p],y=matrix[5*k+q];matrix[5*k+p]=matrix[5*p+k]=c*x-s*y;matrix[5*k+q]=matrix[5*q+k]=s*x+c*y;}
    matrix[6*p]=c*c*a-2*c*s*b+s*s*d;matrix[6*q]=s*s*a+2*c*s*b+c*c*d;matrix[5*p+q]=matrix[5*q+p]=0;
  }
  double out=matrix[0];for(int k=1;k<5;++k)out=std::min(out,matrix[6*k]);return out;
}
} // namespace SlurryFiniteGap
#endif
