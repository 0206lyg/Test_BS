#include "slurry_general_near.h"
#include <algorithm>
#include <vector>
namespace SlurryGeneral {
namespace {
using Poly=std::vector<double>;
using Mat2=Eigen::Matrix2d;
using Tangent=Eigen::Matrix<double,3,2>;
using Mat6=Eigen::Matrix<double,6,6>;
using Vec6=Eigen::Matrix<double,6,1>;
constexpr double pi=3.141592653589793238462643383279502884;
int choose(int n,int k) {
  int out=1;for(int i=1;i<=k;++i)out=out*(n-i+1)/i;return out;
}
Poly convolve(const Poly& a,const Poly& b) {
  Poly out(a.size()+b.size()-1,0.);
  for(std::size_t i=0;i<a.size();++i)for(std::size_t j=0;j<b.size();++j)out[i+j]+=a[i]*b[j];
  return out;
}
Poly rotate_polynomial(const Poly& a,double c,double s) {
  const int n=int(a.size())-1;Poly out(n+1,0.);
  for(int i=0;i<=n;++i)for(int j=0;j<=n-i;++j)for(int k=0;k<=i;++k)
    out[j+k]+=a[i]*choose(n-i,j)*choose(i,k)*std::pow(c,n-i-j)*std::pow(s,j)*std::pow(-s,i-k)*std::pow(c,k);
  return out;
}
struct Polynomials {Poly cubic,quartic;};
Polynomials body_polynomials(const Mat3& shape,const Mat3& inverse,const Tangent& tangent,double sign) {
  const double support=std::sqrt(shape(2,2));
  const Mat2 qt=tangent.transpose()*inverse*tangent;
  const Eigen::Vector2d qn=tangent.transpose()*inverse.col(2);
  const Poly quadratic{qt(0,0),2*qt(0,1),qt(1,1)},normal{qn[0],qn[1]};
  Polynomials out;out.cubic=convolve(normal,quadratic);
  for(double& v:out.cubic)v*=-.5*sign*support*support;
  out.quartic=convolve(convolve(normal,normal),quadratic);
  const Poly q2=convolve(quadratic,quadratic);
  for(std::size_t i=0;i<out.quartic.size();++i)
    out.quartic[i]=support*support*support*(.5*out.quartic[i]+.125*inverse(2,2)*q2[i]);
  return out;
}
Mat3 skew(const Vec3& v) {
  Mat3 out;out<<0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;return out;
}
}
std::array<double,2> compact_near_factors(double h,double cutoff) {
  if(!(h>0&&cutoff>0)||!std::isfinite(h)||!std::isfinite(cutoff))
    throw std::invalid_argument("General lubrication needs finite positive gap and cutoff");
  const double x=h/cutoff;if(x>=1)return {{0.,0.}};
  const double y=1-x;double fb=0.;
  if(y<1e-3){double power=y;for(int k=2;k<=8;++k){power*=y;fb+=power/k;}}
  else fb=-std::log(x)-y;
  return {{y*y/h,fb}};
}
NearResult evaluate_near(const Geometry& g,double cutoff_factor,bool derived_normal_log,bool energy_completion) {
  if(!(g.ar>=1&&g.gap>0&&cutoff_factor>0)||!std::isfinite(g.ar)||!std::isfinite(g.gap)||!std::isfinite(cutoff_factor))
    throw std::invalid_argument("General lubrication requires AR>=1 and finite positive gap/cutoff factor");
  std::array<Mat3,2> shapes,inverses;std::array<Mat2,2> hessian;std::array<Vec3,2> levers;
  for(int i=0;i<2;++i) {
    const Vec3 p=g.rotations[i].col(2);
    if(!g.rotations[i].allFinite()||std::abs(p.squaredNorm()-1)>1e-10)
      throw std::invalid_argument("General lubrication needs finite unit directors");
    shapes[i]=shape_tensor(g.ar,g.rotations[i]);
    inverses[i]=Mat3::Identity()/(g.ar*g.ar)+(1-1/(g.ar*g.ar))*p*p.transpose();
    const double support=std::sqrt(shapes[i](2,2));
    hessian[i]=support*inverses[i].topLeftCorner<2,2>();
    levers[i]=(i==0?1.:-1.)*shapes[i].col(2)/support;
  }
  Eigen::SelfAdjointEigenSolver<Mat2> eig(hessian[0]);
  if(eig.info()!=Eigen::Success)throw std::runtime_error("Contact curvature eigensolve failed");
  Mat2 directions;directions.col(0)=eig.eigenvectors().col(1);directions.col(1)=eig.eigenvectors().col(0);
  if(directions.determinant()<0)directions.col(1)*=-1;
  Tangent tangent=Tangent::Zero();tangent.topRows<2>()=directions;
  Mat3 frame=Mat3::Identity();frame.leftCols<2>()=tangent;
  const double ki1=eig.eigenvalues()[1],ki2=eig.eigenvalues()[0];
  const Mat2 cj=directions.transpose()*hessian[1]*directions;
  const double a=.5*(ki1+cj(0,0)),b=.5*cj(0,1),d=.5*(ki2+cj(1,1));
  const double chi=-.5*std::atan2(2*b,a-d),c=std::cos(chi),s=std::sin(chi);
  const double l1=a*c*c-2*b*s*c+d*s*s,l2=a*s*s+2*b*s*c+d*c*c;
  if(!(std::min({ki1,ki2,l1,l2})>0))throw std::runtime_error("Nondegenerate convex lubrication contact required");
  const double root=std::sqrt(l1*l2),total=l1+l2;
  const auto poly_i=body_polynomials(shapes[0],inverses[0],tangent,1.);
  const auto poly_j=body_polynomials(shapes[1],inverses[1],tangent,-1.);
  const Poly& gi=poly_i.cubic;
  Poly cubic(4),quartic(5);
  for(int k=0;k<4;++k)cubic[k]=poly_i.cubic[k]+poly_j.cubic[k];
  for(int k=0;k<5;++k)quartic[k]=poly_i.quartic[k]+poly_j.quartic[k];
  Poly kappa=rotate_polynomial(cubic,c,s),zeta=rotate_polynomial(quartic,c,s);
  for(int k=0;k<4;++k)kappa[k]/=std::pow(l1,(3-k)/2.)*std::pow(l2,k/2.);
  for(int k=0;k<5;++k)zeta[k]/=std::pow(l1,(4-k)/2.)*std::pow(l2,k/2.);
  Mat6 inverse=Mat6::Zero(),logarithm=Mat6::Zero();inverse(2,2)=3*pi/(root*total);
  // Corrected leading tangential Couette + pressure terms (Gram form).
  const double c0=pi/root;logarithm(0,0)=logarithm(1,1)=c0;
  const double delta11=ki1-a,delta12=-b,delta22=ki2-d;
  for(int k=0;k<2;++k) {
    const double x=k?s:c,y=k?c:-s,lam=k?l2:l1,other=k?l1:l2;
    Vec6 mode;mode<<x*delta11+y*delta12,x*delta12+y*delta22,0.,y,-x,0.;
    logarithm+=3*c0/(lam*(3*lam+2*other))*mode*mode.transpose();
  }
  // Janoschek (2013), Appendix A Eqs. 47, 51, 55--57.
  const double d1=3*l1+2*l2,d2=2*l1+3*l2,nf=3*pi/(2*root*total);
  const double aa=(7*l1+2*l2)*kappa[0]+(l1+2*l2)*kappa[2];
  const double bb=(2*l1+l2)*kappa[1]+(2*l1+7*l2)*kappa[3];
  const double cc=3*l1*kappa[0]+l2*kappa[2],dd=l1*kappa[1]+3*l2*kappa[3];
  const double r1=std::sqrt(l1),r2=std::sqrt(l2);
  Vec6 cross=Vec6::Zero();
  cross[0]=nf*((2*r1*cc-1.5*ki1*aa/r1)*c/d1+(2*r2*dd-1.5*ki1*bb/r2)*s/d2
    +3*gi[0]*(c*c/l1+s*s/l2)+2*gi[1]*s*c*(1/l2-1/l1)+gi[2]*(s*s/l1+c*c/l2));
  cross[1]=nf*((-2*r1*cc+1.5*ki2*aa/r1)*s/d1+(2*r2*dd-1.5*ki2*bb/r2)*c/d2
    +3*gi[3]*(s*s/l1+c*c/l2)+2*gi[2]*s*c*(1/l2-1/l1)+gi[1]*(c*c/l1+s*s/l2));
  cross[3]=-1.5*nf*(bb*c/(r2*d2)-aa*s/(r1*d1));
  cross[4]=1.5*nf*(aa*c/(r1*d1)+bb*s/(r2*d2));
  cross[5]=nf*s*c*(ki1-ki2)*(1/l1-1/l2);
  logarithm.row(2)=cross.transpose();logarithm.col(2)=cross;
  // Independently derived normal logarithm. This is an explicit experimental
  // model option; the default retains the omission in the published correction.
  const double k0=kappa[0],k1=kappa[1],k2=kappa[2],k3=kappa[3];
  const double kk1=cc/d1,kk2=dd/d2;
  const double rh1=c*c*ki1+s*s*ki2,rh2=s*s*ki1+c*c*ki2,rhs=(ki1-ki2)*s*c;
  const double rho1=c*c*ki1*ki1+s*s*ki2*ki2,rho2=s*s*ki1*ki1+c*c*ki2*ki2;
  const double geom2=48./5*(l1*l1+l2*l2)+32./5*l1*l2
    -4*rh1*(3*l1+l2)-4*rh2*(l1+3*l2)+2*rh1*rh2+l2/l1*rho1+l1/l2*rho2
    +3*(rh1*rh1+rho1+rh2*rh2+rho2)+rhs*rhs*std::pow(std::sqrt(l1/l2)+std::sqrt(l2/l1),2);
  const double cubic_p=5*k0*k0+k1*k1+k2*k2+5*k3*k3+2*k0*k2+2*k1*k3;
  const double quartic_term=(5*l1+l2)*zeta[0]+total*zeta[2]+(l1+5*l2)*zeta[4];
  const double corrected_geom34=quartic_term-total*cubic_p/4-aa*kk1-bb*kk2;
  NearResult out;out.derived_normal_log_candidate=pi/(root*total*total)*(1.5*geom2-2.25*corrected_geom34);
  out.normal_log_coefficient=derived_normal_log?out.derived_normal_log_candidate:0.;
  logarithm(2,2)=out.normal_log_coefficient;
  out.inverse_gap6=inverse;out.logarithm6=logarithm;
  auto& t=out.contact_map;
  t.block<3,3>(0,0)=-frame.transpose();t.block<3,3>(0,6)=frame.transpose();
  t.block<3,3>(0,3)=frame.transpose()*skew(levers[0])/g.ar;
  t.block<3,3>(0,9)=-frame.transpose()*skew(levers[1])/g.ar;
  t.block<3,3>(3,3)=-frame.transpose()/g.ar;t.block<3,3>(3,9)=frame.transpose()/g.ar;
  const Vec3 displacement=levers[0]-levers[1];
  for(int k=0;k<5;++k)t.block<3,1>(0,12+k)=frame.transpose()*strain_basis()[k]*displacement/g.ar;
  out.inverse_gap=t.transpose()*inverse*t;out.logarithm=t.transpose()*logarithm*t;
  out.reference_length=4/(hessian[0].trace()+hessian[1].trace());
  Eigen::SelfAdjointEigenSolver<Mat2> eig_j(hessian[1]);
  if(eig_j.info()!=Eigen::Success)throw std::runtime_error("Contact curvature eigensolve failed");
  out.minimum_local_radius=1/std::max(ki1,eig_j.eigenvalues()[1]);
  out.cutoff=cutoff_factor*std::min(out.reference_length,1.);
  const auto factors=compact_near_factors(g.gap,out.cutoff);
  out.grand=factors[0]*out.inverse_gap+factors[1]*out.logarithm;
  if(energy_completion&&g.gap<out.cutoff) {
    // Partition contact modes into normal velocity n and the other five y.
    // K=[k c^T; c C], C=f_B*B_yy is the analytic Gram matrix above.
    // Adding cc^T/k in the y block gives k*(n+c.y/k)^2+y^T C y.
    // It is O(h log^2 h) at contact and C1-zero at the compact cutoff.
    // This policy is applied at every active contact, irrespective of any
    // measured eigenvalue. Retained A/h and B*log(h) terms remain unchanged.
    const double k=factors[0]*inverse(2,2)+factors[1]*logarithm(2,2);
    if(!(k>0&&std::isfinite(k)))
      throw std::runtime_error("Near energy completion requires positive normal resistance; experimental normal log made it nonpositive");
    const Vec6 coupling=factors[1]*cross;
    out.completion6=(coupling*coupling.transpose())/k;
    out.completion=t.transpose()*out.completion6*t;
    out.grand+=out.completion;out.energy_completed=true;
  }
  if(!out.grand.allFinite())throw std::runtime_error("Nonfinite general lubrication response");
  return out;
}
}
