// Leading singular lubrication for smooth convex ellipsoids.
// Janoschek, Harting & Toschi (2013), arXiv:1308.6482, Appendix A.
// This is a resistance model, not an LB correction: no matching subtraction.
// All leading divergent ENTRIES are retained, including cubic-surface normal
// couplings. The subleading logarithm in the normal/normal entry and finite
// Stokes terms are not included (as in the existing Mari sphere model).
//
// The truncated asymptotic matrix need not be positive semidefinite. Completing
// its normal square adds b*b^T/k_n = O(h_reg log(h_reg)^2), preserves every
// retained singular coefficient as h_reg tends to zero, and makes the model
// dissipative. At fixed positive epsilon this addition need not vanish as the
// physical gap tends to zero. It is an explicit finite-gap modelling choice;
// power_completion diagnoses it.
// Forces, torques, imposed-strain drive and stress are derivatives of one
// quadratic. No additional pair-force virial may be added to its stress.
#ifndef SLURRY_SPHEROID_HYDRO_MATH_H
#define SLURRY_SPHEROID_HYDRO_MATH_H

#include "slurry_pair_hydro_math.h"
#include "../vendor/gr_re2/ellipsoidGap.h"
#include "../vendor/gr_re2/nearFieldResistance.h"

namespace SlurrySpheroidHydro {
using SlurryPairHydro::Vec3;
using SlurryPairHydro::Vec12;
using SlurryPairHydro::Mat3;
using SlurryPairHydro::Mat12;
using Assembly=SlurryPairHydro::Assembly;
using Vec6=std::array<double,6>;
using Mat6=std::array<double,36>;
namespace gr=slurry::gr_re2::graphite;

struct Pair {
  bool active=false,exact_sphere=false,uncompleted_asymptotic=false;
  Vec3 normal{0,0,1},tangent1{1,0,0},tangent2{0,1,0};
  Vec3 lever_i{},lever_j{},contact_displacement{};
  double gap=0,regularized_gap=0,epsilon=0,cutoff=0;
  double log_reference_length=0,minimum_curvature_radius=0;
  double normal_resistance=0;
  Vec6 normal_cross{}; // K_3k before the positive completion; K_33 excluded.
  Mat6 squeeze{},shear{},pump{},resistance{};
  SlurryPairHydro::SpherePair sphere{};
};

struct Evaluation : SlurryPairHydro::Evaluation {
  // Added b*b^T/k_n dissipation, already INCLUDED in power_squeeze and power.
  // Its ratio to total power need not be <=1 because of cross cancellation.
  double power_completion=0;
};

namespace detail {
inline double contract(const Vec3&a,const Mat3&m,const Vec3&b)
{return gr::dot(a,gr::mul(m,b));}
inline void outer_add(Mat6&m,const Vec6&a,double weight)
{for(int i=0;i<6;++i)for(int j=0;j<6;++j)m[6*i+j]+=weight*a[i]*a[j];}
inline Vec6 multiply(const Mat6&m,const Vec6&v)
{Vec6 x{};for(int i=0;i<6;++i)for(int j=0;j<6;++j)x[i]+=m[6*i+j]*v[j];return x;}
inline double dot6(const Vec6&a,const Vec6&b)
{double x=0;for(int i=0;i<6;++i)x+=a[i]*b[i];return x;}
inline double quadratic(const Mat6&m,const Vec6&v)
{return dot6(v,multiply(m,v));}
inline bool is_sphere(const gr::Body&b)
{return b.axes[0]==b.axes[1]&&b.axes[1]==b.axes[2];}

// Polynomial coefficients of x^3,x^2*y,x*y^2,y^3 after
// x=c*X+s*Y, y=-s*X+c*Y. No factorials are hidden in the coefficients.
inline std::array<double,4> rotate_cubic(const std::array<double,4>&p,double c,double s)
{
  constexpr int choose[4][4]={{1,0,0,0},{1,1,0,0},{1,2,1,0},{1,3,3,1}};
  std::array<double,4> out{};
  for(int m=0;m<4;++m)for(int a=0;a<=3-m;++a)for(int b=0;b<=m;++b)
    out[a+b]+=p[m]*choose[3-m][a]*choose[m][b]*
      std::pow(c,3-m-a)*std::pow(s,a)*std::pow(-s,m-b)*std::pow(c,b);
  return out;
}

inline Mat3 implicit_shape(const gr::Body&b)
{
  for(double a:b.axes)if(!(a>0)||!std::isfinite(a))
    throw std::invalid_argument("Spheroid lubrication requires positive finite axes");
  return gr::rotatedDiagonal(b.rotation,{1/(b.axes[0]*b.axes[0]),
      1/(b.axes[1]*b.axes[1]),1/(b.axes[2]*b.axes[2])});
}
inline std::array<double,4> cubic(const gr::Body&b,const Mat3&q,const Vec3&n,
                                const Vec3&e1,const Vec3&e2,double k11,double k12,
                                double k22,double outward_sign)
{
  const Mat3 shape=gr::rotatedDiagonal(b.rotation,{b.axes[0]*b.axes[0],
      b.axes[1]*b.axes[1],b.axes[2]*b.axes[2]});
  const double support=std::sqrt(contract(n,shape,n)); // 1/|Q p|
  const double x=-.5*outward_sign*support*contract(e1,q,n);
  const double y=-.5*outward_sign*support*contract(e2,q,n);
  return {x*k11,2*x*k12+y*k11,x*k22+2*y*k12,y*k22};
}

inline Mat3 symmetric_deviatoric(const Mat3&e)
{
  Mat3 d{};const double tr=(e[0]+e[4]+e[8])/3;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    d[3*i+j]=.5*(e[3*i+j]+e[3*j+i])-(i==j?tr:0);
  return d;
}
inline Vec3 world(const Pair&p,const Vec6&v,int offset=0)
{
  Vec3 out{};for(int k=0;k<3;++k)
    out[k]=p.tangent1[k]*v[offset]+p.tangent2[k]*v[offset+1]+p.normal[k]*v[offset+2];
  return out;
}
} // namespace detail

// epsilon and cutoff_cap are PHYSICAL lengths, independent of mechanical
// roughness/contact onset. For a nonspherical pair the actual cutoff is
// min(cutoff_cap, R_ref/2), where R_ref is the mean of the two harmonic mean
// principal curvature radii. This is a local asymptotic model, not a statement
// of accuracy when gap/minimum_curvature_radius is large.
// Exact spheres dispatch to the existing Mari implementation, with the supplied
// cutoff_cap unchanged. This preserves existing explicit sphere settings.
// An explicitly oversized spherical cutoff can therefore differ from the
// nonspherical local cutoff. The near-sphere limit is continuous inside both
// active domains, not across those deliberately different truncation boundaries.
// asymptotic_only is an analytic subtraction basis for a separately matched
// model. It has no cutoff/log clamp or positive-square completion and is NOT
// a standalone dissipative finite-gap resistance. The default path is unchanged.
inline Pair make_pair(const gr::Body&bi,const gr::Body&bj,const gr::GapResult&gap,
                      double eta,double epsilon,double cutoff_cap,
                      bool asymptotic_only=false)
{
  if(!(eta>0)||!(asymptotic_only?epsilon>=0:epsilon>0)||!(cutoff_cap>0)||!std::isfinite(eta)||
     !std::isfinite(epsilon)||!std::isfinite(cutoff_cap)||!std::isfinite(gap.gap))
    throw std::invalid_argument("Invalid spheroid lubrication viscosity, epsilon, cutoff or gap");
  Pair p;p.normal=gr::normalized(gap.normal);p.gap=gap.gap;p.epsilon=epsilon;
  p.regularized_gap=std::max(p.gap,0.0)+epsilon;
  if(!(p.regularized_gap>0))throw std::invalid_argument("Singular lubrication requires positive regularized gap");
  p.uncompleted_asymptotic=asymptotic_only;
  p.lever_i=gap.leverI;p.lever_j=gap.leverJ;
  p.contact_displacement=gr::sub(p.lever_i,p.lever_j);
  const Mat3 qi=detail::implicit_shape(bi),qj=detail::implicit_shape(bj);
  p.exact_sphere=!asymptotic_only&&detail::is_sphere(bi)&&detail::is_sphere(bj);
  if(p.exact_sphere) {
    p.sphere=SlurryPairHydro::make_sphere_pair(bi.axes[0],bj.axes[0],p.normal,
      p.gap,eta,epsilon,cutoff_cap);
    p.cutoff=cutoff_cap;p.active=p.sphere.active;
    p.log_reference_length=.5*(bi.axes[0]+bj.axes[0]);
    p.minimum_curvature_radius=std::min(bi.axes[0],bj.axes[0]);
    p.normal_resistance=p.sphere.squeeze_resistance;
    return p;
  }

  const Mat3 ci=gr::nfCurvature(bi,p.normal),cj=gr::nfCurvature(bj,p.normal);
  const Vec3 reference=std::abs(p.normal[0])<.8?Vec3{1,0,0}:Vec3{0,1,0};
  const Vec3 u=gr::normalized(gr::cross(p.normal,reference)),v=gr::cross(p.normal,u);
  const double ci11=detail::contract(u,ci,u),ci12=detail::contract(u,ci,v),
               ci22=detail::contract(v,ci,v);
  const double angle=.5*std::atan2(2*ci12,ci11-ci22);
  p.tangent1=gr::add(gr::scale(u,std::cos(angle)),gr::scale(v,std::sin(angle)));
  p.tangent2=gr::cross(p.normal,p.tangent1);
  const double ki1=detail::contract(p.tangent1,ci,p.tangent1),
               ki2=detail::contract(p.tangent2,ci,p.tangent2);
  const double kj11=detail::contract(p.tangent1,cj,p.tangent1),
               kj12=detail::contract(p.tangent1,cj,p.tangent2),
               kj22=detail::contract(p.tangent2,cj,p.tangent2);
  const double a=.5*(ki1+kj11),b=.5*kj12,d=.5*(ki2+kj22);
  const double chi=-.5*std::atan2(2*b,a-d),c=std::cos(chi),s=std::sin(chi);
  const double l1=a*c*c-2*b*s*c+d*s*s,l2=a*s*s+2*b*s*c+d*c*c;
  if(!(ki1>0)||!(ki2>0)||!(l1>0)||!(l2>0))
    throw std::domain_error("Degenerate spheroid lubrication curvature");
  const double kjmax=.5*(kj11+kj22+std::hypot(kj11-kj22,2*kj12));
  p.minimum_curvature_radius=1/std::max({ki1,ki2,kjmax});
  p.log_reference_length=1/(ki1+ki2)+1/(kj11+kj22);
  p.cutoff=asymptotic_only?cutoff_cap:std::min(cutoff_cap,.5*p.log_reference_length);
  p.active=asymptotic_only||p.gap<p.cutoff;
  if(!p.active)return p;
  const double pi=std::acos(-1.),root=std::sqrt(l1*l2),sum=l1+l2;
  const double raw_log=std::log(p.log_reference_length/p.regularized_gap);
  const double log=asymptotic_only?raw_log:std::max(raw_log,0.0);
  p.normal_resistance=3*pi*eta/(p.regularized_gap*root*sum);

  // Tangential 4x4 block of Appendix A, written as the local Couette and
  // Reynolds-pressure Gram matrix. This keeps anisotropic curvature and all
  // leading tangential translation/rotation couplings without eigenvalue clips.
  Mat6 lower{};const double c0=pi*eta*log/root;
  lower[0]=c0;lower[7]=c0;
  const double delta11=ki1-a,delta12=-b,delta22=ki2-d;
  for(int axis=0;axis<2;++axis) {
    const double x=axis==0?c:s,y=axis==0?-s:c;
    const double lam=axis==0?l1:l2,other=axis==0?l2:l1;
    const Vec6 mode{x*delta11+y*delta12,x*delta12+y*delta22,0,y,-x,0};
    detail::outer_add(lower,mode,3*c0/(lam*(3*lam+2*other)));
  }

  // Separate sliding and relative rolling/pump through a Schur factorization.
  // For spheres this gives precisely Mari's unequal-radius shear levers/pump.
  if(asymptotic_only) {
    // Preserve the signed logarithmic coefficients even outside the near
    // limit. Their finite-gap replacement is supplied by the residual model.
    p.shear=lower;
  } else if(c0>0) {
    const long double aa=lower[0],ab=lower[1],ad=lower[7],det=aa*ad-ab*ab;
    if(!(det>0))throw std::domain_error("Nonpositive spheroid shear block");
    double m[2][2]{};
    for(int j=0;j<2;++j) {
      m[0][j]=static_cast<double>((ad*lower[3+j]-ab*lower[6+3+j])/det);
      m[1][j]=static_cast<double>((aa*lower[6+3+j]-ab*lower[3+j])/det);
    }
    const Vec6 modes[2]={{1,0,0,m[0][0],m[0][1],0},{0,1,0,m[1][0],m[1][1],0}};
    for(int i=0;i<6;++i)for(int j=0;j<6;++j)
      p.shear[6*i+j]=static_cast<double>(aa*modes[0][i]*modes[0][j]+
        ab*(modes[0][i]*modes[1][j]+modes[1][i]*modes[0][j])+ad*modes[1][i]*modes[1][j]);
    for(int i=3;i<5;++i)for(int j=3;j<5;++j)
      p.pump[6*i+j]=static_cast<double>(static_cast<long double>(lower[6*i+j])-
        static_cast<long double>(lower[i])*m[0][j-3]-
        static_cast<long double>(lower[6+i])*m[1][j-3]);
    p.pump[3*6+4]=p.pump[4*6+3]=.5*(p.pump[3*6+4]+p.pump[4*6+3]);
  }

  // Third-order geometry is essential for the leading normal/tangent and
  // normal/rotation logarithms (A3, A7, A11--A13); it is not dropped.
  const auto gi=detail::cubic(bi,qi,p.normal,p.tangent1,p.tangent2,ki1,0,ki2,1);
  const auto gj=detail::cubic(bj,qj,p.normal,p.tangent1,p.tangent2,kj11,kj12,kj22,-1);
  std::array<double,4> cubic{};for(int k=0;k<4;++k)cubic[k]=gi[k]+gj[k];
  auto kap=detail::rotate_cubic(cubic,c,s);
  for(int k=0;k<4;++k)kap[k]/=std::pow(l1,.5*(3-k))*std::pow(l2,.5*k);
  const double d1=3*l1+2*l2,d2=2*l1+3*l2;
  const double nfac=3*pi*eta*log/(2*root*sum);
  const double A=(7*l1+2*l2)*kap[0]+(l1+2*l2)*kap[2];
  const double B=(2*l1+l2)*kap[1]+(2*l1+7*l2)*kap[3];
  const double C=3*kap[0]*l1+kap[2]*l2,D=kap[1]*l1+3*kap[3]*l2;
  const double r1=std::sqrt(l1),r2=std::sqrt(l2);
  p.normal_cross[0]=nfac*((2*r1*C-1.5*ki1*A/r1)*c/d1+
      (2*r2*D-1.5*ki1*B/r2)*s/d2+3*gi[0]*(c*c/l1+s*s/l2)+
      2*gi[1]*s*c*(1/l2-1/l1)+gi[2]*(s*s/l1+c*c/l2));
  p.normal_cross[1]=nfac*((-2*r1*C+1.5*ki2*A/r1)*s/d1+
      (2*r2*D-1.5*ki2*B/r2)*c/d2+3*gi[3]*(s*s/l1+c*c/l2)+
      2*gi[2]*s*c*(1/l2-1/l1)+gi[1]*(c*c/l1+s*s/l2));
  p.normal_cross[3]=-1.5*nfac*(B*c/(r2*d2)-A*s/(r1*d1));
  p.normal_cross[4]= 1.5*nfac*(A*c/(r1*d1)+B*s/(r2*d2));
  p.normal_cross[5]=nfac*s*c*(ki1-ki2)*(1/l1-1/l2);
  if(asymptotic_only) {
    p.squeeze[6*2+2]=p.normal_resistance;
    for(int k=0;k<6;++k)if(k!=2)
      p.squeeze[6*2+k]=p.squeeze[6*k+2]=p.normal_cross[k];
  } else {
    Vec6 normal_mode{};
    for(int k=0;k<6;++k)normal_mode[k]=p.normal_cross[k]/p.normal_resistance;
    normal_mode[2]=1;
    detail::outer_add(p.squeeze,normal_mode,p.normal_resistance);
  }
  for(int k=0;k<36;++k) {
    p.resistance[k]=p.squeeze[k]+p.shear[k]+p.pump[k];
    if(!std::isfinite(p.resistance[k]))throw std::overflow_error("Spheroid lubrication resistance overflow");
  }
  return p;
}

// Local DOFs are contact slip (j-i) and relative angular velocity (j-i).
// Angular reference is ambient-fluid vorticity, not Jeffery rotation.
// E acts over the CONTACT-LIMIT displacement lever_i-lever_j. Using the actual
// centre displacement instead changes retained sphere G/H/M coefficients.
inline Vec6 local_velocity(const Pair&p,const Vec12&relative_scaled,
                           const Vec3&image_velocity,const Mat3&strain,double length)
{
  if(!(length>0)||!std::isfinite(length))
    throw std::invalid_argument("Spheroid lubrication requires positive reference length");
  Vec3 wi{},wj{},du{};
  for(int k=0;k<3;++k){wi[k]=relative_scaled[3+k]/length;wj[k]=relative_scaled[9+k]/length;
    du[k]=relative_scaled[6+k]-relative_scaled[k]+image_velocity[k];}
  const Vec3 rot=gr::sub(gr::cross(wj,p.lever_j),gr::cross(wi,p.lever_i));
  const Vec3 drive=gr::mul(detail::symmetric_deviatoric(strain),p.contact_displacement);
  du=gr::add(gr::add(du,rot),drive);const Vec3 dw=gr::sub(wj,wi);
  return {gr::dot(du,p.tangent1),gr::dot(du,p.tangent2),gr::dot(du,p.normal),
          gr::dot(dw,p.tangent1),gr::dot(dw,p.tangent2),gr::dot(dw,p.normal)};
}

inline Assembly assemble(const Pair&p,const Vec12&background_scaled,const Vec3&image_velocity,
                         const Mat3&strain,double length)
{
  if(p.exact_sphere)return SlurryPairHydro::assemble(p.sphere,background_scaled,image_velocity,strain,length);
  Assembly out;Vec12 negative{};for(int k=0;k<12;++k)negative[k]=-background_scaled[k];
  const Vec6 offset=local_velocity(p,negative,image_velocity,strain,length);
  const Vec6 weighted=detail::multiply(p.resistance,offset);
  out.energy_constant=.5*detail::dot6(offset,weighted);
  std::array<Vec6,12> basis{},product{};
  for(int k=0;k<12;++k){Vec12 unit{};unit[k]=1;
    basis[k]=local_velocity(p,unit,{}, {},length);product[k]=detail::multiply(p.resistance,basis[k]);
    out.affine[k]=detail::dot6(basis[k],weighted);}
  for(int i=0;i<12;++i)for(int j=0;j<12;++j)
    out.resistance[12*i+j]=detail::dot6(basis[i],product[j]);
  return out;
}

inline Evaluation evaluate(const Pair&p,const Vec12&q_scaled,const Vec12&background_scaled,
                           const Vec3&image_velocity,const Mat3&strain,double length)
{
  Evaluation out;
  if(p.exact_sphere){static_cast<SlurryPairHydro::Evaluation&>(out)=
    SlurryPairHydro::evaluate(p.sphere,q_scaled,background_scaled,image_velocity,strain,length);return out;}
  Vec12 relative{};for(int k=0;k<12;++k)relative[k]=q_scaled[k]-background_scaled[k];
  const Vec6 v=local_velocity(p,relative,image_velocity,strain,length);
  const Vec6 gn=detail::multiply(p.squeeze,v),gt=detail::multiply(p.shear,v),gp=detail::multiply(p.pump,v);
  Vec6 g{};for(int k=0;k<6;++k)g[k]=gn[k]+gt[k]+gp[k];
  const Vec3 force=detail::world(p,g),moment=detail::world(p,g,3);
  out.force_i=force;out.force_j=gr::scale(force,-1);
  out.torque_i=gr::add(gr::cross(p.lever_i,force),moment);
  out.torque_j=gr::scale(gr::add(gr::cross(p.lever_j,force),moment),-1);
  for(int k=0;k<3;++k){out.gradient[k]=-out.force_i[k];out.gradient[6+k]=-out.force_j[k];
    out.gradient[3+k]=-out.torque_i[k]/length;out.gradient[9+k]=-out.torque_j[k]/length;}
  auto stress=[&](const Vec6&component){Mat3 s{};const Vec3 f=detail::world(p,component);
    const double trace=gr::dot(f,p.contact_displacement)/3;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)s[3*i+j]=
      .5*(f[i]*p.contact_displacement[j]+f[j]*p.contact_displacement[i])-(i==j?trace:0);return s;};
  out.stress_squeeze=stress(gn);out.stress_shear=stress(gt);out.stress_pump=stress(gp);
  for(int k=0;k<9;++k)out.stress[k]=out.stress_squeeze[k]+out.stress_shear[k]+out.stress_pump[k];
  out.power_squeeze=detail::dot6(v,gn);out.power_shear=detail::dot6(v,gt);out.power_pump=detail::dot6(v,gp);
  out.power=out.power_squeeze+out.power_shear+out.power_pump;
  const double cross=detail::dot6(p.normal_cross,v);
  out.power_completion=!p.uncompleted_asymptotic&&p.normal_resistance>0?cross*cross/p.normal_resistance:0;
  return out;
}
} // namespace SlurrySpheroidHydro
#endif
