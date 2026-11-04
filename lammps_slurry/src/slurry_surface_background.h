#ifndef SLURRY_SURFACE_BACKGROUND_H
#define SLURRY_SURFACE_BACKGROUND_H

#include "slurry_surface_cohesion.h"

// Background terms from Test_LB@058daeee6854bfcb1c1ad1cc5cabd699446845db,
// olb-1.9r0/src/slurry/gr_re2/reSquaredPotential.h. The near Derjaguin
// attraction/repulsion, RE2 shape factors, and energy switches are unchanged.
// Mechanical g=0 maps to the LB energy reference h0, without moving contact:
// for g>=0 the virtual pair displacement is r+h0*n, whose support gap is
// h=g+h0. All derivatives of this map, including dn, are retained. For g<0
// the near energy is continued linearly in g at h0, at the current curvature.
// Parameters and bodies may be SI or any consistently normalized units.
namespace SlurrySurfaceBackground {
namespace gr=slurry::gr_re2::graphite;
namespace sc=SlurrySurfaceCohesion;
using gr::Vec3;using gr::Mat3;

struct Parameters {
  double hamaker=9.9e-20;             // energy
  double sigma=4.197e-10;             // length
  double reference_gap=2.e-9;        // energy reference only, not contact onset
  double curvature_switch=5.e-9;
  double curvature_cutoff=20.e-9;
  double far_switch=400.e-9;
  double far_cutoff=500.e-9;
};
struct Result {
  double energy=0.,attractive_energy=0.,repulsive_energy=0.,gap=0.;
  double attraction_at_contact=0.;
  Vec3 force_i{},force_j{},torque_i{},torque_j{};
  bool active=false;
};
inline void validate(const Parameters&p) {
  if(!(p.hamaker>=0.)||!std::isfinite(p.hamaker)||!(p.sigma>0.)||
     !(p.reference_gap>0.)||!(p.reference_gap<p.curvature_switch)||
     !(p.curvature_switch<p.curvature_cutoff)||!(p.curvature_cutoff<=p.far_switch)||
     !(p.far_switch<p.far_cutoff)||!std::isfinite(p.sigma+p.reference_gap+
       p.curvature_switch+p.curvature_cutoff+p.far_switch+p.far_cutoff))
    throw std::domain_error("Invalid surface-background parameters");
}
inline double work_of_separation(const Parameters&p) {
  validate(p);
  return p.hamaker/(12.*std::acos(-1.)*p.reference_gap*p.reference_gap)*
      (1.-std::pow(p.sigma/p.reference_gap,6)/30.);
}

namespace detail {
// Nine directional forward derivatives: relative translation, rotation I,
// rotation J. RE2 algebra follows the pinned Test_LB evaluator.
struct AD {double v=0.;std::array<double,9>d{};AD()=default;AD(double x):v(x){}};
inline AD operator+(const AD&a,const AD&b){AD c(a.v+b.v);for(int k=0;k<9;++k)c.d[k]=a.d[k]+b.d[k];return c;}
inline AD operator-(const AD&a,const AD&b){AD c(a.v-b.v);for(int k=0;k<9;++k)c.d[k]=a.d[k]-b.d[k];return c;}
inline AD operator-(const AD&a){AD c(-a.v);for(int k=0;k<9;++k)c.d[k]=-a.d[k];return c;}
inline AD operator*(const AD&a,const AD&b){AD c(a.v*b.v);for(int k=0;k<9;++k)c.d[k]=a.d[k]*b.v+a.v*b.d[k];return c;}
inline AD operator/(const AD&a,const AD&b){AD c(a.v/b.v);for(int k=0;k<9;++k)c.d[k]=(a.d[k]-c.v*b.d[k])/b.v;return c;}
inline AD sqrt(const AD&a){AD c(std::sqrt(a.v));for(int k=0;k<9;++k)c.d[k]=a.d[k]/(2.*c.v);return c;}
inline AD power(AD a,int n){AD b(1.);for(;n;n>>=1,a=a*a)if(n&1)b=b*a;return b;}
using AVec=std::array<AD,3>;using AMat=std::array<AD,9>;
inline AD dot(const AVec&a,const AVec&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline AVec mv(const AMat&a,const AVec&b){AVec c{};for(int i=0;i<3;++i)for(int k=0;k<3;++k)c[i]=c[i]+a[3*i+k]*b[k];return c;}
inline AD determinant(const AMat&m){return m[0]*(m[4]*m[8]-m[5]*m[7])-m[1]*(m[3]*m[8]-m[5]*m[6])+m[2]*(m[3]*m[7]-m[4]*m[6]);}
inline AMat inverse(const AMat&m){
  AMat c{m[4]*m[8]-m[5]*m[7],m[2]*m[7]-m[1]*m[8],m[1]*m[5]-m[2]*m[4],
    m[5]*m[6]-m[3]*m[8],m[0]*m[8]-m[2]*m[6],m[2]*m[3]-m[0]*m[5],
    m[3]*m[7]-m[4]*m[6],m[1]*m[6]-m[0]*m[7],m[0]*m[4]-m[1]*m[3]};
  const AD det=determinant(m);for(auto&x:c)x=x/det;return c;
}
inline AMat rotation(const Mat3&r,int offset){
  AMat m;for(int i=0;i<9;++i)m[i]=AD(r[i]);
  for(int axis=0;axis<3;++axis){Vec3 e{};e[axis]=1.;for(int col=0;col<3;++col){
    const Vec3 v{r[col],r[3+col],r[6+col]};const Vec3 dv=gr::cross(e,v);
    for(int row=0;row<3;++row)m[3*row+col].d[offset+axis]=dv[row];}}
  return m;
}
inline AMat shape(const AMat&r,const Vec3&d){AMat q{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)q[3*i+j]=q[3*i+j]+r[3*i+k]*d[k]*r[3*j+k];return q;}
inline AD orientation_length(const gr::Body&bi,const gr::Body&bj,const AVec&rh,double sigma){
  const AMat ri=rotation(bi.rotation,3),rj=rotation(bj.rotation,6);
  const double di=bi.axes[0]*bi.axes[1]*bi.axes[2],dj=bj.axes[0]*bj.axes[1]*bj.axes[2];
  Vec3 ai2{},aj2{},aiInv{},ajInv{};for(int k=0;k<3;++k){ai2[k]=bi.axes[k]*bi.axes[k];aj2[k]=bj.axes[k]*bj.axes[k];aiInv[k]=1./ai2[k];ajInv[k]=1./aj2[k];}
  const AMat qi=shape(ri,ai2),qj=shape(rj,aj2),ii=shape(ri,aiInv),ij=shape(rj,ajInv);
  const AD pi=1./sqrt(dot(rh,mv(ii,rh))),pj=1./sqrt(dot(rh,mv(ij,rh)));
  AMat h{},b{};for(int k=0;k<9;++k){h[k]=qi[k]/pi+qj[k]/pj;b[k]=qi[k]*(sigma/di)+qj[k]*(sigma/dj);}
  const AD chi=2.*dot(rh,mv(inverse(b),rh));
  const AD eta=(di/(pi*pi)+dj/(pj*pj))/sqrt(determinant(h)/(pi+pj));
  return eta*chi*sigma;
}
inline AD branch(const AD&h,const AD&ell,const gr::Body&i,const gr::Body&j,const Parameters&p,bool repulsive){
  const double m=repulsive?2025.:-36.,o=repulsive?45./56.:3.,shapeScale=repulsive?std::cbrt(60.):2.;
  AD product(1.);for(int k=0;k<3;++k){product=product*(i.axes[k]/(i.axes[k]+h/shapeScale));product=product*(j.axes[k]/(j.axes[k]+h/shapeScale));}
  AD u=(p.hamaker/m)*(1.+o*ell/h)*product;
  if(repulsive)u=u*power(p.sigma/h,6);
  return u;
}
inline AD smooth_switch(const AD&t){
  // Same quintic as Test_LB, evaluated from the nearer endpoint to avoid
  // cancellation of energy and force at the outer cutoff.
  if(t.v<=.5)return 1.-10.*power(t,3)+15.*power(t,4)-6.*power(t,5);
  const AD z=1.-t;return power(z,3)*(10.-15.*z+6.*z*z);
}
struct Geometry {AD gap,radius;AVec normal;};
inline Geometry geometry(const gr::Body&bi,const gr::Body&bj,const gr::GapResult&geo,bool need_curvature){
  Geometry out;out.gap=AD(geo.gap);
  const Vec3 n=geo.normal,ref=std::abs(n[0])<.8?Vec3{1.,0.,0.}:Vec3{0.,1.,0.};
  const Vec3 t1=gr::normalized(gr::cross(n,ref)),t2=gr::cross(n,t1);
  const auto a=sc::detail::shape(bi,n),b=sc::detail::shape(bj,n);
  Mat3 support{},curvature{};
  for(int k=0;k<9;++k){support[k]=a.support_derivative[k]+b.support_derivative[k];curvature[k]=a.s*a.inverse[k]+b.s*b.inverse[k];}
  const double h11=sc::detail::contract(t1,support,t1)+geo.gap,h12=sc::detail::contract(t1,support,t2),
               h22=sc::detail::contract(t2,support,t2)+geo.gap,hdet=h11*h22-h12*h12;
  if(!(h11>0.)||!(h22>0.)||!(hdet>64.*std::numeric_limits<double>::epsilon()*h11*h22))
    throw std::domain_error("Surface-background support normal is degenerate");
  const double c11=sc::detail::contract(t1,curvature,t1),c12=sc::detail::contract(t1,curvature,t2),
               c22=sc::detail::contract(t2,curvature,t2),cdet=c11*c22-c12*c12;
  if(!(cdet>0.)||!std::isfinite(cdet))throw std::domain_error("Invalid surface-background curvature");
  out.radius=AD(1./std::sqrt(cdet));
  for(int k=0;k<3;++k)out.normal[k]=AD(n[k]);
  for(int direction=0;direction<9;++direction){
    Vec3 dr{};Mat3 dqa{},dqb{},dia{},dib{};
    if(direction<3)dr[direction]=1.;
    else if(direction<6){dqa=sc::detail::rotational_derivative(a.q,direction-3);dia=sc::detail::rotational_derivative(a.inverse,direction-3);}
    else {dqb=sc::detail::rotational_derivative(b.q,direction-6);dib=sc::detail::rotational_derivative(b.inverse,direction-6);}
    const Vec3 rhs=gr::sub(gr::sub(dr,sc::detail::fixed_normal_support_derivative(a,dqa,n)),sc::detail::fixed_normal_support_derivative(b,dqb,n));
    const double rhs1=gr::dot(t1,rhs),rhs2=gr::dot(t2,rhs);
    const Vec3 dn=gr::add(gr::scale(t1,(h22*rhs1-h12*rhs2)/hdet),gr::scale(t2,(h11*rhs2-h12*rhs1)/hdet));
    const double fixed_a=.5*sc::detail::contract(n,dqa,n)/a.s,fixed_b=.5*sc::detail::contract(n,dqb,n)/b.s;
    out.gap.d[direction]=gr::dot(n,dr)-fixed_a-fixed_b;
    for(int k=0;k<3;++k)out.normal[k].d[direction]=dn[k];
    if(need_curvature){
      const double dsa=gr::dot(a.qn,dn)/a.s+fixed_a,dsb=gr::dot(b.qn,dn)/b.s+fixed_b;
      Mat3 dc{};for(int k=0;k<9;++k)dc[k]=dsa*a.inverse[k]+a.s*dia[k]+dsb*b.inverse[k]+b.s*dib[k];
      const double dn1=gr::dot(dn,t1),dn2=gr::dot(dn,t2),nc1=sc::detail::contract(n,curvature,t1),nc2=sc::detail::contract(n,curvature,t2);
      const double dc11=sc::detail::contract(t1,dc,t1)-2.*dn1*nc1,dc12=sc::detail::contract(t1,dc,t2)-dn1*nc2-dn2*nc1,
                   dc22=sc::detail::contract(t2,dc,t2)-2.*dn2*nc2;
      out.radius.d[direction]=-.5*out.radius.v*(dc11*c22+c11*dc22-2.*c12*dc12)/cdet;
    }
  }
  return out;
}
} // namespace detail

inline Result evaluate(const gr::Body&bi,const gr::Body&bj,const gr::GapResult&geo,const Parameters&p){
  Result out;
  if(p.hamaker==0.)return out;
  validate(p);
  if(!geo.converged||!std::isfinite(geo.gap)||!gr::finite(geo.normal))throw std::domain_error("Surface background requires converged gap geometry");
  out.gap=geo.gap;
  const double h_value=p.reference_gap+geo.gap;
  if(h_value>=p.far_cutoff)return out;
  using namespace detail;
  const bool near=h_value<p.curvature_cutoff;
  const auto geometry=detail::geometry(bi,bj,geo,near);
  const AD ell=2.*geometry.radius;
  AD ua,ur;
  if(geo.gap<0.){
    const double h=p.reference_gap,s6=std::pow(p.sigma/h,6);
    ua=ell*((-p.hamaker/(12.*h))+geometry.gap*(p.hamaker/(12.*h*h)));
    ur=ell*((p.hamaker/(2520.*h))*s6-geometry.gap*((p.hamaker/(360.*h*h))*s6));
  }else{
    const AD h=geometry.gap+p.reference_gap;
    if(h_value>p.curvature_switch){
      const Vec3 displacement=gr::sub(bj.position,bi.position);AVec r{},rh{};
      for(int k=0;k<3;++k){r[k]=AD(displacement[k]);r[k].d[k]=1.;r[k]=r[k]+p.reference_gap*geometry.normal[k];}
      const AD length=sqrt(dot(r,r));for(int k=0;k<3;++k)rh[k]=r[k]/length;
      const AD ell_far=orientation_length(bi,bj,rh,p.sigma);
      ua=branch(h,ell_far,bi,bj,p,false);ur=branch(h,ell_far,bi,bj,p,true);
    }
    if(near){
      const AD near_a=(-p.hamaker/12.)*(ell/h),near_r=(p.hamaker/2520.)*(ell/h)*power(p.sigma/h,6);
      if(h_value<=p.curvature_switch){ua=near_a;ur=near_r;}
      else{
        const AD sw=smooth_switch((h-p.curvature_switch)/(p.curvature_cutoff-p.curvature_switch));
        ua=ua+sw*(near_a-ua);ur=ur+sw*(near_r-ur);
      }
    }
    if(h_value>p.far_switch){const AD sw=smooth_switch((h-p.far_switch)/(p.far_cutoff-p.far_switch));ua=ua*sw;ur=ur*sw;}
  }
  const AD u=ua+ur;
  out.energy=u.v;out.attractive_energy=ua.v;out.repulsive_energy=ur.v;
  out.attraction_at_contact=2.*std::acos(-1.)*geometry.radius.v*work_of_separation(p);
  for(int k=0;k<3;++k){out.force_i[k]=u.d[k];out.force_j[k]=-u.d[k];out.torque_i[k]=-u.d[3+k];out.torque_j[k]=-u.d[6+k];}
  out.active=true;
  if(!std::isfinite(out.energy)||!gr::finite(out.force_i)||!gr::finite(out.torque_i)||!gr::finite(out.torque_j))throw std::runtime_error("Nonfinite surface-background energy derivative");
  return out;
}
inline Result evaluate(const gr::Body&bi,const gr::Body&bj,const Parameters&p,gr::GapCache*cache=nullptr){
  if(p.hamaker==0.)return {};
  return evaluate(bi,bj,gr::closestEllipsoidGap(bi,bj,cache),p);
}
} // namespace SlurrySurfaceBackground
#endif
