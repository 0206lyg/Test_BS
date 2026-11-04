#ifndef SLURRY_SURFACE_COHESION_H
#define SLURRY_SURFACE_COHESION_H

#include "../vendor/gr_re2/ellipsoidGap.h"
#include <limits>

// Local Derjaguin surface cohesion for the existing signed support gap.
// No lubrication length, contact onset, friction rule, or overlap-created
// contact area enters this kernel. All forces and torques differentiate U.
namespace SlurrySurfaceCohesion {
namespace gr = slurry::gr_re2::graphite;
using gr::Vec3;
using gr::Mat3;

struct Parameters {
  // Test_LB's cohesive excess after its smooth-background work subtraction.
  double work_of_separation = .021243;  // J/m^2
  double opening_range = .67e-9;       // m
};

struct Result {
  double energy = 0.;                 // J
  double gap = 0.;                    // m; supplied mechanical signed gap
  double derjaguin_radius = 0.;       // m
  double attraction_at_contact = 0.; // N, 2 pi R_D W
  double gap_attraction = 0.;        // N, partial U / partial g at fixed R_D
  Vec3 force_i{}, force_j{};          // N, laboratory components
  Vec3 torque_i{}, torque_j{};        // N m about the particle centers
  bool active = false;
};

namespace detail {
inline Mat3 rotational_derivative(const Mat3& q, int axis) {
  Vec3 e{}; e[axis] = 1.;
  const Mat3 omega{0.,-e[2],e[1],e[2],0.,-e[0],-e[1],e[0],0.};
  const auto left=gr::multiply(omega,q),right=gr::multiply(q,omega);
  Mat3 out{};for(int k=0;k<9;++k)out[k]=left[k]-right[k];return out;
}
inline double contract(const Vec3& a,const Mat3& q,const Vec3& b) {
  return gr::dot(a,gr::mul(q,b));
}
struct Shape {
  Mat3 q{}, inverse{}, support_derivative{};
  Vec3 qn{}, support{};
  double s=0.;
};
inline Shape shape(const gr::Body& body,const Vec3& n) {
  Vec3 diagonal{},inverse{};
  for(int k=0;k<3;++k) {
    if(!(body.axes[k]>0.)||!std::isfinite(body.axes[k]))
      throw std::domain_error("Invalid surface-cohesion ellipsoid axis");
    diagonal[k]=body.axes[k]*body.axes[k];inverse[k]=1./diagonal[k];
  }
  Shape x;
  x.q=gr::rotatedDiagonal(body.rotation,diagonal);
  x.inverse=gr::rotatedDiagonal(body.rotation,inverse);
  x.qn=gr::mul(x.q,n);x.s=std::sqrt(gr::dot(n,x.qn));
  x.support=gr::scale(x.qn,1./x.s);
  for(int k=0;k<3;++k)for(int l=0;l<3;++l)
    x.support_derivative[3*k+l]=x.q[3*k+l]/x.s-x.qn[k]*x.qn[l]/(x.s*x.s*x.s);
  return x;
}
inline Vec3 fixed_normal_support_derivative(const Shape& x,const Mat3& dq,const Vec3& n) {
  return gr::sub(gr::scale(gr::mul(dq,n),1./x.s),
                 gr::scale(x.qn,.5*contract(n,dq,n)/(x.s*x.s*x.s)));
}
} // namespace detail

// Reuse the caller's closestEllipsoidGap result. Differentiating its stationary
// normal requires one analytic 2x2 inverse, not more geometry solves or finite
// differences. The nine directional derivatives below are relative position
// (three) and the two laboratory rotation increments (three each).
inline Result evaluate(const gr::Body& bi,const gr::Body& bj,
                       const gr::GapResult& geometry,const Parameters& parameters) {
  Result out;
  if(parameters.work_of_separation==0.)return out; // exact disabled path
  const double W=parameters.work_of_separation,range=parameters.opening_range;
  if(!(W>0.)||!std::isfinite(W)||!(range>0.)||!std::isfinite(range))
    throw std::domain_error("Invalid surface-cohesion energy or opening range");
  if(!geometry.converged||!std::isfinite(geometry.gap)||!gr::finite(geometry.normal))
    throw std::domain_error("Surface cohesion requires converged support geometry");
  out.gap=geometry.gap;
  if(out.gap>=range)return out;
  const Vec3 n=geometry.normal;
  const Vec3 ref=std::abs(n[0])<.8?Vec3{1.,0.,0.}:Vec3{0.,1.,0.};
  const Vec3 t1=gr::normalized(gr::cross(n,ref)),t2=gr::cross(n,t1);
  const auto a=detail::shape(bi,n),b=detail::shape(bj,n);
  Mat3 support{},curvature{};
  for(int k=0;k<9;++k) {
    support[k]=a.support_derivative[k]+b.support_derivative[k];
    curvature[k]=a.s*a.inverse[k]+b.s*b.inverse[k];
  }
  const double h11=detail::contract(t1,support,t1)+out.gap;
  const double h12=detail::contract(t1,support,t2);
  const double h22=detail::contract(t2,support,t2)+out.gap;
  const double hdet=h11*h22-h12*h12;
  // A non-strict support maximum has no unique differentiable normal. Do not
  // silently substitute a different contact shape in this degenerate case.
  if(!(h11>0.)||!(h22>0.)||!(hdet>64.*std::numeric_limits<double>::epsilon()*h11*h22))
    throw std::domain_error("Surface-cohesion support normal is degenerate");
  const double c11=detail::contract(t1,curvature,t1);
  const double c12=detail::contract(t1,curvature,t2);
  const double c22=detail::contract(t2,curvature,t2);
  const double cdet=c11*c22-c12*c12;
  if(!(cdet>0.)||!std::isfinite(cdet))
    throw std::domain_error("Invalid surface-cohesion curvature");
  out.derjaguin_radius=1./std::sqrt(cdet);
  const double pi=std::acos(-1.),R=out.derjaguin_radius;
  const double opening=out.gap<0.?1.:1.-out.gap/range;
  const double energy_per_radius=out.gap<0.?pi*W*(2.*out.gap-range):-pi*W*range*opening*opening;
  const double derivative_per_radius=2.*pi*W*opening;
  out.energy=R*energy_per_radius;
  out.attraction_at_contact=2.*pi*R*W;
  out.gap_attraction=R*derivative_per_radius;
  out.active=true;
  for(int direction=0;direction<9;++direction) {
    Vec3 dr{};Mat3 dqa{},dqb{},dia{},dib{};
    if(direction<3)dr[direction]=1.;
    else if(direction<6) {
      dqa=detail::rotational_derivative(a.q,direction-3);
      dia=detail::rotational_derivative(a.inverse,direction-3);
    } else {
      dqb=detail::rotational_derivative(b.q,direction-6);
      dib=detail::rotational_derivative(b.inverse,direction-6);
    }
    const Vec3 rhs=gr::sub(gr::sub(dr,detail::fixed_normal_support_derivative(a,dqa,n)),
                           detail::fixed_normal_support_derivative(b,dqb,n));
    const double rhs1=gr::dot(t1,rhs),rhs2=gr::dot(t2,rhs);
    const Vec3 dn=gr::add(gr::scale(t1,(h22*rhs1-h12*rhs2)/hdet),
                          gr::scale(t2,(h11*rhs2-h12*rhs1)/hdet));
    const double fixed_a=.5*detail::contract(n,dqa,n)/a.s;
    const double fixed_b=.5*detail::contract(n,dqb,n)/b.s;
    const double dg=gr::dot(n,dr)-fixed_a-fixed_b;
    const double dsa=gr::dot(a.qn,dn)/a.s+fixed_a;
    const double dsb=gr::dot(b.qn,dn)/b.s+fixed_b;
    Mat3 dc{};
    for(int k=0;k<9;++k)
      dc[k]=dsa*a.inverse[k]+a.s*dia[k]+dsb*b.inverse[k]+b.s*dib[k];
    // Parallel transport of the tangent frame: dt_alpha=-n(dn.t_alpha).
    // Any additional rotation within the frame cancels from its determinant.
    const double dn1=gr::dot(dn,t1),dn2=gr::dot(dn,t2);
    const double nc1=detail::contract(n,curvature,t1),nc2=detail::contract(n,curvature,t2);
    const double dc11=detail::contract(t1,dc,t1)-2.*dn1*nc1;
    const double dc12=detail::contract(t1,dc,t2)-dn1*nc2-dn2*nc1;
    const double dc22=detail::contract(t2,dc,t2)-2.*dn2*nc2;
    const double dR=-.5*R*(dc11*c22+c11*dc22-2.*c12*dc12)/cdet;
    const double dU=energy_per_radius*dR+out.gap_attraction*dg;
    if(direction<3){out.force_i[direction]=dU;out.force_j[direction]=-dU;}
    else if(direction<6)out.torque_i[direction-3]=-dU;
    else out.torque_j[direction-6]=-dU;
  }
  if(!std::isfinite(out.energy)||!gr::finite(out.force_i)||!gr::finite(out.torque_i)||!gr::finite(out.torque_j))
    throw std::runtime_error("Nonfinite surface-cohesion energy derivative");
  return out;
}

inline Result evaluate(const gr::Body& bi,const gr::Body& bj,const Parameters& parameters,
                       gr::GapCache* cache=nullptr) {
  if(parameters.work_of_separation==0.)return {};
  return evaluate(bi,bj,gr::closestEllipsoidGap(bi,bj,cache),parameters);
}
} // namespace SlurrySurfaceCohesion
#endif
