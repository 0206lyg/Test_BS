#ifndef SLURRY_CYLINDER_FILM_H
#define SLURRY_CYLINDER_FILM_H

// Finite-domain, fully flooded Reynolds/Couette model for two capped cylinders.
// This is a thin-film model, NOT a full three-dimensional Stokes solution.
// Actual projected planar-face overlap is integrated; p=0 at its open edge.
// Pressure uses a conforming Galerkin basis, hence its finite-basis dissipation
// is a lower approximation to the continuum Reynolds pressure dissipation.
// q_i=(U_i,Omega_i,e_i[STF5]), e_i=-E_inf; moments are ON FLUID.
// Length, velocity, angular-rate and viscosity units are those of the caller.
// No hidden radius rescaling is applied to the 22-by-22 returned matrices.
// Geometry is frozen within an evaluation. The 1 nm floor regularizes the
// constitutive coefficients, NEVER the squeeze/wedge velocity source.

#include "slurry_cylinder_geometry.h"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace SlurryCylinderFilm {
using Body = SlurryCylinderGeometry::Body;
using V2 = Eigen::Vector2d;
using V3 = Eigen::Vector3d;
using M2 = Eigen::Matrix2d;
using M3 = Eigen::Matrix3d;
using Mat22 = Eigen::Matrix<double,22,22>;
using Row22 = Eigen::Matrix<double,1,22>;
using Map322 = Eigen::Matrix<double,3,22>;
using Map222 = Eigen::Matrix<double,2,22>;
constexpr double pi=3.141592653589793238462643383279502884;

struct Options {
  double eta=1.;
  double gap_floor=1./150.;
  // Relative angle between the two face normals, in radians. This is a model
  // applicability limit; the caller supplies the positive rim continuation.
  double max_face_angle=.35;
  int pressure_order=4;
  int quadrature_order=24;
  // Reciprocal-gap enrichment resolves wedge pressure without large polynomial
  // degree. This changes only the trial space, not the Reynolds coefficients.
  int pressure_gap_power=2;
};
struct Result {
  Mat22 resistance=Mat22::Zero(),pressure=Mat22::Zero(),couette=Mat22::Zero();
  bool active=false;
  double area=0.,min_gap=std::numeric_limits<double>::infinity();
  double max_gap=0.,pressure_residual=0.,face_angle=0.;
  double min_pressure_pivot=0.;
  int modes=0,quadrature_points=0;
};

inline const std::array<M3,5>& strain_basis() {
  static const std::array<M3,5> result=[] {
    std::array<M3,5> a;for(auto& e:a)e.setZero();
    a[0].diagonal()<<1/std::sqrt(2.),-1/std::sqrt(2.),0.;
    a[1].diagonal()<<1/std::sqrt(6.),1/std::sqrt(6.),-2/std::sqrt(6.);
    a[2](0,1)=a[2](1,0)=1/std::sqrt(2.);
    a[3](0,2)=a[3](2,0)=1/std::sqrt(2.);
    a[4](1,2)=a[4](2,1)=1/std::sqrt(2.);return a;
  }();return result;
}

namespace detail {
struct Ellipse {
  V2 center=V2::Zero(),slope=V2::Zero();
  M2 Q=M2::Identity(); // (x-center)^T Q (x-center) <= 1
  double z0=0.;
  double q(const V2& x)const {const V2 d=x-center;return 1.-d.dot(Q*d);}
  V2 gradient(const V2& x)const {return -2.*Q*(x-center);}
  bool interval(double x,double& lo,double& hi)const {
    const double dx=x-center.x(),A=Q(1,1),B=Q(0,1)*dx;
    const double disc=B*B-A*(Q(0,0)*dx*dx-1.);
    if(!(disc>0.))return false;
    const double mid=center.y()-B/A,half=std::sqrt(disc)/A;
    lo=std::max(lo,mid-half);hi=std::min(hi,mid+half);return hi>lo;
  }
  std::array<double,2> xrange()const {
    const double half=std::sqrt(Q.inverse()(0,0));
    return {center.x()-half,center.x()+half};
  }
};
struct Rule {std::vector<double> x,w;};
inline Rule gauss_legendre(int n) {
  Rule r;r.x.resize(n);r.w.resize(n);
  for(int i=0;i<(n+1)/2;++i) {
    double z=std::cos(pi*(i+.75)/(n+.5)),derivative=0.;
    for(int k=0;k<40;++k) {
      double p0=1.,p1=z;
      for(int l=2;l<=n;++l){const double p=((2*l-1)*z*p1-(l-1)*p0)/l;p0=p1;p1=p;}
      derivative=n*(z*p1-p0)/(z*z-1.);
      const double dz=p1/derivative;z-=dz;if(std::abs(dz)<4e-16)break;
    }
    // Recompute the derivative at the final root.
    double p0=1.,p1=z;
    for(int l=2;l<=n;++l){const double p=((2*l-1)*z*p1-(l-1)*p0)/l;p0=p1;p1=p;}
    derivative=n*(z*p1-p0)/(z*z-1.);
    const double w=2./((1-z*z)*derivative*derivative);
    r.x[i]=-z;r.x[n-1-i]=z;r.w[i]=r.w[n-1-i]=w;
  }return r;
}
inline const Rule& rule(int n) {
  // Each worker caches its quadrature rule; no global mutable shared state.
  static thread_local int previous=0;
  static thread_local Rule cached;
  if(previous!=n){cached=gauss_legendre(n);previous=n;}return cached;
}

// Boundary intersections supply all changes of the upper/lower ellipse arc.
// Tangent-half-angle substitution reduces this to a polynomial of degree <=4.
inline std::vector<double> intersections_x(const Ellipse& a,const Ellipse& b) {
  Eigen::LLT<M2> factor(a.Q.inverse());
  const M2 M=factor.matrixL();const V2 d=a.center-b.center;
  const double A=M.col(0).dot(b.Q*M.col(0));
  const double B=2.*M.col(0).dot(b.Q*M.col(1));
  const double C=M.col(1).dot(b.Q*M.col(1));
  const double D=2.*d.dot(b.Q*M.col(0));
  const double E=2.*d.dot(b.Q*M.col(1));
  const double F=d.dot(b.Q*d)-1.;
  std::array<double,5> coefficient{A+D+F,2.*(B+E),-2.*A+4.*C+2.*F,
                                   2.*(E-B),A-D+F};
  double scale=0.;for(double v:coefficient)scale=std::max(scale,std::abs(v));
  std::vector<double> result;
  if(scale<2e-13)return result; // coincident boundaries
  for(double& v:coefficient)v/=scale;
  int degree=4;while(degree>0&&std::abs(coefficient[degree])<2e-13)--degree;
  if(degree>0) {
    Eigen::MatrixXd companion=Eigen::MatrixXd::Zero(degree,degree);
    for(int j=0;j<degree;++j)companion(0,j)=-coefficient[degree-1-j]/coefficient[degree];
    for(int j=1;j<degree;++j)companion(j,j-1)=1.;
    Eigen::EigenSolver<Eigen::MatrixXd> roots(companion,false);
    if(roots.info()!=Eigen::Success)throw std::runtime_error("Cylinder face ellipse intersection solve failed");
    for(int j=0;j<degree;++j) {
      const auto z=roots.eigenvalues()[j];
      if(std::abs(z.imag())>1e-8*(1.+std::abs(z.real())))continue;
      const double t=z.real(),den=1.+t*t;
      const V2 point=a.center+M*V2((1.-t*t)/den,2.*t/den);
      if(std::abs(b.q(point))<1e-7)result.push_back(point.x());
    }
  }
  const V2 at_pi=a.center-M.col(0);
  if(std::abs(b.q(at_pi))<1e-10)result.push_back(at_pi.x());
  return result;
}

struct Velocity {
  Map322 value=Map322::Zero(),dx=Map322::Zero(),dy=Map322::Zero();
};
inline Velocity velocity(const Body& body,const Ellipse& face,const V3& origin,
                         const M3& frame,int base) {
  Velocity out;
  const V3 lever=origin+face.z0*frame.col(2)-body.center;
  const V3 tx=frame.col(0)+face.slope.x()*frame.col(2);
  const V3 ty=frame.col(1)+face.slope.y()*frame.col(2);
  out.value.block<3,3>(0,base)=frame.transpose();
  for(int k=0;k<3;++k) {
    out.value.col(base+3+k)=frame.transpose()*V3::Unit(k).cross(lever);
    out.dx.col(base+3+k)=frame.transpose()*V3::Unit(k).cross(tx);
    out.dy.col(base+3+k)=frame.transpose()*V3::Unit(k).cross(ty);
  }
  for(int k=0;k<5;++k) {
    out.value.col(base+6+k)=frame.transpose()*strain_basis()[k]*lever;
    out.dx.col(base+6+k)=frame.transpose()*strain_basis()[k]*tx;
    out.dy.col(base+6+k)=frame.transpose()*strain_basis()[k]*ty;
  }return out;
}
inline void basis(const V2& point,const Ellipse& a,const Ellipse& b,
                  const V2& center,const V2& scale,int order,
                  Eigen::VectorXd& phi,Eigen::MatrixXd& grad) {
  const double q1=std::max(0.,a.q(point)),q2=std::max(0.,b.q(point)),sum=q1+q2;
  if(!(sum>0.)){phi.setZero();grad.setZero();return;}
  const double bubble=q1*q2/sum;
  const V2 gb=(q2*q2*a.gradient(point)+q1*q1*b.gradient(point))/(sum*sum);
  const V2 z=(point-center).cwiseQuotient(scale);
  std::vector<double> x(order+1,1.),y(order+1,1.);
  for(int p=1;p<=order;++p){x[p]=x[p-1]*z.x();y[p]=y[p-1]*z.y();}
  int k=0;
  for(int degree=0;degree<=order;++degree)for(int px=0;px<=degree;++px) {
    const int py=degree-px;const double m=x[px]*y[py];
    phi[k]=bubble*m;grad.row(k)=(gb*m).transpose();
    if(px)grad(k,0)+=bubble*px*x[px-1]*y[py]/scale.x();
    if(py)grad(k,1)+=bubble*py*x[px]*y[py-1]/scale.y();++k;
  }
}
} // namespace detail

inline Result evaluate(const Body& bi,const Body& bj,const Options& options=Options{}) {
  if(!(options.eta>0.)||!(options.gap_floor>0.)||options.pressure_order<0||
     options.pressure_order>10||options.quadrature_order<6||options.quadrature_order>160||
     options.pressure_gap_power<0||options.pressure_gap_power>3||
     !std::isfinite(options.eta)||!std::isfinite(options.gap_floor)||
     !(options.max_face_angle>0.)||!(options.max_face_angle<.5*pi)||
     !(bi.radius>0.)||!(bj.radius>0.)||!(bi.half_thickness>0.)||!(bj.half_thickness>0.)||
     !std::isfinite(bi.radius)||!std::isfinite(bj.radius)||
     !std::isfinite(bi.half_thickness)||!std::isfinite(bj.half_thickness)||
     !bi.center.allFinite()||!bj.center.allFinite()||!bi.axis.allFinite()||!bj.axis.allFinite()||
     !(bi.axis.norm()>0.)||!(bj.axis.norm()>0.))
    throw std::invalid_argument("Invalid cylinder film geometry/options");
  Result out;
  V3 ni=bi.axis.normalized(),nj=bj.axis.normalized();
  if(ni.dot(nj)<0.)nj=-nj;
  const V3 d=bj.center-bi.center;
  V3 n=(ni+nj).normalized();if(n.dot(d)<0.){ni=-ni;nj=-nj;n=-n;}
  out.face_angle=std::acos(std::clamp(ni.dot(nj),-1.,1.));
  if(out.face_angle>options.max_face_angle)return out;
  const V3 ci=bi.center+bi.half_thickness*ni,cj=bj.center-bj.half_thickness*nj;
  const V3 origin=.5*(ci+cj);
  V3 x=d-n*n.dot(d);
  if(x.norm()<1e-10*std::max(bi.radius,bj.radius))x=ni-n*n.dot(ni);
  if(x.norm()<1e-12) {
    int k=0;n.cwiseAbs().minCoeff(&k);x=V3::Unit(k)-n*n[k];
  }
  x.normalize();M3 frame;frame.col(0)=x;frame.col(1)=n.cross(x);frame.col(2)=n;
  auto make_face=[&](const Body& body,const V3& axis,const V3& center) {
    detail::Ellipse f;const V3 v=frame.transpose()*axis,p=frame.transpose()*(center-origin);
    f.center=p.head<2>();f.slope=-v.head<2>()/v.z();
    f.z0=p.z()-f.slope.dot(f.center);
    f.Q=(M2::Identity()+f.slope*f.slope.transpose())/(body.radius*body.radius);return f;
  };
  const auto fi=make_face(bi,ni,ci),fj=make_face(bj,nj,cj);
  const auto xi=fi.xrange(),xj=fj.xrange();
  const double lo=std::max(xi[0],xj[0]),hi=std::min(xi[1],xj[1]);
  if(!(hi>lo))return out;
  std::vector<double> cuts{lo,hi};
  for(double v:detail::intersections_x(fi,fj))if(v>lo&&v<hi)cuts.push_back(v);
  std::sort(cuts.begin(),cuts.end());
  cuts.erase(std::unique(cuts.begin(),cuts.end(),[&](double a,double b){return std::abs(a-b)<1e-11*(hi-lo);}),cuts.end());
  double ylo=std::numeric_limits<double>::infinity(),yhi=-ylo;
  for(std::size_t i=1;i<cuts.size();++i) {
    double a=-std::numeric_limits<double>::infinity(),b=-a;
    if(fi.interval(.5*(cuts[i-1]+cuts[i]),a,b)&&fj.interval(.5*(cuts[i-1]+cuts[i]),a,b)) {
      ylo=std::min(ylo,a);yhi=std::max(yhi,b);
    }
  }
  if(!(yhi>ylo))return out;
  const V2 center(.5*(lo+hi),.5*(ylo+yhi)),scale(.5*(hi-lo),.5*(yhi-ylo));
  const int modes=(options.pressure_order+1)*(options.pressure_order+2)/2;
  Eigen::MatrixXd A=Eigen::MatrixXd::Zero(modes,modes),B=Eigen::MatrixXd::Zero(modes,22);
  Eigen::VectorXd phi(modes);Eigen::MatrixXd grad(modes,2);
  const auto vi=detail::velocity(bi,fi,origin,frame,0),vj=detail::velocity(bj,fj,origin,frame,11);
  const V2 slope=.5*(fi.slope+fj.slope),hslope=fj.slope-fi.slope;
  const double h0=fj.z0-fi.z0;
  const Map322 delta0=vj.value-vi.value,deltaX=vj.dx-vi.dx,deltaY=vj.dy-vi.dy;
  const Row22 div=.5*(vi.dx.row(0)+vj.dx.row(0)+vi.dy.row(1)+vj.dy.row(1));
  auto source=[&](const Map322& delta,double h)->Row22 {
    return delta.row(2)-slope.x()*delta.row(0)-slope.y()*delta.row(1)+h*div;
  };
  const Row22 f0=source(delta0,h0),fx=source(deltaX,hslope.x()),fy=source(deltaY,hslope.y());
  Map222 W;W.row(0)=.5*(vi.dx.row(2)+vj.dx.row(2));W.row(1)=.5*(vi.dy.row(2)+vj.dy.row(2));
  // The 22 inputs are disturbance velocities relative to the imposed affine
  // flow. Restore its common strain across the finite film. Omitting this
  // term would give 2*c*gamma, instead of (2*c+h)*gamma, for two coaxial
  // nonrotating plates translating affinely in simple shear. The added common
  // incompressible affine field has identically zero Reynolds pressure source.
  for(int base:{0,11})for(int k=0;k<5;++k)
    W.col(base+6+k)-=(frame.transpose()*strain_basis()[k]*n).head<2>();
  // H*grad(wbar) supplies the symmetric shear strain and cancels an arbitrary
  // common rigid rotation exactly; it is a finite-order objective completion.
  const Map222 T0=delta0.topRows<2>()+h0*W,Tx=deltaX.topRows<2>()+hslope.x()*W,
               Ty=deltaY.topRows<2>()+hslope.y()*W;
  const auto& rule=detail::rule(options.quadrature_order);
  for(std::size_t segment=1;segment<cuts.size();++segment) {
    const double mid=.5*(cuts[segment-1]+cuts[segment]),half=.5*(cuts[segment]-cuts[segment-1]);
    for(int ix=0;ix<options.quadrature_order;++ix) {
      // Sine mapping resolves the square-root ellipse tips without a staircase.
      const double theta=.5*pi*rule.x[ix],xx=mid+half*std::sin(theta);
      const double wx=rule.w[ix]*half*.5*pi*std::cos(theta);
      double bottom=-std::numeric_limits<double>::infinity(),top=-bottom;
      if(!fi.interval(xx,bottom,top)||!fj.interval(xx,bottom,top))continue;
      for(int iy=0;iy<options.quadrature_order;++iy) {
        const double yy=.5*(bottom+top)+.5*(top-bottom)*rule.x[iy];
        const double weight=wx*.5*(top-bottom)*rule.w[iy];
        const V2 point(xx,yy);const double h=h0+hslope.dot(point);
        out.min_gap=std::min(out.min_gap,h);out.max_gap=std::max(out.max_gap,h);
        if(h<-1e-10*std::max(bi.radius,bj.radius))
          throw std::runtime_error("Intersecting cylinder faces in Reynolds domain");
        const double he=std::max(h,options.gap_floor),K=he*he*he/(12.*options.eta);
        detail::basis(point,fi,fj,center,scale,options.pressure_order,phi,grad);
        if(options.pressure_gap_power) {
          const double reference=std::max(h0+hslope.dot(center),options.gap_floor);
          const double enrich=std::pow(reference/he,options.pressure_gap_power);
          if(h>options.gap_floor)
            grad.noalias()-=(double(options.pressure_gap_power)/he)*phi*hslope.transpose();
          grad*=enrich;phi*=enrich;
        }
        A.noalias()+=weight*K*(grad*grad.transpose());
        B.noalias()-=weight*phi*(f0+xx*fx+yy*fy);
        const Map222 T=T0+xx*Tx+yy*Ty;
        out.couette.noalias()+=(weight*options.eta/he)*(T.transpose()*T);
        out.area+=weight;++out.quadrature_points;
      }
    }
  }
  if(out.quadrature_points==0||!(out.area>0.))return out;
  const Eigen::VectorXd diagonal=A.diagonal().cwiseSqrt();
  if(!(diagonal.minCoeff()>0.))throw std::runtime_error("Degenerate cylinder film pressure basis");
  const Eigen::VectorXd inv=diagonal.cwiseInverse();
  const Eigen::MatrixXd As=inv.asDiagonal()*A*inv.asDiagonal(),Bs=inv.asDiagonal()*B;
  Eigen::LLT<Eigen::MatrixXd> factor(As);
  if(factor.info()!=Eigen::Success)throw std::runtime_error("Cylinder film pressure matrix not positive definite; lower basis order or refine quadrature");
  const Eigen::MatrixXd Y=factor.matrixL().solve(Bs);
  out.pressure.noalias()=Y.transpose()*Y;
  const Eigen::MatrixXd solution=factor.solve(Bs);
  out.pressure_residual=(As*solution-Bs).norm()/std::max(1e-300,Bs.norm());
  out.min_pressure_pivot=Eigen::MatrixXd(factor.matrixL()).diagonal().array().square().minCoeff();
  out.resistance=out.pressure+out.couette;
  out.resistance=.5*(out.resistance+out.resistance.transpose()).eval();
  if(!out.resistance.allFinite())throw std::runtime_error("Nonfinite cylinder film resistance");
  out.active=true;out.modes=modes;return out;
}
} // namespace SlurryCylinderFilm
#endif
