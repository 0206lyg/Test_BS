#ifndef SLURRY_OBJECTIVE_FILM_H
#define SLURRY_OBJECTIVE_FILM_H
// Finite-film Reynolds/Couette approximation only; this is not full Stokes.
// Geometry: equal oblate spheroids, c=eta=1, common closest normal is +z.
// DOFs: [Ui, a*Omega_i, Uj, a*Omega_j, a*E_0,...,a*E_4].
// E uses the orthonormal symmetric trace-free basis below.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace SlurryObjectiveFilm {
using V3=Eigen::Vector3d;
using M3=Eigen::Matrix3d;
using M17=Eigen::Matrix<double,17,17>;
using V17=Eigen::Matrix<double,17,1>;
using M11=Eigen::Matrix<double,8,8>;
using V11=Eigen::Matrix<double,8,1>;
using Velocity=Eigen::Matrix<double,3,8>;
using Tangent=Eigen::Matrix<double,2,8>;
struct Body { M3 M,Q; V3 center; bool lower; };
struct Surface { double z; Eigen::Vector2d gradient; bool valid; };
struct Result {
  M17 resistance=M17::Zero(),pressure=M17::Zero(),couette=M17::Zero();
  V3 center_i=V3::Zero(),center_j=V3::Zero();
  int cells=0; double pressure_relative_residual=0;
  // Source quadrature diagnostics; pressure/Couette retain the staircase domain.
  double source_quadrature_error_estimate=0.,source_quadrature_relative_indicator=0.;
  double source_clipped_area=0.,source_active_rect_area=0.;
  int source_quadrature_depth_limits=0;
};
inline std::array<M3,5> strain_basis() {
  std::array<M3,5> E;for(auto &e:E)e.setZero();
  E[0](0,0)=1/std::sqrt(2.);E[0](1,1)=-1/std::sqrt(2.);
  E[1](0,0)=E[1](1,1)=1/std::sqrt(6.);E[1](2,2)=-2/std::sqrt(6.);
  E[2](0,1)=E[2](1,0)=1/std::sqrt(2.);
  E[3](0,2)=E[3](2,0)=1/std::sqrt(2.);
  E[4](1,2)=E[4](2,1)=1/std::sqrt(2.);return E;
}
inline Body body(V3 axis,bool lower,double ar) {
  if(!axis.allFinite() || !(axis.norm()>0.))throw std::runtime_error("Invalid spheroid axis");
  axis.normalize();Body b;
  b.M=ar*ar*M3::Identity()+(1-ar*ar)*axis*axis.transpose();
  b.Q=M3::Identity()/(ar*ar)+(1-1/(ar*ar))*axis*axis.transpose();
  b.center=(lower?-1.:1.)*b.M.col(2)/std::sqrt(b.M(2,2));b.lower=lower;return b;
}
inline Surface surface(const Body &b,double x,double y) {
  const double dx=x-b.center.x(),dy=y-b.center.y();const auto &Q=b.Q;
  const double q=Q(2,0)*dx+Q(2,1)*dy;
  const double disc=q*q-Q(2,2)*(Q(0,0)*dx*dx+2*Q(0,1)*dx*dy+Q(1,1)*dy*dy-1);
  const double zz=-q/Q(2,2)+(b.lower?1.:-1.)*std::sqrt(std::max(disc,1e-30))/Q(2,2);
  const double den=Q(2,0)*dx+Q(2,1)*dy+Q(2,2)*zz;
  Surface s;s.z=b.center.z()+zz;s.valid=disc>0.;
  s.gradient<<-(Q(0,0)*dx+Q(0,1)*dy+Q(0,2)*zz)/den,
              -(Q(1,0)*dx+Q(1,1)*dy+Q(1,2)*zz)/den;
  return s;
}

// Integrate the existing Reynolds source over a projected-overlap cell.
// Bounded surface heights replace the pointwise derivatives that diverge at
// projected rims. The pressure/Couette midpoint-active domain is unchanged.
using Q6=Eigen::Matrix<double,6,1>;
using Q8=Eigen::Matrix<double,8,1>;
struct SourceQuadrature { Q6 value=Q6::Zero(); double error=0.; int depth_limits=0; };
struct CellSource { Q8 value=Q8::Zero(); double area=0.,error=0.; int depth_limits=0; };
inline double surface_height(const Body &b,double x,double y) {
  const double dx=x-b.center.x(),dy=y-b.center.y();const auto &Q=b.Q;
  const double q=Q(2,0)*dx+Q(2,1)*dy;
  const double disc=q*q-Q(2,2)*(Q(0,0)*dx*dx+2*Q(0,1)*dx*dy+Q(1,1)*dy*dy-1.);
  // Called only inside analytically clipped projected intervals. This max
  // handles roundoff at their exact endpoints; it is not a geometric cutoff.
  return b.center.z()+(-q+(b.lower?1.:-1.)*std::sqrt(std::max(disc,0.)))/Q(2,2);
}
inline bool projected_interval(const Body &b,int d,double other,double &lo,double &hi) {
  const int o=1-d;const auto &Q=b.Q;const double s=other-b.center[o];
  const double A=Q(d,d)-Q(d,2)*Q(2,d)/Q(2,2);
  const double B=Q(d,o)-Q(d,2)*Q(2,o)/Q(2,2);
  const double C=Q(o,o)-Q(o,2)*Q(2,o)/Q(2,2);
  const double r=A-(A*C-B*B)*s*s;
  if(!(r>0.))return false;
  const double mid=b.center[d]-B*s/A,half=std::sqrt(r)/A;
  lo=std::max(lo,mid-half);hi=std::min(hi,mid+half);return hi>lo;
}
inline Q6 source_slice(const Body &bi,const Body &bj,int d,double s,
                       const std::array<double,2> &lo,const std::array<double,2> &hi) {
  double l=lo[d],r=hi[d];Q6 out=Q6::Zero();
  if(!projected_interval(bi,d,s,l,r)||!projected_interval(bj,d,s,l,r))return out;
  const auto heights=[&](double v) {
    const double x=d==0?v:s,y=d==0?s:v;
    const double zi=surface_height(bi,x,y),zj=surface_height(bj,x,y);
    return std::array<double,2>{.5*(zi+zj),(zi-bj.center.z())*(zj-bj.center.z())};
  };
  const auto L=heights(l),R=heights(r);
  out[0]=r-l;
  out[1]=d==0?.5*(r*r-l*l):s*(r-l);
  out[2]=d==0?s*(r-l):.5*(r*r-l*l);
  out[3]=R[0]-L[0];out[4]=R[1]-L[1];
  out[5]=(s-bj.center[1-d])*out[3];return out;
}
inline SourceQuadrature integrate_source_slice(const Body &bi,const Body &bj,int d,
    const std::array<double,2> &lo,const std::array<double,2> &hi,double a,double b,int depth=0) {
  static constexpr double x[8]={
    .991455371120812639206854697526329,.949107912342758524526189684047851,
    .864864423359769072789712788640926,.741531185599394439863864773280788,
    .586087235467691130294144838258730,.405845151377397166906606412076961,
    .207784955007898467600689403773245,0.};
  static constexpr double wk[8]={
    .022935322010529224963732008058970,.063092092629978553290700663189204,
    .104790010322250183839876322541518,.140653259715525918745189590510238,
    .169004726639267902826583426598550,.190350578064785409913256402421014,
    .204432940075298892414161999234649,.209482141084727828012999174891714};
  static constexpr double wg[4]={
    .129484966168869693270611432679082,.279705391489276667901467771423780,
    .381830050505118944950369775488975,.417959183673469387755102040816327};
  SourceQuadrature out;if(!(b>a))return out;
  const double mid=.5*(a+b),half=.5*(b-a);
  const Q6 center=source_slice(bi,bj,d,mid,lo,hi);
  Q6 kronrod=wk[7]*center,gauss=wg[3]*center;
  for(int k=0;k<7;++k) {
    const Q6 pair=source_slice(bi,bj,d,mid-half*x[k],lo,hi)+
                  source_slice(bi,bj,d,mid+half*x[k],lo,hi);
    kronrod+=wk[k]*pair;if(k%2)gauss+=wg[(k-1)/2]*pair;
  }
  out.value=half*kronrod;out.error=(half*(kronrod-gauss)).cwiseAbs().maxCoeff();
  const double tolerance=2e-7*std::max(out.value.cwiseAbs().maxCoeff(),1e-30);
  if(out.error<=tolerance)return out;
  if(depth==8){out.depth_limits=1;return out;}
  const auto L=integrate_source_slice(bi,bj,d,lo,hi,a,mid,depth+1);
  const auto R=integrate_source_slice(bi,bj,d,lo,hi,mid,b,depth+1);
  out.value=L.value+R.value;out.error=L.error+R.error;
  out.depth_limits=L.depth_limits+R.depth_limits;return out;
}
inline CellSource cell_source(const Body &bi,const Body &bj,
    const std::array<double,2> &lo,const std::array<double,2> &hi,double ar) {
  std::array<SourceQuadrature,2> integral;
  for(int d=0;d<2;++d) {
    const int o=1-d;
    const double a=std::max(lo[o],std::max(bi.center[o]-std::sqrt(bi.M(o,o)),
                                         bj.center[o]-std::sqrt(bj.M(o,o))));
    const double b=std::min(hi[o],std::min(bi.center[o]+std::sqrt(bi.M(o,o)),
                                         bj.center[o]+std::sqrt(bj.M(o,o))));
    integral[d]=integrate_source_slice(bi,bj,d,lo,hi,a,b);
  }
  const auto &X=integral[0].value,&Y=integral[1].value;
  CellSource out;out.area=.5*(X[0]+Y[0]);
  const double mx=.5*(X[1]+Y[1]),my=.5*(X[2]+Y[2]),dz=bj.center.z()-bi.center.z();
  out.value[0]=-X[3];out.value[1]=-Y[3];out.value[2]=out.area;
  out.value[3]=(my-bj.center.y()*out.area+.5*Y[4])/ar;
  out.value[4]=(-mx+bj.center.x()*out.area-.5*X[4])/ar;
  out.value[5]=(X[5]-Y[5])/ar;
  out.value[6]=((bj.center.x()-bi.center.x())*out.area-dz*X[3])/(ar*std::sqrt(2.));
  out.value[7]=((bj.center.y()-bi.center.y())*out.area-dz*Y[3])/(ar*std::sqrt(2.));
  out.error=integral[0].error+integral[1].error;
  out.depth_limits=integral[0].depth_limits+integral[1].depth_limits;return out;
}


inline Result assemble(V3 axis_i,V3 axis_j,double h,int n=81,double ar=10.,double eps=.001) {
  if(!(ar>0.)||!std::isfinite(ar)||!(h+eps>0.)||eps<0.||!std::isfinite(h+eps)||n<3)
    throw std::runtime_error("Invalid finite-film dimensions or grid");
  Body bi=body(axis_i,true,ar),bj=body(axis_j,false,ar);bj.center.z()+=h+eps;
  const Eigen::Matrix2d C=.5*(bi.Q.topLeftCorner<2,2>()*std::sqrt(bi.M(2,2))+bj.Q.topLeftCorner<2,2>()*std::sqrt(bj.M(2,2)));
  std::array<std::vector<double>,2> edge,center,width;
  for(int d=0;d<2;++d) {
    const double lo=std::max(bi.center[d]-std::sqrt(bi.M(d,d)),bj.center[d]-std::sqrt(bj.M(d,d)));
    const double hi=std::min(bi.center[d]+std::sqrt(bi.M(d,d)),bj.center[d]+std::sqrt(bj.M(d,d)));
    if(!(hi>lo))throw std::runtime_error("No projected film overlap");
    const double scale=.3*std::sqrt((h+eps)/C(d,d)),v0=std::asinh(lo/scale),v1=std::asinh(hi/scale);
    edge[d].resize(n+1);center[d].resize(n);width[d].resize(n);
    for(int k=0;k<=n;++k)edge[d][k]=scale*std::sinh(v0+(v1-v0)*k/n);
    for(int k=0;k<n;++k){width[d][k]=edge[d][k+1]-edge[d][k];center[d][k]=.5*(edge[d][k+1]+edge[d][k]);}
  }
  struct Cell { Surface si,sj;double h;int id; };
  std::vector<Cell> cells(n*n);int nr=0;
  for(int ix=0;ix<n;++ix)for(int iy=0;iy<n;++iy) {
    auto &c=cells[ix*n+iy];c.si=surface(bi,center[0][ix],center[1][iy]);c.sj=surface(bj,center[0][ix],center[1][iy]);c.h=c.sj.z-c.si.z;
    c.id=c.si.valid&&c.sj.valid&&c.h>0.?nr++:-1;
  }
  if(!nr)throw std::runtime_error("No active finite-film cells");
  using Triplet=Eigen::Triplet<double>;std::vector<Triplet> trips;trips.reserve(5*nr);
  Eigen::MatrixXd B=Eigen::MatrixXd::Zero(nr,8);M11 Rc=M11::Zero();const auto E=strain_basis();
  double source_error=0.,source_scale=0.,source_area=0.,source_rect_area=0.;
  int source_depth_limits=0;
  Tangent W=Tangent::Zero();W(0,4)=-.5/ar;W(1,3)=.5/ar;
  for(int k=3;k<5;++k)W.col(3+k)=-E[k].row(2).head<2>().transpose()/ar;
  for(int ix=0;ix<n;++ix)for(int iy=0;iy<n;++iy) {
    const auto &c=cells[ix*n+iy];if(c.id<0)continue;
    double diagonal=0.;
    for(int d=0;d<2;++d)for(int sign:{-1,1}) {
      const int jx=ix+(d==0?sign:0),jy=iy+(d==1?sign:0),k=d==0?ix:iy;
      const double length=width[1-d][d==0?iy:ix];
      if(jx>=0&&jx<n&&jy>=0&&jy<n&&cells[jx*n+jy].id>=0) {
        const double xx=d==0?edge[0][ix+(sign>0)]:center[0][ix];
        const double yy=d==1?edge[1][iy+(sign>0)]:center[1][iy];
        const double hf=surface(bj,xx,yy).z-surface(bi,xx,yy).z;
        const double delta=std::abs(center[d][k+sign]-center[d][k]);
        const double coefficient=hf*hf*hf*length/delta;
        if(!(coefficient>0.)||!std::isfinite(coefficient))throw std::runtime_error("Invalid finite-volume conductance");
        diagonal+=coefficient;trips.emplace_back(c.id,cells[jx*n+jy].id,-coefficient);
      } else diagonal+=c.h*c.h*c.h*length/(.5*width[d][k]);
    }
    trips.emplace_back(c.id,c.id,diagonal);
    Velocity VI=Velocity::Zero(),VJ=Velocity::Zero();
    const V3 ri=V3(center[0][ix],center[1][iy],c.si.z)-bi.center;
    const V3 rj=V3(center[0][ix],center[1][iy],c.sj.z)-bj.center;
    for(int ib=0;ib<2;++ib) {
      auto &V=ib?VJ:VI;const V3 &r=ib?rj:ri;const int base=0;
      if(ib)for(int k=0;k<3;++k) {
        V(k,base+k)=1.;const V3 omega=V3::Unit(k);V.col(base+3+k)=omega.cross(r)/ar;
      }
      for(int k=3;k<5;++k) {
        V.col(3+k)=-E[k]*r/ar;
      }
    }
    // Integrate the same source functional over the true projected overlap
    // within this active cell. Boundary derivatives are replaced algebraically
    // by bounded endpoint fluxes, including the curved projected-rim boundary.
    const auto integrated=cell_source(bi,bj,{edge[0][ix],edge[1][iy]},
                                      {edge[0][ix+1],edge[1][iy+1]},ar);
    B.row(c.id)=-integrated.value.transpose();
    const double area=width[0][ix]*width[1][iy];
    source_error+=integrated.error;source_scale+=integrated.value.cwiseAbs().sum();
    source_area+=integrated.area;source_rect_area+=area;
    source_depth_limits+=integrated.depth_limits;
    const Tangent T=VJ.topRows<2>()-VI.topRows<2>()+c.h*W;Rc.noalias()+=(area/c.h)*T.transpose()*T;
  }
  Eigen::SparseMatrix<double> A(nr,nr);A.setFromTriplets(trips.begin(),trips.end());
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;solver.compute(A);
  if(solver.info()!=Eigen::Success || solver.vectorD().minCoeff()<=0.)throw std::runtime_error("Finite-film pressure factorization failed");
  const Eigen::MatrixXd X=solver.solve(B);
  if(solver.info()!=Eigen::Success||!X.allFinite())throw std::runtime_error("Finite-film pressure solve failed");
  Result result;result.cells=nr;result.center_i=bi.center;result.center_j=bj.center;
  result.pressure_relative_residual=(A*X-B).norm()/std::max(B.norm(),1e-300);
  result.source_quadrature_error_estimate=source_error;
  result.source_quadrature_relative_indicator=source_error/std::max(source_scale,1e-300);
  result.source_quadrature_depth_limits=source_depth_limits;
  result.source_clipped_area=source_area;result.source_active_rect_area=source_rect_area;
  // Fix body i by subtracting one common rigid motion, then lift the exact
  // 8-coordinate Gram matrix. The three in-plane strain null modes are
  // exact nulls of this lubrication approximation, not of full Stokes flow.
  // No approximation is introduced beyond the specified film functional.
  Eigen::Matrix<double,8,17> P=Eigen::Matrix<double,8,17>::Zero();
  const V3 r=bj.center-bi.center;M3 cross;
  cross<<0.,-r.z(),r.y(),r.z(),0.,-r.x(),-r.y(),r.x(),0.;
  P.block<3,3>(0,0)=-M3::Identity();P.block<3,3>(0,3)=cross/ar;P.block<3,3>(0,6)=M3::Identity();
  P.block<3,3>(3,3)=-M3::Identity();P.block<3,3>(3,9)=M3::Identity();P.block<2,2>(6,15).setIdentity();
  for(int k=0;k<3;++k)P.block<3,1>(0,12+k)=E[k]*r/ar;
  const M11 Rp=12*B.transpose()*X;
  result.pressure=P.transpose()*Rp*P;result.couette=P.transpose()*Rc*P;result.resistance=result.pressure+result.couette;
  if(!result.resistance.allFinite())throw std::runtime_error("Nonfinite film resistance");return result;
}
}
#endif
