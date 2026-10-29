#ifndef SLURRY_SCATTERING_MATH_H
#define SLURRY_SCATTERING_MATH_H
// Generic spheroid two-body exterior-flow surrogate. No pair-geometry fit.
// Isolated boundary-scattering basis is generated offline. Runtime requires
// only the Eigen headers already bundled with the slurry plugin.
// Inputs: U-u_inf(center), a*(Omega-Omega_inf), shared a*E_5.
// Output: forces on fluid, torques/a, intrinsic physical stresslets/a.
// Coefficients use c=eta=1; multiply the returned matrix by eta*c.
// This is an approximation; callers must enforce validated domain and gates.
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace SlurryScattering {
using Matrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using Mat3=Eigen::Matrix3d;
using Vec3=Eigen::Vector3d;
using Mat17=Eigen::Matrix<double,17,17>;
struct Body { Mat3 rotation=Mat3::Identity(); Vec3 center=Vec3::Zero(); };
inline std::array<Mat3,5> strain_basis() {
  std::array<Mat3,5> e; for(auto &x:e)x.setZero();
  const double s=1/std::sqrt(2.); e[0](0,0)=s;e[0](1,1)=-s;
  const double t=1/std::sqrt(6.);e[1](0,0)=t;e[1](1,1)=t;e[1](2,2)=-2*t;
  e[2](0,1)=e[2](1,0)=s;e[3](0,2)=e[3](2,0)=s;e[4](1,2)=e[4](2,1)=s;
  return e;
}
inline Eigen::Matrix<double,5,5> strain_rotation(const Mat3 &Q) {
  const auto e=strain_basis();Eigen::Matrix<double,5,5> r;
  for(int i=0;i<5;++i)for(int j=0;j<5;++j)r(i,j)=(e[i].array()*(Q*e[j]*Q.transpose()).array()).sum();
  return r;
}
struct Result { Mat17 grand; double reciprocity=0,linear_residual=0; };
class Basis {
 public:
  double aspect_ratio=0.;int degree=0;
  Matrix targets,sources,projection,density,rigid,strain,force_torque,stresslet;
  // V2 azimuth contraction retains every stored Fourier coefficient and the
  // identical discrete source nodes. This only reassociates finite sums.
  Matrix fourier_rings,fourier_density,fourier_harmonics;
  int fourier_nphi=0,fourier_modes=0;
  bool use_fourier=false;
  void load(const std::string &path) {
    std::ifstream f(path);if(!f)throw std::runtime_error("Cannot open scattering basis: "+path);load(f);
  }
  void load(std::istream &f) {
    use_fourier=false;fourier_nphi=0;fourier_modes=0;
    std::string magic;
    if(!(f>>magic) || (magic!="SLURRY_SCATTERING_V1" && magic!="SLURRY_SCATTERING_V2"))throw std::runtime_error("Invalid scattering basis header");
    if(!(f>>aspect_ratio>>degree) || !(aspect_ratio>=1) || degree<1 || degree>30)
      throw std::runtime_error("Invalid scattering aspect ratio/degree");
    if(magic=="SLURRY_SCATTERING_V1") {
      read(f,"targets",targets);read(f,"sources",sources);read(f,"projection",projection);read(f,"density",density);
    } else {
      std::string tag;int nphi=0,order=0;f>>tag>>nphi>>order;
      if(tag!="FOURIER" || nphi<4 || order<0 || 2*order>=nphi || nphi>1024)
        throw std::runtime_error("Invalid scattering Fourier dimensions");
      Matrix rings,coef;read(f,"targets",targets);read(f,"rings",rings);
      read(f,"projection",projection);read(f,"fourier_density",coef);
      if(rings.cols()!=2 || coef.rows()!=rings.rows()*(2*order+1)*3 || coef.cols()!=projection.rows())
        throw std::runtime_error("Scattering Fourier coefficient dimensions mismatch");
      const int nm=2*order+1,p=projection.rows();
      if(rings.rows()>100000/nphi)throw std::runtime_error("Too many scattering Fourier nodes");
      use_fourier=nphi>=2*nm;
      fourier_nphi=nphi;fourier_modes=nm;
      if(use_fourier) {
        fourier_rings=rings;fourier_density=coef;fourier_harmonics.resize(nphi,nm);
        for(int j=0;j<nphi;++j) {
          const double phi=2*std::acos(-1.)*j/nphi;
          fourier_harmonics(j,0)=1./nphi;
          for(int m=1;m<=order;++m) {
            fourier_harmonics(j,2*m-1)=std::cos(m*phi)/nphi;
            fourier_harmonics(j,2*m)=std::sin(m*phi)/nphi;
          }
        }
      }
      sources.resize(rings.rows()*nphi,3);
      if(use_fourier)density.resize(0,0);else density.resize(3*sources.rows(),p);
      for(int r=0;r<rings.rows();++r)for(int j=0;j<nphi;++j) {
        const int idx=r*nphi+j;const double phi=2*std::acos(-1.)*j/nphi;
        sources.row(idx)<<rings(r,0)*std::cos(phi),rings(r,0)*std::sin(phi),rings(r,1);
        if(use_fourier)continue;
        density.middleRows(3*idx,3)=coef.middleRows(3*r*nm,3);
        for(int m=1;m<=order;++m) {
          density.middleRows(3*idx,3)+=std::cos(m*phi)*coef.middleRows(3*(r*nm+2*m-1),3);
          density.middleRows(3*idx,3)+=std::sin(m*phi)*coef.middleRows(3*(r*nm+2*m),3);
        }
        density.middleRows(3*idx,3)/=nphi;
      }
    }
    read(f,"rigid",rigid);read(f,"strain",strain);
    read(f,"force_torque",force_torque);read(f,"stresslet",stresslet);
    const int p=projection.rows();
    if(targets.cols()!=3 || sources.cols()!=3 || projection.cols()!=3*targets.rows() ||
       (!use_fourier && (density.rows()!=3*sources.rows() || density.cols()!=p)) || rigid.rows()!=p || rigid.cols()!=6 ||
       strain.rows()!=p || strain.cols()!=5 || force_torque.rows()!=6 || force_torque.cols()!=p ||
       stresslet.rows()!=5 || stresslet.cols()!=p)throw std::runtime_error("Scattering basis dimensions mismatch");
  }
  // Cross field is evaluated in the source body's local frame. This leaves
  // its precomputed density unchanged and rotates only targets and velocities.
  Matrix cross(const Body &target,const Body &source) const {
    if(use_fourier)return cross_fourier(target,source);
    const Mat3 transform=source.rotation.transpose()*target.rotation;
    const Vec3 shift=source.rotation.transpose()*(target.center-source.center);
    Matrix kernel(3*targets.rows(),3*sources.rows());
    const double inv8pi=1./(8*std::acos(-1.));
    for(int t=0;t<targets.rows();++t) {
      const Vec3 x=transform*targets.row(t).transpose()+shift;
      for(int s=0;s<sources.rows();++s) {
        const Vec3 d=x-sources.row(s).transpose();const double r2=d.squaredNorm();
        if(!(r2>1e-25))throw std::runtime_error("Scattering target touches interior source");
        kernel.block<3,3>(3*t,3*s)=(inv8pi/std::sqrt(r2))*(Mat3::Identity()+d*d.transpose()/r2);
      }
    }
    Matrix field=kernel*density;
    for(int t=0;t<targets.rows();++t)field.middleRows(3*t,3)=(transform.transpose()*field.middleRows(3*t,3)).eval();
    return projection*field;
  }
  Matrix cross_fourier(const Body &target,const Body &source) const {
    const Mat3 transform=source.rotation.transpose()*target.rotation;
    const Vec3 shift=source.rotation.transpose()*(target.center-source.center);
    const int nt=targets.rows(),nr=fourier_rings.rows(),nm=fourier_modes;
    // Target quadrature refinement must not allocate its full dense kernel per
    // OpenMP worker. Independent target rows have the same source contraction;
    // blocking them changes no quadrature nodes, basis coefficients or model.
    constexpr int target_block=32;
    Matrix kernel(3*std::min(nt,target_block),nr*nm*3),ring_kernel(9,fourier_nphi),ring_modes(9,nm);
    Matrix field(3*nt,fourier_density.cols());
    const double inv8pi=1./(8*std::acos(-1.));
    for(int begin=0;begin<nt;begin+=target_block) {
      const int count=std::min(target_block,nt-begin);
      for(int local=0;local<count;++local) {
      const int t=begin+local;
      const Vec3 x=transform*targets.row(t).transpose()+shift;
      for(int r=0;r<nr;++r) {
        for(int q=0;q<fourier_nphi;++q) {
          const Vec3 d=x-sources.row(r*fourier_nphi+q).transpose();
          const double r2=d.squaredNorm();
          if(!(r2>1e-25))throw std::runtime_error("Scattering target touches interior source");
          const double z=inv8pi/std::sqrt(r2),w=z/r2;
          for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            ring_kernel(3*i+j,q)=w*d[i]*d[j]+(i==j?z:0.);
        }
        ring_modes.noalias()=ring_kernel*fourier_harmonics;
        for(int m=0;m<nm;++m)for(int i=0;i<3;++i)for(int j=0;j<3;++j)
          kernel(3*local+i,3*(r*nm+m)+j)=ring_modes(3*i+j,m);
      }
      }
      field.middleRows(3*begin,3*count).noalias()=kernel.topRows(3*count)*fourier_density;
    }
    for(int t=0;t<nt;++t)
      field.middleRows(3*t,3)=(transform.transpose()*field.middleRows(3*t,3)).eval();
    return projection*field;
  }
  Result evaluate(const std::array<Body,2> &b) const {
    const int p=projection.rows();Matrix A=Matrix::Identity(2*p,2*p);
    A.block(0,p,p,p)=cross(b[0],b[1]);A.block(p,0,p,p)=cross(b[1],b[0]);
    Matrix rhs=Matrix::Zero(2*p,17),mom=Matrix::Zero(17,2*p);
    for(int i=0;i<2;++i) {
      Eigen::Matrix<double,6,6> Q6=Eigen::Matrix<double,6,6>::Zero();
      Q6.block<3,3>(0,0)=b[i].rotation;Q6.block<3,3>(3,3)=b[i].rotation;
      const auto Q5=strain_rotation(b[i].rotation);
      rhs.block(0+i*p,i*6,p,6)=rigid*Q6.transpose();rhs.block(i*p,12,p,5)=strain*Q5.transpose();
      mom.block(i*6,i*p,6,p)=Q6*force_torque;mom.block(12,i*p,5,p)=Q5*stresslet;
    }
    Eigen::PartialPivLU<Matrix> lu(A);const Matrix x=lu.solve(rhs);
    Result r;r.grand=mom*x;r.linear_residual=(A*x-rhs).norm()/rhs.norm();
    r.reciprocity=(r.grand-r.grand.transpose()).norm()/r.grand.norm();
    if(!r.grand.allFinite() || !std::isfinite(r.linear_residual) || r.linear_residual>1e-8)
      throw std::runtime_error("Scattering pair solve failed residual gate");
    return r; // Preserve raw reciprocity error; do not silently clip eigenvalues.
  }
 private:
  static void read(std::istream &f,const char *expected,Matrix &m) {
    std::string name;int r=0,c=0;
    if(!(f>>name>>r>>c) || name!=expected || r<=0 || c<=0 || r>100000 || c>100000 || (long long)r*c>16000000)
      throw std::runtime_error(std::string("Invalid scattering section ")+expected);
    m.resize(r,c);for(int i=0;i<r;++i)for(int j=0;j<c;++j)
      if(!(f>>m(i,j)) || !std::isfinite(m(i,j)))throw std::runtime_error("Invalid scattering coefficient");
  }
};
}
#endif
