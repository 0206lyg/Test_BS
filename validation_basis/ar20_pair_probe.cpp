// One exact recorded AR20 pose; direct scattering only, no film/matching path.
// Reuses the generator's quadrature definitions to retarget a fixed fitted basis.
// Local copies of the generator quadrature helpers keep this standalone probe
// from compiling its unrelated fitting/serialization executable as well.
#include "../lammps_slurry/src/slurry_scattering_provider.h"
#include "../lammps_slurry/tools/basis_cylindrical_kernel.h"
#include <Eigen/Eigenvalues>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/resource.h>
using M=Eigen::MatrixXd;
using V=Eigen::VectorXd;
constexpr double pi=3.141592653589793238462643383279502884;
struct GL { V x,w; };
GL gauss(int n) {
  GL g{V(n),V(n)};
  for(int k=0;k<(n+1)/2;++k) {
    double z=std::cos(pi*(k+.75)/(n+.5)),derivative=0.;
    for(int it=0;it<100;++it) {
      double a=1,b=z;
      if(n==0)b=a;
      for(int l=2;l<=n;++l){const double c=((2*l-1)*z*b-(l-1)*a)/l;a=b;b=c;}
      derivative=n*(z*b-a)/(z*z-1.);
      const double next=z-b/derivative;
      if(std::abs(next-z)<3e-16){z=next;break;}z=next;
    }
    // Re-evaluate derivative at the accepted node.
    double a=1,b=z;for(int l=2;l<=n;++l){double c=((2*l-1)*z*b-(l-1)*a)/l;a=b;b=c;}
    derivative=n*(z*b-a)/(z*z-1.);
    g.x[k]=-z;g.x[n-1-k]=z;
    g.w[k]=g.w[n-1-k]=2/((1-z*z)*derivative*derivative);
  }
  return g;
}
// Orthonormal real scalar harmonics in dmu dphi, l-major order:
// m=0, cos(m phi), sin(m phi), m=1..l. Condon--Shortley phase is immaterial.
V harmonics(int L,double mu,double phi) {
  V out((L+1)*(L+1));out.setZero();
  double pmm=1.;
  for(int m=0;m<=L;++m) {
    if(m)pmm*=-(2*m-1)*std::sqrt(std::max(0.,1-mu*mu));
    double prev=0.,p=pmm;
    for(int l=m;l<=L;++l) {
      if(l>m){double next=((2*l-1)*mu*p-(l+m-1)*prev)/(l-m);prev=p;p=next;}
      const double norm=std::sqrt((2*l+1)/(4*pi)*std::exp(std::lgamma(l-m+1.)-std::lgamma(l+m+1.)));
      if(m==0)out[l*l]=norm*p;
      else {out[l*l+2*m-1]=std::sqrt(2.)*norm*p*std::cos(m*phi);out[l*l+2*m]=std::sqrt(2.)*norm*p*std::sin(m*phi);}
    }
  }
  return out;
}
struct Surface {M points,H;V weights;GL gl;};
Surface surface(double a,int L,int nlat,int nphi,double phase=0.) {
  Surface s;s.gl=gauss(nlat);const int n=nlat*nphi,p0=3*(L+1)*(L+1);
  s.points.resize(n,3);s.H=M::Zero(3*n,p0);s.weights.resize(3*n);
  for(int i=0;i<nlat;++i)for(int j=0;j<nphi;++j) {
    const int q=i*nphi+j;const double phi=2*pi*(j+phase)/nphi,mu=s.gl.x[i],rho=a*std::sqrt(1-mu*mu);
    s.points.row(q)<<rho*std::cos(phi),rho*std::sin(phi),mu;
    const V y=harmonics(L,mu,phi);
    for(int k=0;k<y.size();++k)for(int c=0;c<3;++c)s.H(3*q+c,3*k+c)=y[k];
    for(int c=0;c<3;++c)s.weights[3*q+c]=s.gl.w[i]*2*pi/nphi;
  }
  return s;
}
M null_flux(const Surface& s,double a) {
  V f=V::Zero(s.H.cols());
  for(int q=0;q<s.points.rows();++q) {
    // n dS/(dmu dphi)=(x,y,a^2 z); no normalized-normal approximation.
    Eigen::Vector3d nds(s.points(q,0),s.points(q,1),a*a*s.points(q,2));
    f.noalias()+=s.weights[3*q]*s.H.middleRows(3*q,3).transpose()*nds;
  }
  if(!(f.norm()>0))throw std::runtime_error("Zero net-flux functional");
  f.normalize();V v=f;v[0]+=f[0]>=0?1.:-1.;v.normalize();
  M Q=M::Identity(f.size(),f.size())-2*v*v.transpose();
  M basis=Q.rightCols(f.size()-1);
  if((f.transpose()*basis).norm()>1e-12)throw std::runtime_error("Net-flux elimination failed");
  return basis;
}
M rigid_values(const M& points,double a) {
  M v=M::Zero(3*points.rows(),6);
  for(int q=0;q<points.rows();++q){v.block<3,3>(3*q,0).setIdentity();for(int k=0;k<3;++k)v.block<3,1>(3*q,3+k)=Eigen::Vector3d::Unit(k).cross(Eigen::Vector3d(points.row(q).transpose()))/a;}
  return v;
}
M strain_values(const M& points,double a) {
  M v(3*points.rows(),5);const auto es=SlurryScattering::strain_basis();
  for(int q=0;q<points.rows();++q)for(int k=0;k<5;++k)v.block<3,1>(3*q,k)=-es[k]*points.row(q).transpose()/a;
  return v;
}
M exact_isolated(double a) {
  M h=M::Zero(11,11);const auto coeff=SlurryStress::unit_stresslet({a,a,1.});const auto drag=SlurryHydro::unit_resistance({a,a,1.});
  for(int k=0;k<3;++k){h(k,k)=drag.translation[k];h(k+3,k+3)=drag.rotation[k]/(a*a);}
  const SlurryStress::Mat3 Q{1,0,0,0,1,0,0,0,1};const auto es=SlurryScattering::strain_basis();
  for(int k=0;k<5;++k){SlurryStress::Mat3 E{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)E[3*i+j]=es[k](i,j)/a;
    const auto s=SlurryStress::stresslet(coeff,Q,1.,E,{0,0,0},{0,0,0});
    for(int i=0;i<3;++i)h(3+i,6+k)=h(6+k,3+i)=-s.hydro_torque[i]/a;
    for(int i=0;i<5;++i)for(int r=0;r<3;++r)for(int c=0;c<3;++c)h(6+i,6+k)+=es[i](r,c)*s.total[3*r+c]/a;
  }return h;
}

namespace sc=SlurryScattering;
using CM=Eigen::MatrixXcd;
template<class Derived> void matrix_json(std::ostream& f,const Eigen::MatrixBase<Derived>& x) {
  f<<'[';for(int i=0;i<x.rows();++i){if(i)f<<',';f<<'[';for(int j=0;j<x.cols();++j){if(j)f<<',';f<<x(i,j);}f<<']';}f<<']';
}
void change_source_phi(sc::Basis& b,int nphi) {
  const int nr=b.fourier_rings.rows(),nm=b.fourier_modes;
  if(!b.use_fourier||nphi<2*nm||nr*nphi>100000)throw std::runtime_error("Unsupported source quadrature");
  b.fourier_nphi=nphi;b.sources.resize(nr*nphi,3);b.fourier_harmonics.resize(nphi,nm);
  for(int j=0;j<nphi;++j){const double phi=2*pi*j/nphi;b.fourier_harmonics(j,0)=1./nphi;
    for(int m=1;m<=(nm-1)/2;++m){b.fourier_harmonics(j,2*m-1)=std::cos(m*phi)/nphi;b.fourier_harmonics(j,2*m)=std::sin(m*phi)/nphi;}
    for(int r=0;r<nr;++r)b.sources.row(r*nphi+j)<<b.fourier_rings(r,0)*std::cos(phi),b.fourier_rings(r,0)*std::sin(phi),b.fourier_rings(r,1);}
}
void change_targets(sc::Basis& b,int factor) {
  if(factor<1||factor>8)throw std::runtime_error("Unsupported target factor");
  int old_phi=1;while(old_phi<b.targets.rows()&&b.targets(old_phi,2)==b.targets(0,2))++old_phi;
  if(b.targets.rows()%old_phi)throw std::runtime_error("Cannot recover original target grid");
  if(old_phi==factor*(2*b.degree+5)&&b.targets.rows()/old_phi==factor*(b.degree+3))return;
  const auto original=surface(b.aspect_ratio,b.degree,b.targets.rows()/old_phi,old_phi);
  // Recover the SAVED null-flux coordinate gauge. Recomputing its Householder
  // basis on another quadrature can change a sign through roundoff in f[0].
  // Such a gauge change must not be applied without rotating stored density.
  const M F=(b.projection*original.H).transpose();
  const auto surf=surface(b.aspect_ratio,b.degree,factor*(b.degree+3),factor*(2*b.degree+5));
  const M H=surf.H*F,wh=surf.weights.asDiagonal()*H;
  b.targets=surf.points;b.projection=(H.transpose()*wh).ldlt().solve(wh.transpose());
  b.rigid=b.projection*rigid_values(b.targets,b.aspect_ratio);b.strain=b.projection*strain_values(b.targets,b.aspect_ratio);
}
void restrict_degree(sc::Basis& b,int degree,int factor) {
  if(degree==b.degree){change_targets(b,factor);return;}
  if(degree<1||degree>b.degree)throw std::runtime_error("Restriction degree must not exceed fitted basis degree");
  int old_phi=1;while(old_phi<b.targets.rows()&&b.targets(old_phi,2)==b.targets(0,2))++old_phi;
  const auto original=surface(b.aspect_ratio,b.degree,b.targets.rows()/old_phi,old_phi);
  const M high_flux=(b.projection*original.H).transpose();
  const auto low=surface(b.aspect_ratio,degree,factor*(degree+3),factor*(2*degree+5));
  const M low_flux=null_flux(low,b.aspect_ratio);
  // l-major vector spherical harmonics nest exactly. This changes retained
  // boundary resolution while keeping the fitted isolated source operator.
  const M transform=high_flux.topRows(low_flux.rows()).transpose()*low_flux;
  b.fourier_density=(b.fourier_density*transform).eval();
  b.force_torque=(b.force_torque*transform).eval();b.stresslet=(b.stresslet*transform).eval();
  const M H=low.H*low_flux,wh=low.weights.asDiagonal()*H;
  b.degree=degree;b.targets=low.points;b.projection=(H.transpose()*wh).ldlt().solve(wh.transpose());
  b.rigid=b.projection*rigid_values(b.targets,b.aspect_ratio);b.strain=b.projection*strain_values(b.targets,b.aspect_ratio);
}
Eigen::RowVectorXcd cartesian_mode(const sc::Basis& b,int ring,int m,int component) {
  const int order=(b.fourier_modes-1)/2,p=b.projection.rows();
  if(std::abs(m)>order)return Eigen::RowVectorXcd::Zero(p);
  const int start=3*ring*b.fourier_modes;
  if(m==0)return b.fourier_density.row(start+component).cast<std::complex<double>>();
  const int k=std::abs(m);const std::complex<double> I(0.,m>0?-1.:1.);
  return .5*(b.fourier_density.row(start+3*(2*k-1)+component).cast<std::complex<double>>()
          +I*b.fourier_density.row(start+3*(2*k)+component).cast<std::complex<double>>());
}
sc::Matrix actual_self_projection(const sc::Basis& b,int nquad) {
  int nphi=1;while(nphi<b.targets.rows()&&b.targets(nphi,2)==b.targets(0,2))++nphi;
  const int nlat=b.targets.rows()/nphi,p=b.projection.rows(),nr=b.fourier_rings.rows(),maxm=b.degree+1;
  M rhoz(nlat,2);for(int t=0;t<nlat;++t)rhoz.row(t)<<b.targets.row(t*nphi).head<2>().norm(),b.targets(t*nphi,2);
  const auto kernel=SlurryBasisCylindrical::modal_kernel(rhoz,b.fourier_rings,maxm,nquad);
  sc::Matrix response=sc::Matrix::Zero(p,p);const std::complex<double> I(0.,1.);
  for(int m=0;m<=maxm;++m) {
    CM force(3*nr,p);
    for(int r=0;r<nr;++r){
      const auto xm=cartesian_mode(b,r,m-1,0),xp=cartesian_mode(b,r,m+1,0);
      const auto ym=cartesian_mode(b,r,m-1,1),yp=cartesian_mode(b,r,m+1,1);
      const double factor=m==0?1.:2.;
      force.row(3*r)=factor*(.5*(xm+xp)-.5*I*(ym-yp));
      force.row(3*r+1)=factor*(.5*I*(xm-xp)+.5*(ym+yp));
      force.row(3*r+2)=factor*cartesian_mode(b,r,m,2);
    }
    const CM velocity=kernel[m]*force;
    CM projection=CM::Zero(p,3*nlat);
    for(int t=0;t<nlat;++t)for(int j=0;j<nphi;++j){
      const double phi=2*pi*j/nphi,c=std::cos(phi),s=std::sin(phi);const std::complex<double> phase=std::exp(I*double(m)*phi);
      const int q=3*(t*nphi+j);
      projection.col(3*t)+=phase*(c*b.projection.col(q)+s*b.projection.col(q+1)).cast<std::complex<double>>();
      projection.col(3*t+1)+=phase*(-s*b.projection.col(q)+c*b.projection.col(q+1)).cast<std::complex<double>>();
      projection.col(3*t+2)+=phase*b.projection.col(q+2).cast<std::complex<double>>();
    }
    response+=(projection*velocity).real();
  }
  return response;
}
int main(int argc,char**argv) {
  try {
    if(argc<6||argc>8)throw std::runtime_error("usage: ar20_pair_probe basis.ssb h_over_c target_factor source_phi result.json [retained_degree [self_phi]]");
    const auto started=std::chrono::steady_clock::now();sc::Basis b;b.load(argv[1]);
    if(b.aspect_ratio!=20.)throw std::runtime_error("This probe is restricted to recorded AR20 pose");
    const double h=std::stod(argv[2]);const int factor=std::stoi(argv[3]),nphi=std::stoi(argv[4]);
    if(!(std::isfinite(h)&&h>0.))throw std::runtime_error("Positive direct fluid gap required");
    const double source_depth=b.fourier_rings.col(1).cwiseAbs().maxCoeff()/gauss(b.fourier_rings.rows()).x.maxCoeff();
    const int fitted_degree=b.degree;restrict_degree(b,argc>=7?std::stoi(argv[6]):b.degree,factor);change_source_phi(b,nphi);
    const double ti=.0049245303592113559,tj=.041010932014513654,az=1.7016960206944713;
    const std::array<sc::Vec3,2> axes{{sc::Vec3(std::sin(ti),0.,std::cos(ti)),sc::Vec3(std::sin(tj)*std::cos(az),std::sin(tj)*std::sin(az),std::cos(tj))}};
    const auto bodies=sc::geometry(axes,h,20.);const int p=b.projection.rows();
    sc::Matrix A=sc::Matrix::Identity(2*p,2*p);A.block(0,p,p,p)=b.cross(bodies[0],bodies[1]);A.block(p,0,p,p)=b.cross(bodies[1],bodies[0]);
    const int self_phi=argc==8?std::stoi(argv[7]):0;if(self_phi!=0&&(self_phi<64||self_phi>131072))throw std::runtime_error("Self quadrature must be zero or in [64,131072]");double self_error=0.;
    if(self_phi){const sc::Matrix S=actual_self_projection(b,self_phi);self_error=(S-sc::Matrix::Identity(p,p)).norm()/std::sqrt(double(p));A.topLeftCorner(p,p)=S;A.bottomRightCorner(p,p)=S;}
    sc::Matrix rhs=sc::Matrix::Zero(2*p,17),mom=sc::Matrix::Zero(17,2*p);
    for(int i=0;i<2;++i){Eigen::Matrix<double,6,6> Q6=Eigen::Matrix<double,6,6>::Zero();Q6.topLeftCorner<3,3>()=bodies[i].rotation;Q6.bottomRightCorner<3,3>()=bodies[i].rotation;
      const auto Q5=sc::strain_rotation(bodies[i].rotation);rhs.block(i*p,6*i,p,6)=b.rigid*Q6.transpose();rhs.block(i*p,12,p,5)=b.strain*Q5.transpose();
      mom.block(6*i,i*p,6,p)=Q6*b.force_torque;mom.block(12,i*p,5,p)=Q5*b.stresslet;}
    Eigen::PartialPivLU<sc::Matrix> lu(A);const sc::Matrix x=lu.solve(rhs);const sc::Mat17 raw=mom*x,sym=.5*(raw+raw.transpose()).eval();
    const double residual=(A*x-rhs).norm()/rhs.norm(),reciprocity=(raw-raw.transpose()).norm()/raw.norm();
    Eigen::LLT<sc::Mat17> positive(sym);if(!raw.allFinite()||positive.info()!=Eigen::Success||residual>1e-8)throw std::runtime_error("Direct matrix residual/positivity failed");
    const sc::Mat11 reduced=sc::reduced(sym,sc::midpoint_map(bodies,h,20.));
    const M local_exact=exact_isolated(20.);sc::Mat17 self=sc::Mat17::Zero();
    for(int i=0;i<2;++i){sc::Mat11 Q=sc::Mat11::Zero();Q.topLeftCorner<3,3>()=bodies[i].rotation;Q.block<3,3>(3,3)=bodies[i].rotation;Q.bottomRightCorner<5,5>()=sc::strain_rotation(bodies[i].rotation);const sc::Mat11 H=Q*local_exact*Q.transpose();self.block<6,6>(6*i,6*i)=H.topLeftCorner<6,6>();self.block<6,5>(6*i,12)=H.topRightCorner<6,5>();self.block<5,6>(12,6*i)=H.bottomLeftCorner<5,6>();self.bottomRightCorner<5,5>()+=H.bottomRightCorner<5,5>();}
    // Store physical world-Cartesian ring coefficients for independent analytic
    // angular integration of the actual pair solution on off-grid surface points.
    sc::Matrix coefficients(2*b.fourier_density.rows(),17);
    for(int i=0;i<2;++i){sc::Matrix v=b.fourier_density*x.middleRows(i*p,p);for(int j=0;j<v.rows();j+=3)v.middleRows(j,3)=(bodies[i].rotation*v.middleRows(j,3)).eval();coefficients.middleRows(i*v.rows(),v.rows())=v;}
    const std::string coeff_path=std::string(argv[5])+".coefficients.bin";std::ofstream binary(coeff_path,std::ios::binary);binary.write(reinterpret_cast<const char*>(coefficients.data()),coefficients.size()*sizeof(double));binary.close();
    int target_phi=1;while(target_phi<b.targets.rows()&&b.targets(target_phi,2)==b.targets(0,2))++target_phi;
    const auto original=surface(b.aspect_ratio,b.degree,b.targets.rows()/target_phi,target_phi);
    const M gauge=(b.projection*original.H).transpose();
    sc::Matrix harmonics_coefficients(2*gauge.rows(),17);
    for(int i=0;i<2;++i)harmonics_coefficients.middleRows(i*gauge.rows(),gauge.rows())=gauge*x.middleRows(i*p,p);
    const std::string harmonic_path=std::string(argv[5])+".harmonics.bin";std::ofstream harmonic_file(harmonic_path,std::ios::binary);harmonic_file.write(reinterpret_cast<const char*>(harmonics_coefficients.data()),harmonics_coefficients.size()*sizeof(double));harmonic_file.close();
    std::ofstream f(argv[5]);f<<std::setprecision(17)<<"{\n\"schema\":\"ar20_recorded_pose_direct_v1\",\"physical_pair_gate_passed\":false,\"h_over_c\":"<<h<<",\"effective_gap_nm\":"<<150*h<<",\"aspect_ratio\":20,\"degree\":"<<b.degree<<",\"fitted_degree\":"<<fitted_degree<<",\"target_factor\":"<<factor<<",\"target_nodes\":"<<b.targets.rows()<<",\"source_phi\":"<<nphi<<",\"source_rings\":"<<b.fourier_rings.rows()<<",\"source_order\":"<<(b.fourier_modes-1)/2<<",\"source_depth\":"<<source_depth<<",\"theta_i\":"<<ti<<",\"theta_j\":"<<tj<<",\"phi\":"<<az<<",\"linear_residual\":"<<residual<<",\"raw_reciprocity\":"<<reciprocity<<",\"lu_rcond\":"<<lu.rcond()<<",\"positive\":true,\"coefficients_file\":\""<<coeff_path<<"\",\"coefficients_rows\":"<<coefficients.rows()<<",\"coefficients_cols\":17,\"axes\":[";matrix_json(f,axes[0].transpose());f<<',';matrix_json(f,axes[1].transpose());f<<"],\"centers\":[";matrix_json(f,bodies[0].center.transpose());f<<',';matrix_json(f,bodies[1].center.transpose());f<<"],\n\"raw_full17\":";matrix_json(f,raw);f<<",\n\"symmetric_full17\":";matrix_json(f,sym);f<<",\n\"isolated_full17\":";matrix_json(f,self);f<<",\n\"reduced11\":";matrix_json(f,reduced);
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();f<<",\"harmonics_file\":\""<<harmonic_path<<"\",\"harmonics_rows\":"<<harmonics_coefficients.rows()<<",\"self_quadrature\":"<<self_phi<<",\"self_projection_relative_identity_error\":"<<self_error<<",\"wall_seconds\":"<<seconds<<",\"peak_rss_kib\":"<<usage.ru_maxrss<<"}\n";f.close();
    std::cout<<std::setprecision(10)<<"degree="<<b.degree<<" h="<<h<<" target="<<factor<<" reciprocity="<<reciprocity<<" residual="<<residual<<" seconds="<<seconds<<" rss_kib="<<usage.ru_maxrss<<std::endl;
    return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
