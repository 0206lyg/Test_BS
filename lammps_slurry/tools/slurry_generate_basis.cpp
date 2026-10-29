// Offline Stokes MFS boundary-scattering generator. No pair-reference fitting.
// C++17 + the Eigen headers bundled with this project; no Python runtime.
// Units: polar semi-axis c=1, viscosity eta=1, equatorial semi-axis a=AR.
#include "slurry_scattering_math.h"
#include "slurry_stress_math.h"
#include "basis_cylindrical_kernel.h"
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace {
using M=Eigen::MatrixXd;
using C=Eigen::MatrixXcd;
using V=Eigen::VectorXd;
using Z=std::complex<double>;
constexpr double pi=3.141592653589793238462643383279502884;
struct Settings {
  double ar=10.,depth=.5,rcond=1e-11;
  int degree=12,nsource=96,source_phi=128,source_order=-1,ntarget=-1,target_phi=-1;
  int nfit=-1,fit_phi=4096,ncheck=-1,check_phi=8192;
  int hybrid_version=3;
  std::array<double,3> anchor_gaps{{.5,1.,2.5}};
  std::string output,report,far;
};
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
M meridian(double a,const V& mu) {
  M x(mu.size(),2);for(int i=0;i<mu.size();++i)x.row(i)<<a*std::sqrt(1-mu[i]*mu[i]),mu[i];return x;
}
std::vector<C> boundary_modes(double a,int L,const V& mu,const M& flux_basis) {
  const int maxm=L+1,nphi=2*maxm+3,p0=flux_basis.rows();
  std::vector<C> modes(maxm+1,C::Zero(3*mu.size(),p0));
  for(int i=0;i<mu.size();++i)for(int q=0;q<nphi;++q) {
    const double phi=2*pi*q/nphi,cp=std::cos(phi),sp=std::sin(phi);
    const V Y=harmonics(L,mu[i],phi);
    for(int m=0;m<=maxm;++m) {
      const Z f=std::exp(Z(0,-m*phi))*((m==0?1.:2.)/nphi);
      for(int k=0;k<Y.size();++k) {
        const Z y=f*Y[k];
        modes[m](3*i,3*k)+=cp*y;modes[m](3*i,3*k+1)+=sp*y;
        modes[m](3*i+1,3*k)-=sp*y;modes[m](3*i+1,3*k+1)+=cp*y;
        modes[m](3*i+2,3*k+2)+=y;
      }
    }
  }
  for(auto& x:modes)x=(x*flux_basis.cast<Z>()).eval();
  return modes;
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
void write_matrix(std::ostream& out,const std::string& name,const M& x) {
  out<<name<<' '<<x.rows()<<' '<<x.cols()<<'\n';for(int i=0;i<x.rows();++i){for(int j=0;j<x.cols();++j)out<<x(i,j)<<(j+1==x.cols()?'\n':' ');}
}
struct Generated {M targets,rings,projection,density,rigid,strain,force,stress;};
void write_basis(std::ostream& out,const Settings& s,const Generated& b) {
  out<<"SLURRY_SCATTERING_V2\n"<<s.ar<<' '<<s.degree<<"\nFOURIER "<<s.source_phi<<' '<<s.source_order<<'\n';
  write_matrix(out,"targets",b.targets);write_matrix(out,"rings",b.rings);write_matrix(out,"projection",b.projection);
  write_matrix(out,"fourier_density",b.density);write_matrix(out,"rigid",b.rigid);write_matrix(out,"strain",b.strain);
  write_matrix(out,"force_torque",b.force);write_matrix(out,"stresslet",b.stress);
}
Settings parse(int argc,char**argv) {
  Settings s;
  std::string anchor_values;
  std::map<std::string,int*> integers{{"--hybrid-version",&s.hybrid_version},{"--degree",&s.degree},{"--source-rings",&s.nsource},{"--source-phi",&s.source_phi},{"--source-order",&s.source_order},{"--target-rings",&s.ntarget},{"--target-phi",&s.target_phi},{"--fit-rings",&s.nfit},{"--fit-phi",&s.fit_phi},{"--check-rings",&s.ncheck},{"--check-phi",&s.check_phi}};
  std::map<std::string,double*> reals{{"--ar",&s.ar},{"--source-depth",&s.depth},{"--rcond",&s.rcond}};
  for(int k=1;k<argc;++k){std::string key=argv[k];if(key=="--help"){std::cout<<"slurry_generate_basis --ar A --degree L --source-rings N --source-phi N --source-depth D --target-rings N --target-phi N --fit-rings N --fit-phi N --check-rings N --check-phi N --rcond R --output FILE --report FILE [--far-basis FILE --hybrid-version 3 --anchor-gaps 0.5,1]\n";std::exit(0);}if(k+1>=argc)throw std::runtime_error("Missing value for "+key);std::string v=argv[++k];
    if(integers.count(key))*integers.at(key)=std::stoi(v);else if(reals.count(key))*reals.at(key)=std::stod(v);else if(key=="--output")s.output=v;else if(key=="--report")s.report=v;else if(key=="--far-basis")s.far=v;else if(key=="--anchor-gaps")anchor_values=v;else throw std::runtime_error("Unknown option "+key);
  }
  if(s.hybrid_version<1||s.hybrid_version>3)throw std::runtime_error("Hybrid version must be1,2 or3");
  const int declared_anchors=s.hybrid_version==3?2:3;
  if(!anchor_values.empty()) {
    std::replace(anchor_values.begin(),anchor_values.end(),',',' ');std::istringstream values(anchor_values);
    for(int k=0;k<declared_anchors;++k)if(!(values>>s.anchor_gaps[k]))throw std::runtime_error("Wrong number of comma-separated matching gaps");
    std::string extra;if(values>>extra)throw std::runtime_error("Unexpected extra matching gap");
  }
  for(int k=0;k<declared_anchors;++k)if(!std::isfinite(s.anchor_gaps[k])||!(s.anchor_gaps[k]>0.)||(k&&!(s.anchor_gaps[k]>s.anchor_gaps[k-1])))throw std::runtime_error("Matching gaps must be positive and strictly increasing");
  if(s.hybrid_version==3) {
    if(s.anchor_gaps[0]!=.5||s.anchor_gaps[1]!=1.)throw std::runtime_error("Hybrid V3 requires anchors0.5,1");
    s.anchor_gaps[2]=0.;
  }
  // Isolated fitting and cross-body surface projection are separate quadratures.
  // A grid only just resolving the retained harmonics aliases the neighbouring
  // body's non-polynomial field; oversample that projection independently.
  if(s.source_order<0)s.source_order=s.degree+2;
  if(s.ntarget<0)s.ntarget=3*(s.degree+3);
  if(s.target_phi<0)s.target_phi=3*(2*s.degree+5);
  if(s.nfit<0)s.nfit=std::max(2*s.nsource,4*s.degree+24);if(s.ncheck<0)s.ncheck=s.nfit+7;
  if(!std::isfinite(s.ar)||s.ar<1||s.degree<1||s.degree>30||!(s.depth>0&&s.depth<1)||!(s.rcond>0&&s.rcond<1e-2)||s.output.empty()||s.report.empty())throw std::runtime_error("Invalid generator physical parameters or output/report path");
  if(s.nsource<4||s.nsource>512||s.ntarget>512||s.target_phi>1024||s.nfit>2048||s.ncheck>2048||s.nsource*s.source_phi>100000||static_cast<long long>(3*s.ntarget)*s.target_phi*(3*(s.degree+1)*(s.degree+1)-1)>16000000||s.source_order>32||s.fit_phi>1048576||s.check_phi>1048576||s.ntarget<s.degree+1||s.target_phi<2*s.degree+1||s.nfit<s.nsource||s.ncheck<1||s.source_order<s.degree+2||s.source_phi<=2*s.source_order||s.source_phi>1024||s.fit_phi<4*s.source_order+4||s.check_phi<s.fit_phi||static_cast<long long>(s.nsource)*(2*s.source_order+1)*3*(3*(s.degree+1)*(s.degree+1)-1)>16000000)throw std::runtime_error("Invalid/too large numerical resolution or storage dimensions");
  return s;
}
}
int main(int argc,char**argv) {
 try {
  const auto started=std::chrono::steady_clock::now();const Settings s=parse(argc,argv);const int maxm=s.degree+1,p0=3*(s.degree+1)*(s.degree+1),p=p0-1,nm=2*s.source_order+1;
  std::cout<<std::setprecision(12)<<"GENERATE ar="<<s.ar<<" degree="<<s.degree<<" p="<<p<<" modal_system="<<3*s.nfit<<'x'<<3*s.nsource<<" modes="<<maxm+1<<std::endl;
  const Surface surf=surface(s.ar,s.degree,s.ntarget,s.target_phi);const M F=null_flux(surf,s.ar);const M H=surf.H*F;
  const M wh=surf.weights.asDiagonal()*H;const M gram=H.transpose()*wh;
  Generated b;b.targets=surf.points;b.projection=gram.ldlt().solve(wh.transpose());
  const double projector_error=(b.projection*H-M::Identity(p,p)).norm()/std::sqrt(double(p));
  b.rigid=b.projection*rigid_values(b.targets,s.ar);b.strain=b.projection*strain_values(b.targets,s.ar);
  const GL sg=gauss(s.nsource),fg=gauss(s.nfit),cg=gauss(s.ncheck);
  b.rings.resize(s.nsource,2);const double source_a=std::sqrt(s.ar*s.ar-1+s.depth*s.depth);
  for(int i=0;i<s.nsource;++i)b.rings.row(i)<<source_a*std::sqrt(1-sg.x[i]*sg.x[i]),s.depth*sg.x[i];
  std::cout<<"Building isolated modal kernel (periodic quadrature "<<s.fit_phi<<")"<<std::endl;
  auto K=SlurryBasisCylindrical::modal_kernel(meridian(s.ar,fg.x),b.rings,maxm,s.fit_phi);
  auto U=boundary_modes(s.ar,s.degree,fg.x,F);
  std::vector<C> forces(maxm+1);std::vector<int> ranks(maxm+1);std::vector<double> fits(maxm+1),checks(maxm+1);
  V weights(3*s.nfit);for(int i=0;i<s.nfit;++i)for(int j=0;j<3;++j)weights[3*i+j]=std::sqrt(fg.w[i]*s.ar*std::sqrt(1+(s.ar*s.ar-1)*fg.x[i]*fg.x[i]));
  b.density=M::Zero(s.nsource*nm*3,p);
  double maxfit=0.;
  for(int m=0;m<=maxm;++m) {
    C kw=weights.asDiagonal()*K[m],uw=weights.asDiagonal()*U[m];
    Eigen::CompleteOrthogonalDecomposition<C> qr;qr.setThreshold(s.rcond);qr.compute(kw);forces[m]=qr.solve(uw);ranks[m]=qr.rank();
    fits[m]=(kw*forces[m]-uw).norm()/std::max(uw.norm(),1e-30);maxfit=std::max(maxfit,fits[m]);
    const C D=SlurryBasisCylindrical::cylindrical_to_cartesian(m,s.source_order);
    for(int r=0;r<s.nsource;++r)b.density.middleRows(3*r*nm,3*nm)+=(D*forces[m].middleRows(3*r,3)).real();
    std::cout<<"mode="<<m<<" rank="<<ranks[m]<<" weighted_bc="<<fits[m]<<std::endl;
  }
  // Moments are exact integrals of the stored Cartesian force harmonics.
  b.force=M::Zero(6,p);b.stress=M::Zero(5,p);const auto es=SlurryScattering::strain_basis();
  for(int r=0;r<s.nsource;++r) {
    const M f0=b.density.middleRows(3*r*nm,3),fc=b.density.middleRows(3*(r*nm+1),3),fs=b.density.middleRows(3*(r*nm+2),3);
    b.force.topRows(3)+=f0;
    for(int k=0;k<p;++k){Eigen::Matrix3d moment;moment.col(0)=.5*b.rings(r,0)*fc.col(k);moment.col(1)=.5*b.rings(r,0)*fs.col(k);moment.col(2)=b.rings(r,1)*f0.col(k);
      b.force(3,k)+=(moment(2,1)-moment(1,2))/s.ar;b.force(4,k)+=(moment(0,2)-moment(2,0))/s.ar;b.force(5,k)+=(moment(1,0)-moment(0,1))/s.ar;
      for(int j=0;j<5;++j)b.stress(j,k)-=(es[j].array()*moment.array()).sum()/s.ar;
    }
  }
  M rhs(p,11);rhs<<b.rigid,b.strain;M moments(11,p);moments<<b.force,b.stress;const M grand=moments*rhs,exact=exact_isolated(s.ar);
  const double reciprocal=(grand-grand.transpose()).norm()/grand.norm();
  Eigen::GeneralizedSelfAdjointEigenSolver<M> eig(.5*(grand+grand.transpose()).eval(),exact);
  if(eig.info()!=Eigen::Success)throw std::runtime_error("Isolated generalized eigenproblem failed");
  const double isolated_error=(eig.eigenvalues().array()-1).abs().maxCoeff();
  // Independent meridional nodes and doubled angular kernel quadrature.
  std::cout<<"Checking off-grid boundary (rings "<<s.ncheck<<", quadrature "<<s.check_phi<<")"<<std::endl;
  K.clear();U.clear();
  auto CK=SlurryBasisCylindrical::modal_kernel(meridian(s.ar,cg.x),b.rings,maxm,s.check_phi);
  auto CU=boundary_modes(s.ar,s.degree,cg.x,F);
  double maxcheck=0.,physical_error2=0.,physical_norm2=0.,basis_error2=0.,basis_norm2=0.;
  for(int m=0;m<=maxm;++m){const C error=CK[m]*forces[m]-CU[m];checks[m]=error.norm()/std::max(CU[m].norm(),1e-30);maxcheck=std::max(maxcheck,checks[m]);
    const C ep=error*rhs.cast<Z>(),up=CU[m]*rhs.cast<Z>();
    for(int i=0;i<s.ncheck;++i){const double w=cg.w[i]*s.ar*std::sqrt(1+(s.ar*s.ar-1)*cg.x[i]*cg.x[i])*(m==0?1.:.5);
      physical_error2+=w*ep.middleRows(3*i,3).squaredNorm();physical_norm2+=w*up.middleRows(3*i,3).squaredNorm();basis_error2+=w*error.middleRows(3*i,3).squaredNorm();basis_norm2+=w*CU[m].middleRows(3*i,3).squaredNorm();}
  }
  const double offgrid_physical=std::sqrt(physical_error2/physical_norm2),offgrid_basis=std::sqrt(basis_error2/basis_norm2);
  if(!b.density.allFinite()||!std::isfinite(offgrid_basis)||!(eig.eigenvalues().minCoeff()>0))throw std::runtime_error("Nonfinite density/check or nonpositive isolated resistance");
  std::ostringstream contents;contents<<std::setprecision(17);
  if(!s.far.empty()) {
    SlurryScattering::Basis far;far.load(s.far);if(far.aspect_ratio!=s.ar)throw std::runtime_error("Far-basis aspect ratio mismatch");
    std::ifstream input(s.far);contents<<"SLURRY_HYBRID_V"<<s.hybrid_version<<'\n'<<s.ar<<' '<<s.degree<<'\n';
    if(s.hybrid_version>=2) {
      const int count=s.hybrid_version==3?2:3;contents<<"ANCHORS "<<count;
      for(int k=0;k<count;++k)contents<<' '<<s.anchor_gaps[k];contents<<'\n';
    }
    contents<<input.rdbuf()<<'\n';
  }
  std::ostringstream near_text;near_text<<std::setprecision(17);write_basis(near_text,s,b);
  std::istringstream roundtrip(near_text.str());SlurryScattering::Basis loaded;loaded.load(roundtrip);
  if(loaded.aspect_ratio!=s.ar||loaded.degree!=s.degree||loaded.fourier_nphi!=s.source_phi||
     (loaded.targets-b.targets).norm()!=0.||(loaded.use_fourier&&(loaded.fourier_density-b.density).norm()!=0.)||
     (loaded.projection-b.projection).norm()!=0.||(loaded.rigid-b.rigid).norm()!=0.||
     (loaded.strain-b.strain).norm()!=0.||(loaded.force_torque-b.force).norm()!=0.||
     (loaded.stresslet-b.stress).norm()!=0.)throw std::runtime_error("Serialized basis loader roundtrip failed");
  contents<<near_text.str();
  const std::string temporary=s.output+".tmp";std::ofstream output(temporary);if(!output)throw std::runtime_error("Cannot write basis output");output<<contents.str();output.close();
  if(!output||std::rename(temporary.c_str(),s.output.c_str())!=0)throw std::runtime_error("Cannot atomically publish generated basis");
  const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  std::ofstream report(s.report);if(!report)throw std::runtime_error("Cannot write generation report");report<<std::setprecision(17);
  report<<"{\n  \"format\": \"slurry_stokes_basis_generation_v1\",\n  \"status\": \"candidate_unqualified\",\n  \"method\": \"confocal interior Stokeslets, cylindrical Fourier least squares; no pair data fit\",\n  \"aspect_ratio\": "<<s.ar<<",\n  \"degree\": "<<s.degree<<",\n  \"coefficient_count\": "<<p<<",\n  \"source_rings\": "<<s.nsource<<",\n  \"source_depth\": "<<s.depth<<",\n  \"source_order\": "<<s.source_order<<",\n  \"source_phi_runtime\": "<<s.source_phi<<",\n  \"target_rings\": "<<s.ntarget<<",\n  \"target_phi\": "<<s.target_phi<<",\n  \"fit_rings\": "<<s.nfit<<",\n  \"fit_phi_kernel\": "<<s.fit_phi<<",\n  \"check_rings\": "<<s.ncheck<<",\n  \"check_phi_kernel\": "<<s.check_phi<<",\n  \"rcond\": "<<s.rcond<<",\n  \"projection_identity_relative\": "<<projector_error<<",\n  \"max_modal_fit_relative\": "<<maxfit<<",\n  \"max_modal_offgrid_relative\": "<<maxcheck<<",\n  \"offgrid_basis_weighted_relative\": "<<offgrid_basis<<",\n  \"offgrid_physical11_weighted_relative\": "<<offgrid_physical<<",\n  \"isolated_all11_energy_error\": "<<isolated_error<<",\n  \"isolated_reciprocity\": "<<reciprocal<<",\n  \"isolated_min_relative_eigenvalue\": "<<eig.eigenvalues().minCoeff()<<",\n  \"hybrid_version\": "<<(s.far.empty()?0:s.hybrid_version)<<",\n  \"anchor_gaps\": [";
  if(!s.far.empty()){if(s.hybrid_version==1)report<<"1,2.5";else for(int k=0;k<(s.hybrid_version==3?2:3);++k)report<<(k?",":"")<<s.anchor_gaps[k];}
  report<<"],\n  \"finite_gap_policy\": \""<<(s.far.empty()?"plain_stokes":(s.hybrid_version==3?"direct_near_basis_at_query_gap":"matched_residual"))<<"\",\n  \"serialized_far_slot_active\": "<<(!s.far.empty()&&s.hybrid_version!=3?"true":"false")<<",\n  \"physical_pair_gate_run\": false,\n  \"runtime_source_phi_convergence_checked\": false,\n  \"loader_roundtrip_pass\": true,\n  \"seconds\": "<<elapsed<<",\n  \"modes\": [";
  for(int m=0;m<=maxm;++m)report<<(m?",":"")<<"{\"m\":"<<m<<",\"rank\":"<<ranks[m]<<",\"fit\":"<<fits[m]<<",\"offgrid\":"<<checks[m]<<"}";
  report<<"]\n}\n";
  std::cout<<"GENERATED isolated_all11_error="<<isolated_error<<" reciprocity="<<reciprocal<<" offgrid_basis="<<offgrid_basis<<" offgrid_physical11="<<offgrid_physical<<" seconds="<<elapsed<<" physical_pair_gate=NOT_RUN"<<std::endl;
  return 0;
 } catch(const std::exception& e){std::cerr<<"Basis generation failed: "<<e.what()<<'\n';return 1;}
}
