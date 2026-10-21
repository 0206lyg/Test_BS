// Diagnostic repair: oversampled no-slip least squares in the SAME fitted
// isolated basis. Only the recorded AR20 pose is supported; no table output.
#define main ar20_existing_probe_main
#include "ar20_pair_probe.cpp"
#undef main
#include <Eigen/QR>

sc::Matrix self_field(const sc::Basis& b,int nquad) {
  int nphi=1;while(nphi<b.targets.rows()&&b.targets(nphi,2)==b.targets(0,2))++nphi;
  const int nlat=b.targets.rows()/nphi,p=b.projection.rows(),nr=b.fourier_rings.rows(),maxm=b.degree+1;
  M rhoz(nlat,2);for(int t=0;t<nlat;++t)rhoz.row(t)<<b.targets.row(t*nphi).head<2>().norm(),b.targets(t*nphi,2);
  const auto kernel=SlurryBasisCylindrical::modal_kernel(rhoz,b.fourier_rings,maxm,nquad);
  sc::Matrix field=sc::Matrix::Zero(3*b.targets.rows(),p);const std::complex<double> I(0.,1.);
  for(int m=0;m<=maxm;++m){CM force(3*nr,p);
    for(int r=0;r<nr;++r){const auto xm=cartesian_mode(b,r,m-1,0),xp=cartesian_mode(b,r,m+1,0),ym=cartesian_mode(b,r,m-1,1),yp=cartesian_mode(b,r,m+1,1);const double factor=m==0?1.:2.;
      force.row(3*r)=factor*(.5*(xm+xp)-.5*I*(ym-yp));force.row(3*r+1)=factor*(.5*I*(xm-xp)+.5*(ym+yp));force.row(3*r+2)=factor*cartesian_mode(b,r,m,2);}
    const CM vel=kernel[m]*force;
    for(int t=0;t<nlat;++t)for(int j=0;j<nphi;++j){const double phi=2*pi*j/nphi,c=std::cos(phi),s=std::sin(phi);const std::complex<double> phase=std::exp(I*double(m)*phi);const int q=3*(t*nphi+j);
      field.row(q)+=(phase*(c*vel.row(3*t)-s*vel.row(3*t+1))).real();field.row(q+1)+=(phase*(s*vel.row(3*t)+c*vel.row(3*t+1))).real();field.row(q+2)+=(phase*vel.row(3*t+2)).real();}
  }return field;
}
sc::Matrix cross_field(const sc::Basis& b,const sc::Body& target,const sc::Body& source) {
 const sc::Mat3 transform=source.rotation.transpose()*target.rotation;const sc::Vec3 shift=source.rotation.transpose()*(target.center-source.center);
 const int nt=b.targets.rows(),nr=b.fourier_rings.rows(),nm=b.fourier_modes;constexpr int block=32;
 sc::Matrix kernel(3*std::min(nt,block),nr*nm*3),ring_kernel(9,b.fourier_nphi),ring_modes(9,nm),field(3*nt,b.fourier_density.cols());
 for(int begin=0;begin<nt;begin+=block){const int count=std::min(block,nt-begin);
  for(int local=0;local<count;++local){const sc::Vec3 x=transform*b.targets.row(begin+local).transpose()+shift;
   for(int r=0;r<nr;++r){for(int q=0;q<b.fourier_nphi;++q){const sc::Vec3 d=x-b.sources.row(r*b.fourier_nphi+q).transpose();const double r2=d.squaredNorm(),z=1./(8*pi*std::sqrt(r2)),w=z/r2;for(int i=0;i<3;++i)for(int j=0;j<3;++j)ring_kernel(3*i+j,q)=w*d[i]*d[j]+(i==j?z:0.);}
    ring_modes.noalias()=ring_kernel*b.fourier_harmonics;for(int m=0;m<nm;++m)for(int i=0;i<3;++i)for(int j=0;j<3;++j)kernel(3*local+i,3*(r*nm+m)+j)=ring_modes(3*i+j,m);}
  }field.middleRows(3*begin,3*count).noalias()=kernel.topRows(3*count)*b.fourier_density;
 }for(int t=0;t<nt;++t)field.middleRows(3*t,3)=(transform.transpose()*field.middleRows(3*t,3)).eval();return field;
}
int main(int argc,char**argv){try{
 if(argc!=8)throw std::runtime_error("usage: ar20_collocation_probe basis h_over_c retained_degree target_factor source_phi self_phi output.json");
 auto started=std::chrono::steady_clock::now();sc::Basis b;b.load(argv[1]);if(b.aspect_ratio!=20.)throw std::runtime_error("AR20 only");
 const int fitted=b.degree,degree=std::stoi(argv[3]),factor=std::stoi(argv[4]),nphi_source=std::stoi(argv[5]),self_phi=std::stoi(argv[6]);const double h=std::stod(argv[2]);if(!(std::isfinite(h)&&h>0.)||self_phi<64||self_phi>131072)throw std::runtime_error("Positive finite gap and self quadrature in [64,131072] required");restrict_degree(b,degree,factor);change_source_phi(b,nphi_source);
 const double ti=.0049245303592113559,tj=.041010932014513654,az=1.7016960206944713;
 const std::array<sc::Vec3,2> axes{{{std::sin(ti),0.,std::cos(ti)},{std::sin(tj)*std::cos(az),std::sin(tj)*std::sin(az),std::cos(tj)}}};const auto bodies=sc::geometry(axes,h,20.);
 const int p=b.projection.rows(),n=3*b.targets.rows();M A(2*n,2*p),rhs=M::Zero(2*n,17);sc::Matrix mom=sc::Matrix::Zero(17,2*p);
 {const sc::Matrix own=self_field(b,self_phi);A.topLeftCorner(n,p)=own;A.bottomRightCorner(n,p)=own;}
 A.topRightCorner(n,p)=cross_field(b,bodies[0],bodies[1]);A.bottomLeftCorner(n,p)=cross_field(b,bodies[1],bodies[0]);
 int nphi=1;while(nphi<b.targets.rows()&&b.targets(nphi,2)==b.targets(0,2))++nphi;const auto gl=gauss(b.targets.rows()/nphi);
 for(int i=0;i<2;++i){Eigen::Matrix<double,6,6> Q6=Eigen::Matrix<double,6,6>::Zero();Q6.topLeftCorner<3,3>()=bodies[i].rotation;Q6.bottomRightCorner<3,3>()=bodies[i].rotation;const auto Q5=sc::strain_rotation(bodies[i].rotation);
  rhs.block(i*n,6*i,n,6)=rigid_values(b.targets,20.)*Q6.transpose();rhs.block(i*n,12,n,5)=strain_values(b.targets,20.)*Q5.transpose();mom.block(6*i,i*p,6,p)=Q6*b.force_torque;mom.block(12,i*p,5,p)=Q5*b.stresslet;
  for(int q=0;q<b.targets.rows();++q){const double mu=b.targets(q,2),weight=std::sqrt(gl.w[q/nphi]*2*pi/nphi*20.*std::sqrt(1.+399.*mu*mu));A.middleRows(i*n+3*q,3)*=weight;rhs.middleRows(i*n+3*q,3)*=weight;}
 }
 std::cerr<<"COLLOCATION rows="<<A.rows()<<" cols="<<A.cols()<<" QR start\n";
 Eigen::ColPivHouseholderQR<M> qr(A);const sc::Matrix x=qr.solve(rhs);const double residual=(A*x-rhs).norm()/rhs.norm();const sc::Mat17 raw=mom*x,sym=.5*(raw+raw.transpose()).eval();const double reciprocity=(raw-raw.transpose()).norm()/raw.norm();Eigen::LLT<sc::Mat17> positive(sym);
 const M local_exact=exact_isolated(20.);sc::Mat17 self=sc::Mat17::Zero();for(int i=0;i<2;++i){sc::Mat11 Q=sc::Mat11::Zero();Q.topLeftCorner<3,3>()=bodies[i].rotation;Q.block<3,3>(3,3)=bodies[i].rotation;Q.bottomRightCorner<5,5>()=sc::strain_rotation(bodies[i].rotation);const sc::Mat11 H=Q*local_exact*Q.transpose();self.block<6,6>(6*i,6*i)=H.topLeftCorner<6,6>();self.block<6,5>(6*i,12)=H.topRightCorner<6,5>();self.block<5,6>(12,6*i)=H.bottomLeftCorner<5,6>();self.bottomRightCorner<5,5>()+=H.bottomRightCorner<5,5>();}
 sc::Matrix coefficients(2*b.fourier_density.rows(),17);for(int i=0;i<2;++i){sc::Matrix v=b.fourier_density*x.middleRows(i*p,p);for(int j=0;j<v.rows();j+=3)v.middleRows(j,3)=(bodies[i].rotation*v.middleRows(j,3)).eval();coefficients.middleRows(i*v.rows(),v.rows())=v;}
 const std::string path=std::string(argv[7])+".coefficients.bin";std::ofstream binary(path,std::ios::binary);if(!binary)throw std::runtime_error("Cannot open coefficient output");binary.write(reinterpret_cast<const char*>(coefficients.data()),coefficients.size()*sizeof(double));binary.flush();if(!binary)throw std::runtime_error("Cannot write coefficient output");binary.close();
 const double depth=b.fourier_rings.col(1).cwiseAbs().maxCoeff()/gauss(b.fourier_rings.rows()).x.maxCoeff();struct rusage usage{};getrusage(RUSAGE_SELF,&usage);const double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
 std::ofstream f(argv[7]);if(!f)throw std::runtime_error("Cannot open JSON output");f<<std::setprecision(17)<<"{\"schema\":\"ar20_recorded_pose_collocation_v1\",\"solver\":\"surface area weighted oversampled QR\",\"physical_pair_gate_passed\":false,\"h_over_c\":"<<h<<",\"effective_gap_nm\":"<<150*h<<",\"aspect_ratio\":20,\"degree\":"<<degree<<",\"fitted_degree\":"<<fitted<<",\"target_factor\":"<<factor<<",\"target_nodes\":"<<b.targets.rows()<<",\"source_phi\":"<<nphi_source<<",\"source_rings\":"<<b.fourier_rings.rows()<<",\"source_order\":"<<(b.fourier_modes-1)/2<<",\"source_depth\":"<<depth<<",\"theta_i\":"<<ti<<",\"theta_j\":"<<tj<<",\"phi\":"<<az<<",\"weighted_boundary_residual\":"<<residual<<",\"raw_reciprocity\":"<<reciprocity<<",\"rank\":"<<qr.rank()<<",\"columns\":"<<A.cols()<<",\"positive\":"<<(positive.info()==Eigen::Success?"true":"false")<<",\"coefficients_file\":\""<<path<<"\",\"coefficients_rows\":"<<coefficients.rows()<<",\"coefficients_cols\":17,\"axes\":[";matrix_json(f,axes[0].transpose());f<<',';matrix_json(f,axes[1].transpose());f<<"],\"centers\":[";matrix_json(f,bodies[0].center.transpose());f<<',';matrix_json(f,bodies[1].center.transpose());f<<"],\"raw_full17\":";matrix_json(f,raw);f<<",\"symmetric_full17\":";matrix_json(f,sym);f<<",\"isolated_full17\":";matrix_json(f,self);f<<",\"reduced11\":";matrix_json(f,sc::reduced(sym,sc::midpoint_map(bodies,h,20.)));f<<",\"self_quadrature\":"<<self_phi<<",\"wall_seconds\":"<<sec<<",\"peak_rss_kib\":"<<usage.ru_maxrss<<"}\n";
 f.flush();if(!f)throw std::runtime_error("Cannot write JSON output");
 std::cout<<"degree="<<degree<<" weightedBC="<<residual<<" reciprocity="<<reciprocity<<" seconds="<<sec<<" rss="<<usage.ru_maxrss<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
