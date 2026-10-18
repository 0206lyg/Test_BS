// V2 representative full-resolution interpolation check at one frozen 3nm pose.
// Same-source interpolation only: no independent physics or full-table claim.
// Never writes its partially populated in-memory table.
#include "slurry_spd_table.h"
#include "slurry_scattering_provider.h"
#include <Eigen/Eigenvalues>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <omp.h>
#include <vector>
#ifndef SLURRY_NUMERICAL_SOURCE_FINGERPRINT
#define SLURRY_NUMERICAL_SOURCE_FINGERPRINT "unbound"
#endif
namespace sc=SlurryScattering;
namespace st=SlurrySPDTable;
namespace coord=SlurrySPDCoordinates;
using Mat11=sc::Mat11;
using Axes=std::array<sc::Vec3,2>;
Axes axes_at(double ti,double tj,double phi) {
  return {{sc::Vec3(std::sin(ti),0.,std::cos(ti)),
           sc::Vec3(std::sin(tj)*std::cos(phi),std::sin(tj)*std::sin(phi),std::cos(tj))}};
}
template<int N> double energy_error(const Eigen::Matrix<double,N,N>& a,const Eigen::Matrix<double,N,N>& b) {
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double,N,N>> e(a,b);
  if(e.info()!=Eigen::Success)throw std::runtime_error("Generalized energy solve failed");
  return (e.eigenvalues().array()-1.).abs().maxCoeff();
}
Eigen::Matrix<double,5,5> free_strain(const Mat11& h) {
  const auto a=h.topLeftCorner<6,6>().eval();
  Eigen::LLT<Eigen::Matrix<double,6,6>> l(a);
  if(l.info()!=Eigen::Success)throw std::runtime_error("Velocity block not SPD");
  const Eigen::Matrix<double,5,5> result=h.bottomRightCorner<5,5>()-h.bottomLeftCorner<5,6>()*l.solve(h.topRightCorner<6,5>());
  return .5*(result+result.transpose()).eval();
}
int main(int argc,char**argv) {
 try {
  if(argc<2)throw std::runtime_error("Usage: local_integrated_gate matched.ssb [threads=4] [film_grid=161] [source_check_grid=321]");
  const int threads=argc>2?std::stoi(argv[2]):4;
  const int film_grid=argc>3?std::stoi(argv[3]):161;
  const int source_check_grid=argc>4?std::stoi(argv[4]):321;
  if(film_grid<21 || film_grid%2==0)throw std::runtime_error("Film grid must be an odd integer >=21");
  if(source_check_grid<film_grid||source_check_grid%2==0)throw std::runtime_error("Source check grid must be odd and at least film_grid");
  if(threads<1||threads>8)throw std::runtime_error("Thread count outside 1..8");
  std::cout<<std::setprecision(17);
  std::cout<<"COMPILED_SOURCE_FINGERPRINT "<<SLURRY_NUMERICAL_SOURCE_FINGERPRINT<<"\n";
  st::Provider fast;fast.ar=10.;fast.shape=std::make_unique<coord::Shape>(10.);
  const double pi=std::acos(-1.);
  for(int k=0;k<=16;++k){const double psi=pi*k/32.;fast.theta.push_back(std::atan2(std::sin(psi),10.*std::cos(psi)));}
  for(double deg:{2.,5.,10.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.})fast.theta.push_back(deg*pi/180.);
  std::sort(fast.theta.begin(),fast.theta.end());
  fast.theta.erase(std::unique(fast.theta.begin(),fast.theta.end(),[](double a,double b){return std::abs(a-b)<1e-12;}),fast.theta.end());
  for(double t:fast.theta)fast.angular_coordinate.push_back(std::atan2(10.*std::sin(t),std::cos(t)));
  for(int k=0;k<48;++k)fast.phi.push_back(-pi+2*pi*k/48.);
  for(int k=0;k<=25;++k)fast.gap.push_back(.001*std::pow(2.,.5*k));
  fast.format_version=2;fast.joins={.5,1.,2.5};
  for(double h:fast.joins)fast.gap.push_back(h);fast.gap.push_back(5.001);
  std::sort(fast.gap.begin(),fast.gap.end());
  fast.gap.erase(std::unique(fast.gap.begin(),fast.gap.end(),[](double a,double b){return std::abs(a-b)<1e-12*std::max(a,b);}),fast.gap.end());
  for(double h:fast.gap)fast.log_gap.push_back(std::log(h));
  fast.nt=int(fast.theta.size());fast.np=int(fast.phi.size());fast.nh=int(fast.gap.size());
  if(fast.nt!=34||fast.np!=48||fast.nh!=30)throw std::runtime_error("Unexpected trial grid dimensions");
  fast.hmin=fast.gap.front();fast.hmax=fast.gap.back();
  const std::size_t nodes=std::size_t(fast.nt)*fast.nt*fast.np*fast.nh;
  const double ti=.073,tj=.137,phi=.79,actual_h=.020,epsilon=.001,heff=actual_h+epsilon;
  const auto ai=st::stencil(fast.angular_coordinate,std::atan2(10.*std::sin(ti),std::cos(ti)),0,fast.nt-1);
  const auto aj=st::stencil(fast.angular_coordinate,std::atan2(10.*std::sin(tj),std::cos(tj)),0,fast.nt-1);
  const auto ah=fast.gap_stencil(heff);
  for(int k=0;k<ai.size;++k)if(ai.index[k]==0)throw std::runtime_error("First-pose harness unexpectedly touches a pole row; implement pole conditioning before using this grid");
  std::vector<std::unique_ptr<sc::Provider>> direct;
  for(int k=0;k<threads;++k){direct.push_back(std::make_unique<sc::Provider>());direct.back()->load(argv[1]);direct.back()->film_grid=film_grid;
    if(direct.back()->hybrid_version!=2 || direct.back()->anchor_count!=3)throw std::runtime_error("V2 source with three anchors required");
    for(int q=0;q<3;++q)if(std::abs(direct.back()->anchor_gaps[q]-fast.joins[q])>1e-14)throw std::runtime_error("Source joins mismatch");}
  const Axes query_axes=axes_at(ti,tj,phi);
  const Mat11 query_self=fast.shape->self(query_axes,actual_h);
  const Mat11 checked_reference=query_self+direct[0]->correction(query_axes,actual_h,epsilon);
  {
    sc::Provider fine;fine.load(argv[1]);fine.film_grid=source_check_grid;
    const Mat11 higher=query_self+fine.correction(query_axes,actual_h,epsilon);
    Eigen::LLT<Mat11> coarse_factor(checked_reference),fine_factor(higher);
    if(coarse_factor.info()!=Eigen::Success||fine_factor.info()!=Eigen::Success)
      throw std::runtime_error("Direct source at held-out pose is not SPD");
    const double source_error=energy_error<11>(checked_reference,higher);
    const double source_stress_error=energy_error<5>(free_strain(checked_reference),free_strain(higher));
    const bool source_pass=source_error<=.01&&source_stress_error<=.01;
    std::cout<<"QUERY_SOURCE_CONVERGENCE "<<(source_pass?"PASS":"FAIL")
      <<" film_grid="<<film_grid<<" check_grid="<<source_check_grid
      <<" all11_error="<<source_error<<" free_strain_error="<<source_stress_error
      <<" actual_h="<<actual_h<<" epsilon="<<epsilon<<"\n"<<std::flush;
    if(!source_pass){std::cout<<"STENCIL_GENERATION=NOT_RUN EMULATOR_GATE=NOT_RUN\n";return 2;}
  }
  sc::Mat3 mirror=sc::Mat3::Identity();mirror(1,1)=-1.;
  Mat11 reflection=Mat11::Zero();reflection.topLeftCorner<3,3>()=mirror;
  reflection.block<3,3>(3,3)=-mirror;reflection.bottomRightCorner<5,5>()=sc::strain_rotation(mirror);
  {
    const Axes mirrored_axes=axes_at(ti,tj,-phi);
    const Mat11 mirrored_direct=fast.shape->self(mirrored_axes,actual_h)+direct[0]->correction(mirrored_axes,actual_h,epsilon);
    const Mat11 mirrored_from_positive=reflection*checked_reference*reflection.transpose();
    const double symmetry_error=energy_error<11>(mirrored_from_positive,mirrored_direct);
    const double symmetry_strain_error=energy_error<5>(free_strain(mirrored_from_positive),free_strain(mirrored_direct));
    const bool symmetry_pass=symmetry_error<=1e-8&&symmetry_strain_error<=1e-8;
    std::cout<<"SOURCE_REFLECTION_GATE "<<(symmetry_pass?"PASS":"FAIL")<<" all11_error="<<symmetry_error
      <<" free_strain_error="<<symmetry_strain_error<<" limit=1e-8\n"<<std::flush;
    if(!symmetry_pass)return 2;
  }
  fast.data.assign(nodes*66,0.);std::vector<unsigned char> populated(nodes,0);
  std::atomic<bool> failed{false};std::atomic<int> done{0};std::mutex failure_mutex;std::string failure;
  double minimum_source_eigenvalue=1e300;
  const int half_phi=fast.np/2+1;
  const int jobs=ai.size*aj.size*half_phi;
  const auto start=std::chrono::steady_clock::now();
  std::cout<<"IN_MEMORY_STENCIL_ONLY nt="<<fast.nt<<" np="<<fast.np<<" nh="<<fast.nh<<" orientation_jobs="<<jobs<<" film_grid="<<film_grid<<" table_written=0\n"<<std::flush;
  #pragma omp parallel for num_threads(threads) schedule(dynamic,1) reduction(min:minimum_source_eigenvalue)
  for(int job=0;job<jobs;++job) {
    if(failed.load())continue;
    const int ipos=job%half_phi,j=(job/half_phi)%aj.size,i=job/(half_phi*aj.size);
    const int ip=(fast.np/2+ipos)%fast.np;
    const int ii=ai.index[i],jj=aj.index[j];const Axes axes=axes_at(fast.theta[ii],fast.theta[jj],2*pi*ipos/fast.np);
    auto& oracle=*direct[omp_get_thread_num()];oracle.anchor_cache.clear();double active_h=0.;
    try {
      for(int hk=0;hk<ah.size;++hk) {
        const int ih=ah.index[hk];const double h=fast.gap[ih];active_h=h;
        const Mat11 total=fast.shape->self(axes,h)+oracle.correction(axes,h,0.);
        const auto store_node=[&](int target_ip,const Mat11& target_total) {
        const Axes target_axes=axes_at(fast.theta[ii],fast.theta[jj],fast.phi[target_ip]);
        const Mat11 M=fast.shape->contact_map(target_axes,h),Mi=coord::inverse_contact_map(M);
        const Mat11 Pc=Mi.transpose()*fast.shape->preconditioner(target_axes,h,h)*Mi;
        if(!(Pc(2,2)>0.))throw std::runtime_error("Invalid normal conditioning metric");
        Mat11 N=Mi;N.col(2)/=std::sqrt(Pc(2,2));
        const Mat11 raw=N.transpose()*target_total*N,T=.5*(raw+raw.transpose()).eval();
        Eigen::SelfAdjointEigenSolver<Mat11> eig(T);
        if(eig.info()!=Eigen::Success||!(eig.eigenvalues().minCoeff()>0.))throw std::runtime_error("Direct source total is not SPD; no clipping permitted");
        minimum_source_eigenvalue=std::min(minimum_source_eigenvalue,eig.eigenvalues().minCoeff());
        const Mat11 Z=eig.eigenvectors()*eig.eigenvalues().array().log().matrix().asDiagonal()*eig.eigenvectors().transpose();
        const std::size_t node=((std::size_t(ii)*fast.nt+jj)*fast.np+target_ip)*fast.nh+ih;
        int p=0;for(int r=0;r<11;++r)for(int c=0;c<=r;++c)fast.data[node*66+p++]=.5*(Z(r,c)+Z(c,r));
        if(populated[node])throw std::runtime_error("Duplicate reflected lookup node");
        populated[node]=1;
        };
        store_node(ip,total);
        if(ipos>0&&ipos<fast.np/2)store_node(fast.np/2-ipos,reflection*total*reflection.transpose());
      }
      const int finished=++done;if(finished%32==0)std::cout<<"GENERATED_ORIENTATIONS "<<finished<<"/"<<jobs<<"\n"<<std::flush;
    } catch(const std::exception& e) {
      failed=true;std::lock_guard<std::mutex> guard(failure_mutex);
      if(failure.empty())failure="theta_i="+std::to_string(fast.theta[ii])+" theta_j="+std::to_string(fast.theta[jj])+" phi="+std::to_string(fast.phi[ip])+" heff="+std::to_string(active_h)+": "+e.what();
    }
  }
  if(failed)throw std::runtime_error(failure);
  for(int i=0;i<ai.size;++i)for(int j=0;j<aj.size;++j)for(int ip=0;ip<fast.np;++ip)for(int k=0;k<ah.size;++k) {
    const std::size_t node=((std::size_t(ai.index[i])*fast.nt+aj.index[j])*fast.np+ip)*fast.nh+ah.index[k];
    if(!populated[node])throw std::runtime_error("Unpopulated lookup stencil; refusing evaluation");
  }
  const Axes axes=axes_at(ti,tj,phi);const Mat11 self=fast.shape->self(axes,actual_h);
  const Mat11 reference=checked_reference;
  const Mat11 predicted=self+fast.correction(axes,actual_h,epsilon);
  Eigen::LLT<Mat11> check_reference(reference),check_prediction(predicted);
  if(check_reference.info()!=Eigen::Success||check_prediction.info()!=Eigen::Success)throw std::runtime_error("Held-out total resistance not SPD");
  const double error=energy_error<11>(predicted,reference),stress_error=energy_error<5>(free_strain(predicted),free_strain(reference));
  const double reciprocity=(predicted-predicted.transpose()).norm()/predicted.norm();
  const bool pass=error<=.01&&stress_error<=.01&&reciprocity<1e-12;
  const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  std::cout<<"V2_ORIGINAL_3NM_INTERPOLATION_GATE "<<(pass?"PASS":"FAIL")<<" all11_error="<<error<<" free_strain_error="<<stress_error<<" reciprocity="<<reciprocity<<" min_normalized_source_eigenvalue="<<minimum_source_eigenvalue<<" total_seconds="<<seconds<<" film_grid="<<film_grid<<" reflection_source_jobs="<<jobs<<" table_written=0 populated_nodes="<<ai.size*aj.size*fast.np*ah.size<<" full_table_nodes="<<nodes<<" independent_physics=NOT_TESTED other_poses=NOT_RUN\n";
  return pass?0:2;
 }catch(const std::exception&e){std::cerr<<"FAIL_STOP "<<e.what()<<"\n";return 2;}
}

