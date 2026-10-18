#include "slurry_spd_table.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using SlurryScattering::Mat11;
using SlurryScattering::Vec3;
using Axes=std::array<Vec3,2>;
namespace checkpoint {
// Cache-integrity and identity checksums, explicitly FNV-1a64 (not SHA-256).
// The executing binary fingerprints the compiled source/header implementation.
constexpr std::uint64_t initial_hash=14695981039346656037ull;
inline void hash_bytes(std::uint64_t& h,const void* pointer,std::size_t count) {
  const auto* bytes=static_cast<const unsigned char*>(pointer);
  for(std::size_t i=0;i<count;++i){h^=bytes[i];h*=1099511628211ull;}
}
inline std::string hex(std::uint64_t h){std::ostringstream s;s<<std::hex<<std::setfill('0')<<std::setw(16)<<h;return s.str();}
inline std::string hash_file(const std::string& path) {
  std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Cannot fingerprint file: "+path);
  std::uint64_t h=initial_hash;char buffer[65536];
  while(f){f.read(buffer,sizeof(buffer));hash_bytes(h,buffer,std::size_t(f.gcount()));}
  if(!f.eof())throw std::runtime_error("Cannot finish fingerprint: "+path);return hex(h);
}
inline std::string hash_text(const std::string& text){auto h=initial_hash;hash_bytes(h,text.data(),text.size());return hex(h);}
inline std::string json(const std::string& text){
  std::ostringstream s;s<<'"';for(unsigned char c:text){
    if(c=='"'||c=='\\')s<<'\\'<<char(c);
    else if(c<32)s<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
    else s<<char(c);
  }s<<'"';return s.str();
}
inline std::string temporary(const std::string& path){return path+".tmp."+std::to_string(::getpid());}
inline void install(const std::string& temporary_path,const std::string& final_path){
  if(std::rename(temporary_path.c_str(),final_path.c_str()))throw std::runtime_error("Cannot atomically install "+final_path);
}
struct Bounds {double minimum=std::numeric_limits<double>::infinity(),pole_change=0.;};
inline void write(const std::string& path,const std::string& identity,int chunk,
                  const std::vector<std::uint64_t>& rows,std::size_t stride,
                  const std::vector<double>& data,const Bounds& bounds){
  const auto pending=temporary(path);std::ofstream f(pending,std::ios::binary);
  if(!f)throw std::runtime_error("Cannot write checkpoint: "+pending);
  f<<std::setprecision(17)<<"SLURRY_SPD_CHECKPOINT_V1\n"<<identity<<' '<<chunk<<' '<<rows.size()<<' '<<stride<<' '
    <<bounds.minimum<<' '<<bounds.pole_change<<"\nDATA\n";
  auto hash=initial_hash;
  for(auto row:rows){
    const auto* source=data.data()+std::size_t(row)*stride;
    f.write(reinterpret_cast<const char*>(&row),sizeof(row));hash_bytes(hash,&row,sizeof(row));
    f.write(reinterpret_cast<const char*>(source),std::streamsize(stride*sizeof(double)));hash_bytes(hash,source,stride*sizeof(double));
  }
  f.write(reinterpret_cast<const char*>(&hash),sizeof(hash));f.close();
  if(!f)throw std::runtime_error("Incomplete checkpoint write: "+pending);
  install(pending,path);
}
inline Bounds read(const std::string& path,const std::string& identity,int chunk,
                   const std::vector<std::uint64_t>& rows,std::size_t stride,std::vector<double>& data){
  std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Cannot read checkpoint: "+path);
  std::string magic,key,marker;int stored_chunk=-1;std::size_t count=0,stored_stride=0;Bounds bounds;
  f>>magic>>key>>stored_chunk>>count>>stored_stride>>bounds.minimum>>bounds.pole_change>>marker;
  if(!f||magic!="SLURRY_SPD_CHECKPOINT_V1"||key!=identity||stored_chunk!=chunk||count!=rows.size()||stored_stride!=stride||marker!="DATA"||
     !(bounds.minimum>0.)||!std::isfinite(bounds.minimum)||bounds.pole_change<0.||!std::isfinite(bounds.pole_change))
    throw std::runtime_error("Checkpoint identity/header mismatch: "+path);
  f.ignore(std::numeric_limits<std::streamsize>::max(),'\n');auto hash=initial_hash;
  for(auto expected:rows){
    std::uint64_t row=0;f.read(reinterpret_cast<char*>(&row),sizeof(row));
    if(!f||row!=expected)throw std::runtime_error("Checkpoint row mismatch: "+path);
    hash_bytes(hash,&row,sizeof(row));auto* destination=data.data()+std::size_t(row)*stride;
    f.read(reinterpret_cast<char*>(destination),std::streamsize(stride*sizeof(double)));
    if(!f)throw std::runtime_error("Truncated checkpoint: "+path);
    hash_bytes(hash,destination,stride*sizeof(double));
    for(std::size_t k=0;k<stride;++k)if(!std::isfinite(destination[k]))throw std::runtime_error("Nonfinite checkpoint payload: "+path);
  }
  std::uint64_t stored_hash=0;f.read(reinterpret_cast<char*>(&stored_hash),sizeof(stored_hash));
  if(!f||stored_hash!=hash||f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Checkpoint checksum/length mismatch: "+path);
  return bounds;
}
}
int main(int argc,char** argv) {try {
  std::string basis_path,output,checkpoint_dir,source_fingerprint;double ar=10.,epsilon=.001,range=.5;
  bool smoke=false;
  int threads=1,angle_intervals=16,np=48,film_grid=161;
  for(int i=1;i<argc;++i){std::string k=argv[i];if(i+1==argc)throw std::runtime_error("Missing option value");std::string v=argv[++i];
    if(k=="--basis")basis_path=v;else if(k=="--output")output=v;else if(k=="--ar")ar=std::stod(v);
    else if(k=="--epsilon-over-c")epsilon=std::stod(v);else if(k=="--range")range=std::stod(v);
    else if(k=="--threads")threads=std::stoi(v);else if(k=="--angle-intervals")angle_intervals=std::stoi(v);
    else if(k=="--phi-nodes")np=std::stoi(v);else if(k=="--film-grid")film_grid=std::stoi(v);
    else if(k=="--checkpoint-dir")checkpoint_dir=v;else if(k=="--source-fingerprint")source_fingerprint=v;
    else if(k=="--smoke"){if(v!="0"&&v!="1")throw std::runtime_error("--smoke must be 0 or 1");smoke=v=="1";}
    else throw std::runtime_error("Unknown option: "+k);}
  if(basis_path.empty()||output.empty()||!(ar>=1.)||!std::isfinite(ar)||!(epsilon>0)||!std::isfinite(epsilon)||!(range>0)||!std::isfinite(range)||
     threads<1||angle_intervals<4||np<8||np>256||np%2||film_grid<21||film_grid%2==0)
    throw std::runtime_error("Invalid SPD generation options");
  const double pi=std::acos(-1.),requested_max=ar*range+epsilon;
  if(!std::isfinite(requested_max)||!(requested_max>epsilon))throw std::runtime_error("Invalid effective outer gap");
  auto first_evaluator=std::make_unique<SlurryScattering::Provider>();first_evaluator->load(basis_path);
  if(std::abs(first_evaluator->basis.aspect_ratio-ar)>1e-12)throw std::runtime_error("AR mismatch; no substitution");
  const bool explicit_joins=first_evaluator->hybrid_version>=2;
  const bool legacy_outer_join=first_evaluator->hybrid_version<3;
  const std::string table_format=explicit_joins?"SLURRY_SPD_LOGFOURIER_V2":"SLURRY_SPD_LOGFOURIER_V1";
  double hmax=requested_max;
  std::vector<double> theta;
  if(smoke){np=8;theta={0.,std::atan(1./ar),pi/4.,pi/2.};
    // AR=1 remains the separate Mari route, but retain a well-formed smoke grid.
    if(std::abs(theta[1]-theta[2])<1e-12)theta[1]=pi/8.;
  }else{
    for(int k=0;k<=angle_intervals;++k){double psi=.5*pi*k/angle_intervals;theta.push_back(std::atan2(std::sin(psi)/ar,std::cos(psi)));}
    for(double d:{0.,2.,5.,10.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.})theta.push_back(d*pi/180.);
  }
  std::sort(theta.begin(),theta.end());
  theta.erase(std::unique(theta.begin(),theta.end(),[](double a,double b){return std::abs(a-b)<1e-12;}),theta.end());
  theta.front()=0.;theta.back()=.5*pi;const int nt=int(theta.size());std::vector<double> phi(np);
  for(int k=0;k<np;++k)phi[k]=-pi+2*pi*k/np;
  std::vector<double> gaps{epsilon};
  if(smoke){
    for(double h:{.02,.5,2.5})if(h>epsilon&&h<requested_max&&(h!=2.5||legacy_outer_join))gaps.push_back(h);
    gaps.push_back(requested_max);
    if(gaps.size()<4)for(int k=1;k<4;++k)gaps.push_back(epsilon*std::pow(requested_max/epsilon,.25*k));
  }else{
    for(int k=1;;++k){double h=epsilon*std::pow(2.,.5*k);if(!std::isfinite(h)||k>256)throw std::runtime_error("Unsupported gap grid size");gaps.push_back(h);if(h>=requested_max){hmax=h;break;}}
    if(legacy_outer_join&&epsilon<2.5&&hmax>2.5)gaps.push_back(2.5);gaps.push_back(requested_max);
  }
  std::vector<double> joins;
  if(explicit_joins)for(int k=0;k<first_evaluator->anchor_count;++k){
    const double h=first_evaluator->anchor_gaps[k];
    if(h>=epsilon&&h<=hmax){
      joins.push_back(h);
      // Make the source join itself exact even if a geometric grid node is
      // equal within roundoff; do not let de-duplication discard the join.
      for(double& node:gaps)if(std::abs(node-h)<1e-12*std::max(node,h))node=h;
      gaps.push_back(h);
    }
  }
  if(!explicit_joins&&std::find(gaps.begin(),gaps.end(),2.5)!=gaps.end())joins.push_back(2.5);
  std::sort(gaps.begin(),gaps.end());
  gaps.erase(std::unique(gaps.begin(),gaps.end(),[](double a,double b){return std::abs(a-b)<1e-12*std::max(a,b);}),gaps.end());
  const int nh=int(gaps.size());
  if(nt<4||nt>128||nh<4||nh>256)throw std::runtime_error("Generated grid exceeds supported table dimensions");
  const std::size_t payload_count=std::size_t(nt)*nt*np*nh*66;
  if(payload_count>(std::size_t(1)<<29))throw std::runtime_error("Generated table exceeds runtime payload limit");
  const auto basis_hash=checkpoint::hash_file(basis_path);
  const auto executable_hash=checkpoint::hash_file("/proc/self/exe");
  std::ostringstream description;description<<std::setprecision(17)<<table_format<<'|'<<(smoke?"smoke":"full")<<'|'
    <<basis_hash<<'|'<<executable_hash<<'|'<<source_fingerprint<<'|'<<ar<<'|'<<epsilon<<'|'<<range<<'|'<<film_grid<<'|'<<nt<<'|'<<np<<'|'<<nh;
  for(const auto* v:{&theta,&phi,&gaps})for(double x:*v)description<<'|'<<x;
  for(double x:joins)description<<"|join="<<x;
  const auto identity=checkpoint::hash_text(description.str());
  std::filesystem::path checkpoint_root;
  if(!checkpoint_dir.empty()){checkpoint_root=std::filesystem::path(checkpoint_dir)/("spd_"+identity);std::filesystem::create_directories(checkpoint_root);}
#ifndef _OPENMP
  threads=1;
#endif
  std::vector<std::unique_ptr<SlurryScattering::Provider>> evaluators;
  for(int k=0;k<threads;++k){auto p=k==0?std::move(first_evaluator):std::make_unique<SlurryScattering::Provider>();
    if(k)p->load(basis_path);
    if(std::abs(p->basis.aspect_ratio-ar)>1e-12)throw std::runtime_error("AR mismatch; no substitution");
    p->film_grid=film_grid;evaluators.push_back(std::move(p));}
  const SlurrySPDCoordinates::Shape shape(ar);
  std::vector<double> data(payload_count);
  std::atomic<bool> failed{false};std::atomic<int> done{0};std::string failure;
  std::atomic<double> minimum_normalized_eigenvalue{std::numeric_limits<double>::infinity()};
  std::atomic<double> maximum_pole_log_relative_change{0.};
  std::atomic<double> chunk_minimum{std::numeric_limits<double>::infinity()},chunk_pole_change{0.};
  auto record_min=[](std::atomic<double>& value,double x){double old=value.load();while(x<old&&!value.compare_exchange_weak(old,x)){};};
  auto record_max=[](std::atomic<double>& value,double x){double old=value.load();while(x>old&&!value.compare_exchange_weak(old,x)){};};
  auto describe=[](double x){std::ostringstream stream;stream<<std::setprecision(17)<<x;return stream.str();};
  const auto start=std::chrono::steady_clock::now();
  std::cout<<"Building matrix-log/Fourier table profile="<<(smoke?"smoke":"full")<<" physics_ready=0 AR="<<ar<<" theta="<<nt<<" phi="<<np<<" gaps="<<nh<<" threads="<<threads<<" film_grid="<<film_grid<<" bytes="<<data.size()*8<<" identity_fnv1a64="<<identity<<std::endl;
  auto axes_for=[&](int i,int j,double p){return Axes{Vec3(std::sin(theta[i]),0,std::cos(theta[i])),
    Vec3(std::sin(theta[j])*std::cos(p),std::sin(theta[j])*std::sin(p),std::cos(theta[j]))};};
  auto store_node=[&](int i,int j,int ip,int k,const Mat11& total){
    const Axes axes=axes_for(i,j,phi[ip]);const double h=gaps[k];
    const Mat11 symmetric_total=.5*(total+total.transpose()).eval();
    if(!symmetric_total.allFinite())throw std::runtime_error("Non-finite source total: h="+describe(h)+" minEig=nan");
    Eigen::LLT<Mat11> gate(symmetric_total);
    if(gate.info()!=Eigen::Success){Eigen::SelfAdjointEigenSolver<Mat11> s(symmetric_total,Eigen::EigenvaluesOnly);
      const double minimum=s.info()==Eigen::Success?s.eigenvalues().minCoeff():std::numeric_limits<double>::quiet_NaN();
      throw std::runtime_error("Unmodified source total is not SPD: h="+describe(h)+" minEig="+describe(minimum));}
    const Mat11 M=shape.contact_map(axes,h),Mi=SlurrySPDCoordinates::inverse_contact_map(M);
    const Mat11 metric=Mi.transpose()*shape.preconditioner(axes,h,h)*Mi;
    Eigen::Matrix<double,11,1> scale=Eigen::Matrix<double,11,1>::Ones();
    scale[2]=std::sqrt(metric(2,2));
    if(!scale.allFinite()||scale.minCoeff()<=0.)throw std::runtime_error("Invalid conditioning scale");
    const Mat11 N=Mi*scale.cwiseInverse().asDiagonal();
    const Mat11 raw=N.transpose()*symmetric_total*N;
    const Mat11 T=.5*(raw+raw.transpose()).eval();
    if(!T.allFinite())throw std::runtime_error("Non-finite normalized total: h="+describe(h)+" minEig=nan");
    Eigen::SelfAdjointEigenSolver<Mat11> spectrum(T);
    const double minimum=spectrum.info()==Eigen::Success?spectrum.eigenvalues().minCoeff():std::numeric_limits<double>::quiet_NaN();
    if(spectrum.info()!=Eigen::Success||!spectrum.eigenvalues().allFinite()||!(minimum>0.))
      throw std::runtime_error("Normalized source total is not SPD: h="+describe(h)+" minEig="+describe(minimum));
    record_min(minimum_normalized_eigenvalue,minimum);
    record_min(chunk_minimum,minimum);
    Mat11 logarithm=spectrum.eigenvectors()*spectrum.eigenvalues().array().log().matrix().asDiagonal()*spectrum.eigenvectors().transpose();
    logarithm=.5*(logarithm+logarithm.transpose()).eval();
    if(i==0&&j==0){
      // The exact double-pole geometry has no azimuth. Project only this node
      // onto the rotationally invariant logarithm; retain and report the change.
      // Q has angular modes |m|<=2, hence the conjugated matrix has |m|<=4.
      Mat11 average=Mat11::Zero();
      for(int p=0;p<np;++p){const Mat11 Q=SlurrySPDTable::rotation11(SlurrySPDTable::z_rotation(2*pi*p/np));
        average+=Q*logarithm*Q.transpose()/np;}
      average=.5*(average+average.transpose()).eval();
      const double change=(average-logarithm).norm()/std::max(1e-30,logarithm.norm());
      record_max(maximum_pole_log_relative_change,change);record_max(chunk_pole_change,change);logarithm=average;
    }
    if(!logarithm.allFinite())throw std::runtime_error("Non-finite source matrix logarithm: h="+describe(h)+" minEig="+describe(minimum));
    double* out=data.data()+(((std::size_t(i)*nt+j)*np+ip)*nh+k)*66;
    int n=0;for(int r=0;r<11;++r)for(int c=0;c<=r;++c)out[n++]=logarithm(r,c);
  };
  // Evaluate one azimuth half-plane. Reflection changes an axial spin by det(R).
  SlurryScattering::Mat3 mirror=SlurryScattering::Mat3::Identity();mirror(1,1)=-1.;
  Mat11 reflection=Mat11::Zero();reflection.topLeftCorner<3,3>()=mirror;reflection.block<3,3>(3,3)=-mirror;
  reflection.bottomRightCorner<5,5>()=SlurryScattering::strain_rotation(mirror);
  struct SourcePose {int i,j,ipos;};
  std::vector<SourcePose> poses;
  for(int i=0;i<nt;++i)for(int j=0;j<nt;++j)for(int ipos=0;ipos<=np/2;++ipos)
    if((i!=0&&j!=0)||ipos==0)poses.push_back({i,j,ipos});
  constexpr int chunk_poses=128;
  const int chunks=(int(poses.size())+chunk_poses-1)/chunk_poses;
  const std::size_t row_stride=std::size_t(nh)*66;
  auto rows_for=[&](int begin,int end){std::vector<std::uint64_t> rows;
    for(int index=begin;index<end;++index){const auto& p=poses[index];
      const std::uint64_t base=(std::uint64_t(p.i)*nt+p.j)*np;
      if(p.i==0||p.j==0){for(int ip=0;ip<np;++ip)rows.push_back(base+ip);}
      else{rows.push_back(base+(np/2+p.ipos)%np);if(p.ipos>0&&p.ipos<np/2)rows.push_back(base+np/2-p.ipos);}
    }return rows;
  };
  // Ensure source symmetry expansion covers every row exactly once before any
  // costly solve. Missing rows must never be supplied as zero matrix logarithms.
  std::vector<unsigned char> coverage(std::size_t(nt)*nt*np,0);
  for(auto row:rows_for(0,int(poses.size())))if(row>=coverage.size()||coverage[row]++)
    throw std::runtime_error("Duplicate or invalid symmetry-expanded table row");
  for(auto flag:coverage)if(flag!=1)throw std::runtime_error("Incomplete symmetry-expanded table domain");
  int resumed_chunks=0,resumed_poses=0;
  for(int chunk=0;chunk<chunks;++chunk){
    const int begin=chunk*chunk_poses,end=std::min(begin+chunk_poses,int(poses.size()));
    const auto rows=rows_for(begin,end);
    const auto shard=checkpoint_root.empty()?std::filesystem::path{}:checkpoint_root/("chunk_"+std::to_string(chunk)+".bin");
    if(!shard.empty()&&std::filesystem::exists(shard)){
      const auto bounds=checkpoint::read(shard.string(),identity,chunk,rows,row_stride,data);
      record_min(minimum_normalized_eigenvalue,bounds.minimum);record_max(maximum_pole_log_relative_change,bounds.pole_change);
      ++resumed_chunks;resumed_poses+=end-begin;done+=end-begin;
      std::cout<<"CHECKPOINT_RESUMED chunk="<<chunk+1<<'/'<<chunks<<" completed_source_poses="<<done.load()<<std::endl;
      continue;
    }
    chunk_minimum=std::numeric_limits<double>::infinity();chunk_pole_change=0.;
#pragma omp parallel for schedule(dynamic) num_threads(threads)
  for(int index=begin;index<end;++index){
    if(failed.load())continue;
    const auto pose=poses[index];const int i=pose.i,j=pose.j,ipos=pose.ipos;
    int worker=0;
#ifdef _OPENMP
    worker=omp_get_thread_num();
#endif
    const double az=2*pi*ipos/np;const Axes axes=axes_for(i,j,az);
    double current_h=std::numeric_limits<double>::quiet_NaN();
    try{
      auto& p=*evaluators[worker];p.anchor_cache.clear();
      for(int k=0;k<nh;++k){const double h=gaps[k];current_h=h;
        const Mat11 total=shape.self(axes,h)+p.correction(axes,h,0.,0,1);
        if(i==0||j==0){
          for(int ip=0;ip<np;++ip){const auto Q=SlurrySPDTable::rotation11(SlurrySPDTable::z_rotation(i==0?phi[ip]:0.));
            store_node(i,j,ip,k,Q*total*Q.transpose());}
        }else{
          const int positive=(np/2+ipos)%np;store_node(i,j,positive,k,total);
          if(ipos>0&&ipos<np/2)store_node(i,j,np/2-ipos,k,reflection*total*reflection.transpose());
        }
      }
      int count=++done;if(count%128==0){
#pragma omp critical(precompute_output)
        std::cout<<"completed_source_poses="<<count<<" elapsed_s="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<std::endl;
      }
    }catch(const std::exception& e){
#pragma omp critical(precompute_error)
      {if(!failed.exchange(true))failure="theta_i="+describe(theta[i])+" theta_j="+describe(theta[j])+" phi="+describe(az)+" h="+describe(current_h)+": "+e.what();}
    }
  }
    if(failed)throw std::runtime_error("SPD SOURCE GATE FAILED; no table written; completed checkpoints retained: "+failure);
    if(!shard.empty())checkpoint::write(shard.string(),identity,chunk,rows,row_stride,data,{chunk_minimum.load(),chunk_pole_change.load()});
    std::cout<<"CHUNK_COMPLETED chunk="<<chunk+1<<'/'<<chunks<<" completed_source_poses="<<done.load()<<" checkpoint="<<(!shard.empty())<<std::endl;
  }
  if(done.load()!=int(poses.size()))throw std::runtime_error("Not all source orientations completed; refusing final table");
  const std::string temporary=checkpoint::temporary(output);std::ofstream f(temporary,std::ios::binary);if(!f)throw std::runtime_error("Cannot write SPD table");
  f<<std::setprecision(17)<<table_format<<'\n'<<ar<<' '<<epsilon<<' '<<hmax<<' '<<nt<<' '<<np<<' '<<nh<<'\n';
  if(explicit_joins){f<<joins.size();for(double h:joins)f<<' '<<h;f<<'\n';}
  for(const auto* v:{&theta,&phi,&gaps}){for(double x:*v)f<<x<<' ';f<<'\n';}f<<"DATA\n";
  f.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()*sizeof(double)));f.close();
  if(!f)throw std::runtime_error("Incomplete SPD table write");
  checkpoint::install(temporary,output);
  const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  // This permanent provenance identifies a computational profile. Neither full
  // coverage nor node SPD alone certifies interpolation/physical/bulk accuracy.
  const auto metadata_path=output+".json",metadata_temporary=checkpoint::temporary(metadata_path);
  std::ofstream metadata(metadata_temporary);
  if(!metadata)throw std::runtime_error("Cannot write generator provenance");
  metadata<<std::setprecision(17)<<"{\n  \"schema_version\": 1,\n  \"profile\": "<<checkpoint::json(smoke?"smoke":"full")
    <<",\n  \"physics_ready\": false,\n  \"scope\": "<<checkpoint::json(smoke?"Coarse global table for executable plumbing only; no quantitative rheology claim":"Full selected grid coverage; separate source, interpolation and bulk gates required")
    <<",\n  \"source_gate\": \"Direct-source internal checks and finite/SPD conditioned node matrices; no independent physical-accuracy certification\""
    <<",\n  \"table_format\": "<<checkpoint::json(table_format)<<",\n  \"matching_joins\": [";
  for(std::size_t k=0;k<joins.size();++k)metadata<<(k?", ":"")<<joins[k];
  metadata<<"],\n  \"ar\": "<<ar<<",\n  \"epsilon_over_c\": "<<epsilon<<",\n  \"outer_range_over_a\": "<<range
    <<",\n  \"h_min\": "<<epsilon<<",\n  \"h_max\": "<<hmax<<",\n  \"n_theta\": "<<nt<<",\n  \"n_phi\": "<<np<<",\n  \"n_gap\": "<<nh
    <<",\n  \"film_grid\": "<<film_grid<<",\n  \"threads\": "<<threads<<",\n  \"source_basis\": "<<checkpoint::json(basis_path)
    <<",\n  \"basis_fnv1a64\": "<<checkpoint::json(basis_hash)<<",\n  \"executable_fnv1a64\": "<<checkpoint::json(executable_hash)
    <<",\n  \"provided_source_fingerprint\": "<<checkpoint::json(source_fingerprint)<<",\n  \"cache_identity_fnv1a64\": "<<checkpoint::json(identity)
    <<",\n  \"payload_bytes\": "<<data.size()*sizeof(double)<<",\n  \"source_poses\": "<<poses.size()<<",\n  \"chunk_source_poses\": "<<chunk_poses
    <<",\n  \"chunks\": "<<chunks<<",\n  \"resumed_chunks\": "<<resumed_chunks<<",\n  \"resumed_source_poses\": "<<resumed_poses
    <<",\n  \"minimum_normalized_eigenvalue\": "<<minimum_normalized_eigenvalue.load()<<",\n  \"maximum_double_pole_log_relative_change\": "<<maximum_pole_log_relative_change.load()
    <<",\n  \"elapsed_seconds\": "<<seconds<<"\n}\n";
  metadata.close();if(!metadata)throw std::runtime_error("Incomplete generator provenance write");checkpoint::install(metadata_temporary,metadata_path);
  std::cout<<std::setprecision(17)<<"minimum_normalized_eigenvalue="<<minimum_normalized_eigenvalue.load()
    <<" maximum_double_pole_log_relative_change="<<maximum_pole_log_relative_change.load()<<std::endl;
  std::cout<<"TABLE_WRITTEN "<<output<<" profile="<<(smoke?"smoke":"full")<<" physics_ready=0 seconds="<<seconds<<std::endl;
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}}
