// Query a serialized table through the runtime Provider. This adapter never
// calls the imported-source surrogate or solves a Stokes boundary problem.
//
// Input (radians, physical nm in the last column):
// QUERY_REFERENCE_TABLE17_V1
// CONVENTION slurry_full17_c_eta_1_v1
// UNITS c_m floor_m
// AR aspect_ratio
// QUERIES count
// id theta_i theta_j phi gap_nm
// ...
// END
#include "slurry_spd_table17.h"
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#ifndef SLURRY_NUMERICAL_SOURCE_FINGERPRINT
#define SLURRY_NUMERICAL_SOURCE_FINGERPRINT "unbound_standalone_build"
#endif

namespace table=SlurrySPDTable17;
namespace sc=SlurryScattering;
using table::Axes;using table::Mat17;using table::Mat3;using table::Vec3;
namespace {
constexpr const char* convention="slurry_full17_c_eta_1_v1";
struct Query {std::string id;double theta_i=0.,theta_j=0.,phi=0.,gap_nm=0.;};
struct Batch {double ar=0.,c_m=0.,floor_m=0.;std::vector<Query> rows;};
struct Response {double h=0.,heff=0.;Mat17 total,correction,isolated;};
void require(bool pass,const std::string& message){if(!pass)throw std::runtime_error(message);}
bool close_scalar(double a,double b){return std::abs(a-b)<=1e-10*std::max(std::abs(a),std::abs(b));}
void marker(std::istream& in,const char* expected){
  std::string token;in>>token;require(in&&token==expected,std::string("Expected ")+expected);
}
std::string json(const std::string& value){
  std::ostringstream out;out<<'"';
  for(unsigned char c:value){
    if(c=='"'||c=='\\')out<<'\\'<<char(c);
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
    else out<<char(c);
  }
  out<<'"';return out.str();
}
std::string file_fnv1a64(const std::string& path){
  std::ifstream in(path,std::ios::binary);require(bool(in),"Cannot hash "+path);
  std::array<char,65536> buffer;std::uint64_t hash=14695981039346656037ull;
  while(in){in.read(buffer.data(),buffer.size());for(std::streamsize i=0;i<in.gcount();++i){
    hash^=static_cast<unsigned char>(buffer[std::size_t(i)]);hash*=1099511628211ull;}}
  require(in.eof(),"Incomplete hash read: "+path);
  std::ostringstream out;out<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;return out.str();
}
template<class Matrix> void matrix_json(std::ostream& out,const Matrix& matrix){
  require(matrix.allFinite(),"Nonfinite query output matrix");out<<'[';
  for(int i=0;i<matrix.rows();++i){if(i)out<<',';out<<'[';
    for(int j=0;j<matrix.cols();++j){if(j)out<<',';out<<matrix(i,j);}out<<']';}out<<']';
}
Axes axes_for(const Query& q){
  return {Vec3(std::sin(q.theta_i),0.,std::cos(q.theta_i)),
    Vec3(std::sin(q.theta_j)*std::cos(q.phi),std::sin(q.theta_j)*std::sin(q.phi),std::cos(q.theta_j))};
}
Batch read_batch(std::istream& in){
  Batch batch;marker(in,"QUERY_REFERENCE_TABLE17_V1");marker(in,"CONVENTION");
  std::string id;in>>id;require(id==convention,"Unsupported full17 work-coordinate convention");
  marker(in,"UNITS");in>>batch.c_m>>batch.floor_m;
  marker(in,"AR");in>>batch.ar;marker(in,"QUERIES");long long count=0;in>>count;
  require(in&&std::isfinite(batch.ar)&&batch.ar>=1.&&std::isfinite(batch.c_m)&&batch.c_m>0.&&
    std::isfinite(batch.floor_m)&&batch.floor_m>0.&&std::isfinite(batch.ar*batch.c_m),
    "Expected finite AR >= 1 and positive physical lengths");
  require(count>0&&count<=100000,"Query count must be in [1,100000]");
  batch.rows.resize(std::size_t(count));std::set<std::string> ids;const double half_pi=std::acos(-1.)/2.;
  for(auto& q:batch.rows){
    in>>q.id>>q.theta_i>>q.theta_j>>q.phi>>q.gap_nm;
    const bool valid_id=!q.id.empty()&&q.id.size()<=128&&std::all_of(q.id.begin(),q.id.end(),[](char c){
      return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'||c=='.'||c==':';});
    require(in&&valid_id&&ids.insert(q.id).second,"Invalid or duplicate stable query id");
    require(std::isfinite(q.theta_i)&&std::isfinite(q.theta_j)&&std::isfinite(q.phi)&&std::isfinite(q.gap_nm)&&
      q.theta_i>=0.&&q.theta_i<=half_pi&&q.theta_j>=0.&&q.theta_j<=half_pi,
      "Expected finite gap/azimuth and canonical polar angles in [0,pi/2]");
  }
  marker(in,"END");std::string extra;require(!(in>>extra),"Extra tokens after query END");return batch;
}
void validate_units(const table::Provider& provider,const Batch& batch){
  require(close_scalar(provider.ar,batch.ar),"Query/table aspect ratio mismatch");
  require(close_scalar(batch.floor_m/batch.c_m,provider.hmin),"Physical floor/c_m differs from table minimum gap");
  if(provider.sparse_reference){
    require(close_scalar(batch.c_m,provider.reference_length_m)&&close_scalar(batch.floor_m,provider.reference_floor_m),
      "Query/sparse table physical size or floor mismatch");
  }
}
Response evaluate(const table::Provider& provider,const Batch& batch,const Axes& axes,double gap_nm){
  Response r;r.h=(gap_nm*1e-9)/batch.c_m;r.heff=std::max(r.h,batch.floor_m/batch.c_m);
  require(std::isfinite(r.h)&&std::isfinite(r.heff),"Nonfinite normalized query gap");
  // correction17 is the serialized runtime path, including angular/gap
  // interpolation, canonical-frame transport, and physical hard-floor handling.
  r.correction=provider.correction17(axes,r.h,batch.floor_m/batch.c_m);
  r.isolated=provider.shape->self(axes,r.heff);
  r.total=r.isolated+r.correction;
  require(r.total.allFinite()&&r.correction.allFinite()&&r.isolated.allFinite(),"Nonfinite restored full17 response");
  return r;
}
bool stored_angle(const table::Provider& provider,const Query& query){
  if(provider.sparse_reference)return true;
  auto near_node=[](const std::vector<double>& nodes,double x){return std::any_of(nodes.begin(),nodes.end(),
    [&](double node){return std::abs(node-x)<=1e-10;});};
  // Match the same polar fallback used by Provider::correction17.
  const Axes axes=axes_for(query);double az=query.phi;
  if(axes[0].head<2>().norm()<=1e-13)az=0.;
  if(axes[1].head<2>().norm()<=1e-13)az=0.;
  const double two_pi=2.*std::acos(-1.);
  const bool az_node=std::any_of(provider.phi.begin(),provider.phi.end(),
    [&](double node){return std::abs(std::remainder(node-az,two_pi))<=1e-10;});
  return near_node(provider.theta,query.theta_i)&&near_node(provider.theta,query.theta_j)&&az_node;
}
void write_batch(const std::string& table_path,const std::string& query_path,const std::string& output_path){
  const auto output=std::filesystem::absolute(output_path).lexically_normal();
  for(const auto& input:{table_path,query_path}){
    const auto normalized=std::filesystem::absolute(input).lexically_normal();
    require(output!=normalized&&(!std::filesystem::exists(output)||!std::filesystem::equivalent(output,normalized)),
      "Output must not overwrite the table or query input");
  }
  std::ifstream input(query_path);require(bool(input),"Cannot open query batch: "+query_path);
  const Batch batch=read_batch(input);table::Provider provider;provider.load(table_path);validate_units(provider,batch);
  const std::string table_hash=file_fnv1a64(table_path),query_hash=file_fnv1a64(query_path);
  const std::string temporary=output_path+".tmp";
  require(!std::filesystem::exists(temporary),"Temporary output already exists: "+temporary);
  try{
    std::ofstream out(temporary);require(bool(out),"Cannot open query output: "+temporary);out<<std::setprecision(17);
    out<<"{\n\"schema\":\"reference17_query_v1\",\n\"convention\":"<<json(convention)
      <<",\n\"operator\":\"symmetric_full17_total\",\n\"independent_reference\":false,"
      <<"\n\"physical_accuracy_qualified\":false,\n\"contains_new_stokes_solve\":false,"
      <<"\n\"units\":\"c=eta=1; input [Ui-uinf_i,a(Omegai-Omegainf),Uj-uinf_j,a(Omegaj-Omegainf),aE5]; output [Fsource_i,Tsource_i/a,Fsource_j,Tsource_j/a,S_intrinsic_sum/a]\","
      <<"\n\"frame\":\"Cartesian contact-normal +z; first director in xz plane; body centers from exact support geometry\","
      <<"\n\"ar\":"<<batch.ar<<",\n\"c_m\":"<<batch.c_m<<",\n\"floor_m\":"<<batch.floor_m
      <<",\n\"model_revision\":"<<json(provider.loaded_model_revision)
      <<",\n\"layout\":"<<json(provider.sparse_reference?"sparse_exact_poses":"full_grid")
      <<",\n\"reference_scope\":"<<json(provider.reference_scope)
      <<",\n\"source_identity\":"<<json(provider.source_identity)
      <<",\n\"source_binding\":"<<json(provider.sparse_reference?"table_header_fnv1a64":"external_pipeline_manifest_required")
      <<",\n\"physical_units_binding\":"<<json(provider.sparse_reference?"validated_table_header":"query_input_ratio_checked_external_manifest_required")
      <<",\n\"table_fnv1a64\":"<<json(table_hash)<<",\n\"query_input_fnv1a64\":"<<json(query_hash)
      <<",\n\"numerical_source_fingerprint\":"<<json(SLURRY_NUMERICAL_SOURCE_FINGERPRINT)
      <<",\n\"compiled_source_binding\":"<<json(std::string(SLURRY_NUMERICAL_SOURCE_FINGERPRINT)=="unbound_standalone_build"?"unbound":"build_manifest")
      <<",\n\"hmin_over_c\":"<<provider.hmin<<",\n\"hmax_over_c\":"<<provider.hmax<<",\n\"rows\":[\n";
    for(std::size_t index=0;index<batch.rows.size();++index){
      const Query& q=batch.rows[index];const Axes axes=axes_for(q);const Response r=evaluate(provider,batch,axes,q.gap_nm);
      const auto bodies=sc::geometry(axes,r.h,batch.ar);
      Eigen::Matrix<double,2,3> directors,centers;
      for(int b=0;b<2;++b){directors.row(b)=axes[b].transpose();centers.row(b)=bodies[b].center.transpose();}
      if(index)out<<",\n";
      out<<"{\"id\":"<<json(q.id)<<",\"theta_i\":"<<q.theta_i<<",\"theta_j\":"<<q.theta_j<<",\"phi\":"<<q.phi
        <<",\"gap_nm\":"<<q.gap_nm<<",\"h_over_c\":"<<r.h<<",\"effective_h_over_c\":"<<r.heff
        <<",\"actual_gap_m\":"<<q.gap_nm*1e-9<<",\"effective_gap_m\":"<<r.heff*batch.c_m
        <<",\"query_mode\":"<<json(stored_angle(provider,q)?"stored_pose":"interpolated_angle")
        <<",\"physical_accuracy_qualified\":false,\"surrogate_available\":false,"
        <<"\"surrogate_unavailable_reason\":\"This adapter evaluates the serialized runtime table only\",\"axes\":";
      matrix_json(out,directors);out<<",\"centers_over_c\":";matrix_json(out,centers);
      out<<",\"total_full17\":";matrix_json(out,r.total);
      out<<",\"correction_full17\":";matrix_json(out,r.correction);
      out<<",\"isolated_full17\":";matrix_json(out,r.isolated);out<<'}';
    }
    out<<"\n]}\n";out.close();require(bool(out),"Incomplete query output write");
    std::filesystem::rename(temporary,output_path);
  }catch(...){std::error_code ignored;std::filesystem::remove(temporary,ignored);throw;}
}

// Small synthetic contract check. The prescribed conditioned logarithm is a
// rank-one matrix with a closed-form exponential, so expected TOTALs do not
// call the Provider interpolation or matrix-exponential implementation.
Eigen::Matrix<double,17,1> test_direction(){
  Eigen::Matrix<double,17,1> v;for(int k=0;k<17;++k)v[k]=k+1.;return v.normalized();
}
double test_log_amplitude(double ar,double ti,double tj,double phi,double h){
  return .02+.01*std::atan2(ar*std::sin(ti),std::cos(ti))+
    .013*std::atan2(ar*std::sin(tj),std::cos(tj))+.007*std::cos(phi)+.004*std::sin(phi)+.002*std::log(h);
}
Mat17 test_expected(const table::Provider& provider,const Query& q,double h){
  const auto v=test_direction();const double s=test_log_amplitude(provider.ar,q.theta_i,q.theta_j,q.phi,h);
  const Mat17 L=provider.shape->cholesky(axes_for(q),h);
  return L*(Mat17::Identity()+std::expm1(s)*(v*v.transpose()))*L.transpose();
}
void test_fixture(const std::filesystem::path& path,double ar,int nh=4){
  const double pi=std::acos(-1.);const std::array<double,4> theta{0.,.21,.64,pi/2.};
  const std::array<double,4> phi{-pi,-pi/2.,0.,pi/2.};std::vector<double> gap(std::size_t(nh),0.);
  for(int k=0;k<nh;++k)gap[k]=std::exp(std::log(1./150.)*(1.-double(k)/(nh-1)));
  gap.front()=1./150.;gap.back()=1.;
  std::ofstream out(path,std::ios::binary);out<<std::setprecision(17)<<table::table_magic
    <<"\nMODEL "<<table::dissipative_model_revision<<'\n'<<ar<<' '<<gap.front()<<" 1 4 4 "<<nh<<"\n0\n";
  for(const auto* nodes:{&theta,&phi}){for(double value:*nodes)out<<value<<' ';out<<'\n';}
  for(double value:gap)out<<value<<' ';out<<"\nDATA\n";
  const auto v=test_direction();
  for(double ti:theta)for(double tj:theta)for(double az:phi)for(double h:gap){
    const Mat17 log=test_log_amplitude(ar,ti,tj,az,h)*(v*v.transpose());
    std::array<double,153> lower;int index=0;
    for(int i=0;i<17;++i)for(int j=0;j<=i;++j)lower[index++]=log(i,j);
    out.write(reinterpret_cast<const char*>(lower.data()),std::streamsize(lower.size()*sizeof(double)));}
  require(bool(out),"Cannot write synthetic query fixture");
}
template<class Function> void expect_rejection(Function operation,const char* message){
  bool rejected=false;try{operation();}catch(const std::exception&){rejected=true;}require(rejected,message);
}
void test_close(const Mat17& actual,const Mat17& expected,const char* message){
  require((actual-expected).norm()/std::max(expected.norm(),1e-300)<5e-11,message);
}
void self_test(const std::string& directory){
  std::filesystem::create_directories(directory);const auto path=std::filesystem::path(directory)/"synthetic_query.sft";
  for(double ar:{2.,5.,10.,20.}){
    test_fixture(path,ar);table::Provider provider;provider.load(path.string());Batch batch;batch.ar=ar;batch.c_m=150e-9;batch.floor_m=1e-9;
    validate_units(provider,batch);const Query q{"heldout",.37,.55,.41,15.};const Axes axes=axes_for(q);
    for(double gap_nm:{.5,15.,150.}){
      const auto actual=evaluate(provider,batch,axes,gap_nm);const Mat17 expected=test_expected(provider,q,actual.heff);
      test_close(actual.total,expected,"Off-grid TOTAL differs from closed-form matrix oracle");
      test_close(actual.correction+actual.isolated,expected,"TOTAL reconstruction lost isolated response");
      const Mat3 R=table::z_rotation(.73);const Mat17 Q=table::rotation17(R);
      for(int signs=0;signs<4;++signs){Axes rotated;
        for(int b=0;b<2;++b)rotated[b]=(signs&(1<<b)?-1.:1.)*R*axes[b];
        test_close(evaluate(provider,batch,rotated,gap_nm).total,Q*expected*Q.transpose(),"Cartesian force/torque/strain frame transport failed");}
    }
    test_close(evaluate(provider,batch,axes,-.5).total,evaluate(provider,batch,axes,1.).total,"Hard floor changed TOTAL");
    expect_rejection([&]{evaluate(provider,batch,axes,151.);},"Above-anchor extrapolation was accepted");
    Batch wrong=batch;wrong.ar+=1.;expect_rejection([&]{validate_units(provider,wrong);},"Wrong AR was accepted");
    wrong=batch;wrong.floor_m*=2.;expect_rejection([&]{validate_units(provider,wrong);},"Wrong floor was accepted");
  }
  test_fixture(path,2.,257);table::Provider refined;refined.load(path.string());
  require(refined.nh==257,"Full-grid loader rejected more than256 gap nodes");
  Batch refined_batch;refined_batch.ar=2.;refined_batch.c_m=150e-9;refined_batch.floor_m=1e-9;
  const Query refined_query{"refined",.37,.55,.41,15.};
  test_close(evaluate(refined,refined_batch,axes_for(refined_query),15.).total,test_expected(refined,refined_query,.1),
    "Refined full-grid lookup changed the analytic matrix");
  std::istringstream valid("QUERY_REFERENCE_TABLE17_V1 CONVENTION slurry_full17_c_eta_1_v1 UNITS 1.5e-7 1e-9 AR 20 QUERIES 1 h1 .3 .4 .5 150 END");
  require(read_batch(valid).rows[0].id=="h1","Stable query ID lost");
  expect_rejection([]{std::istringstream in("QUERY_REFERENCE_TABLE17_V1 CONVENTION bad UNITS 1 1 AR 20 QUERIES 1 h1 .3 .4 .5 150 END");read_batch(in);},"Wrong convention accepted");
  expect_rejection([]{std::istringstream in("QUERY_REFERENCE_TABLE17_V1 CONVENTION slurry_full17_c_eta_1_v1 UNITS 1.5e-7 1e-9 AR 20 QUERIES 2 h1 .3 .4 .5 150 h1 .3 .4 .5 50 END");read_batch(in);},"Duplicate query IDs accepted");
  std::cout<<"PASS synthetic AR2/5/10/20 serialized-table TOTAL queries, heldout interpolation, Cartesian F/T/S covariance, hard floor and input guards; physical_accuracy_qualified=false\n";
}
}
int main(int argc,char** argv){
  try{
    std::string table_path,query_path,output_path,self_test_path;
    for(int k=1;k<argc;++k){const std::string option=argv[k];
      if(option=="--table"&&k+1<argc)table_path=argv[++k];
      else if(option=="--queries"&&k+1<argc)query_path=argv[++k];
      else if(option=="--output"&&k+1<argc)output_path=argv[++k];
      else if(option=="--self-test"&&k+1<argc)self_test_path=argv[++k];
      else if(option=="--help"){
        std::cout<<"query_reference_table --table FILE --queries QUERY_REFERENCE_TABLE17_V1.txt --output JSON\n"
          <<"query_reference_table --self-test TEMP_DIRECTORY\n";return 0;
      }else throw std::runtime_error("Unknown or incomplete option: "+option);
    }
    if(!self_test_path.empty()){
      require(table_path.empty()&&query_path.empty()&&output_path.empty(),"--self-test cannot be combined with query options");self_test(self_test_path);
    }else{
      require(!table_path.empty()&&!query_path.empty()&&!output_path.empty(),"Required: --table FILE --queries FILE --output FILE");
      write_batch(table_path,query_path,output_path);
    }
    return 0;
  }catch(const std::exception& error){std::cerr<<"Reference table query: "<<error.what()<<'\n';return 1;}
}
