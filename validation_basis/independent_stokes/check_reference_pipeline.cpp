// Actual imported reference rows -> generated table -> Provider::make_pair
// -> gp::evaluate. These are algebra/serialization/runtime tests, not a new BVP.
#include "slurry_reference_source17.h"
#include "slurry_spd_table17.h"
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <functional>
#include <unistd.h>

#ifndef SLURRY_NUMERICAL_SOURCE_FINGERPRINT
#define SLURRY_NUMERICAL_SOURCE_FINGERPRINT "unbound_standalone_build"
#endif

namespace rs=SlurryReferenceSource17;
namespace sc=SlurryScattering;
namespace gp=SlurryGrandPair;
namespace gr=SlurryFiniteGap::gr;
using V17=Eigen::Matrix<double,17,1>;
using sc::Mat17; using sc::Mat3; using sc::Vec3;

static std::string bytes(const std::string& path) {
  std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot open "+path);
  std::ostringstream out;out<<in.rdbuf();return out.str();
}
static std::string file_fnv1a64(const std::string& path) {
  std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot open "+path);
  std::array<char,65536> buffer;std::uint64_t hash=14695981039346656037ull;
  while(in){in.read(buffer.data(),buffer.size());for(std::streamsize i=0;i<in.gcount();++i){hash^=static_cast<unsigned char>(buffer[std::size_t(i)]);hash*=1099511628211ull;}}
  if(!in.eof())throw std::runtime_error("Cannot hash complete table");
  std::ostringstream out;out<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;return out.str();
}
static void require(bool value,const std::string& message) {
  if(!value)throw std::runtime_error(message);
}
static double relative(const Eigen::MatrixXd& a,const Eigen::MatrixXd& b) {
  return (a-b).norm()/std::max(b.norm(),1e-300);
}
static double energy_error(const Mat17& actual,const Mat17& expected) {
  Eigen::GeneralizedSelfAdjointEigenSolver<Mat17> solver(actual,expected);
  require(solver.info()==Eigen::Success,"Generalized energy eigensolve failed");
  return (solver.eigenvalues().array()-1.).abs().maxCoeff();
}
static double minimum(const Mat17& matrix) {
  Eigen::SelfAdjointEigenSolver<Mat17> solver(matrix,Eigen::EigenvaluesOnly);
  require(solver.info()==Eigen::Success,"Minimum eigenvalue solve failed");
  return solver.eigenvalues().minCoeff();
}
static bool rejects(const std::function<void()>& operation) {
  try{operation();return false;}catch(const std::exception&){return true;}
}
static gr::Vec3 array(const Vec3& v){return {v[0],v[1],v[2]};}
static Vec3 vector(const gp::Vec3& v){return {v[0],v[1],v[2]};}

struct Geometry {
  std::array<gr::Body,2> body;
  gr::GapResult gap;
  std::array<Vec3,2> centers;
};
static Geometry geometry(const rs::Source& source,std::size_t index,double actual_h,const Mat3& rotation,
                         bool headtail) {
  const auto normalized=sc::geometry(source.poses[index].axes,actual_h,source.ar);
  Geometry out;const Vec3 origin(.23e-6,-.17e-6,.11e-6);
  for(int b=0;b<2;++b){
    auto& body=out.body[b];body.axes={source.ar*source.c_m,source.ar*source.c_m,source.c_m};
    out.centers[b]=rotation*(source.c_m*normalized[b].center)+origin;
    body.position=array(out.centers[b]);
    Mat3 orientation=rotation*normalized[b].rotation;
    if(headtail){Mat3 flip=Mat3::Identity();flip(1,1)=flip(2,2)=-1.;orientation=orientation*flip;}
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)body.rotation[3*i+j]=orientation(i,j);
  }
  out.gap.normal=array(rotation*Vec3::UnitZ());out.gap.gap=actual_h*source.c_m;
  out.gap.converged=true;return out;
}
static gp::Inputs inputs(const V17& z,const Geometry& geometry,const Mat3& rotation,
                         double ell,double solver_length) {
  const auto basis=sc::strain_basis();Mat3 strain=Mat3::Zero();
  for(int k=0;k<5;++k)strain+=z[12+k]*basis[k]/ell;
  strain=(rotation*strain*rotation.transpose()).eval();
  const Vec3 translation(.031,-.027,.019),ambient_omega(.13/ell,-.07/ell,.11/ell);
  const Vec3 origin=(geometry.centers[0]+geometry.centers[1])/2.;gp::Inputs x;
  for(int b=0;b<2;++b){
    const Vec3 bg=translation+strain*(geometry.centers[b]-origin)+ambient_omega.cross(geometry.centers[b]-origin);
    const Vec3 du=rotation*z.segment<3>(6*b),dw=rotation*z.segment<3>(6*b+3)/ell;
    for(int k=0;k<3;++k){
      x.background[6*b+k]=bg[k];x.velocity[6*b+k]=bg[k]+du[k];
      x.background[6*b+3+k]=solver_length*ambient_omega[k];
      x.velocity[6*b+3+k]=solver_length*(ambient_omega[k]+dw[k]);
    }
  }
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)x.strain[3*i+j]=strain(i,j);
  return x;
}
static V17 response(const gp::Evaluation& value,const Mat3& rotation,double eta_c,double ell) {
  V17 out;out.segment<3>(0)=-rotation.transpose()*vector(value.force_i)/eta_c;
  out.segment<3>(6)=-rotation.transpose()*vector(value.force_j)/eta_c;
  out.segment<3>(3)=-rotation.transpose()*vector(value.torque_i)/(eta_c*ell);
  out.segment<3>(9)=-rotation.transpose()*vector(value.torque_j)/(eta_c*ell);
  Mat3 stress;for(int i=0;i<3;++i)for(int j=0;j<3;++j)stress(i,j)=value.stress[3*i+j];
  stress=(rotation.transpose()*stress*rotation/(eta_c*ell)).eval();
  const auto basis=sc::strain_basis();for(int k=0;k<5;++k)out[12+k]=(stress.array()*basis[k].array()).sum();
  return out;
}
struct Result {
  std::size_t pose_index=0;
  double actual_nm=0.,effective_nm=0.,weight=0.,operator_error=0.,energy_error=0.;
  double force_error=0.,torque_error=0.,stress_error=0.,power_error=0.,derivative_error=0.;
  double minimum_eigenvalue=0.,constant_minimum=0.,completion_metric_norm=0.;
  std::string transform;
};
static Result check(const rs::Source& source,const SlurrySPDTable17::Provider& table,
                    rs::MatchedModel model,std::size_t index,double h,const Mat3& rotation,
                    bool headtail,const std::string& transform) {
  const double heff=std::max(h,source.hmin);
  const double eta=.73,ell=source.ar*source.c_m,solver_length=1.7*ell;
  const auto shape=geometry(source,index,h,rotation,headtail);
  const double range=2.*source.hmax/source.ar;
  const auto pair=table.make_pair(shape.body[0],shape.body[1],shape.gap,eta,solver_length,range,.75);
  require(pair.active&&std::abs(pair.weight-1.)<1e-14,"Runtime cutoff hid an active reference query");
  require(pair.model.metadata.model_id==table.loaded_model_revision,"Runtime lost selected model identity");
  if(table.sparse_reference||model==rs::MatchedModel::DissipativeIncrement)
    require(pair.model.metadata.reference_id.find("physical_accuracy_qualified=0")!=std::string::npos,
            "Runtime incorrectly lost provisional qualification metadata");
  const Mat17 expected=source.total(index,h,model),self=source.shape->self(source.poses[index].axes,heff);
  const Mat17 correction=expected-self;
  Mat17 actual;double power_error=0.;
  for(int k=0;k<17;++k){
    V17 z=V17::Zero();z[k]=1.;const auto x=inputs(z,shape,rotation,ell,solver_length);
    const auto value=gp::evaluate(pair.model,pair.frame,x);
    actual.col(k)=response(value,rotation,eta*source.c_m,ell);
    const double expected_power=eta*source.c_m*z.dot(correction*z);
    power_error=std::max(power_error,std::abs(value.power-expected_power)/(eta*source.c_m*correction.norm()));
    require(std::abs(value.power-2.*value.energy)<1e-12*std::max(1e-300,std::abs(value.power)),
            "Runtime power is not twice quadratic energy");
  }
  V17 z;for(int k=0;k<17;++k)z[k]=std::sin(.53*(k+1))+.2*std::cos(.31*k);
  const auto x=inputs(z,shape,rotation,ell,solver_length);const auto value=gp::evaluate(pair.model,pair.frame,x);
  const double expected_power=eta*source.c_m*z.dot(correction*z);
  power_error=std::max(power_error,std::abs(value.power-expected_power)/(eta*source.c_m*correction.norm()*z.squaredNorm()));
  double derivative_error=0.;const double step=1e-3,scale=eta*source.c_m*std::max(1.,actual.norm());
  // Differentiate the actual runtime quadratic independently of its output
  // mapping. Hold background center velocities fixed for intrinsic stress.
  for(int k=0;k<12;++k){auto plus=x,minus=x;plus.velocity[k]+=step;minus.velocity[k]-=step;
    const double fd=(gp::evaluate(pair.model,pair.frame,plus).energy-gp::evaluate(pair.model,pair.frame,minus).energy)/(2*step);
    derivative_error=std::max(derivative_error,std::abs(fd-value.gradient[k])/scale);
  }
  const auto basis=sc::strain_basis();
  for(int k=0;k<5;++k){auto plus=x,minus=x;const Mat3 direction=rotation*basis[k]*rotation.transpose()/ell;
    double expected_derivative=0.;for(int i=0;i<3;++i)for(int j=0;j<3;++j){
      plus.strain[3*i+j]+=step*direction(i,j);minus.strain[3*i+j]-=step*direction(i,j);
      expected_derivative+=value.stress[3*i+j]*direction(i,j);
    }
    const double fd=(gp::evaluate(pair.model,pair.frame,plus).energy-gp::evaluate(pair.model,pair.frame,minus).energy)/(2*step);
    derivative_error=std::max(derivative_error,std::abs(fd-expected_derivative)/scale);
  }
  const Mat17 total=.5*(actual+actual.transpose()).eval()+self;
  Result out;out.pose_index=index;out.actual_nm=h*source.c_m*1e9;out.effective_nm=heff*source.c_m*1e9;out.transform=transform;out.weight=pair.weight;
  out.operator_error=relative(actual,correction);out.energy_error=energy_error(total,expected);
  Eigen::Matrix<double,6,17> forces_a,forces_b,torques_a,torques_b;
  forces_a<<actual.middleRows<3>(0),actual.middleRows<3>(6);forces_b<<correction.middleRows<3>(0),correction.middleRows<3>(6);
  torques_a<<actual.middleRows<3>(3),actual.middleRows<3>(9);torques_b<<correction.middleRows<3>(3),correction.middleRows<3>(9);
  out.force_error=relative(forces_a,forces_b);out.torque_error=relative(torques_a,torques_b);
  out.stress_error=relative(actual.bottomRows<5>(),correction.bottomRows<5>());
  out.power_error=power_error;out.derivative_error=derivative_error;out.minimum_eigenvalue=minimum(total);
  const Mat17 constant=source.constant_candidate(index,h);out.constant_minimum=minimum(constant);
  Eigen::LLT<Mat17> metric(expected);require(metric.info()==Eigen::Success,"Matched model must be SPD");
  out.completion_metric_norm=rs::whiten(expected-constant,metric.matrixL()).norm();
  return out;
}

int main(int argc,char**argv){try{
  std::string source_path,table_path,output_path,matched_model,layout="sparse";double tolerance=2e-6;
  for(int i=1;i<argc;++i){const std::string arg=argv[i];
    if(i+1>=argc)throw std::runtime_error("Expected value after "+arg);
    if(arg=="--source")source_path=argv[++i];else if(arg=="--table")table_path=argv[++i];
    else if(arg=="--matched-model")matched_model=argv[++i];
    else if(arg=="--layout")layout=argv[++i];
    else if(arg=="--output")output_path=argv[++i];else if(arg=="--tolerance")tolerance=std::stod(argv[++i]);
    else throw std::runtime_error("Unknown argument "+arg);
  }
  require(!source_path.empty()&&!table_path.empty(),"Usage: check_reference_pipeline --source source17 --table table.sft --matched-model dissipative-increment|constant-residual [--output report.json]");
  require(matched_model=="dissipative-increment"||matched_model=="constant-residual","Explicit valid --matched-model is required");
  require(layout=="sparse"||layout=="full-grid","Invalid --layout");const bool sparse=layout=="sparse";
  require(tolerance>0.&&tolerance<=1e-4,"Invalid algebra tolerance");
  rs::Source source;source.load(source_path,sparse);SlurrySPDTable17::Provider table;table.load(table_path);table.epsilon_m=source.floor_m;
  require(table.sparse_reference==sparse&&!table.physical_accuracy_qualified,"Layout or qualification metadata mismatch");
  require(std::abs(table.ar-source.ar)<1e-12*source.ar&&std::abs(table.hmin-source.hmin)<1e-12*source.hmin&&
          std::abs(table.hmax-source.hmax)<1e-12*source.hmax,"Source/table shape or normalized gap domain mismatch");
  if(sparse){
    require(table.provisional&&table.reference_scope=="exact_poses_only","Sparse source lost provisional exact-pose metadata");
    require(table.source_identity=="fnv1a64:"+source.source_fnv1a64,"Source/table identity mismatch");
    require(table.reference_poses.size()==source.poses.size(),"Sparse source/table pose count mismatch");
  }else require(table.reference_scope=="complete_angular_grid"&&
                 std::size_t(table.nt)*table.nt*table.np==source.poses.size(),"Full grid does not contain all actual source rows");
  const rs::MatchedModel model=matched_model=="dissipative-increment"?rs::MatchedModel::DissipativeIncrement:rs::MatchedModel::ConstantResidual;
  require(table.loaded_model_revision==rs::model_revision_for(model),"Table model revision differs from explicit --matched-model");
  const Mat3 identity=Mat3::Identity();
  const Mat3 rotation=Eigen::AngleAxisd(.83,Vec3(.31,-.47,.67).normalized()).toRotationMatrix();
  std::vector<Result> results;
  std::vector<double> queries{source.hmax,source.hmin,.5*source.hmin,0.};
  auto add_gap=[&](double h){if(h<source.hmin||h>source.hmax)return;for(double old:queries)if(std::abs(h-old)<1e-13*source.hmax)return;queries.push_back(h);};
  add_gap(.25*source.hmax);add_gap(std::sqrt(source.hmin*source.hmax));add_gap(3.15e-9/source.c_m);
  for(std::size_t index=0;index<source.poses.size();++index)for(double h:queries){
    results.push_back(check(source,table,model,index,h,identity,false,"identity"));
    results.push_back(check(source,table,model,index,h,rotation,false,"generic_rotation"));
    results.push_back(check(source,table,model,index,h,rotation,true,"generic_rotation_headtail"));
  }
  bool pass=true;for(const auto& result:results)pass&=result.energy_error<=tolerance&&result.operator_error<=tolerance&&
    result.force_error<=tolerance&&result.torque_error<=tolerance&&result.stress_error<=tolerance&&
    result.power_error<=tolerance&&result.derivative_error<2e-9&&result.minimum_eigenvalue>0.;
  const auto ordinary=geometry(source,0,std::sqrt(source.hmin*source.hmax),identity,false);
  std::vector<std::pair<std::string,bool>> rejection;
  std::vector<std::string> skipped;
  auto call=[&](const Geometry& g){table.make_pair(g.body[0],g.body[1],g.gap,.73,1.7*source.ar*source.c_m,2.*source.hmax/source.ar,.75);};
  auto wrong=ordinary;const Mat3 tilt=Eigen::AngleAxisd(.01,Vec3::UnitY()).toRotationMatrix();
  Mat3 orientation;for(int i=0;i<3;++i)for(int j=0;j<3;++j)orientation(i,j)=wrong.body[1].rotation[3*i+j];
  orientation=(tilt*orientation).eval();for(int i=0;i<3;++i)for(int j=0;j<3;++j)wrong.body[1].rotation[3*i+j]=orientation(i,j);
  if(sparse)rejection.push_back({"uncovered_angle",rejects([&]{call(wrong);})});
  else skipped.push_back("uncovered_angle: complete angular grid covers these angles");
  wrong=ordinary;for(auto& b:wrong.body)b.axes[0]=b.axes[1]=1.05*source.ar*source.c_m;
  rejection.push_back({"wrong_AR",rejects([&]{call(wrong);})});
  wrong=ordinary;for(auto& b:wrong.body)for(double& a:b.axes)a*=1.01;
  if(sparse)rejection.push_back({"wrong_physical_size",rejects([&]{call(wrong);})});
  else skipped.push_back("wrong_physical_size: normalized full-grid format has no stored SI particle length");
  wrong=ordinary;wrong.body[1].axes[0]*=1.01;
  rejection.push_back({"unequal_size",rejects([&]{call(wrong);})});
  if(sparse)rejection.push_back({"wrong_hydrodynamic_floor",rejects([&]{table.correction17(source.poses[0].axes,source.hmin,2.*source.hmin);})});
  else skipped.push_back("wrong_hydrodynamic_floor: full-grid uses explicit runtime epsilon; normalized hmin was checked against the source");
  rejection.push_back({"above_anchor",rejects([&]{table.correction17(source.poses[0].axes,1.01*source.hmax,source.hmin);})});
  const std::string table_hash=file_fnv1a64(table_path);
  const std::filesystem::path temporary=std::filesystem::path(table_path).parent_path()/("pipeline_corrupt_"+std::to_string(getpid())+".sft");
  struct Cleanup{std::filesystem::path path;~Cleanup(){std::error_code ec;std::filesystem::remove(path,ec);}} cleanup{temporary};
  if(std::filesystem::file_size(table_path)<=32*1024*1024){
  const std::string payload=bytes(table_path);for(bool truncate:{false,true}){
    {std::ofstream out(temporary,std::ios::binary);if(truncate)out.write(payload.data(),payload.size()-1);else{out.write(payload.data(),payload.size());out.put('x');}require(bool(out),"Cannot write temporary malformed table");}
    rejection.push_back({truncate?"truncated_payload":"extra_payload",rejects([&]{SlurrySPDTable17::Provider bad;bad.load(temporary.string());})});
  }}else skipped.push_back("malformed_payload: >32 MiB table; covered separately by loader unit tests without duplicating the large payload");
  for(const auto& item:rejection)pass&=item.second;
  double anchor_error=0.,floor_error=0.;
  for(std::size_t index=0;index<source.poses.size();++index){
    anchor_error=std::max(anchor_error,relative(source.total(index,source.hmax,model),.5*(source.poses[index].raw+source.poses[index].raw.transpose()).eval()));
    floor_error=std::max(floor_error,relative(source.total(index,0.,model),source.total(index,source.hmin,model)));
  }
  pass&=anchor_error<1e-13&&floor_error==0.;
  std::ostringstream report;report<<std::setprecision(17)<<std::boolalpha;
  report<<"{\n\"schema\":\"reference17_actual_pipeline_v1\",\n\"physical_accuracy_qualified\":false,\n"
    <<"\"numerical_source_fingerprint\":\""<<SLURRY_NUMERICAL_SOURCE_FINGERPRINT<<"\",\n"
    <<"\"scope\":\"imported actual reference rows; algebra, storage and runtime mapping only; no narrow-gap BVP or model-error qualification\",\n"
    <<"\"runtime_layout\":\""<<(sparse?"sparse_exact_pose":"complete_actual_source_grid")<<"\",\n\"poses_checked\":"<<source.poses.size()<<",\n"
    <<"\"source_binding\":\""<<(sparse?"table_header_fnv1a64":"external_pipeline_manifest_required")<<"\",\n"
    <<"\"source_fnv1a64\":\""<<source.source_fnv1a64<<"\",\n\"table_fnv1a64\":\""<<table_hash<<"\",\n"
    <<"\"source_json_sha256\":\""<<source.poses[0].source_json_sha256<<"\",\n\"model_revision\":\""<<table.loaded_model_revision<<"\",\n"
    <<"\"source_anchor_relative_error\":"<<anchor_error<<",\n\"subfloor_source_relative_error\":"<<floor_error<<",\n"
    <<"\"gap_rule\":\"max(actual_surface_gap, stored_floor); no mechanical roughness subtraction\",\n\"algebra_tolerance\":"<<tolerance<<",\n\"queries\":[\n";
  for(std::size_t i=0;i<results.size();++i){const auto& r=results[i];report<<(i?",\n":"")
    <<"{\"pose_index\":"<<r.pose_index<<",\"actual_gap_nm\":"<<r.actual_nm<<",\"effective_gap_nm\":"<<r.effective_nm<<",\"transform\":\""<<r.transform
    <<"\",\"window_weight\":"<<r.weight<<",\"full17_energy_error\":"<<r.energy_error<<",\"correction_operator_relative_error\":"<<r.operator_error
    <<",\"force_operator_relative_error\":"<<r.force_error<<",\"torque_operator_relative_error\":"<<r.torque_error
    <<",\"intrinsic_stress_operator_relative_error\":"<<r.stress_error<<",\"power_backward_scaled_error\":"<<r.power_error
    <<",\"runtime_quadratic_derivative_scaled_error\":"<<r.derivative_error<<",\"minimum_full17_eigenvalue\":"<<r.minimum_eigenvalue
    <<",\"raw_constant_candidate_minimum_eigenvalue\":"<<r.constant_minimum<<",\"completion_change_frobenius_in_selected_model_metric\":"<<r.completion_metric_norm<<'}';}
  report<<"\n],\n\"completion_change_is_physical_error_bound\":false,\n\"rejections\":{";
  for(std::size_t i=0;i<rejection.size();++i)report<<(i?",":"")<<'"'<<rejection[i].first<<"\":"<<rejection[i].second;
  report<<"},\n\"rejection_checks_not_applicable\":[";
  for(std::size_t i=0;i<skipped.size();++i)report<<(i?",":"")<<'"'<<skipped[i]<<'"';
  report<<"],\n\"pipeline_algebra_passed\":"<<pass<<"\n}\n";
  if(output_path.empty())std::cout<<report.str();else{std::ofstream out(output_path);out<<report.str();require(bool(out),"Cannot write report");}
  return pass?0:2;
}catch(const std::exception& error){std::cerr<<"Reference pipeline: "<<error.what()<<'\n';return 1;}}
