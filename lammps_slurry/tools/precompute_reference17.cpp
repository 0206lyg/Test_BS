#include "slurry_reference_source17.h"
#include "slurry_spd_table17.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#ifndef SLURRY_NUMERICAL_SOURCE_FINGERPRINT
#define SLURRY_NUMERICAL_SOURCE_FINGERPRINT "unbound_standalone_build"
#endif

namespace reference=SlurryReferenceSource17;
namespace table=SlurrySPDTable17;
using reference::Mat17;

namespace {
std::string json(const std::string& value) {
  std::ostringstream out;out<<'"';
  for(unsigned char c:value){if(c=='"'||c=='\\')out<<'\\'<<char(c);
    else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
    else out<<char(c);}
  out<<'"';return out.str();
}
void install_new(const std::string& pending,const std::string& output) {
  // A link installation is atomic and refuses to replace an existing table.
  if(::link(pending.c_str(),output.c_str())!=0)
    throw std::runtime_error("Cannot install fresh output (existing path or filesystem error): "+output);
  if(::unlink(pending.c_str())!=0)throw std::runtime_error("Cannot remove installed output temporary link");
}
double relative_energy_error(const Mat17& expected,const Mat17& actual) {
  Eigen::LLT<Mat17> llt(expected);
  if(llt.info()!=Eigen::Success)throw std::runtime_error("Interpolation comparison target is not SPD");
  const Mat17 error=reference::whiten(actual-expected,llt.matrixL());
  Eigen::SelfAdjointEigenSolver<Mat17> eig(.5*(error+error.transpose()).eval(),Eigen::EigenvaluesOnly);
  if(eig.info()!=Eigen::Success||!eig.eigenvalues().allFinite())
    throw std::runtime_error("Interpolation relative-energy error evaluation failed");
  return eig.eigenvalues().cwiseAbs().maxCoeff();
}
Mat17 logarithm_at(const std::vector<double>& payload,std::size_t pose,int nh,int node) {
  Mat17 matrix;const auto offset=(pose*std::size_t(nh)+std::size_t(node))*153;
  int q=0;for(int r=0;r<17;++r)for(int c=0;c<=r;++c)matrix(r,c)=matrix(c,r)=payload[offset+q++];
  return matrix;
}
Mat17 interpolate_log(const std::vector<double>& payload,std::size_t pose,
                      const std::vector<double>& log_gap,double h) {
  const auto stencil=table::stencil(log_gap,std::log(h),0,int(log_gap.size())-1);
  Mat17 value=Mat17::Zero();
  for(int k=0;k<stencil.size;++k)value+=stencil.weight[k]*logarithm_at(payload,pose,int(log_gap.size()),stencil.index[k]);
  return .5*(value+value.transpose()).eval();
}
struct Grid {
  std::vector<double> theta,phi;
  std::vector<std::size_t> source_order;
};
Grid complete_grid(const reference::Source& source) {
  Grid grid;std::vector<double> theta_j;
  for(const auto& p:source.poses){grid.theta.push_back(p.theta_i);theta_j.push_back(p.theta_j);grid.phi.push_back(p.phi);}
  const auto unique=[](std::vector<double>& values){std::sort(values.begin(),values.end());
    values.erase(std::unique(values.begin(),values.end(),[](double a,double b){return std::abs(a-b)<1e-12;}),values.end());};
  unique(grid.theta);unique(theta_j);unique(grid.phi);
  const int nt=int(grid.theta.size()),np=int(grid.phi.size());const double pi=std::acos(-1.);
  if(nt<4||nt>128||np<4||np>256||np%2||grid.theta.size()!=theta_j.size()||
     std::abs(grid.theta.front())>1e-12||std::abs(grid.theta.back()-.5*pi)>1e-12)
    throw std::runtime_error("Full grid requires shared theta endpoints 0/pi2, >=4 theta and >=4 even periodic phi nodes");
  for(int i=0;i<nt;++i)if(std::abs(grid.theta[i]-theta_j[i])>1e-12)
    throw std::runtime_error("Full grid requires identical theta_i and theta_j grids");
  for(int p=0;p<np;++p)if(std::abs(grid.phi[p]-(-pi+2.*pi*p/np))>1e-12)
    throw std::runtime_error("Full grid phi must equal -pi+2*pi*k/np; arbitrary offsets are unsupported");
  const std::size_t expected=std::size_t(nt)*nt*np;
  if(source.poses.size()!=expected)throw std::runtime_error("Full grid requires actual source evidence for every theta_i/theta_j/phi row; missing rows are never fabricated");
  grid.source_order.assign(expected,std::numeric_limits<std::size_t>::max());
  const auto find=[](const std::vector<double>& nodes,double value){
    const auto at=std::lower_bound(nodes.begin(),nodes.end(),value-1e-12);
    if(at==nodes.end()||std::abs(*at-value)>1e-12)throw std::runtime_error("Full grid source key is not on its numeric grid");
    return std::size_t(at-nodes.begin());};
  for(std::size_t row=0;row<source.poses.size();++row){const auto& p=source.poses[row];
    const auto index=(find(grid.theta,p.theta_i)*nt+find(grid.theta,p.theta_j))*np+find(grid.phi,p.phi);
    if(grid.source_order[index]!=std::numeric_limits<std::size_t>::max())throw std::runtime_error("Duplicate numeric full-grid source row");
    grid.source_order[index]=row;
  }
  return grid;
}
}

int main(int argc,char** argv) {
  std::string pending,report_pending;
  try {
    std::string source_path,output,report,layout="sparse",model_name;int nh=129,max_nh=256;double tolerance=1e-6;
    for(int i=1;i<argc;++i){const std::string option=argv[i];
      if(option=="--help"){
        std::cout<<"slurry_precompute_reference17 --source IMPORTED_REFERENCE --output TABLE --matched-model constant-residual|dissipative-increment [--gap-nodes 129] [--max-gap-nodes 256] [--interpolation-tolerance 1e-6] [--report FILE] [--layout sparse|full-grid]\n"
          <<"Gap nodes are refined near measured interpolation errors, with seven interior checks per interval. This checks numerical transport of the selected model, not physical accuracy. No Basis solve or angular extrapolation. Fresh output paths are required.\n";return 0;
      }
      if(i+1==argc)throw std::runtime_error("Missing argument for "+option);
      const std::string value=argv[++i];
      if(option=="--source")source_path=value;else if(option=="--output")output=value;
      else if(option=="--report")report=value;else if(option=="--layout")layout=value;
      else if(option=="--matched-model")model_name=value;
      else if(option=="--gap-nodes")nh=std::stoi(value);
      else if(option=="--max-gap-nodes")max_nh=std::stoi(value);
      else if(option=="--interpolation-tolerance")tolerance=std::stod(value);
      else throw std::runtime_error("Unknown argument: "+option);
    }
    if(source_path.empty()||output.empty()||nh<4||nh>max_nh||max_nh>4096||
       !std::isfinite(tolerance)||tolerance<=0.||tolerance>1e-3)
      throw std::runtime_error("Invalid reference table generation options");
    if(layout!="sparse"&&layout!="full-grid")throw std::runtime_error("--layout must be sparse or full-grid");
    reference::MatchedModel model;
    if(model_name=="constant-residual")model=reference::MatchedModel::ConstantResidual;
    else if(model_name=="dissipative-increment")model=reference::MatchedModel::DissipativeIncrement;
    else throw std::runtime_error("Select an explicit --matched-model constant-residual or dissipative-increment; neither is physically certified");
    const std::string revision=reference::model_revision_for(model);
    if(report.empty())report=output+".validation.json";
    if(output==report||std::filesystem::exists(output)||std::filesystem::exists(report))
      throw std::runtime_error("Reference table and report require distinct fresh output paths");
    reference::Source source;source.load(source_path,layout=="sparse");
    Grid grid;if(layout=="full-grid")grid=complete_grid(source);
    const int initial_nh=nh;
    std::vector<double> gap(nh),log_gap(nh),payload;
    const double log_min=std::log(source.hmin),log_max=std::log(source.hmax);
    for(int k=0;k<nh;++k){log_gap[k]=log_min+(log_max-log_min)*double(k)/(nh-1);gap[k]=std::exp(log_gap[k]);}
    gap.front()=source.hmin;gap.back()=source.hmax;
    log_gap.front()=log_min;log_gap.back()=log_max;
    double maximum_roundtrip=0.,maximum_interpolation=0.,minimum_conditioned_eigenvalue=std::numeric_limits<double>::infinity();
    double minimum_constant_eigenvalue=std::numeric_limits<double>::infinity(),maximum_closure_deviation=0.;
    double worst_constant_gap=0.;std::size_t worst_constant_pose=0;
    const auto compare_constant=[&](std::size_t p,double h,const Mat17& total){
      const Mat17 constant=source.constant_candidate(p,h);
      Eigen::SelfAdjointEigenSolver<Mat17> eig(constant,Eigen::EigenvaluesOnly);
      if(eig.info()!=Eigen::Success||!eig.eigenvalues().allFinite())throw std::runtime_error("Constant-residual diagnostic eigensolve failed");
      if(eig.eigenvalues().minCoeff()<minimum_constant_eigenvalue){minimum_constant_eigenvalue=eig.eigenvalues().minCoeff();worst_constant_gap=h;worst_constant_pose=p;}
      maximum_closure_deviation=std::max(maximum_closure_deviation,relative_energy_error(total,constant));
    };
    std::vector<double> pose_interpolation(source.poses.size(),0.);
    struct RefinementPass {int nodes;double error,h,left,right;std::size_t pose;};
    std::vector<RefinementPass> refinement_history;
    for(;;){
      nh=int(gap.size());log_gap.resize(nh);
      for(int k=0;k<nh;++k)log_gap[k]=std::log(gap[k]);
      const std::size_t count=source.poses.size()*std::size_t(nh)*153;
      if(count>(std::size_t(1)<<26))throw std::runtime_error("Reference table payload exceeds 512 MiB bound");
      payload.resize(count);maximum_interpolation=0.;std::fill(pose_interpolation.begin(),pose_interpolation.end(),0.);
      std::vector<double> interval_error(nh-1,0.);
      RefinementPass pass{nh,0.,gap.front(),gap.front(),gap[1],0};
      for(std::size_t p=0;p<source.poses.size();++p){
      const auto& pose=source.poses[p];
      for(int k=0;k<nh;++k){
        const Mat17 total=source.total(p,gap[k],model);compare_constant(p,gap[k],total);
        minimum_conditioned_eigenvalue=std::min(minimum_conditioned_eigenvalue,
          reference::positive_minimum(table::normalized_total(*source.shape,pose.axes,gap[k],total)));
        const Mat17 logarithm=table::encode_total(*source.shape,pose.axes,gap[k],total);
        int q=0;const auto offset=(p*std::size_t(nh)+std::size_t(k))*153;
        for(int r=0;r<17;++r)for(int c=0;c<=r;++c)payload[offset+q++]=logarithm(r,c);
        const Mat17 restored=table::decode_total(*source.shape,pose.axes,gap[k],logarithm_at(payload,p,nh,k));
        maximum_roundtrip=std::max(maximum_roundtrip,relative_energy_error(total,restored));
      }
      for(int k=0;k<nh-1;++k)for(double fraction:{.125,.25,.375,.5,.625,.75,.875}){
        const double h=std::exp((1.-fraction)*log_gap[k]+fraction*log_gap[k+1]);
        const Mat17 expected=source.total(p,h,model);compare_constant(p,h,expected);
        const Mat17 restored=table::decode_total(*source.shape,pose.axes,h,interpolate_log(payload,p,log_gap,h));
        const double error=relative_energy_error(expected,restored);
        pose_interpolation[p]=std::max(pose_interpolation[p],error);interval_error[k]=std::max(interval_error[k],error);
        if(error>pass.error)pass={nh,error,h,gap[k],gap[k+1],p};
      }
      maximum_interpolation=std::max(maximum_interpolation,pose_interpolation[p]);
      }
      refinement_history.push_back(pass);
      std::cerr<<std::setprecision(17)<<"{\"event\":\"reference_gap_interpolation_check\",\"gap_nodes\":"<<nh
        <<",\"maximum_relative_energy_error\":"<<pass.error<<",\"worst_gap_over_c\":"<<pass.h
        <<",\"interval_left_over_c\":"<<pass.left<<",\"interval_right_over_c\":"<<pass.right
        <<",\"source_pose\":"<<pass.pose<<",\"physical_accuracy_qualified\":false}\n";
      if(maximum_roundtrip>1e-8)throw std::runtime_error("153-value reference round-trip exceeds relative-energy tolerance 1e-8");
      if(maximum_interpolation<=tolerance)break;
      if(nh==max_nh){
        std::ostringstream error;error<<std::setprecision(17)<<"Gap interpolation error "<<pass.error<<" exceeds "<<tolerance
          <<" at h/c="<<pass.h<<" in ["<<pass.left<<','<<pass.right<<"] after adaptive refinement to "<<nh
          <<" nodes; no table installed. A different explicit numerical tolerance or model is required.";
        throw std::runtime_error(error.str());
      }
      std::vector<int> selected;
      // Refine the dominant errors first rather than spend the bounded node
      // budget uniformly while the anchored logarithm varies rapidly nearby.
      const double trigger=std::max(tolerance,.25*maximum_interpolation);
      for(int k=0;k<nh-1;++k)if(interval_error[k]>=trigger)selected.push_back(k);
      std::sort(selected.begin(),selected.end(),[&](int a,int b){return interval_error[a]>interval_error[b];});
      if(selected.size()>std::size_t(max_nh-nh))selected.resize(max_nh-nh);
      if(selected.empty())throw std::runtime_error("Adaptive gap refinement found no resolvable failing interval");
      for(int k:selected){
        const double midpoint=std::exp(.5*(log_gap[k]+log_gap[k+1]));
        if(!(midpoint>gap[k]&&midpoint<gap[k+1]))throw std::runtime_error("Gap interpolation refinement reached floating-point spacing");
        gap.push_back(midpoint);
      }
      std::sort(gap.begin(),gap.end());
    }
    for(std::size_t p=0;p<source.poses.size();++p){const auto& pose=source.poses[p];
      if((source.total(p,source.hmax,model)-pose.anchor_total).norm()!=0.)
        throw std::runtime_error("Selected closure does not reproduce the supplied anchor exactly");
      if((source.total(p,source.hmin*.5,model)-source.total(p,source.hmin,model)).norm()!=0.)
        throw std::runtime_error("Selected closure does not apply the hard floor consistently");
      const double adjacent=std::nextafter(source.hmax,source.hmin);
      if(relative_energy_error(pose.anchor_total,source.total(p,adjacent,model))>1e-8)
        throw std::runtime_error("Selected closure is discontinuous immediately below its anchor");
    }
    pending=output+".tmp."+std::to_string(::getpid());
    if(std::filesystem::exists(pending))throw std::runtime_error("Temporary output already exists");
    std::ofstream file(pending,std::ios::binary);
    if(!file)throw std::runtime_error("Cannot create temporary reference table");
    file<<std::setprecision(17);
    if(layout=="sparse"){
    file<<"SLURRY_SPD_REFERENCE_SPARSE_V1\nMODEL "<<revision
        <<"\nSTATUS PROVISIONAL 0\nSOURCE_FNV1A64 "<<source.source_fnv1a64
        <<"\nUNITS "<<source.c_m<<' '<<source.floor_m<<'\n'
        <<source.ar<<' '<<source.hmin<<' '<<source.hmax<<' '<<source.poses.size()<<' '<<nh<<"\nPOSES\n";
    for(const auto& pose:source.poses)file<<pose.theta_i<<' '<<pose.theta_j<<' '<<pose.phi<<' '<<pose.anchor_h
      <<' '<<pose.source_json_sha256<<' '<<pose.solution_sha256<<'\n';
    file<<"GAPS\n";for(double h:gap)file<<h<<' ';file<<"\nDATA\n";
    }else{
      file<<table::table_magic<<"\nMODEL "<<revision<<'\n'<<source.ar<<' '<<source.hmin<<' '<<source.hmax<<' '
        <<grid.theta.size()<<' '<<grid.phi.size()<<' '<<nh<<"\n0\n";
      for(double value:grid.theta)file<<value<<' ';file<<'\n';
      for(double value:grid.phi)file<<value<<' ';file<<'\n';
      for(double value:gap)file<<value<<' ';file<<"\nDATA\n";
    }
    static_assert(sizeof(double)==8&&std::numeric_limits<double>::is_iec559,"Table requires IEEE binary64");
    if(layout=="sparse")file.write(reinterpret_cast<const char*>(payload.data()),std::streamsize(payload.size()*sizeof(double)));
    else for(std::size_t row:grid.source_order)file.write(reinterpret_cast<const char*>(payload.data()+row*std::size_t(nh)*153),std::streamsize(nh*153*sizeof(double)));
    file.close();
    if(!file)throw std::runtime_error("Incomplete reference table write");
    std::ostringstream metadata;metadata<<std::setprecision(17)
      <<"{\n  \"schema\": \"reference_source17_table_generation_v1\",\n  \"status\": \"provisional_numerical_transport_passed\",\n"
      <<"  \"physical_accuracy_qualified\": false,\n  \"continuation_physically_validated\": false,\n"
      <<"  \"numerical_source_fingerprint\": "<<json(SLURRY_NUMERICAL_SOURCE_FINGERPRINT)<<",\n"
      <<"  \"contains_new_stokes_solve\": false,\n  \"angular_interpolation_supported\": "<<(layout=="full-grid"?"true":"false")<<",\n"
      <<"  \"source_file\": "<<json(source_path)<<",\n  \"source_fnv1a64\": "<<json(source.source_fnv1a64)
      <<",\n  \"model_revision\": "<<json(revision)<<",\n  \"matched_model\": "<<json(model_name)<<",\n  \"convention\": "<<json(reference::convention)
      <<",\n  \"layout\": "<<json(layout=="sparse"?"sparse_exact_pose":"complete_actual_source_grid")<<",\n  \"aspect_ratio\": "<<source.ar
      <<",\n  \"c_m\": "<<source.c_m<<",\n  \"floor_m\": "<<source.floor_m
      <<",\n  \"hmin_over_c\": "<<source.hmin<<",\n  \"hmax_over_c\": "<<source.hmax
      <<",\n  \"pose_count\": "<<source.poses.size()<<",\n  \"gap_nodes\": "<<nh
      <<",\n  \"initial_gap_nodes\": "<<initial_nh<<",\n  \"maximum_gap_nodes\": "<<max_nh
      <<",\n  \"matrix_dimension\": 17,\n  \"stored_values_per_node\": 153,\n  \"payload_bytes\": "<<payload.size()*sizeof(double)
      <<",\n  \"maximum_roundtrip_relative_energy_error\": "<<maximum_roundtrip
      <<",\n  \"maximum_gap_interpolation_relative_energy_error\": "<<maximum_interpolation
      <<",\n  \"interpolation_tolerance\": "<<tolerance
      <<",\n  \"interior_samples_per_interval\": 7,\n  \"minimum_conditioned_eigenvalue_at_knots\": "<<minimum_conditioned_eigenvalue
      <<",\n  \"minimum_raw_constant_residual_eigenvalue\": "<<minimum_constant_eigenvalue
      <<",\n  \"worst_raw_constant_residual_gap_over_c\": "<<worst_constant_gap
      <<",\n  \"worst_raw_constant_residual_source_pose\": "<<worst_constant_pose
      <<",\n  \"maximum_closure_deviation_from_constant_in_candidate_energy_metric\": "<<maximum_closure_deviation
      <<",\n  \"angular_interpolation_physically_validated\": false,\n"
      <<"  \"scope\": \"Checks storage and gap interpolation against the explicitly selected provisional closure. Closure-vs-constant deviation is a model diagnostic, not a physical error bound. No Stokes accuracy or angular interpolation accuracy certification.\",\n  \"refinement_history\": [\n";
    for(std::size_t k=0;k<refinement_history.size();++k){const auto& p=refinement_history[k];if(k)metadata<<",\n";
      metadata<<"    {\"gap_nodes\": "<<p.nodes<<", \"maximum_relative_energy_error\": "<<p.error
        <<", \"worst_gap_over_c\": "<<p.h<<", \"interval_left_over_c\": "<<p.left
        <<", \"interval_right_over_c\": "<<p.right<<", \"source_pose\": "<<p.pose<<'}';}
    metadata<<"\n  ],\n  \"poses\": [\n";
    for(std::size_t p=0;p<source.poses.size();++p){const auto& pose=source.poses[p];
      if(p)metadata<<",\n";
      metadata<<"    {\"theta_i\": "<<pose.theta_i<<", \"theta_j\": "<<pose.theta_j<<", \"phi\": "<<pose.phi
        <<", \"anchor_h_over_c\": "<<pose.anchor_h<<", \"source_json_sha256\": "<<json(pose.source_json_sha256)
        <<", \"solution_sha256\": "<<json(pose.solution_sha256)<<", \"raw_reciprocity\": "<<pose.raw_reciprocity
        <<", \"self_metric_reciprocity\": "<<pose.metric_reciprocity
        <<", \"minimum_anchor_self_metric_eigenvalue\": "<<pose.minimum_metric_eigenvalue
        <<", \"maximum_gap_interpolation_relative_energy_error\": "<<pose_interpolation[p]<<'}';
    }
    metadata<<"\n  ]\n}\n";
    report_pending=report+".tmp."+std::to_string(::getpid());
    if(std::filesystem::exists(report_pending))throw std::runtime_error("Temporary report already exists");
    std::ofstream report_file(report_pending);report_file<<metadata.str();report_file.close();
    if(!report_file)throw std::runtime_error("Incomplete reference table report write");
    install_new(pending,output);pending.clear();install_new(report_pending,report);report_pending.clear();
    std::cout<<metadata.str();return 0;
  }catch(const std::exception& error){
    if(!pending.empty())std::remove(pending.c_str());if(!report_pending.empty())std::remove(report_pending.c_str());
    std::cerr<<"Reference table: "<<error.what()<<'\n';return 1;
  }
}
