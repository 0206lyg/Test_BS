// Sparse table transport/coverage contract only. Synthetic logs certify no physics.
#include "slurry_spd_table17.h"
#include <Eigen/Geometry>
#include <filesystem>
#include <iomanip>
#include <iostream>

namespace table=SlurrySPDTable17;
namespace gr=SlurryFiniteGap::gr;
namespace gp=SlurryGrandPair;
using table::Axes;using table::Vec3;using table::Mat3;using table::Mat17;
namespace {
void require(bool pass,const char* reason){if(!pass)throw std::runtime_error(reason);}
template<class F> void rejected(F operation,const char* reason){
  bool caught=false;try{operation();}catch(const std::exception&){caught=true;}require(caught,reason);
}
void close(const Mat17& actual,const Mat17& expected,const char* reason){
  require((actual-expected).norm()<3e-11*std::max(1.,expected.norm()),reason);
}
table::ReferencePose pose(double ti,double tj,double az){
  table::ReferencePose p;p.theta_i=ti;p.theta_j=tj;p.phi=az;p.anchor_h_over_c=1.;
  p.source_json_sha256=std::string(64,'a');p.solution_sha256=std::string(64,'b');return p;
}
Mat17 logarithm(std::size_t row,double h){
  Mat17 result=Mat17::Zero();for(int k=0;k<17;++k)result(k,k)=.03*(row+1)+.02*std::log(h)+.001*k;
  result(0,6)=result(6,0)=.003;return result;
}
struct Options {
  double ar=20.,c_m=150e-9,floor_m=1e-9;
  std::string status="PROVISIONAL 0",identity="0123456789abcdef";
  std::string revision=table::model_revision;
  bool nan=false,truncated=false,extra=false;
};
void fixture(const std::filesystem::path& path,const std::vector<table::ReferencePose>& poses,
             const Options& options=Options{}){
  const std::array<double,4> gaps{1./150.,.02,.2,1.};std::ofstream f(path,std::ios::binary);
  f<<std::setprecision(17)<<table::reference_table_magic<<"\nMODEL "<<options.revision
    <<"\nSTATUS "<<options.status<<"\nSOURCE_FNV1A64 "<<options.identity
    <<"\nUNITS "<<options.c_m<<' '<<options.floor_m<<'\n'
    <<options.ar<<' '<<gaps.front()<<' '<<gaps.back()<<' '<<poses.size()<<" 4\nPOSES\n";
  for(const auto&p:poses)f<<p.theta_i<<' '<<p.theta_j<<' '<<p.phi<<' '<<p.anchor_h_over_c
    <<' '<<p.source_json_sha256<<' '<<p.solution_sha256<<'\n';
  f<<"GAPS\n";for(double h:gaps)f<<h<<' ';f<<"\nDATA\n";
  for(std::size_t p=0;p<poses.size();++p)for(std::size_t k=0;k<gaps.size();++k){
    const Mat17 log=logarithm(p,gaps[k]);
    for(int i=0;i<17;++i)for(int j=0;j<=i;++j){
      if(options.truncated&&p+1==poses.size()&&k+1==gaps.size()&&i==16&&j==16)return;
      const double value=options.nan&&p==0&&k==0&&i==0&&j==0?
        std::numeric_limits<double>::quiet_NaN():log(i,j);
      f.write(reinterpret_cast<const char*>(&value),sizeof(value));}}
  if(options.extra)f<<'x';
}
void complete_fixture(const std::filesystem::path& path,const std::string& revision=table::model_revision){
  const double pi=std::acos(-1.);std::ofstream f(path,std::ios::binary);
  f<<std::setprecision(17)<<table::table_magic<<"\nMODEL "<<revision
    <<"\n20 "<<1./150.<<" 1 4 4 4\n0\n0 "<<pi/6<<' '<<pi/3<<' '<<pi/2<<'\n'
    <<-pi<<' '<<-pi/2<<" 0 "<<pi/2<<"\n"<<1./150.<<" .02 .2 1\nDATA\n";
  const double value=0.;for(int i=0;i<4*4*4*4*153;++i)f.write(reinterpret_cast<const char*>(&value),sizeof(value));
}
void query_checks(const std::filesystem::path& path){
  const std::vector<table::ReferencePose> poses{pose(.3,.4,.8),pose(std::acos(-1.)/2,.23,-1.2)};
  fixture(path,poses);table::Provider p;p.load(path.string());
  require(p.sparse_reference&&p.provisional&&!p.physical_accuracy_qualified,"Sparse qualification metadata lost");
  require(p.reference_scope=="exact_poses_only"&&p.source_identity=="fnv1a64:0123456789abcdef","Sparse scope/source identity lost");
  require(p.reference_poses.size()==2&&p.nh==4&&p.nt==0&&p.np==0,"Sparse poses became a fabricated angular grid");
  require(p.reference_length_m==150e-9&&p.reference_floor_m==1e-9&&p.epsilon_m==1e-9,"Physical units/floor lost");
  for(std::size_t row=0;row<poses.size();++row){const auto axes=poses[row].axes();
    for(double h:{p.min_gap(),.013,.1,.6,p.max_gap()}){
      close(p.residual(poses[row].theta_i,poses[row].theta_j,poses[row].phi,h),logarithm(row,h),"Log-gap interpolation failed");
      const Mat17 expected=table::decode_total(*p.shape,axes,h,logarithm(row,h))-p.shape->self(axes,h);
      close(p.correction17(axes,h,p.min_gap()),expected,"Sparse correction differs from decoded total minus self");
      for(double angle:{-.71,.27,2.4}){const Mat3 R=table::z_rotation(angle);const Mat17 Q=table::rotation17(R);
        for(int signs=0;signs<4;++signs){Axes rotated;
          for(int k=0;k<2;++k)rotated[k]=(signs&(1<<k)?-1.:1.)*R*axes[k];
          close(p.correction17(rotated,h,p.min_gap()),Q*expected*Q.transpose(),"Director signs or global rotation changed response");}}
    }
    const Mat17 floored=p.correction17(axes,p.min_gap(),p.min_gap());
    for(double h:{-.02,0.,.001,p.min_gap()}){
      close(p.correction17(axes,h,p.min_gap()),floored,"Sparse hard floor changed response");
      close(p.correction17(axes,h,0.),floored,"Zero epsilon bypassed stored physical floor");}
    rejected([&]{p.correction17(axes,p.max_gap()*1.01,p.min_gap());},"Gap extrapolation accepted");
    rejected([&]{p.correction17(axes,.1,2*p.min_gap());},"Physical floor override accepted");
  }
  Axes unknown=poses.front().axes();unknown[1]=Eigen::AngleAxisd(.002,Vec3::UnitX())*unknown[1];
  rejected([&]{p.correction17(unknown,.1,p.min_gap());},"Missing angular row silently interpolated");
  unknown[1]=Vec3::Zero();rejected([&]{p.correction17(unknown,.1,p.min_gap());},"Zero director accepted");
  table::Provider unloaded;rejected([&]{unloaded.correction17(poses[0].axes(),.1,0.);},"Unloaded table accepted");
  complete_fixture(path);p.load(path.string());
  require(!p.sparse_reference&&!p.provisional&&p.reference_poses.empty(),"Complete V4 inherited sparse restrictions");
  close(p.residual(.1,.2,.3,.1),Mat17::Zero(),"Existing complete V4 lookup changed");
  complete_fixture(path,table::dissipative_model_revision);p.load(path.string());
  require(!p.sparse_reference&&p.provisional&&!p.physical_accuracy_qualified&&
    p.loaded_model_revision==table::dissipative_model_revision,"New full-grid physical revision mislabeled");
}
void parser_checks(const std::filesystem::path& path){
  const std::vector<table::ReferencePose> poses{pose(.3,.4,.8)};
  auto rejects=[&](const Options&o){fixture(path,poses,o);table::Provider p;rejected([&]{p.load(path.string());},"Malformed sparse file accepted");};
  Options o;o.status="PROVISIONAL 1";rejects(o);o=Options{};o.status="QUALIFIED 0";rejects(o);
  o=Options{};o.identity="not_a_hash";rejects(o);o=Options{};o.floor_m=2e-9;rejects(o);
  o=Options{};o.nan=true;rejects(o);o=Options{};o.truncated=true;rejects(o);o=Options{};o.extra=true;rejects(o);
  auto invalid=poses;invalid[0].anchor_h_over_c=.5;fixture(path,invalid);table::Provider p;
  rejected([&]{p.load(path.string());},"Wrong common anchor accepted");
  invalid=poses;invalid[0].source_json_sha256="abc";fixture(path,invalid);
  rejected([&]{p.load(path.string());},"Invalid source SHA accepted");
  invalid=poses;invalid.push_back(invalid.front());fixture(path,invalid);
  rejected([&]{p.load(path.string());},"Duplicate canonical pose accepted");
  invalid={pose(0.,.4,.2)};fixture(path,invalid);
  rejected([&]{p.load(path.string());},"Noncanonical polar pose accepted");
  // Generic aspect-ratio support is a synthetic file/coordinate test, not a
  // physical solve or accuracy assertion for a second aspect ratio.
  o=Options{};o.ar=7.;fixture(path,poses,o);p.load(path.string());
  require(p.aspect_ratio()==7.,"Sparse loader hardcodes AR20");
  require(p.correction17(poses[0].axes(),.1,p.min_gap()).allFinite(),"Generic sparse AR query failed");
  o=Options{};o.revision=table::dissipative_model_revision;fixture(path,poses,o);p.load(path.string());
  require(p.loaded_model_revision==table::dissipative_model_revision&&p.provisional&&!p.physical_accuracy_qualified,
    "Sparse new-model revision mislabeled");
  o.revision="UNKNOWN_MODEL";rejects(o);
}
void runtime_checks(const std::filesystem::path& path){
  const auto pose0=pose(.3,.4,.8);Options options;options.revision=table::dissipative_model_revision;
  fixture(path,{pose0},options);table::Provider p;p.load(path.string());
  const double h=.2,c=p.reference_length_m;const auto pair=SlurryFastSingular::canonical_pair(pose0.axes(),h,p.ar);
  std::array<gr::Body,2> bodies;
  for(int k=0;k<2;++k){bodies[k].axes={p.ar*c,p.ar*c,c};
    for(int i=0;i<3;++i){bodies[k].position[i]=c*pair.bodies[k].center[i];
      for(int j=0;j<3;++j)bodies[k].rotation[3*i+j]=pair.bodies[k].rotation(i,j);}}
  gr::GapResult gap;gap.normal={0.,0.,1.};gap.gap=h*c;
  for(int i=0;i<3;++i){gap.leverI[i]=c*pair.pair.lever_i[i];gap.leverJ[i]=c*pair.pair.lever_j[i];}
  const auto result=p.make_pair(bodies[0],bodies[1],gap,1.,p.ar*c,.1,.75);
  require(result.active&&result.weight==1.,"Sparse runtime pair inactive or unexpectedly tapered");
  require(result.model.metadata.reference_id.find("provisional;physical_accuracy_qualified=0;exact_poses_only;")==0,"Runtime lost provisional scope");
  require(result.model.metadata.model_id==table::dissipative_model_revision,"Runtime physical model revision mislabeled");
  Mat17 actual;for(int i=0;i<17;++i)for(int j=0;j<17;++j)actual(i,j)=result.model.resistance[17*i+j];
  close(actual,p.correction17(pose0.axes(),h,p.min_gap())/p.ar,"Runtime pair changed correction scaling");
  auto wrong=bodies;for(auto& b:wrong)for(double& a:b.axes)a*=2.;
  rejected([&]{p.make_pair(wrong[0],wrong[1],gap,1.,p.ar*c,.1,.75);},"Sparse physical particle length mismatch accepted");
  p.epsilon_m*=2.;rejected([&]{p.make_pair(bodies[0],bodies[1],gap,1.,p.ar*c,.1,.75);},"Sparse runtime floor mismatch accepted");
}
}
int main(int argc,char**argv){try{
  if(argc!=2)throw std::runtime_error("Usage: reference_table17_test TEMP_DIRECTORY");
  std::filesystem::path root=argv[1];std::filesystem::create_directories(root);const auto path=root/"sparse_reference.sft";
  query_checks(path);parser_checks(path);runtime_checks(path);
  std::cout<<"PASS sparse reference scope, parsing, exact poses, generic AR, rotation/director covariance, hardfloor and runtime mapping\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
