// Artifact/lookup contract only: synthetic logarithms do not certify physics.
#define main slurry_precompute_program_main
#include "../tools/precompute_fast.cpp"
#undef main

namespace {
void require(bool passed,const std::string& message) {
  if(!passed)throw std::runtime_error(message);
}
template<class F> void rejected(F operation,const std::string& message) {
  bool caught=false;try{operation();}catch(const std::exception&){caught=true;}
  require(caught,message);
}
void fixture(const std::string& path,const std::string& magic=SlurrySPDTable17::table_magic,
             const std::string& revision=SlurrySPDTable17::model_revision) {
  const double pi=std::acos(-1.),floor=1./150.;
  std::ofstream f(path,std::ios::binary);
  f<<std::setprecision(17)<<magic<<"\nMODEL "<<revision<<"\n20 "<<floor<<" 10 4 4 4\n1 0.5\n";
  f<<0.<<' '<<pi/6<<' '<<pi/3<<' '<<pi/2<<'\n';
  f<<-pi<<' '<<-pi/2<<' '<<0.<<' '<<pi/2<<'\n';
  const std::array<double,4> gaps{floor,.02,.5,10.};
  for(double h:gaps)f<<h<<' ';f<<"\nDATA\n";
  for(int i=0;i<4*4*4;++i)for(double h:gaps)for(int r=0;r<17;++r)for(int c=0;c<=r;++c){
    const double value=r==c?.1*std::log(h):0.;
    f.write(reinterpret_cast<const char*>(&value),sizeof(value));
  }
}
}
int main(int argc,char**argv) {try {
  if(argc!=2)throw std::runtime_error("Usage: hardfloor_table_test TEMP_DIRECTORY");
  const auto root=std::filesystem::path(argv[1]);std::filesystem::create_directories(root);
  const auto table=(root/"hardfloor_fixture.sft").string();fixture(table);
  SlurrySPDTable17::Provider p;p.load(table);
  require(p.format_version==4&&p.epsilon_m==1e-9,"Wrong default floor/version");
  const Axes axes{Vec3(.1,0.,std::sqrt(.99)),Vec3(0.,.2,std::sqrt(.96))};
  const double floor=1./150.,gap=3.15/150.;
  const auto base=p.correction17(axes,floor,floor);
  for(double h:{-floor,0.,.15/150.,.9/150.,floor})
    require((p.correction17(axes,h,floor)-base).norm()==0.,"Below-floor query changed resistance");
  const auto actual=p.correction17(axes,gap,floor),unregularized=p.correction17(axes,gap,0.);
  require((actual-unregularized).norm()==0.,"Above-floor query added an offset");
  require((actual-p.correction17(axes,gap+floor,0.)).norm()>1e-8,"Fixture cannot distinguish additive offset");
  const auto log=p.residual(.1,.2,.3,gap);
  require((log-.1*std::log(gap)*Mat17::Identity()).norm()<1e-12,"Stored log interpolation did not roundtrip");
  for(const auto* old:{"SLURRY_SPD_LOGFOURIER_V1","SLURRY_SPD_LOGFOURIER_V2","SLURRY_SPD_LOGFOURIER_V3"}){
    fixture(table,old);rejected([&]{p.load(table);},"Old physical table accepted");
  }
  fixture(table,SlurrySPDTable17::table_magic,"OLD_MODEL");
  rejected([&]{p.load(table);},"Wrong physical revision accepted");
  // A common translation is an independent physical mode. Encoding must
  // retain its response instead of minimizing it out of eleven coordinates.
  SlurrySPDCoordinates17::Shape shape(20.);
  Eigen::Matrix<double,17,1> common=Eigen::Matrix<double,17,1>::Zero();common[0]=common[6]=1.;
  const Mat17 total=shape.self(axes,gap)+3.*common*common.transpose();
  const Mat17 encoded=SlurrySPDTable17::encode_total(shape,axes,gap,total);
  const Mat17 decoded=SlurrySPDTable17::decode_total(shape,axes,gap,encoded);
  require((decoded-total).norm()/total.norm()<2e-12,"Full17 encode/decode lost matrix response");
  const Mat17 correction=decoded-shape.self(axes,gap);
  require(std::abs(common.dot(correction*common)-12.)<1e-7,"Full17 encode/decode lost common translation shielding mode");
  const auto shard=(root/"checkpoint.bin").string();
  const std::vector<std::uint64_t> rows{0,2};const std::vector<double> source{1,2,3,4,5,6};
  std::vector<double> restored(6,0.);
  checkpoint::write(shard,"test_identity",7,rows,2,source,{1.,0.});
  checkpoint::read(shard,"test_identity",7,rows,2,restored);
  require(restored==std::vector<double>({1,2,0,0,5,6}),"Checkpoint roundtrip failed");
  rejected([&]{checkpoint::read(shard,"different_model_identity",7,rows,2,restored);},"Mismatched checkpoint reused");
  {std::fstream f(shard,std::ios::in|std::ios::out|std::ios::binary);f<<"SLURRY_SPD_CHECKPOINT_V1";}
  rejected([&]{checkpoint::read(shard,"test_identity",7,rows,2,restored);},"Legacy checkpoint reused");
  std::cout<<"PASS hardfloor table serialization, lookup, old-model rejection, checkpoint identity\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
