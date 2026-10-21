// Synthetic matrices only: this checks I/O identity without a Stokes solve.
#define main slurry_basis_generator_program_main
#include "../tools/slurry_generate_basis.cpp"
#undef main
#include <filesystem>

namespace {
std::string file_contents(const std::string& path) {
  std::ifstream file(path);std::ostringstream text;text<<file.rdbuf();return text.str();
}
M values(int rows,int cols) {
  M x(rows,cols);
  for(int i=0;i<rows;++i)for(int j=0;j<cols;++j)x(i,j)=((i+j)%2?-1.:1.)*(1.+i)/(3.+j);
  return x;
}
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
}
int main(int argc,char**argv) {try {
  if(argc!=2)throw std::runtime_error("Usage: basis_serialization_test TEMP_DIRECTORY");
  std::filesystem::create_directories(argv[1]);
  Settings s;s.ar=20.;s.degree=1;s.source_phi=8;s.source_order=1;
  s.output=(std::filesystem::path(argv[1])/"basis.ssb").string();
  Generated b;b.targets=values(2,3);b.rings=values(2,2);b.projection=values(11,6);
  b.density=values(18,11);b.rigid=values(11,6);b.strain=values(11,5);
  b.force=values(6,11);b.stress=values(5,11);
  std::ostringstream original;original<<std::setprecision(17);write_basis(original,s,b);
  write_verified_basis(s,b);
  require(file_contents(s.output)==original.str(),"Plain streamed output differs from original serializer");
  require(!std::filesystem::exists(s.output+".tmp"),"Pending file remained after successful publish");
  s.far=s.output;s.output=(std::filesystem::path(argv[1])/"hybrid.ssb").string();
  std::ostringstream hybrid;hybrid<<std::setprecision(17)<<"SLURRY_HYBRID_V4\n20 1\nANCHORS 2 0.5 1\n"
    <<original.str()<<'\n'<<original.str();
  write_verified_basis(s,b);
  require(file_contents(s.output)==hybrid.str(),"Hybrid streamed output differs from original serializer");
  b.rigid.resize(2,6);bool rejected=false;
  try{write_verified_basis(s,b);}catch(const std::exception&){rejected=true;}
  require(rejected,"Invalid pending basis was not rejected");
  require(file_contents(s.output)==hybrid.str(),"Failed validation overwrote published basis");
  std::cout<<"PASS streamed basis byte identity, file roundtrip and atomic publication\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
