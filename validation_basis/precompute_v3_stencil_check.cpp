// Validation-only wrapper: ordinary arguments execute the actual generator.
// --check-stencil exercises the actual table Provider without a source solve.
#define main slurry_precompute_main
#include "../lammps_slurry/tools/precompute_fast.cpp"
#undef main
#ifndef SLURRY_NUMERICAL_SOURCE_FINGERPRINT
#define SLURRY_NUMERICAL_SOURCE_FINGERPRINT "unbound"
#endif

static void fill_grid(SlurrySPDTable::Provider& p,int source_version){
  const double pi=std::acos(-1.);p.ar=10.;p.format_version=2;
  for(int k=0;k<=16;++k){const double psi=.5*pi*k/16;p.theta.push_back(std::atan2(std::sin(psi)/10.,std::cos(psi)));}
  for(double d:{0.,2.,5.,10.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.})p.theta.push_back(d*pi/180.);
  std::sort(p.theta.begin(),p.theta.end());
  p.theta.erase(std::unique(p.theta.begin(),p.theta.end(),[](double a,double b){return std::abs(a-b)<1e-12;}),p.theta.end());
  p.theta.front()=0.;p.theta.back()=.5*pi;
  for(double t:p.theta)p.angular_coordinate.push_back(std::atan2(10.*std::sin(t),std::cos(t)));
  for(int k=0;k<48;++k)p.phi.push_back(-pi+2*pi*k/48);
  for(int k=0;k<=25;++k)p.gap.push_back(.001*std::pow(2.,.5*k));
  p.gap.push_back(5.001);p.joins={.5,1.};if(source_version==2)p.joins.push_back(2.5);
  for(double h:p.joins)p.gap.push_back(h);
  std::sort(p.gap.begin(),p.gap.end());
  p.gap.erase(std::unique(p.gap.begin(),p.gap.end(),[](double a,double b){return std::abs(a-b)<1e-12*std::max(a,b);}),p.gap.end());
  for(double h:p.gap)p.log_gap.push_back(std::log(h));
  p.nt=p.theta.size();p.np=p.phi.size();p.nh=p.gap.size();p.hmin=p.gap.front();p.hmax=p.gap.back();
}

int main(int argc,char** argv){
  if(argc!=2||std::string(argv[1])!="--check-stencil")return slurry_precompute_main(argc,argv);
  try{
    SlurrySPDTable::Provider old,current;fill_grid(old,2);fill_grid(current,3);
    const auto a=old.gap_stencil(.021),b=current.gap_stencil(.021);
    bool same=old.nt==34&&old.np==48&&old.nh==30&&current.nh==29&&
      old.theta==current.theta&&old.phi==current.phi&&old.angular_coordinate==current.angular_coordinate&&
      a.size==b.size&&a.index==b.index&&a.weight==b.weight;
    for(int k=0;k<a.size;++k)same=same&&old.gap[a.index[k]]==current.gap[b.index[k]];
    if(!same)throw std::runtime_error("V2/V3 original3nm interpolation stencil differs");
    std::cout<<std::setprecision(17)<<"{\n  \"passed\": true,\n  \"scope\": \"Actual Provider gap_stencil and grid identity only; no physical source solves\",\n"
      <<"  \"compiled_source_fingerprint\": \""<<SLURRY_NUMERICAL_SOURCE_FINGERPRINT<<"\",\n"
      <<"  \"theta_nodes_identical\": true,\n  \"phi_nodes_identical\": true,\n  \"weights_bitwise_identical\": true,\n"
      <<"  \"theta_count\": "<<old.nt<<",\n  \"phi_count\": "<<old.np<<",\n  \"v2_gap_count\": "<<old.nh<<",\n  \"v3_gap_count\": "<<current.nh<<",\n"
      <<"  \"actual_gap_over_c\": 0.02,\n  \"epsilon_over_c\": 0.001,\n  \"effective_gap_over_c\": 0.021,\n  \"indices\": [";
    for(int k=0;k<a.size;++k)std::cout<<(k?", ":"")<<a.index[k];std::cout<<"],\n  \"gap_nodes\": [";
    for(int k=0;k<a.size;++k)std::cout<<(k?", ":"")<<old.gap[a.index[k]];std::cout<<"],\n  \"weights\": [";
    for(int k=0;k<a.size;++k)std::cout<<(k?", ":"")<<a.weight[k];std::cout<<"]\n}\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
