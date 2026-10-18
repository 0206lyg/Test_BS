// Synthetic format/interpolation infrastructure check; no physical source gate.
#include "slurry_spd_table.h"
#include <cstdio>
#include <iomanip>
#include <iostream>

namespace {
const std::vector<double> gaps{.001,.02,.1,.5,.7,1.,1.5,2.5,4.,5.};
double value(double h){
  const double x=std::log(h);
  return .03+.2*x+.7*std::max(0.,x-std::log(.5))-.9*std::max(0.,x)+.4*std::max(0.,x-std::log(2.5));
}
void write_table(const std::string& path,int version,const std::vector<double>& joins){
  std::ofstream f(path,std::ios::binary);const double pi=std::acos(-1.);
  f<<std::setprecision(17)<<"SLURRY_SPD_LOGFOURIER_V"<<version<<"\n10 .001 5 4 8 "<<gaps.size()<<'\n';
  if(version==2){f<<joins.size();for(double h:joins)f<<' '<<h;f<<'\n';}
  for(double theta:{0.,.1,.7,pi/2})f<<theta<<' ';f<<'\n';
  for(int k=0;k<8;++k)f<<-pi+2*pi*k/8<<' ';f<<'\n';
  for(double h:gaps)f<<h<<' ';f<<"\nDATA\n";
  for(int pose=0;pose<4*4*8;++pose)for(double h:gaps){
    std::array<double,66> data{};data[0]=value(h);
    f.write(reinterpret_cast<const char*>(data.data()),sizeof(data));
  }
  if(!f)throw std::runtime_error("Synthetic fixture write failed");
}
}

int main(){try{
  const std::string path="synthetic_spd_format.spt";
  write_table(path,2,{.5,1.,2.5});SlurrySPDTable::Provider v2;v2.load(path);
  if(v2.format_version!=2||v2.joins!=std::vector<double>({.5,1.,2.5}))throw std::runtime_error("V2 header roundtrip failed");
  for(int pose=0;pose<4*4*8;++pose)for(std::size_t k=0;k<gaps.size();++k)
    for(int q=0;q<66;++q)if(v2.data[(pose*gaps.size()+k)*66+q]!=(q?0.:value(gaps[k])))
      throw std::runtime_error("V2 binary payload roundtrip failed");
  std::vector<double> queries{.001,.01,.3,.6,.9,1.2,2.,3.,4.5,5.};
  for(double join:v2.joins)for(double delta:{-1e-10,0.,1e-10})queries.push_back(join*(1+delta));
  double maximum_error=0.;
  for(double h:queries){
    double low=gaps.front(),high=gaps.back();
    for(double join:v2.joins){if(h<=join){high=join;break;}low=join;}
    const auto s=v2.gap_stencil(h);
    for(int k=0;k<s.size;++k)if(gaps[s.index[k]]<low||gaps[s.index[k]]>high)
      throw std::runtime_error("V2 stencil crossed an explicit source join");
    const auto actual=v2.residual(.22,.51,-1.23,h);
    maximum_error=std::max(maximum_error,std::abs(actual(0,0)-value(h)));
    if(maximum_error>1e-12)throw std::runtime_error("Piecewise log-gap field interpolation failed");
  }
  write_table(path,1,{});SlurrySPDTable::Provider v1;v1.load(path);
  if(v1.format_version!=1||v1.joins!=std::vector<double>({2.5}))throw std::runtime_error("V1 compatibility failed");
  for(double h:queries){
    // This is the original V1 segment selection, including its single join.
    const int at=int(std::lower_bound(gaps.begin(),gaps.end(),2.5)-gaps.begin());
    const auto old=SlurrySPDTable::stencil(v1.log_gap,std::log(h),h<=2.5?0:at,h<=2.5?at:int(gaps.size())-1);
    const auto now=v1.gap_stencil(h);
    if(old.size!=now.size||old.index!=now.index||old.weight!=now.weight)
      throw std::runtime_error("V1 interpolation behavior changed");
  }
  int rejected=0;
  for(const auto& invalid:std::vector<std::vector<double>>{{.5,.5,2.5},{1.,.5,2.5},{.5,1.1,2.5},{.5,1.,6.}}){
    write_table(path,2,invalid);
    try{SlurrySPDTable::Provider bad;bad.load(path);}catch(const std::exception&){++rejected;}
  }
  if(rejected!=4)throw std::runtime_error("Malformed V2 joins were accepted");
  std::remove(path.c_str());
  std::cout<<std::setprecision(17)<<"{\n  \"passed\": true,\n  \"scope\": \"Synthetic table format and join interpolation only; no physical source validation\",\n"
    <<"  \"v2_binary_roundtrip\": true,\n  \"v1_stencils_unchanged\": true,\n  \"joins_checked\": [0.5, 1, 2.5],\n"
    <<"  \"query_count\": "<<queries.size()<<",\n  \"max_piecewise_interpolation_error\": "<<maximum_error
    <<",\n  \"malformed_join_cases_rejected\": "<<rejected<<"\n}\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
