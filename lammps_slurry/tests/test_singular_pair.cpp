#include "slurry_singular_pair.h"
#include <iostream>
#include <iomanip>
int main(){try {
  using namespace SlurryScattering;namespace old=SlurrySpheroidHydro;
  namespace gr=slurry::gr_re2::graphite;
  const double a=10.,pi=std::acos(-1.);
  const std::array<Vec3,2> face{Vec3::UnitZ(),Vec3::UnitZ()};
  double normal_error=0.;
  for(double h:{.001,.02,1.,6.,2000.}){
    const Mat11 H=SlurryFastSingular::leading11(face,h,a);
    normal_error=std::max(normal_error,std::abs(H(2,2)*h/(1.5*pi*a*a*a*a)-1.));
    if((H-H.transpose()).norm()>1e-12*H.norm())throw std::runtime_error("reciprocity failure");
  }
  if(normal_error>1e-12)throw std::runtime_error("face squeeze leading coefficient failure");
  const std::array<Vec3,2> axes{Vec3(std::sin(.13),0,std::cos(.13)),Vec3(std::sin(.35)*std::cos(.7),std::sin(.35)*std::sin(.7),std::cos(.35))};
  const auto b=geometry(axes,.01,a);gr::Body bi,bj;gr::Body* bs[]={&bi,&bj};
  for(int k=0;k<2;++k){bs[k]->axes={a,a,1};for(int i=0;i<3;++i){bs[k]->position[i]=b[k].center[i];for(int j=0;j<3;++j)bs[k]->rotation[3*i+j]=b[k].rotation(i,j);}}
  gr::GapResult gap;gap.gap=.01;gap.normal={0,0,1};gap.converged=true;
  for(int i=0;i<3;++i){gap.leverI[i]=-b[0].center[i];gap.leverJ[i]=(i==2?.01:0)-b[1].center[i];}
  auto normal=old::make_pair(bi,bj,gap,1.,.001,1.);
  auto raw=old::make_pair(bi,bj,gap,1.,.001,1.,true);
  if(!normal.active||!raw.active)throw std::runtime_error("unexpected inactivity");
  double diff=0.,scale=0.;
  for(int i=0;i<6;++i)for(int j=0;j<6;++j){const double expected=raw.normal_cross[i]*raw.normal_cross[j]/raw.normal_resistance;
    diff=std::max(diff,std::abs(normal.resistance[6*i+j]-raw.resistance[6*i+j]-expected));scale=std::max(scale,std::abs(normal.resistance[6*i+j]));}
  if(diff>1e-12*scale)throw std::runtime_error("completion subtraction mismatch");
  std::cout<<std::setprecision(17)<<"PASS normal coefficient max relative error "<<normal_error<<"; completion removal relative error "<<diff/scale<<"\n";
}catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
