#include "slurry_spd_table.h"
#include <iomanip>
#include <iostream>

namespace st=SlurrySPDTable;
using M=st::Mat11;
static st::Axes pose(double ti,double tj,double phi) {
  return {{st::Vec3(std::sin(ti),0.,std::cos(ti)),
           st::Vec3(std::sin(tj)*std::cos(phi),std::sin(tj)*std::sin(phi),std::cos(tj))}};
}
int main(int argc,char**argv) {
  try {
    if(argc!=2)throw std::runtime_error("Expected actual generated table");
    st::Provider p;p.load(argv[1]);
    double maximum_knot_error=0.,maximum_covariance_error=0.;int nodes=0,offgrid=0;
    for(int i=0;i<p.nt;++i)for(int j=0;j<p.nt;++j)for(int q=0;q<p.np;++q)for(int k=0;k<p.nh;++k) {
      const auto a=pose(p.theta[i],p.theta[j],p.phi[q]);
      const M logarithm=p.residual(p.theta[i],p.theta[j],p.phi[q],p.gap[k]);
      const std::size_t node=((std::size_t(i)*p.nt+j)*p.np+q)*p.nh+k;
      M stored;int n=0;for(int r=0;r<11;++r)for(int c=0;c<=r;++c)stored(r,c)=stored(c,r)=p.data[66*node+n++];
      maximum_knot_error=std::max(maximum_knot_error,(logarithm-stored).norm()/std::max(1.,stored.norm()));
      const M total=p.correction(a,p.gap[k],0.)+p.shape->self(a,p.gap[k]);
      Eigen::LLT<M> spd(total);if(!total.allFinite()||spd.info()!=Eigen::Success)throw std::runtime_error("Knot query failed positive total check");
      ++nodes;
    }
    const double pi=std::acos(-1.);const auto R=st::z_rotation(.371);const auto Q=st::rotation11(R);
    for(int k=0;k<32;++k) {
      const double ti=.5*pi*(k+.37)/32.,tj=.5*pi*((13*k+7)%32+.21)/32.,phi=-pi+2*pi*((7*k+3)%32+.19)/32.;
      const double h=p.hmin*std::pow(p.hmax/p.hmin,(k+.31)/32.);
      auto a=pose(ti,tj,phi),b=a;for(auto& x:b)x=R*x;
      const M base=p.correction(a,h,0.),rotated=p.correction(b,h,0.);
      const M total=base+p.shape->self(a,h);
      Eigen::LLT<M> spd(total);if(!total.allFinite()||spd.info()!=Eigen::Success)throw std::runtime_error("Offgrid query failed positive total check");
      maximum_covariance_error=std::max(maximum_covariance_error,(rotated-Q*base*Q.transpose()).norm()/std::max(1.,base.norm()));
      ++offgrid;
    }
    if(maximum_knot_error>1e-10||maximum_covariance_error>1e-10)throw std::runtime_error("Table knot or covariance tolerance exceeded");
    std::cout<<std::setprecision(17)<<"{\"status\":\"passed\",\"scope\":\"actual completed table loader and reconstruction only; no interpolation accuracy or independent Stokes certification\",\"aspect_ratio\":"<<p.ar<<",\"grid_nodes_queried\":"<<nodes<<",\"offgrid_queries\":"<<offgrid<<",\"maximum_knot_log_relative_error\":"<<maximum_knot_error<<",\"maximum_rotation_covariance_relative_error\":"<<maximum_covariance_error<<"}\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
