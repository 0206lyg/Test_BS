#include "slurry_scattering_provider.h"
#include <Eigen/Eigenvalues>
#include <iomanip>
#include <iostream>

namespace sc=SlurryScattering;
void resample(sc::Basis& b,int nphi) {
  const int nr=b.fourier_rings.rows(),nm=b.fourier_modes;
  if(!b.use_fourier||nphi<2*nm||nr*nphi>100000)throw std::runtime_error("Unsupported resampling dimensions");
  b.fourier_nphi=nphi;b.fourier_harmonics.resize(nphi,nm);b.sources.resize(nr*nphi,3);
  const double pi=std::acos(-1.);
  for(int j=0;j<nphi;++j) {
    const double phi=2*pi*j/nphi;b.fourier_harmonics(j,0)=1./nphi;
    for(int m=1;m<=(nm-1)/2;++m){b.fourier_harmonics(j,2*m-1)=std::cos(m*phi)/nphi;b.fourier_harmonics(j,2*m)=std::sin(m*phi)/nphi;}
    for(int r=0;r<nr;++r)b.sources.row(r*nphi+j)<<b.fourier_rings(r,0)*std::cos(phi),b.fourier_rings(r,0)*std::sin(phi),b.fourier_rings(r,1);
  }
}
int main(int argc,char**argv) {
 try {
  if(argc!=2&&argc!=3)throw std::runtime_error("Expected plain near basis file [face_edge]");
  sc::Basis b;b.load(argv[1]);const double a=b.aspect_ratio,h=.5;
  const bool edge=argc==3;
  const std::array<sc::Vec3,2> axes=edge ? std::array<sc::Vec3,2>{{
    sc::Vec3(std::sin(.0397),0,std::cos(.0397)),
    sc::Vec3(std::sin(1.4893)*std::cos(-.931),std::sin(1.4893)*std::sin(-.931),std::cos(1.4893))}}
    :std::array<sc::Vec3,2>{{sc::Vec3(0,0,1),sc::Vec3(0,0,1)}};
  const auto bodies=sc::geometry(axes,h,a);const auto map=sc::midpoint_map(bodies,h,a);
  std::vector<sc::Mat11> matrices;std::cout<<std::setprecision(17);
  for(int nphi:{128,256,512}) {
    resample(b,nphi);const auto raw=b.evaluate(bodies);const sc::Mat17 full=.5*(raw.grand+raw.grand.transpose()).eval();
    const sc::Mat11 reduced=sc::reduced(full,map);Eigen::LLT<sc::Mat17> pos(full);
    if(pos.info()!=Eigen::Success)throw std::runtime_error("Pair grand resistance is not SPD");
    matrices.push_back(reduced);
    std::cout<<"{\"ar\":"<<a<<",\"pose\":\""<<(edge?"face_edge_h0.5":"face_face_h0.5")<<"\",\"runtime_phi\":"<<nphi<<",\"raw_reciprocity\":"<<raw.reciprocity<<",\"linear_residual\":"<<raw.linear_residual<<",\"spd\":true";
    if(matrices.size()>1){Eigen::GeneralizedSelfAdjointEigenSolver<sc::Mat11> eig(matrices[matrices.size()-2],reduced);if(eig.info()!=Eigen::Success)throw std::runtime_error("Comparison eigenproblem failed");std::cout<<",\"previous_to_current_all11_change\":"<<(eig.eigenvalues().array()-1).abs().maxCoeff();}
    std::cout<<"}"<<std::endl;
  }
  return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
