// Port parity against the separately evaluated Python general operator, and
// analytic/invariance checks. No test fixture is accessed by the runtime.
#include "slurry_general_outer.h"
#include <Eigen/Geometry>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace sg=SlurryGeneral;
namespace {
template<class A,class B> double relative(const A& a,const B& b) {
  return (a-b).norm()/std::max(1.e-300,b.norm());
}
void require(bool ok,const std::string& message) {
  if(!ok)throw std::runtime_error(message);
}
template<class Matrix> void read_matrix(std::istream& f,Matrix& a) {
  for(int i=0;i<a.rows();++i)for(int j=0;j<a.cols();++j)f>>a(i,j);
}
sg::Mat17 transform17(const sg::Mat3& w) {
  sg::Mat17 t=sg::Mat17::Zero();
  for(int k=0;k<4;++k)t.block<3,3>(3*k,3*k)=w;
  const auto& e=sg::strain_basis();
  for(int a=0;a<5;++a)for(int b=0;b<5;++b)
    t(12+a,12+b)=(e[a].array()*(w*e[b]*w.transpose()).array()).sum();
  return t;
}
}
int main(int argc,char**argv) {
 try {
  const char* path=argc>1?argv[1]:"lammps_slurry/tests/data/general_outer_reference.dat";
  std::ifstream in(path);require(bool(in),std::string("Cannot read fixture ")+path);
  std::string magic;int n;in>>magic>>n;
  require(magic=="SLURRY_GENERAL_OUTER_REFERENCE_V1"&&n==6,"Invalid general outer fixture");
  double maximum=0.;sg::Geometry generic,face;
  for(int k=0;k<n;++k) {
    std::string name;sg::Geometry g;int nr,np,far;
    in>>name>>g.ar>>g.gap>>nr>>np>>far;
    for(auto& c:g.centers)for(int d=0;d<3;++d)in>>c[d];
    for(auto& q:g.rotations)read_matrix(in,q);
    sg::Mat17 grand,isolated;sg::Mat22 mobility;
    read_matrix(in,grand);read_matrix(in,isolated);read_matrix(in,mobility);
    require(bool(in),"Truncated fixture "+name);
    const auto result=sg::evaluate_outer(g,nr,np,bool(far));
    const double error=relative(result.grand,grand);
    maximum=std::max(maximum,error);
    require(error<2.e-9,name+": C++/Python grand17 mismatch");
    require(relative(result.isolated,isolated)<2.e-9,name+": isolated mismatch");
    require(relative(result.mobility,mobility)<2.e-9,name+": mobility mismatch");
    require(result.linear_residual<1.e-10,name+": inverse residual");
    require(relative(result.grand,result.grand.transpose())<1.e-11,name+": reciprocity");
    std::cout<<name<<" relative="<<std::setprecision(12)<<error<<"\n";
    if(name=="ar2_generic_disk")generic=g;
    if(name=="ar20_face_disk")face=g;
  }
  const double pi=std::acos(-1.);
  sg::Mat11 expected=sg::Mat11::Zero();
  for(int i=0;i<3;++i){expected(i,i)=6*pi;expected(i+3,i+3)=8*pi;}
  for(int i=6;i<11;++i)expected(i,i)=20*pi/3.;
  require(relative(sg::isolated11(1.,sg::Mat3::Identity()),expected)<1.e-10,"Isolated sphere analytic limit");

  // Rotate the same physical pair and all tensor/vector modes together.
  const sg::Mat3 w=Eigen::AngleAxisd(.73,sg::Vec3(.3,-.4,.5).normalized()).toRotationMatrix();
  auto rotated=generic;
  for(int b=0;b<2;++b){rotated.centers[b]=w*generic.centers[b];rotated.rotations[b]=w*generic.rotations[b];}
  const auto a=sg::evaluate_outer(generic,4,16);
  const auto b=sg::evaluate_outer(rotated,4,16);
  const auto t=transform17(w);
  require(relative(b.grand,(t*a.grand*t.transpose()).eval())<2.e-10,"Global rotation covariance");

  // The exact common-azimuth reduction must also work about a tilted axis.
  auto tilted_face=face;
  for(int k=0;k<2;++k){tilted_face.centers[k]=w*face.centers[k];tilted_face.rotations[k]=w*face.rotations[k];}
  const auto face_a=sg::evaluate_outer(face,16,64);
  const auto face_b=sg::evaluate_outer(tilted_face,16,64);
  require(face_a.kernel.axisymmetric_azimuth_reduction&&face_b.kernel.axisymmetric_azimuth_reduction,
          "Coaxial reduction should be frame independent");
  require(relative(face_b.grand,(t*face_a.grand*t.transpose()).eval())<2.e-10,
          "Tilted coaxial reduction covariance");

  const sg::Vec3 r=generic.centers[0]-generic.centers[1];
  const auto forward=sg::cross_mobility_disk(r,generic.rotations[0],generic.rotations[1],generic.ar,4,16);
  const auto reverse=sg::cross_mobility_disk(-r,generic.rotations[1],generic.rotations[0],generic.ar,4,16);
  require(relative(forward,reverse.transpose())<2.e-11,"Independently recomputed reverse cross block");

  auto exchanged=generic;
  std::swap(exchanged.centers[0],exchanged.centers[1]);
  std::swap(exchanged.rotations[0],exchanged.rotations[1]);
  const auto c=sg::evaluate_outer(exchanged,4,16);
  sg::Mat17 p=sg::Mat17::Zero();
  for(int i=0;i<6;++i){p(i,i+6)=p(i+6,i)=1.;}
  for(int i=12;i<17;++i)p(i,i)=1.;
  require(relative(c.grand,(p*a.grand*p.transpose()).eval())<2.e-11,"Particle exchange");
  std::cout<<"general outer: 6 fixtures + sphere, rotation, reverse, exchange passed; max_relative="<<maximum<<"\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"general outer FAILED: "<<e.what()<<"\n";return 1;}
}
