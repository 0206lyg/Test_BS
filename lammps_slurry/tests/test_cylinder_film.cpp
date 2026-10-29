#include "slurry_cylinder_film.h"
#include <Eigen/Eigenvalues>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
namespace cf=SlurryCylinderFilm;
using V22=Eigen::Matrix<double,22,1>;
using V3=Eigen::Vector3d;
void require(bool x,const char* s){if(!x)throw std::runtime_error(s);}
double power(const cf::Mat22& r,const V22& q){return q.dot(r*q);}
std::pair<cf::Body,cf::Body> fixture(double offset,double half_angle,double h=.02) {
  cf::Body a,b;a.radius=b.radius=20.;a.half_thickness=b.half_thickness=1.;
  a.axis=V3(std::sin(half_angle),0.,std::cos(half_angle));
  b.axis=V3(-std::sin(half_angle),0.,std::cos(half_angle));
  const double span=20.*std::cos(half_angle)-.5*offset;
  const double d=2.*std::cos(half_angle)+h+2.*std::tan(half_angle)*span;
  a.center=V3(-.5*offset,0.,-.5*d);b.center=V3(.5*offset,0.,.5*d);return {a,b};
}
int main(int argc,char** argv) {
  const bool probe=argc>1&&std::string(argv[1])=="--probe";
  if(probe) {
    const int order=argc>2?std::stoi(argv[2]):4,quad=argc>3?std::stoi(argv[3]):24;
    std::cout<<std::setprecision(17)<<"[";bool first=true;
    for(const auto& values:std::vector<std::pair<double,double>>{{0,0},{.5,0},{.5,.1},{.5,1.},{1.5,1.}}) {
      const auto bodies=fixture(values.first*20.,values.second*cf::pi/180.);
      cf::Options o;o.pressure_order=order;o.quadrature_order=quad;
      const auto start=std::chrono::steady_clock::now();
      const auto r=cf::evaluate(bodies.first,bodies.second,o);
      const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      V22 q=V22::Zero();q[2]=-.5;q[13]=.5;
      const double alpha=values.second*cf::pi/180.,t=std::tan(alpha);
      const double hmid=(bodies.second.center-bodies.first.center).z()-2.*std::cos(alpha);
      Eigen::Matrix<double,22,3> modes=Eigen::Matrix<double,22,3>::Zero();modes.col(0)=q;
      modes(3,1)=-.5;modes(14,1)=.5;
      modes(4,2)=-.5/(1.-t*t);modes(15,2)=.5/(1.-t*t);
      modes.col(2)+=((std::sin(alpha)-.5*hmid*t)/(1.-t*t))*q;
      const Eigen::Matrix3d pm=modes.transpose()*r.pressure*modes;
      Eigen::Matrix<double,22,3> sm=Eigen::Matrix<double,22,3>::Zero();
      sm(0,0)=-.5;sm(11,0)=.5;sm(1,1)=-.5;sm(12,1)=.5;
      sm(5,2)=-.5;sm(16,2)=.5;sm.col(2)-=std::sin(alpha)*sm.col(1);
      const Eigen::Matrix3d cm=sm.transpose()*r.couette*sm;
      if(!first)std::cout<<",";first=false;
      std::cout<<"{\"offset_over_a\":"<<values.first<<",\"half_angle_deg\":"<<values.second
        <<",\"order\":"<<order<<",\"quadrature\":"<<quad<<",\"pressure_normal\":"<<power(r.pressure,q)
        <<",\"area\":"<<r.area<<",\"minimum_sample_gap\":"<<r.min_gap
        <<",\"seconds\":"<<seconds<<",\"pressure_residual\":"<<r.pressure_residual;
      auto matrix=[&](const char* name,const Eigen::Matrix3d& value) {
        std::cout<<",\""<<name<<"\":[";
        for(int row=0;row<3;++row){if(row)std::cout<<",";std::cout<<"[";
          for(int col=0;col<3;++col){if(col)std::cout<<",";std::cout<<value(row,col);}std::cout<<"]";}
        std::cout<<"]";
      };
      matrix("pressure_matrix_normal_y_minusx",pm);matrix("couette_matrix_x_y_spin",cm);std::cout<<"}";
    }std::cout<<"]\n";return 0;
  }
  cf::Options o;
  auto bodies=fixture(0.,0.);auto r=cf::evaluate(bodies.first,bodies.second,o);
  require(r.active,"coaxial film inactive");
  V22 normal=V22::Zero(),slide=normal,spin=normal,rock=normal;
  normal[2]=-.5;normal[13]=.5;slide[0]=-.5;slide[11]=.5;
  spin[5]=-.5;spin[16]=.5;rock[3]=-.5;rock[14]=.5;
  const double a=20.,h=.02;
  const double rn=1.5*cf::pi*std::pow(a,4)/std::pow(h,3),rt=cf::pi*a*a/h;
  const double rz=.5*cf::pi*std::pow(a,4)/h,rr=cf::pi*std::pow(a,6)/(8.*std::pow(h,3));
  require(std::abs(power(r.pressure,normal)/rn-1)<1e-10,"coaxial squeeze coefficient");
  require(std::abs(power(r.couette,slide)/rt-1)<1e-10,"coaxial sliding coefficient");
  require(std::abs(power(r.couette,spin)/rz-1)<1e-10,"coaxial axial-spin coefficient");
  require(std::abs(power(r.pressure,rock)/rr-1)<1e-10,"coaxial rocking coefficient");
  // Actual center motion is affine simple shear, but actual Omega is zero.
  // qU=0, qOmega=-Omega_inf, qe=-E_inf. Full slide speed is gamma*d.
  V22 shear=V22::Zero();shear[4]=shear[15]=-.5;
  shear[9]=shear[20]=-1./std::sqrt(2.);
  const double d=2.+h;
  require(std::abs(power(r.couette,shear)/(rt*d*d)-1.)<1e-10,"finite-gap affine shear strain convention");
  // Rigid translation and rotation of both bodies must have zero pair loss.
  const V3 omega(.13,-.27,.32),translation(.7,.1,-.3);
  V22 rigid=V22::Zero();rigid.segment<3>(0)=translation+omega.cross(bodies.first.center);
  rigid.segment<3>(11)=translation+omega.cross(bodies.second.center);
  rigid.segment<3>(3)=rigid.segment<3>(14)=omega;
  require((r.resistance*rigid).norm()<1e-11*r.resistance.norm()*rigid.norm(),"common rigid motion is not null");
  for(const auto& values:std::vector<std::pair<double,double>>{{.5,0.},{.5,.001},{1.5,.02}}) {
    bodies=fixture(values.first*20.,values.second);r=cf::evaluate(bodies.first,bodies.second,o);
    require((r.resistance-r.resistance.transpose()).norm()<1e-14*r.resistance.norm(),"asymmetric film resistance");
    Eigen::SelfAdjointEigenSolver<cf::Mat22> eig(r.resistance);
    require(eig.eigenvalues().minCoeff()>-1e-12*eig.eigenvalues().maxCoeff(),"negative film dissipation");
    const auto swapped=cf::evaluate(bodies.second,bodies.first,o);
    Eigen::Matrix<double,22,22> P=cf::Mat22::Zero();P.block<11,11>(0,11).setIdentity();P.block<11,11>(11,0).setIdentity();
    require((r.resistance-P*swapped.resistance*P).norm()<1e-9*r.resistance.norm(),"film pair exchange symmetry");
    const Eigen::Matrix3d Q=Eigen::AngleAxisd(.73,V3(.3,.4,.8).normalized()).toRotationMatrix();
    auto ra=bodies.first,rb=bodies.second;ra.center=Q*ra.center;rb.center=Q*rb.center;
    ra.axis=Q*ra.axis;rb.axis=Q*rb.axis;
    const auto rotated=cf::evaluate(ra,rb,o);cf::Mat22 transform=cf::Mat22::Zero();
    Eigen::Matrix<double,5,5> transformE;
    for(int i=0;i<5;++i)for(int j=0;j<5;++j)
      transformE(i,j)=(cf::strain_basis()[i].array()*(Q*cf::strain_basis()[j]*Q.transpose()).array()).sum();
    for(int base:{0,11}){transform.block<3,3>(base,base)=Q;transform.block<3,3>(base+3,base+3)=Q;
      transform.block<5,5>(base+6,base+6)=transformE;}
    require((r.resistance-transform.transpose()*rotated.resistance*transform).norm()<1e-9*r.resistance.norm(),
            "film rotation covariance");
  }
  // Constitutive floor leaves squeeze forcing active below the floor.
  bodies=fixture(0.,0.,.001);r=cf::evaluate(bodies.first,bodies.second,o);
  require(std::abs(power(r.pressure,normal)/(1.5*cf::pi*std::pow(a,4)/std::pow(o.gap_floor,3))-1)<1e-10,
          "floor incorrectly suppresses squeeze source");
  // Independent P1 FEM (512 subdivisions, 523264 triangles), generated by
  // validate_cylinder_reference.py. This mixed-mode check catches the >40%
  // wedge underestimate of an unenriched low-order pressure trial space.
  const double alpha=cf::pi/180.,t=std::tan(alpha);
  bodies=fixture(10.,alpha);r=cf::evaluate(bodies.first,bodies.second,o);
  const double hmid=(bodies.second.center-bodies.first.center).z()-2.*std::cos(alpha);
  Eigen::Matrix<double,22,3> modes=Eigen::Matrix<double,22,3>::Zero();modes.col(0)=normal;
  modes(3,1)=-.5;modes(14,1)=.5;
  modes(4,2)=-.5/(1.-t*t);modes(15,2)=.5/(1.-t*t);
  modes.col(2)+=((std::sin(alpha)-.5*hmid*t)/(1.-t*t))*normal;
  Eigen::Matrix3d reference;
  reference<<6578205.398727499,0.,53271784.7676034,
             0.,154462406.288681,0.,
             53271784.7676034,0.,557158809.5868585;
  Eigen::LLT<Eigen::Matrix3d> rf(reference);
  const Eigen::Matrix3d inverse=rf.matrixL().solve(Eigen::Matrix3d::Identity());
  const Eigen::Matrix3d difference=inverse*(modes.transpose()*r.pressure*modes-reference)*inverse.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> error(difference);
  require(error.eigenvalues().cwiseAbs().maxCoeff()<.012,"tilted mixed-mode pressure accuracy exceeds 1.2 percent");
  std::cout<<"PASS: cylinder film analytic modes, affine strain, rigid null, symmetry, PSD and floor\n";
}
