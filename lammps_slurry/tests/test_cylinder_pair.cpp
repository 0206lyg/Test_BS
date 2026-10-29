#include "slurry_cylinder_pair.h"
#include <Eigen/Eigenvalues>
#include <iostream>
#include <stdexcept>
namespace cg=SlurryCylinderGeometry;
namespace cp=SlurryCylinderPair;
void require(bool c,const char*m){if(!c)throw std::runtime_error(m);}
int main(){
  cg::Body a,b;a.radius=b.radius=20;a.half_thickness=b.half_thickness=1;
  cp::Options o;o.film.quadrature_order=24;
  for(double separation:{2.02,4.,10.,21.,23.}){
    b.center={3.,0.,separation};const auto g=cg::closest_gap(a,b);
    const auto r=cp::evaluate(a,b,g,o);
    Eigen::SelfAdjointEigenSolver<cp::Mat22> e(r.resistance);
    require(e.info()==Eigen::Success,"pair eigencheck failed");
    require(e.eigenvalues().minCoeff()>=-1e-10*std::max(1.,r.resistance.norm()),"pair is not positive semidefinite");
    if(g.gap>=20.)require(r.resistance.norm()==0.,"cutoff is not compact");
  }
  b.axis={1.,0.,0.};b.center={22.,0.,0.};
  auto gap=cg::closest_gap(a,b);auto r=cp::evaluate(a,b,gap,o);
  require(r.active&&!r.face_active,"rim fixture is not active");
  Eigen::Matrix<double,22,1> q=Eigen::Matrix<double,22,1>::Zero();
  const cp::V3 omega(.2,-.3,.5),shift(1.,2.,3.);
  q.segment<3>(0)=shift+omega.cross(a.center);q.segment<3>(3)=omega;
  q.segment<3>(11)=shift+omega.cross(b.center);q.segment<3>(14)=omega;
  require((r.resistance*q).norm()<1e-9*std::max(1.,r.resistance.norm()),"rim common rigid motion is not null");
  for(const auto&body:{a,b}){
    Eigen::SelfAdjointEigenSolver<cp::Mat11> s(cp::self_mobility(body));
    require(s.eigenvalues().minCoeff()>0.,"self mobility is not positive definite");
  }
  std::cout<<"PASS cylinder pair: positive blending, finite cutoff, rim rigid null, positive self\n";
}
