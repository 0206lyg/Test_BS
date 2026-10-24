#include "slurry_bridge_step.h"
#include "slurry_cylinder_pair.h"
#include <iostream>
#include <stdexcept>
namespace st=SlurryBridgeStep;
namespace sh=SlurryBridgeShape;
namespace bs=SlurryBridgeSolver;
using V3=Eigen::Vector3d;
using M3=Eigen::Matrix3d;
void require(bool ok,const char *message) { if(!ok)throw std::runtime_error(message); }
int main() {
  sh::gr::Body a,b;
  a.axes=b.axes={20.,20.,1.};
  a.rotation=b.rotation={1,0,0,0,1,0,0,0,1};
  a.position={0,0,0}; b.position={20.,0,2.02};
  const auto cylinder=sh::closest_gap(a,b,true,1e-10);
  const auto spheroid=sh::closest_gap(a,b,false);
  require(std::abs(cylinder.gap-.02)<1e-12,"flake offset cap gap is not physical cylinder geometry");
  require(spheroid.gap>cylinder.gap+.1,"fixture does not distinguish cylinder and spheroid surfaces");
  const V3 separation=sh::vector(b.position)+sh::vector(cylinder.leverJ)
      -sh::vector(a.position)-sh::vector(cylinder.leverI);
  require((separation-cylinder.gap*sh::vector(cylinder.normal)).norm()<1e-12,"cylinder contact levers inconsistent with gap");
  require(std::abs(sh::bounding_radius(a,true)-std::sqrt(401.))<1e-12,"flake broad phase misses cap rim");
  const auto inertia=sh::principal_inertia(3.,a.axes.data(),true);
  require(std::abs(inertia[0]-301.)<1e-12&&std::abs(inertia[1]-301.)<1e-12&&std::abs(inertia[2]-600.)<1e-12,"cylinder inertia incorrect");
  auto overlap=b; overlap.position={0,0,1.9}; bool rejected=false;
  try { sh::closest_gap(a,overlap,true); } catch(const std::runtime_error &) { rejected=true; }
  require(rejected,"overlapping cylinders were accepted as zero gap");

  bs::Problem p; p.count=2;p.dt=1e-4;
  p.prescribed=bs::Vector::Zero(22);p.external=bs::Vector::Zero(12);
  p.near_resistance.resize(22,22);p.self_mobility.assign(2,bs::Matrix11::Identity());
  p.mobility=[](const bs::Vector&x){return x;};
  p.external[4]=.5;p.external[10]=-.5;
  bs::Contact c;c.i=0;c.j=1;c.gap=.02;c.onset=.02;c.kn=100./(.01*150e-9*100.);
  c.normal=V3::UnitZ();c.Bi.leftCols<3>()=-M3::Identity();c.Bj.leftCols<3>()=M3::Identity();
  M3 cross; cross<<0,-1.01,0,1.01,0,0,0,0,0;
  c.Bi.rightCols<3>()=cross;c.Bj.rightCols<3>()=cross;p.contacts.push_back(c);
  st::Geometry geometry;geometry.flake=true;geometry.gap_tolerance=1e-10;
  geometry.lattice=100.*M3::Identity();geometry.contact_onset=.02;
  geometry.bodies={a,a};geometry.bodies[1].position={0,0,2.02};
  geometry.quaternions.assign(2,Eigen::Quaterniond::Identity());geometry.images.emplace_back(0,0,0);
  bs::Options options;options.relative_tolerance=1e-8;options.absolute_tolerance=1e-12;options.linear_relative_tolerance=1e-10;
  const auto result=st::solve(p,geometry,options);
  const double theta=p.dt*.5,reach=20.*std::sin(theta)+std::cos(theta);
  const double load=c.kn*(c.onset-2.02+2.*reach)/(1.+2.*p.dt*c.kn);
  require(result.geometry_iterations>=2,"cylinder rotation did not trigger endpoint correction");
  require(std::abs(result.solution.contacts[0].normal_load-load)<1e-5,"cylinder rotation load disagrees with analytic cap support");
  require(std::abs(result.endpoint.gaps[0]-(2.02+2*p.dt*load-2*reach))<2e-10,"cylinder endpoint used spheroid support");
  require(result.endpoint.gaps[0]>0.,"accepted cylinder endpoint overlaps");
  require(geometry.bodies[1].position[2]==2.02&&p.contacts[0].history.isZero(),"trial mutated original state");
  std::cout<<"flake geometry, inertia and endpoint-force consistency passed\n";
}
