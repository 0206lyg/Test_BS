#include "slurry_nearfield_step.h"
#include "slurry_nearfield_hydro.h"
#include <iostream>
#include <stdexcept>

namespace bs=SlurryBridgeSolver;
namespace ns=SlurryNearfieldSolver;
namespace ni=SlurryNearfieldImplicitSolver;
namespace imex=SlurryNearfieldImexSolver;
namespace step=SlurryBridgeStep;
namespace gr=slurry::gr_re2::graphite;
using V3=Eigen::Vector3d;
using M3=Eigen::Matrix3d;
void require(bool condition,const char *message) {
  if(!condition)throw std::runtime_error(message);
}

int main() {
  // Physical eta=.01 Pa s, a_ref=4um, rate=100/s, dt=2.5us and
  // kn=3..300, kt=kr=.8*kn N/m. These finite preloads excite fast elastic
  // modes; the high-stiffness cases retain exactly the same physical timestep.
  for(double stiffness : {3.,30.,100.,300.})
  for(double aspect : {1.,5.})
  for(int scheme=0;scheme<2;++scheme)for(int mode=0;mode<4;++mode) {
    // Retain the coupled-reference regression at its previous scope. The
    // production IMEX scheme is tested across the full stiffness/shape matrix.
    if(!scheme&&(stiffness!=3.||aspect!=1.))continue;
    bs::Problem p;p.count=2;p.dt=2.5e-4;
    p.prescribed=bs::Vector::Zero(22);p.external=bs::Vector::Zero(12);
    step::Geometry geometry;
    geometry.lattice=20.*M3::Identity();geometry.bodies.resize(2);
    geometry.quaternions.assign(2,Eigen::Quaterniond::Identity());
    geometry.images.emplace_back(0,0,0);
    const double a=std::cbrt(aspect),c_axis=1./(a*a);
    for(auto &body:geometry.bodies){body.axes={a,a,c_axis};body.rotation=gr::identity3;}
    geometry.bodies[1].position={2.*a-.001,0.,0.};
    const auto gap=gr::closestEllipsoidGap(geometry.bodies[0],geometry.bodies[1]);
    const auto near=SlurryNearfieldHydro::evaluate(geometry.bodies[0],geometry.bodies[1],gap,1.,1.,.001,.5);
    p.near_resistance=near.resistance.sparseView();
    ni::Workspace workspace;
    p.self_mobility=ns::build_self_mobility(geometry.bodies,workspace);
    bs::Contact c;c.i=0;c.j=1;c.gap=gap.gap;c.normal=V3::UnitX();
    c.kn=stiffness/4e-6;c.kt=c.kr=.8*c.kn;c.mu=c.mu_roll=mode==3?.01:1e8;c.roll_length=1.;
    c.Bi.leftCols<3>()=-M3::Identity();c.Bj.leftCols<3>()=M3::Identity();
    const double lever=a-.0005;
    M3 cross;cross<<0.,0.,0.,0.,0.,-lever,0.,lever,0.;
    c.Bi.rightCols<3>()=cross;c.Bj.rightCols<3>()=cross;
    if(mode!=1)c.history=1e-4*V3::UnitY();
    if(mode!=0)c.rolling_history=1e-4*V3::UnitZ();
    const double preload=-c.kn*c.gap;
    p.external.segment<3>(0)=preload*c.normal;
    p.external.segment<3>(6)=-preload*c.normal;
    p.contacts.push_back(c);
    const double initial_energy=.5*c.kt*c.history.squaredNorm()+.5*c.kr*c.rolling_history.squaredNorm();
    bs::Options options;options.relative_tolerance=1e-9;
    imex::Workspace imex_workspace;
    imex_workspace.use_contact_preconditioner=true;
    const auto caps=imex::freeze_caps(p);
    const auto accepted=scheme ? SlurryNearfieldStep::solve_imex(p,caps,geometry,options,imex_workspace)
                               : SlurryNearfieldStep::solve(p,geometry,options,workspace);
    const auto &response=accepted.solution.contacts[0];
    const double analytic_gap=aspect==1. ?
        (step::vector(accepted.endpoint.bodies[1].position)-
         step::vector(accepted.endpoint.bodies[0].position)).norm()-2. :
        gr::closestEllipsoidGap(accepted.endpoint.bodies[0],accepted.endpoint.bodies[1]).gap;
    require(std::abs(response.predicted_gap-analytic_gap)<2e-12,
            "BE accepted load does not use the actual spherical endpoint gap");
    require(std::abs(response.normal_load-c.kn*std::max(0.,-analytic_gap))<2e-6,
            "BE normal penalty differs from actual endpoint compression");
    require(response.tangent_energy+response.rolling_energy<initial_energy,
            "BE finite-prestress elastic energy increased at the requested timestep");
    require(accepted.solution.relative_residual<=options.relative_tolerance+1e-12,
            "BE endpoint failed original physical force/mobility tolerance");
    require(accepted.geometry_iterations>=1&&accepted.geometry_iterations<=24,
            "BE endpoint iteration bound was not respected");
    require((p.contacts[0].history-c.history).norm()==0.&&
            (p.contacts[0].rolling_history-c.rolling_history).norm()==0.,
            "BE trial solve mutated committed history");
    // The accepted history is one BE update, never an extra explicit advance.
    const auto law=scheme ? imex::contact_response(c,caps[0],accepted.solution.velocity,p.dt,analytic_gap)
                          : bs::contact_response(c,accepted.solution.velocity,p.dt,analytic_gap);
    if(scheme&&mode==3&&stiffness==3.&&aspect==1.) {
      require(response.sliding&&response.rolling,"IMEX cap regression did not reach sliding/rolling");
      require(std::abs(response.tangent_force.norm()-c.mu*caps[0])<1e-10&&
              std::abs(response.torque.norm()-c.mu_roll*c.roll_length*caps[0])<1e-10,
              "IMEX endpoint correction changed frozen original friction caps");
      require(accepted.geometry_iterations>1,"IMEX cap regression did not exercise geometry correction");
    }
    require((response.tangent_history-law.tangent_history).norm()<1e-12&&
            (response.rolling_history-law.rolling_history).norm()<1e-12,
            "BE history was advanced more than once");
    if(scheme&&stiffness==300.&&aspect==1.&&mode==0) {
      imex::Workspace old_workspace;
      old_workspace.use_contact_preconditioner=true;
      old_workspace.refine_damped_line_search=false;
      const auto old=SlurryNearfieldStep::solve_imex(p,caps,geometry,options,old_workspace);
      require(imex_workspace.damped_line_refinements==0,
              "smooth preload case unexpectedly needed damped-step refinement");
      require((accepted.solution.velocity-old.solution.velocity).norm()<1e-13,
              "damped-step safeguard changed the ordinary full-step solution");
    }
  }
  std::cout<<"nearfield coupled and frozen-cap IMEX endpoint/prestress/history checks passed; physical kn=3,30,100,300 N/m, AR=1,5, dt=2.5us\n";
}
