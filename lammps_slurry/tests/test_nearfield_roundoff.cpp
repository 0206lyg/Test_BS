#include "slurry_nearfield_step.h"
#include "slurry_nearfield_hydro.h"
#include "slurry_surface_cohesion.h"
#include "slurry_compact_bridge.h"
#include <iostream>
#include <iomanip>

namespace bs=SlurryBridgeSolver;
namespace ns=SlurryNearfieldSolver;
namespace im=SlurryNearfieldImexSolver;
namespace st=SlurryBridgeStep;
namespace nh=SlurryNearfieldHydro;
namespace co=SlurrySurfaceCohesion;
namespace cb=SlurryCompactBridge;
namespace gr=slurry::gr_re2::graphite;
using V=Eigen::Vector3d;
using M=Eigen::Matrix3d;
static V vec(const gr::Vec3& a){return {a[0],a[1],a[2]};}
static gr::Vec3 arr(const V& a){return {a[0],a[1],a[2]};}
static void require(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}

int main() {
  try {
    constexpr double length=4e-6,eta=.00089,rate=100.,physical_dt=1e-6;
    const double stiffness_scale=eta*rate*length,force_scale=stiffness_scale*length;
    const double major=std::cbrt(4.),minor=1./(major*major);
    const co::Parameters adhesion{.021243/stiffness_scale,.67e-9/length};
    std::cout<<std::setprecision(12);
    for(double kn:{300.,900.})for(int placement=0;placement<3;++placement) {
      st::Geometry geometry;geometry.bodies.resize(2);
      geometry.quaternions.assign(2,Eigen::Quaterniond::Identity());
      geometry.lattice=1e4*M::Identity();geometry.images.emplace_back(0,0,0);
      for(auto& b:geometry.bodies){b.axes={major,major,minor};b.rotation=gr::identity3;}
      V normal=placement==2?V::UnitX():V::UnitZ();
      double touching=placement==2?2.*major:2.*minor;
      if(placement==1) {
        geometry.quaternions[1]=Eigen::Quaterniond(Eigen::AngleAxisd(std::acos(-1.)/2,V::UnitY()));
        const M rotation=geometry.quaternions[1].toRotationMatrix();
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)geometry.bodies[1].rotation[3*i+j]=rotation(i,j);
        touching=major+minor;
      }
      geometry.bodies[1].position=arr(touching*normal);
      const double f0=co::evaluate(geometry.bodies[0],geometry.bodies[1],adhesion).attraction_at_contact;
      geometry.bodies[1].position=arr((touching+.5*adhesion.opening_range)*normal);
      im::Workspace workspace;workspace.use_roundoff_budget=true;
      workspace.defer_candidate_validation=true;
      workspace.use_contact_preconditioner=true;
      bs::Options options;options.relative_tolerance=1e-6;options.absolute_tolerance=1e-11;
      bs::Result warm;double max_force_error=0.,last_gap=0.,last_budget=0.;
      for(int iteration=0;iteration<80;++iteration) {
        const auto gap=gr::closestEllipsoidGap(geometry.bodies[0],geometry.bodies[1]);
        const auto attraction=co::evaluate(geometry.bodies[0],geometry.bodies[1],gap,adhesion);
        const auto lengths=nh::pair_lengths(1.,1.,.001);
        const auto resistance=nh::evaluate(geometry.bodies[0],geometry.bodies[1],gap,1.,1.,lengths.epsilon,lengths.cutoff);
        bs::Problem p;p.count=2;p.dt=physical_dt*rate;p.prescribed=bs::Vector::Zero(22);
        p.external=bs::Vector::Zero(12);p.near_resistance=resistance.resistance.sparseView();
        p.self_mobility=ns::build_self_mobility(geometry.bodies,workspace);
        p.external.segment<3>(0)=vec(attraction.force_i);p.external.segment<3>(3)=vec(attraction.torque_i);
        p.external.segment<3>(6)=vec(attraction.force_j);p.external.segment<3>(9)=vec(attraction.torque_j);
        bs::Contact c;c.i=0;c.j=1;c.gap=gap.gap;c.normal=vec(gap.normal);c.kn=kn/stiffness_scale;
        const V d=vec(geometry.bodies[1].position)-vec(geometry.bodies[0].position);
        const V li=.5*(vec(gap.leverI)+d+vec(gap.leverJ)),lj=li-d;
        c.Bi.leftCols<3>()=-M::Identity();c.Bi.rightCols<3>()=cb::cross_matrix(li);
        c.Bj.leftCols<3>()=M::Identity();c.Bj.rightCols<3>()=-cb::cross_matrix(lj);p.contacts={c};
        const auto caps=im::freeze_caps(p);
        auto result=SlurryNearfieldStep::solve_imex(p,caps,geometry,options,workspace,iteration?&warm:nullptr);
        if(iteration==0) {
          im::Workspace reference;reference.use_roundoff_budget=true;
          reference.use_contact_preconditioner=true;
          const auto original=SlurryNearfieldStep::solve_imex(p,caps,geometry,options,reference);
          require((original.solution.velocity.array()==result.solution.velocity.array()).all(),
                  "deferred candidate check changed accepted velocity");
          require(original.endpoint.gaps==result.endpoint.gaps,
                  "deferred candidate check changed accepted gaps");
          require(original.geometry_iterations==result.geometry_iterations,
                  "deferred candidate check changed endpoint iterations");
        }
        const auto metric=im::acceptance_metric(p,caps,result.solution,result.solution.contacts,workspace,options);
        require(metric.residual_norm<=metric.tolerance,"accepted residual exceeds explicitly reported budget");
        require(metric.roundoff_tolerance>=0.&&metric.roundoff_tolerance<1e-2,"roundoff allowance is not microscopic");
        max_force_error=std::max(max_force_error,metric.force_norm*force_scale);
        last_gap=result.endpoint.gaps[0];last_budget=metric.roundoff_tolerance;
        if(iteration==79) {
          require(std::abs(last_gap+f0/c.kn)<1e-8*std::abs(f0/c.kn)+1e-12,
                  "roundoff-aware static preload is not F0/kn");
          // A real force error must still fail even with the budget enabled.
          auto bad=result.solution;bad.total_moment[0]+=1e-4*f0;
          const auto rejected=im::acceptance_metric(p,caps,bad,bad.contacts,workspace,options);
          require(rejected.residual_norm>100.*rejected.tolerance,"physical residual was accepted as roundoff");
          workspace.use_roundoff_budget=false;
          const auto disabled=im::acceptance_metric(p,caps,result.solution,result.solution.contacts,workspace,options);
          require(disabled.roundoff_tolerance==0.&&disabled.tolerance==options.absolute_tolerance+
                  options.relative_tolerance*disabled.rhs_norm,"disabled budget changed original tolerance");
          workspace.use_roundoff_budget=true;
        }
        for(int i=0;i<2;++i)geometry.quaternions[i]=st::rotate(geometry.quaternions[i],
            result.solution.velocity.segment<3>(6*i+3),p.dt);
        geometry.bodies=result.endpoint.bodies;geometry.lattice=result.endpoint.lattice;warm=std::move(result.solution);
      }
      std::cout<<"kn="<<kn<<" placement="<<placement<<" equilibrium_overlap_nm="<<-last_gap*length*1e9
               <<" max_raw_force_residual_N="<<max_force_error<<" last_mixed_roundoff_budget="<<last_budget
               <<" roundoff_acceptances="<<workspace.roundoff_acceptances<<'\n';
    }
    // Candidate validation remains mandatory before commit, even if the old
    // listed-contact solve itself has no residual at all.
    for(bool deferred:{false,true}) {
      st::Geometry geometry;geometry.bodies.resize(2);
      geometry.quaternions.assign(2,Eigen::Quaterniond::Identity());geometry.lattice=100.*M::Identity();
      for(auto& b:geometry.bodies){b.axes={1.,1.,1.};b.rotation=gr::identity3;}
      geometry.bodies[1].position={3.,0.,0.};
      bs::Problem p;p.count=2;p.dt=1.;p.prescribed=bs::Vector::Zero(22);p.external=bs::Vector::Zero(12);
      p.external[0]=1.;p.external[6]=-1.;p.self_mobility.assign(2,bs::Matrix11::Identity());
      p.near_resistance.resize(22,22);p.near_resistance.setZero();
      im::Workspace workspace;workspace.defer_candidate_validation=deferred;
      bool rejected=false;
      try {SlurryNearfieldStep::solve_imex(p,{},geometry,bs::Options{},workspace);}
      catch(const std::runtime_error& e){rejected=std::string(e.what()).find("unlisted contact")!=std::string::npos;}
      require(rejected,"unlisted new contact escaped endpoint validation");
    }
    std::cout<<"nearfield roundoff budget PASS\n";return 0;
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
