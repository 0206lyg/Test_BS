#include "slurry_nearfield_imex_solver.h"
#include "slurry_nearfield_hydro.h"
#include "slurry_compact_bridge.h"
#include <iostream>
#include <iomanip>

namespace bs=SlurryBridgeSolver;
namespace ns=SlurryNearfieldSolver;
namespace ni=SlurryNearfieldImexSolver;
namespace nh=SlurryNearfieldHydro;
namespace cb=SlurryCompactBridge;
namespace gr=slurry::gr_re2::graphite;
using V3=Eigen::Vector3d;using M3=Eigen::Matrix3d;
using V5=Eigen::Matrix<double,5,1>;using M5=Eigen::Matrix<double,5,5>;
using G5=Eigen::Matrix<double,5,12>;
void require(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
gr::Vec3 arr(const V3&v){return {v[0],v[1],v[2]};}
V3 vec(const gr::Vec3&v){return {v[0],v[1],v[2]};}
bs::Options accuracy(){bs::Options o;o.relative_tolerance=1e-11;o.absolute_tolerance=1e-14;return o;}
bs::Problem simple(){
  bs::Problem p;p.count=2;p.dt=.02;p.prescribed=bs::Vector::Zero(22);
  p.external=bs::Vector::Zero(12);p.near_resistance.resize(22,22);
  p.self_mobility.assign(2,bs::Matrix11::Identity());return p;
}

void frozen_caps_and_convex_jacobian(){
  auto p=simple();bs::Contact c;c.i=0;c.j=1;c.normal=V3::UnitZ();c.gap=-.02;
  c.kn=500.;c.kt=300.;c.kr=100.;c.mu=.2;c.mu_roll=.3;c.roll_length=.7;
  c.history=V3(.2,.1,0.);c.rolling_history=V3(-.15,.08,0.);
  c.Bi.leftCols<3>()=-M3::Identity();c.Bj.leftCols<3>()=M3::Identity();
  p.contacts={c};const auto caps=ni::freeze_caps(p);ni::Workspace workspace;
  const auto result=ni::solve(p,caps,accuracy(),workspace);
  const double fn=-c.kn*c.gap/(1+2*p.dt*c.kn);
  const V3 tangent=c.mu*caps[0]*c.history.normalized();
  const V3 rolling=c.mu_roll*caps[0]*c.roll_length*c.rolling_history.normalized();
  const V3 traction=tangent-fn*c.normal;
  require(result.contacts[0].active&&result.contacts[0].sliding&&result.contacts[0].rolling,
          "frozen-cap reference did not exercise both Coulomb caps");
  require(std::abs(result.contacts[0].normal_load-fn)<1e-10&&
          (result.contacts[0].traction-traction).norm()<1e-10&&
          (result.contacts[0].torque-rolling).norm()<1e-10,
          "frozen-cap loads differ from independent closed form");
  require((result.velocity.head<3>()-traction).norm()<1e-10&&
          (result.velocity.segment<3>(6)+traction).norm()<1e-10&&
          (result.velocity.segment<3>(3)-rolling).norm()<1e-10&&
          (result.velocity.segment<3>(9)+rolling).norm()<1e-10,
          "frozen-cap force or couple sign is incorrect");
  ni::Workspace direct_workspace;direct_workspace.use_krylov=false;
  const auto direct=ni::solve(p,caps,accuracy(),direct_workspace);
  require((direct.velocity-result.velocity).norm()<1e-9&&direct.linear_iterations==0,
          "sparse Krylov changed the independently solved contact root");
  ni::Workspace contact_workspace;contact_workspace.use_contact_preconditioner=true;
  const auto preconditioned=ni::solve(p,caps,accuracy(),contact_workspace);
  require((preconditioned.velocity-result.velocity).norm()<1e-9&&
          contact_workspace.contact_preconditioner_refreshes>0&&
          contact_workspace.contact_preconditioner_solves>0&&
          contact_workspace.maximum_true_relative_linear_residual<=contact_workspace.linear_relative_target,
          "contact preconditioner changed the solved root or bypassed the true residual gate");
  require(workspace.maximum_true_relative_linear_residual<=workspace.linear_relative_target,
          "accepted sparse Newton direction failed true residual check");
  auto point=result.velocity;point[0]+=.12;point[4]-=.04;
  const auto at=ni::detail::evaluate(p,caps,point,workspace);
  const Eigen::MatrixXd analytic=ni::detail::jacobian(p,at,workspace);
  Eigen::MatrixXd numerical(12,12);const double epsilon=1e-7;
  for(int j=0;j<12;++j){auto plus=point,minus=point;plus[j]+=epsilon;minus[j]-=epsilon;
    numerical.col(j)=(ni::detail::evaluate(p,caps,plus,workspace).residual-
                     ni::detail::evaluate(p,caps,minus,workspace).residual)/(2*epsilon);
    bs::Vector du=bs::Vector::Zero(12);du[j]=epsilon;
    const double slope=(ni::detail::objective_increment(p,caps,at,du,workspace)-
                        ni::detail::objective_increment(p,caps,at,-du,workspace))/(2*epsilon);
    require(std::abs(slope-at.residual[j])<1e-7,"convex potential gradient differs from physical residual");
  }
  require((analytic-numerical).norm()<2e-7*analytic.norm(),
          "frozen-cap Jacobian differs from independent finite differences");
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(analytic);
  require((analytic-analytic.transpose()).norm()<1e-12&&eigen.eigenvalues().minCoeff()>.99999,
          "fixed-cap Jacobian is not symmetric positive definite");
  auto majorizer_problem=p;auto&mc=majorizer_problem.contacts[0];
  mc.kw=90.;mc.mu_twist=.2;mc.twist_length=.9;mc.twisting_history=.1;
  const auto majorizer_at=ni::detail::evaluate(majorizer_problem,caps,point,workspace);
  const auto&mr=majorizer_at.state.contacts[0];
  const Eigen::MatrixXd majorizer=ni::detail::jacobian(majorizer_problem,majorizer_at,workspace,true);
  const Eigen::MatrixXd exact_hessian=ni::detail::jacobian(majorizer_problem,majorizer_at,workspace);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> added(majorizer-exact_hessian);
  require(added.eigenvalues().minCoeff()>-1e-10&&(majorizer-exact_hessian).norm()>0.,
          "Huber globalization did not add positive semidefinite radial curvature");
  // Transverse displacements keep the normal hinge unchanged. The Huber
  // quadratic must bound the ORIGINAL potential across sliding, rolling and
  // twisting yield radii, including cancellation of each old trial history.
  for(double ratio:{-2.,-.5,0.,.5,2.}) {
    bs::Vector du=bs::Vector::Zero(12);
    const V3 tangent_target=ratio*mc.mu*caps[0]/mc.kt*V3::UnitX();
    const V3 rolling_target=ratio*mc.mu_roll*mc.roll_length*caps[0]/mc.kr*V3::UnitY();
    const double twisting_target=ratio*mc.mu_twist*mc.twist_length*caps[0]/mc.kw;
    du.segment<3>(6)=(tangent_target-mr.trial_history)/p.dt;
    du.segment<3>(9)=(rolling_target-mr.rolling_trial)/p.dt+
                    (twisting_target-mr.twisting_trial)/p.dt*mc.normal;
    const double exact_change=ni::detail::objective_increment(majorizer_problem,caps,majorizer_at,du,workspace);
    const double bound=majorizer_at.residual.dot(du)+.5*du.dot(majorizer*du);
    require(exact_change<=bound+1e-10*std::max(1.,std::abs(bound)),
            "Huber direction model does not majorize the unchanged potential");
  }
  const auto metric=ni::acceptance_metric(p,caps,result,result.contacts,workspace,accuracy());
  require(metric.residual_norm<=metric.tolerance&&result.mobility_applications==0,
          "IMEX stage bypassed physical residual gate or used a periodic mobility");
  auto wrong=result.contacts;wrong[0].traction[0]+=.01;
  require(ni::acceptance_metric(p,caps,result,wrong,workspace,accuracy()).residual_norm>metric.tolerance,
          "physical gate accepted a post-clipped force inconsistent with solved moments");
  const auto&r=result.contacts[0];
  const double initial=.5*c.kn*c.gap*c.gap+.5*c.kt*c.history.squaredNorm()+.5*c.kr*c.rolling_history.squaredNorm();
  const double final=r.normal_energy+r.tangent_energy+r.rolling_energy;
  require(final+p.dt*result.hydro_power<=initial+1e-10,"unforced frozen-cap step created energy");
  auto corrected=p;corrected.contacts[0].gap*=.2;
  const auto original_response=ni::contact_response(c,caps[0],point,p.dt);
  const auto shifted_response=ni::contact_response(corrected.contacts[0],caps[0],point,p.dt);
  require((original_response.tangent_force-shifted_response.tangent_force).norm()==0.&&
          (original_response.torque-shifted_response.torque).norm()==0.,
          "artificial endpoint gap correction changed the original friction caps");
  // Opening releases histories only during commit, not while solving interval forces.
  p.external[2]=-20.;p.external[8]=20.;
  const auto opened=ni::solve(p,caps,accuracy(),workspace,&result);
  require(!opened.contacts[0].active&&opened.contacts[0].tangent_history.norm()>0.&&
          opened.contacts[0].rolling_history.norm()>0.,"opening erased balanced interval friction forces");
  require(ni::acceptance_metric(p,caps,opened,opened.contacts,workspace,accuracy()).residual_norm<=
          ni::acceptance_metric(p,caps,opened,opened.contacts,workspace,accuracy()).tolerance,
          "opening contact violated interval force balance");
  ni::Workspace fallback_workspace;fallback_workspace.krylov_limit=1;
  auto mixed_seed=result;mixed_seed.velocity[0]+=.1;mixed_seed.velocity[1]-=.2;
  mixed_seed.velocity[3]+=.11;mixed_seed.velocity[4]+=.23;
  const auto fallback=ni::solve(p,caps,accuracy(),fallback_workspace,&mixed_seed);
  require(fallback_workspace.krylov_fallbacks>0&&fallback_workspace.newton_numeric_factorizations>0&&
          (fallback.velocity-opened.velocity).norm()<1e-9,
          "bounded sparse Krylov failure did not recover through SPD direct solve");
  auto newly=p;newly.contacts[0].gap=.01;newly.external.setZero();newly.external[2]=1.;newly.external[8]=-1.;
  const auto newcaps=ni::freeze_caps(newly);ni::Workspace new_workspace;
  const auto closing=ni::solve(newly,newcaps,accuracy(),new_workspace);
  require(newcaps[0]==0.&&closing.contacts[0].active&&closing.contacts[0].tangent_force.isZero(0.)&&
          closing.contacts[0].torque.isZero(0.),"new contact used an endpoint friction activation mask");
  std::cout<<"frozen caps, potential gradient, SPD Jacobian, opening, new contact and passivity passed; Newton="
           <<result.newton_iterations<<" relative_residual="<<result.relative_residual<<'\n';
}

void damped_huber_globalization(){
  auto p=simple();bs::Contact c;c.i=0;c.j=1;c.normal=V3::UnitZ();c.gap=-.02;
  c.kn=500.;c.kt=300.;c.kr=100.;c.kw=80.;
  c.mu=.2;c.mu_roll=.3;c.mu_twist=.15;c.roll_length=.7;c.twist_length=.9;
  c.history=.04*V3::UnitX();c.rolling_history=.04*V3::UnitY();c.twisting_history=.04;
  c.Bi.leftCols<3>()=-M3::Identity();c.Bj.leftCols<3>()=M3::Identity();p.contacts={c};
  bs::Result warm;warm.velocity=bs::Vector::Zero(12);
  warm.velocity[0]=warm.velocity[4]=warm.velocity[5]=100.;
  warm.velocity[6]=warm.velocity[10]=warm.velocity[11]=-100.;
  ni::Workspace workspace;workspace.use_contact_preconditioner=true;
  auto options=accuracy();bool traced_majorizer=false;
  options.trace_callback=[&](const bs::IterationTrace&t){
    traced_majorizer=traced_majorizer||std::string(t.phase)=="nearfield_imex_huber_majorized";
  };
  const auto caps=ni::freeze_caps(p);
  const auto result=ni::solve(p,caps,options,workspace,&warm);
  auto capped=[](double stiffness,double history,double dt,double cap){
    return std::clamp(stiffness*history/(1+2*stiffness*dt),-cap,cap);
  };
  const double normal=-c.kn*c.gap/(1+2*c.kn*p.dt);
  const V3 traction=capped(c.kt,.04,p.dt,c.mu*caps[0])*V3::UnitX()-normal*c.normal;
  const V3 moment=capped(c.kr,.04,p.dt,c.mu_roll*c.roll_length*caps[0])*V3::UnitY()+
                  capped(c.kw,.04,p.dt,c.mu_twist*c.twist_length*caps[0])*c.normal;
  require(workspace.huber_majorized_directions>0&&traced_majorizer,
          "damped contact regression did not exercise Huber globalization");
  require((result.velocity.head<3>()-traction).norm()<1e-9&&
          (result.velocity.segment<3>(6)+traction).norm()<1e-9&&
          (result.velocity.segment<3>(3)-moment).norm()<1e-9&&
          (result.velocity.segment<3>(9)+moment).norm()<1e-9,
          "Huber globalization changed the independent closed-form root");
  ni::Workspace reference_workspace;reference_workspace.use_huber_globalization=false;
  reference_workspace.maximum_recovery_iterations=0;
  options.trace_callback={};options.maximum_newton_iterations=200;
  const auto reference=ni::solve(p,caps,options,reference_workspace,&warm);
  require((reference.velocity-result.velocity).norm()<1e-9&&
          reference_workspace.huber_majorized_directions==0,
          "globalized and exact-Newton references solve different equations");
  std::cout<<"damped Huber globalization majorized="<<workspace.huber_majorized_directions
           <<" Newton="<<result.newton_iterations<<" reference_Newton="<<reference.newton_iterations<<'\n';
  // Exercise the rare-step recovery without weakening the physical equation:
  // use a deliberately tiny primary budget on this independent analytic root.
  options.maximum_newton_iterations=1;bool traced_recovery=false,entry_majorized=false;
  options.trace_callback=[&](const bs::IterationTrace&t){
    traced_recovery=traced_recovery||std::string(t.phase).find("nearfield_imex_recovery")==0;
    if(t.iteration==options.maximum_newton_iterations+1)
      entry_majorized=std::string(t.phase)=="nearfield_imex_recovery_majorized";
  };
  ni::Workspace recovery_workspace;recovery_workspace.use_contact_preconditioner=true;
  recovery_workspace.use_huber_globalization=false;
  const auto recovered=ni::solve(p,caps,options,recovery_workspace,&warm);
  require(recovery_workspace.nonlinear_recoveries==1&&
          recovery_workspace.nonlinear_recovery_iterations>0&&traced_recovery&&entry_majorized&&
          recovery_workspace.huber_majorized_directions==1&&
          (recovered.velocity-result.velocity).norm()<1e-9&&
          recovered.relative_residual<=options.relative_tolerance,
          "bounded recovery failed to preserve the independently known root and tolerance");
  for(int budget:{0,1}) {
    ni::Workspace limited;limited.maximum_recovery_iterations=budget;
    bool stopped=false;
    try{ni::solve(p,caps,options,limited,&warm);}
    catch(const std::runtime_error&e){const std::string message=e.what();
      stopped=message.find("IMEX: convex Newton limit;")==0&&
              message.find("; iterations="+std::to_string(1+budget)+";")!=std::string::npos;
    }
    require(stopped&&limited.nonlinear_recoveries==std::size_t(budget)&&
            limited.nonlinear_recovery_iterations==std::size_t(budget),
            "disabled or exhausted recovery bypassed its bounded iteration limit");
  }
  ni::Workspace invalid;invalid.maximum_recovery_iterations=-1;bool rejected=false;
  try{ni::solve(p,caps,options,invalid,&warm);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"negative nonlinear recovery budget was accepted");
  std::cout<<"bounded recovery additional_iterations="<<recovery_workspace.nonlinear_recovery_iterations
           <<"; disabled, exhausted and negative budgets checked\n";
}

void cap_lag_convergence(){
  auto error=[](int steps){
    auto p=simple();p.dt=.01/steps;bs::Contact c;c.i=0;c.j=1;c.normal=V3::UnitZ();
    c.gap=-.02;c.kn=50.;c.kt=5.;c.mu=.2;c.history=V3(1.,0,0);
    c.Bi.leftCols<3>()=-M3::Identity();c.Bj.leftCols<3>()=M3::Identity();p.contacts={c};
    ni::Workspace workspace;double tangential_displacement=0.;
    for(int step=0;step<steps;++step){
      const auto caps=ni::freeze_caps(p);const auto result=ni::solve(p,caps,accuracy(),workspace);
      const auto&r=result.contacts[0];require(r.sliding,"cap lag convergence lost its known sliding branch");
      tangential_displacement+=p.dt*r.relative[0];
      p.contacts[0].gap=r.predicted_gap;p.contacts[0].history=r.tangent_history;
    }
    const double exact=-c.mu*(-c.gap)*(1.-std::exp(-2*c.kn*.01));
    return std::abs(tangential_displacement-exact);
  };
  const double e20=error(20),e40=error(40),e80=error(80);
  require(e20/e40>1.8&&e20/e40<2.2&&e40/e80>1.8&&e40/e80<2.2,
          "current-load Coulomb cap lag failed first-order convergence to the continuous sliding ODE");
  std::cout<<"cap-lag first-order ratios="<<e20/e40<<","<<e40/e80<<'\n';
}

struct Relaxation {
  bs::Problem p;bs::Contact base;ni::Workspace workspace;
  G5 map;V5 stiffness,initial;V3 t1,t2;double lambda=0.;
  Relaxation(double ar,bool edge,bool contact_preconditioner=false){
    workspace.use_contact_preconditioner=contact_preconditioner;
    p=simple();p.dt=100*2.5e-6;
    const double a=std::cbrt(ar),c=a/ar;
    std::vector<gr::Body>b(2);
    for(auto&body:b){body.axes={a,a,c};body.rotation={1,0,0,0,1,0,0,0,1};}
    b[0].position={0,0,0};b[1].position=edge?gr::Vec3{2*a-2.5e-5,0,0}:gr::Vec3{0,0,2*c-2.5e-5};
    const auto gap=gr::closestEllipsoidGap(b[0],b[1]);
    p.near_resistance=nh::evaluate(b[0],b[1],gap,1.,1.,.001,.5).resistance.sparseView();
    p.self_mobility=ns::build_self_mobility(b,workspace);
    auto&x=base;x.i=0;x.j=1;x.normal=vec(gap.normal);x.gap=gap.gap;
    // eta=.01 Pa s,L=4um,rate=100/s: kn=3,kt=kr=2.4 N/m.
    x.kn=750000.;x.kt=600000.;x.kr=600000.;x.mu=x.mu_roll=10.;x.roll_length=1.;
    const V3 d=vec(b[1].position)-vec(b[0].position);
    const V3 li=.5*(vec(gap.leverI)+d+vec(gap.leverJ)),lj=li-d;
    x.Bi.leftCols<3>()=-M3::Identity();x.Bi.rightCols<3>()=cb::cross_matrix(li);
    x.Bj.leftCols<3>()=M3::Identity();x.Bj.rightCols<3>()=-cb::cross_matrix(lj);
    const auto preload=bs::contact_response(x,bs::Vector::Zero(12),0.);
    p.external.segment<6>(0)=bs::generalized_contact_map(x,true).transpose()*bs::contact_wrench(preload);
    p.external.segment<6>(6)=bs::generalized_contact_map(x,false).transpose()*bs::contact_wrench(preload);
    p.contacts={x};ns::solve(p,accuracy(),workspace);
    t1=x.normal.unitOrthogonal();t2=x.normal.cross(t1);map.setZero();
    for(int k=0;k<3;++k){const V3 axis=k==0?t1:k==1?t2:x.normal;
      map.block<1,6>(k,0)=axis.transpose()*x.Bi;map.block<1,6>(k,6)=axis.transpose()*x.Bj;}
    map.block<1,3>(3,3)=-t1.transpose();map.block<1,3>(3,9)=t1.transpose();
    map.block<1,3>(4,3)=-t2.transpose();map.block<1,3>(4,9)=t2.transpose();
    stiffness<<x.kt,x.kt,x.kn,x.kr,x.kr;
    const M5 rate=stiffness.cwiseSqrt().asDiagonal()*map*workspace.factor.solve(map.transpose())*stiffness.cwiseSqrt().asDiagonal();
    Eigen::SelfAdjointEigenSolver<M5> eigen(.5*(rate+rate.transpose()));lambda=eigen.eigenvalues().maxCoeff();
    initial=stiffness.cwiseSqrt().cwiseInverse().asDiagonal()*eigen.eigenvectors().col(4);
    initial*=1e-8/initial.cwiseAbs().maxCoeff();
  }
  double energy(const V5&h){return .5*(stiffness.array()*h.array().square()).sum();}
  V5 advance(const V5&h,double physical_dt,int*iterations=nullptr){
    p.dt=100*physical_dt;p.contacts[0]=base;
    p.contacts[0].gap=base.gap+h[2];p.contacts[0].history=t1*h[0]+t2*h[1];
    p.contacts[0].rolling_history=t1*h[3]+t2*h[4];
    const auto result=ni::solve(p,ni::freeze_caps(p),accuracy(),workspace);
    if(iterations)*iterations=result.newton_iterations;
    const auto&r=result.contacts[0];
    require(r.active&&!r.sliding&&!r.rolling,"finite relaxation unexpectedly hit a Coulomb cap");
    V5 next;next<<t1.dot(r.tangent_history),t2.dot(r.tangent_history),r.predicted_gap-base.gap,
                   t1.dot(r.rolling_history),t2.dot(r.rolling_history);
    return next;
  }
  double error(int steps){
    const double duration=2./(100*lambda),dt=duration/steps;V5 h=initial;
    for(int i=0;i<steps;++i)h=advance(h,dt);
    return (h-initial*std::exp(-2.)).norm()/initial.norm();
  }
};

void finite_prestress_decay_and_convergence(){
  for(bool contact_preconditioner:{false,true})for(double ar:{1.,2.,5.})for(bool edge:{false,true}){
    Relaxation test(ar,edge,contact_preconditioner);const double z=test.lambda*100*2.5e-6;
    int iterations=0;const V5 next=test.advance(test.initial,2.5e-6,&iterations);
    const double ratio=test.energy(next)/test.energy(test.initial);
    const double expected=1./((1+z)*(1+z));
    require(ratio<1.&&(next-test.initial/(1+z)).norm()<2e-6*test.initial.norm(),
            "finite-prestress implicit update failed exact backward-Euler relaxation");
    const double coarse=test.error(20),fine=test.error(40);
    require(coarse/fine>1.8&&coarse/fine<2.2,"implicit relaxation failed first-order exact-ODE convergence");
    std::cout<<"preconditioner="<<(contact_preconditioner?"contact":"hydro")<<" AR="<<ar<<" placement="<<(edge?"edge":"face")<<" dt_lambda="<<z
      <<" Heun_energy_factor="<<std::pow(1-z+.5*z*z,2)
      <<" BE_energy_factor="<<ratio<<" expected="<<expected<<" Newton="<<iterations
      <<" convergence_ratio="<<coarse/fine<<'\n';
  }
}
int main(){std::cout<<std::setprecision(12);frozen_caps_and_convex_jacobian();damped_huber_globalization();cap_lag_convergence();
  finite_prestress_decay_and_convergence();std::cout<<"nearfield frozen-cap IMEX checks passed\n";}
