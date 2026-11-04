#include "slurry_nearfield_step.h"
#include <iostream>
#include <limits>
#include <stdexcept>
namespace bs=SlurryBridgeSolver;
namespace imex=SlurryNearfieldImexSolver;
namespace ns=SlurryNearfieldStep;
namespace st=SlurryBridgeStep;
namespace gr=slurry::gr_re2::graphite;
using V=Eigen::Vector3d;using M=Eigen::Matrix3d;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Fixture {bs::Problem p;st::Geometry g;};
static Fixture symmetric_images() {
  Fixture f;auto&p=f.p;auto&g=f.g;
  p.count=2;p.dt=.1;p.prescribed=bs::Vector::Zero(22);p.external=bs::Vector::Zero(12);
  p.near_resistance.resize(22,22);p.self_mobility.assign(2,bs::Matrix11::Identity());
  g.bodies.resize(2);g.quaternions.assign(2,Eigen::Quaterniond::Identity());
  for(auto&b:g.bodies){b.axes={1.,1.,1.};b.rotation=gr::identity3;}
  g.bodies[1].position={1.99,0.,0.};g.lattice=20.*M::Identity();g.lattice(0,0)=3.98;
  g.kappa(1,0)=1.;
  // Two physical periodic images are opposite and equally loaded. Their
  // exact endpoint loads differ from the linear predictor by the same amount,
  // while their resultant forces cancel. This separates a pairwise iterate
  // match from global force balance without changing either force law.
  for(int sign:{1,-1}) {
    bs::Contact c;c.i=0;c.j=1;c.normal=sign*V::UnitX();c.gap=-.01;c.kn=100.;
    c.Bi.leftCols<3>()=-M::Identity();c.Bj.leftCols<3>()=M::Identity();
    c.affine=sign*1.99*V::UnitY();p.contacts.push_back(c);
    g.images.emplace_back(sign==1?0:-1,0,0);
  }
  return f;
}
int main(){try{
  auto f=symmetric_images();const auto caps=imex::freeze_caps(f.p);
  bs::Options options;options.relative_tolerance=1e-8;options.absolute_tolerance=1e-12;
  imex::Workspace strict,explicit_defaults,practical;
  require(strict.endpoint_inner_tolerance_fraction==.1&&strict.require_endpoint_constitutive_match,
          "endpoint defaults changed");
  explicit_defaults.endpoint_inner_tolerance_fraction=.1;
  explicit_defaults.require_endpoint_constitutive_match=true;
  const auto a=ns::solve_imex(f.p,caps,f.g,options,strict);
  const auto b=ns::solve_imex(f.p,caps,f.g,options,explicit_defaults);
  require((a.solution.velocity.array()==b.solution.velocity.array()).all()&&
          a.endpoint.gaps==b.endpoint.gaps&&a.geometry_iterations==b.geometry_iterations,
          "explicit historical controls differ from defaults");
  practical.endpoint_inner_tolerance_fraction=1.;practical.require_endpoint_constitutive_match=false;
  const auto c=ns::solve_imex(f.p,caps,f.g,options,practical);
  require(a.geometry_iterations>1&&c.geometry_iterations==1,
          "symmetric-image fixture did not separate the optional pairwise gate");
  require((a.solution.velocity.array()==c.solution.velocity.array()).all(),
          "balanced symmetric images changed velocity");
  const auto metric=imex::acceptance_metric(f.p,caps,c.solution,c.solution.contacts,practical,options);
  require(metric.residual_norm<=metric.tolerance,"practical endpoint bypassed global physical residual");
  for(std::size_t k=0;k<f.p.contacts.size();++k) {
    const auto exact=imex::contact_response(f.p.contacts[k],caps[k],c.solution.velocity,f.p.dt,c.endpoint.gaps[k]);
    require((bs::contact_wrench(exact).array()==bs::contact_wrench(c.solution.contacts[k]).array()).all()&&
            c.solution.contacts[k].predicted_gap==c.endpoint.gaps[k],
            "practical endpoint retained predictor forces instead of exact endpoint forces");
    require(std::abs(c.solution.contacts[k].normal_load-caps[k])>.5,
            "fixture did not exercise a substantial pairwise force correction");
  }
  // A full physical residual failure must remain a failure even when pairwise
  // matching is disabled. An intentionally corrupt inner result cannot pass.
  auto corrupt=c.solution;corrupt.total_moment[0]+=1.;bool rejected=false;
  try {
    ns::detail::solve_endpoint(f.p,f.g,options,practical,nullptr,
      [&](const auto&,const auto&,auto&,const auto*){return corrupt;},
      [&](std::size_t k,const auto&contact,const auto&u,double dt,double gap){return imex::contact_response(contact,caps[k],u,dt,gap);},
      [&](const auto&p,const auto&s,const auto&contacts,const auto&w,const auto&o){return imex::acceptance_metric(p,caps,s,contacts,w,o);},
      false,1.,false);
  } catch(const std::runtime_error&e){rejected=std::string(e.what()).find("endpoint correction did not converge")!=std::string::npos;}
  require(rejected,"disabled pairwise matching admitted corrupt global force balance");
  for(double invalid:{0.,-1.,1.01,std::numeric_limits<double>::quiet_NaN()}) {
    imex::Workspace w;w.endpoint_inner_tolerance_fraction=invalid;bool threw=false;
    try{ns::solve_imex(f.p,caps,f.g,options,w);}catch(const std::invalid_argument&){threw=true;}
    require(threw,"invalid endpoint inner fraction accepted");
  }
  // The optional pairwise gate must not bypass independent new-contact checks.
  for(bool deferred:{false,true}) {
    auto missing=symmetric_images();missing.p.contacts.clear();missing.g.images.clear();
    missing.g.kappa.setZero();missing.g.lattice=20.*M::Identity();missing.g.bodies[1].position={3.,0.,0.};
    missing.p.dt=1.;missing.p.external[0]=1.;missing.p.external[6]=-1.;
    imex::Workspace w;w.endpoint_inner_tolerance_fraction=1.;w.require_endpoint_constitutive_match=false;
    w.defer_candidate_validation=deferred;bool threw=false;
    try{ns::solve_imex(missing.p,{},missing.g,options,w);}
    catch(const std::runtime_error&e){threw=std::string(e.what()).find("unlisted contact")!=std::string::npos;}
    require(threw,"practical endpoint bypassed independent candidate validation");
  }
  std::cout<<"endpoint controls PASS; strict_iterations="<<a.geometry_iterations
           <<" practical_iterations="<<c.geometry_iterations<<" physical_residual="<<metric.residual_norm
           <<" exact_normal_load="<<c.solution.contacts[0].normal_load<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
