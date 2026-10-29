#include "slurry_bridge_solver.h"
#include <Eigen/LU>
#include <iostream>
using namespace SlurryBridgeSolver;
static void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
static Problem fixture() {
  Problem p; p.count=3; p.dt=.03;
  p.self_mobility.resize(p.count);
  for (int i=0;i<p.count;++i) {
    Matrix11 b=Matrix11::Identity();
    b(0,7)=.04; b(2,8)=.09; b(3,6)=.03;
    p.self_mobility[i]=(1.+.1*i)*b*b.transpose();
  }
  const int nm=11*p.count;
  Eigen::MatrixXd b=Eigen::MatrixXd::Zero(nm,nm);
  for(int k=0;k<nm;++k) b(k,k)=.2+.001*k;
  for(int i=0;i<2;++i) for(int k=0;k<6;++k) {
    b(11*i+k,11*i+k)+=.5;
    b(11*(i+1)+k,11*(i+1)+k)+=.5;
    b(11*i+k,11*(i+1)+k)-=.5;
    b(11*(i+1)+k,11*i+k)-=.5;
  }
  p.near_resistance=b.sparseView();
  p.prescribed=Vector::Zero(nm); p.external=Vector::Zero(6*p.count);
  for(int i=0;i<p.count;++i) p.prescribed[11*i+6]=-.3;
  p.external[0]=.4; p.external[8]=-.2;
  const auto self=p.self_mobility;
  p.mobility=[self](const Vector &v)->Vector {
    Vector result(v.size());
    for(std::size_t i=0;i<self.size();++i) result.segment<11>(11*i)=self[i]*v.segment<11>(11*i);
    return result;
  };
  Contact c; c.i=0;c.j=1;c.normal=V3::UnitZ();c.kn=40.;c.kt=15.;c.mu=.2;
  c.gap=-.02;c.onset=0.;c.history=V3(.03,.02,0.);
  c.Bi.leftCols<3>()=-Eigen::Matrix3d::Identity();
  c.Bj.leftCols<3>()=Eigen::Matrix3d::Identity();
  c.Bi(0,4)=.3;c.Bj(1,3)=-.2;
  c.affine=V3(.6,.3,-.2);p.contacts.push_back(c);
  return p;
}
static std::vector<ContactResponse> states(const Problem&p) {
  std::vector<ContactResponse> r;
  for(const auto&c:p.contacts) r.push_back(contact_response(c,Vector::Zero(6*p.count),p.dt));
  return r;
}
static Eigen::MatrixXd exact_mixed(const Problem &p,const std::vector<ContactResponse>&s) {
  int nm=11*p.count,nu=6*p.count;
  Eigen::MatrixXd k=Eigen::MatrixXd::Zero(nm+nu,nm+nu),projection=Eigen::MatrixXd::Zero(nm,nu);
  for(int i=0;i<p.count;++i) {k.block<11,11>(11*i,11*i)=p.self_mobility[i];projection.block<6,6>(11*i,6*i).setIdentity();}
  Eigen::MatrixXd a=projection.transpose()*p.near_resistance*projection;
  for(std::size_t j=0;j<p.contacts.size();++j) {
    const auto &c=p.contacts[j];Eigen::MatrixXd b=Eigen::MatrixXd::Zero(3,nu);
    b.block<3,6>(0,6*c.i)+=c.Bi;b.block<3,6>(0,6*c.j)+=c.Bj;
    a+=b.transpose()*s[j].derivative*b;
  }
  k.topRightCorner(nm,nu)=-projection;k.bottomLeftCorner(nu,nm)=-projection.transpose();
  k.bottomRightCorner(nu,nu)=-a;return k;
}
int main() {
  auto p=fixture();auto s=states(p);auto w=std::make_shared<detail::SolverWorkspace>();
  const Vector rhs=Vector::LinSpaced(17*p.count,-.7,.9);
  const auto check=[&](const Problem&problem,const std::vector<ContactResponse>&state,const Vector&got) {
    const auto k=exact_mixed(problem,state);
    require((k*got-rhs).norm()<2e-11,"cached Schur inverse failed true mixed residual");
    require((got-k.fullPivLu().solve(rhs)).norm()<2e-11,"cached Schur inverse disagrees with independent dense solve");
  };
  Vector original;
  {detail::CoupledPreconditioner pre(p,s,w);original=pre.apply(rhs);check(p,s,original);}
  const auto symbolic=w->coupled_symbolic_analyses,numeric=w->coupled_numeric_factorizations;
  {detail::CoupledPreconditioner pre(p,s,w);check(p,s,pre.apply(rhs));}
  require(w->coupled_numeric_factorizations==numeric && w->coupled_numeric_reuses==1,"identical numeric factors not reused");
  auto changed=s;changed[0].derivative*=1.2;
  {detail::CoupledPreconditioner pre(p,changed,w);check(p,changed,pre.apply(rhs));}
  require(w->coupled_symbolic_analyses==symbolic && w->coupled_numeric_factorizations==numeric+1,"same-pattern numeric change not refactorized correctly");
  // A live old preconditioner must remain valid after a new state is prepared.
  detail::CoupledPreconditioner old(p,changed,w);
  detail::CoupledPreconditioner newer(p,s,w);
  check(p,changed,old.apply(rhs));check(p,s,newer.apply(rhs));
  detail::Preconditioner monitor(p,Vector::Zero(6*p.count),w);
  const Vector probe=monitor.left(rhs);
  detail::Preconditioner monitor_same(p,Vector::Zero(6*p.count),w);
  require(w->monitor_reuses==1,"identical norm monitor was not reused");
  auto p2=p;p2.self_mobility[0]*=1.1;p2.contacts[0].gap=-.1;
  detail::Preconditioner monitor_changed(p2,Vector::Zero(6*p.count),w);
  require((monitor.left(rhs)-probe).norm()==0.,"live acceptance norm was mutated by another problem");
  require((monitor_changed.left(rhs)-probe).norm()>1e-4,"changed self/geometry failed cache invalidation");
  require((monitor.inverse_left(probe)-rhs).norm()<1e-12,"cached norm transformations are not inverse");
  {detail::CoupledPreconditioner pre(p2,states(p2),w);check(p2,states(p2),pre.apply(rhs));}
  auto p3=p;p3.near_resistance.coeffRef(0,11)+=.04;
  p3.near_resistance.makeCompressed();
  {detail::CoupledPreconditioner pre(p3,s,w);check(p3,s,pre.apply(rhs));}
  auto p4=p3;p4.near_resistance.coeffRef(1,23)=.02;
  p4.near_resistance.makeCompressed();
  {detail::CoupledPreconditioner pre(p4,s,w);check(p4,s,pre.apply(rhs));}
  Options opt;opt.relative_tolerance=1e-9;opt.linear_relative_tolerance=1e-10;opt.workspace=w;
  auto cached=solve_direct(p,opt);opt.workspace.reset();auto fresh=solve_direct(p,opt);
  require((cached.velocity-fresh.velocity).norm()<1e-10 && (cached.mid_moment-fresh.mid_moment).norm()<1e-10,"cached and fresh nonlinear physical solutions disagree");
  require(cached.relative_residual<1e-9,"cached nonlinear physical residual failed");
  std::cout<<"PASS backend="<<detail::preconditioner_backend()<<" symbolic="<<w->coupled_symbolic_analyses<<" numeric="<<w->coupled_numeric_factorizations<<" reuse="<<w->coupled_numeric_reuses<<'\n';
}
