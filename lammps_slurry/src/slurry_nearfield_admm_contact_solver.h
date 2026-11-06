#ifndef SLURRY_NEARFIELD_ADMM_CONTACT_SOLVER_H
#define SLURRY_NEARFIELD_ADMM_CONTACT_SOLVER_H
// Bounded numerical recovery for the ORIGINAL finite-spring, frozen-cap IMEX
// objective at fixed geometry. No change to stiffness, gap law, history, cap, timestep or
// acceptance tolerance. Inactive constitutive subspaces are omitted from the
// split. Exactly one additional SPD factor is used for all ADMM iterations.
#include "slurry_nearfield_imex_solver.h"

namespace SlurryNearfieldAdmmContactSolver {
namespace bs=SlurryBridgeSolver;
namespace ns=SlurryNearfieldSolver;
namespace imex=SlurryNearfieldImexSolver;
struct Options {
  bs::Options nonlinear;
  int maximum_iterations=2000, check_interval=10;
  double penalty_scale=1., relaxation=1.6;
  imex::FrozenCaps frozen_normal_loads;
};
struct Result {
  bs::Result state;
  bool converged=false; // A false result MUST NOT advance physical time/history.
  int iterations=0;
  double physical_tolerance=0., primal_norm=0., dual_norm=0.;
};
struct Workspace : imex::Workspace {
  bs::Sparse augmented;
  Eigen::SimplicialLDLT<bs::Sparse> augmented_factor;
  std::size_t augmented_factorizations=0, augmented_solves=0;
  double maximum_true_linear_relative_residual=0.;
};
namespace detail {
struct Penalty {
  double normal=0., tangent=0., rolling=0., twisting=0.;
  bs::Matrix6 matrix=bs::Matrix6::Zero();
};
inline double coefficient(double mobility,double stiffness,double dt,double scale) {
  if(stiffness==0.)return 0.;
  if(!(mobility>0.)||!std::isfinite(mobility))
    throw std::runtime_error("ADMM: nonpositive contact mobility scale");
  return scale*std::sqrt(stiffness*dt/mobility);
}
inline Penalty penalty(const bs::Contact&c,double cap,double dt,double scale,
                       const bs::Matrix6&mobility) {
  Penalty out;const Eigen::Matrix3d nn=c.normal*c.normal.transpose(),
      P=Eigen::Matrix3d::Identity()-nn;
  const Eigen::Matrix3d v=mobility.topLeftCorner<3,3>(),
      r=mobility.bottomRightCorner<3,3>();
  out.normal=coefficient(c.normal.dot(v*c.normal),c.kn,dt,scale);
  if(cap>0.&&c.kt>0.&&c.mu>0.)
    out.tangent=coefficient(.5*(P*v).trace(),c.kt,dt,scale);
  if(cap>0.&&c.kr>0.&&c.mu_roll>0.&&c.roll_length>0.)
    out.rolling=coefficient(.5*(P*r).trace(),c.kr,dt,scale);
  if(cap>0.&&c.kw>0.&&c.mu_twist>0.&&c.twist_length>0.)
    out.twisting=coefficient(c.normal.dot(r*c.normal),c.kw,dt,scale);
  out.matrix.topLeftCorner<3,3>()=out.normal*nn+out.tangent*P;
  out.matrix.bottomRightCorner<3,3>()=out.rolling*P+out.twisting*nn;
  if(!out.matrix.allFinite())throw std::runtime_error("ADMM: nonfinite penalty");
  return out;
}
// Prox of H_{k,cap}(history+dt*v)/dt in a subspace with velocity penalty rho.
inline bs::V3 huber_prox(const bs::V3&v,const bs::V3&history,
                       double k,double cap,double dt,double rho) {
  if(rho==0.)return v;
  const bs::V3 trial=history+dt*v;const double length=trial.norm();
  const double threshold=cap/k+cap*dt/rho;
  const bs::V3 corrected=length<=threshold ? bs::V3((rho/(rho+k*dt))*trial) :
      bs::V3((1.-cap*dt/(rho*length))*trial);
  return (corrected-history)/dt;
}
inline bs::V6 prox(const bs::Contact&c,double cap,double dt,
                  const Penalty&r,const bs::V6&v) {
  const Eigen::Matrix3d P=Eigen::Matrix3d::Identity()-c.normal*c.normal.transpose();
  bs::V6 z=v;
  if(r.normal>0.) {
    const double gap=c.gap-c.onset+dt*c.normal.dot(v.head<3>());
    z.head<3>()-=c.normal*(c.kn/(r.normal+c.kn*dt)*std::min(gap,0.));
  }
  if(r.tangent>0.)z.head<3>()+=huber_prox(P*v.head<3>(),P*c.history,
      c.kt,c.mu*cap,dt,r.tangent)-P*v.head<3>();
  if(r.rolling>0.)z.tail<3>()+=huber_prox(P*v.tail<3>(),P*c.rolling_history,
      c.kr,c.mu_roll*c.roll_length*cap,dt,r.rolling)-P*v.tail<3>();
  if(r.twisting>0.) {
    const double v0=c.normal.dot(v.tail<3>()),trial=c.twisting_history+dt*v0,
        limit=c.mu_twist*c.twist_length*cap;
    const double corrected=std::abs(trial)<=limit/c.kw+limit*dt/r.twisting ?
        r.twisting/(r.twisting+c.kw*dt)*trial :
        trial-std::copysign(limit*dt/r.twisting,trial);
    z.tail<3>()+=c.normal*((corrected-c.twisting_history)/dt-v0);
  }
  return z;
}
inline bs::V6 affine(const bs::Contact&c) {
  bs::V6 a=bs::V6::Zero();a.head<3>()=c.affine;return a;
}
inline bs::V6 scaled_dual(const bs::Contact&c,const Penalty&r,const bs::V6&wrench) {
  const Eigen::Matrix3d nn=c.normal*c.normal.transpose(),P=Eigen::Matrix3d::Identity()-nn;
  bs::V6 dual=bs::V6::Zero();
  if(r.normal>0.)dual.head<3>()+=(nn*wrench.head<3>())/r.normal;
  if(r.tangent>0.)dual.head<3>()+=(P*wrench.head<3>())/r.tangent;
  if(r.rolling>0.)dual.tail<3>()+=(P*wrench.tail<3>())/r.rolling;
  if(r.twisting>0.)dual.tail<3>()+=(nn*wrench.tail<3>())/r.twisting;
  return dual;
}
} // namespace detail

inline Result solve(const bs::Problem&p,const Options&o,Workspace&w,
                    const bs::Result*warm=nullptr) {
  if(o.maximum_iterations<=0||o.check_interval<=0||!(o.penalty_scale>0.)||
      !std::isfinite(o.penalty_scale)||!(o.relaxation>0.&&o.relaxation<2.))
    throw std::invalid_argument("ADMM: invalid options");
  const auto caps=o.frozen_normal_loads.empty()?imex::freeze_caps(p):o.frozen_normal_loads;
  if(caps.size()!=p.contacts.size())throw std::invalid_argument("ADMM: cap size mismatch");
  for(double cap:caps)if(cap<0.||!std::isfinite(cap))throw std::invalid_argument("ADMM: invalid cap");
  const auto start=std::chrono::steady_clock::now();
  bs::Vector u;
  const bool valid_warm=warm&&warm->velocity.size()==6*p.count&&warm->velocity.allFinite();
  if(valid_warm) {
    ns::prepare_resistance(p,w,o.nonlinear);u=warm->velocity;
  } else u=ns::solve(p,o.nonlinear,w).velocity;
  const int nv=6*p.count,nc=int(p.contacts.size());
  Result result;
  auto check=[&](int iteration) {
    auto evaluation=imex::detail::evaluate(p,caps,u,w);
    const auto metric=imex::acceptance_metric(p,caps,evaluation.state,evaluation.state.contacts,w,o.nonlinear);
    result.state=std::move(evaluation.state);result.iterations=iteration;
    result.physical_tolerance=metric.tolerance;result.converged=metric.residual_norm<=metric.tolerance;
    auto&s=result.state;s.residual_norm=metric.residual_norm;
    s.relative_residual=metric.residual_norm/std::max({metric.rhs_norm,
        o.nonlinear.absolute_tolerance,std::numeric_limits<double>::min()});
    s.force_residual_norm=metric.force_norm;s.mobility_residual_norm=metric.mobility_norm;
    s.linear_iterations=iteration;s.hydro_power=s.q.dot(s.total_moment);s.contact_power=0.;
    for(const auto&r:s.contacts)s.contact_power+=bs::contact_power(r);
    if(!std::isfinite(s.hydro_power+s.contact_power))throw std::runtime_error("ADMM: nonfinite power");
    if(o.nonlinear.trace_callback) {bs::IterationTrace t;t.phase="nearfield_admm";
      t.iteration=iteration;t.linear_iterations=1;t.total_linear_iterations=iteration;
      t.residual=metric.residual_norm;t.tolerance=metric.tolerance;
      t.elapsed_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      o.nonlinear.trace_callback(t);}
    return result.converged;
  };
  if(check(0))return result;
  std::vector<bs::Matrix6> local_inverse(p.count),gi(nc),gj(nc);
  for(int i=0;i<p.count;++i) {
    const bs::Matrix6 block=w.matrix.block(6*i,6*i,6,6);Eigen::LLT<bs::Matrix6> factor(block);
    if(factor.info()!=Eigen::Success)throw std::runtime_error("ADMM: local resistance is not SPD");
    local_inverse[i]=factor.solve(bs::Matrix6::Identity());
  }
  std::vector<detail::Penalty> penalties(nc);
  std::vector<bs::V6> z(nc),dual(nc,bs::V6::Zero()),a(nc);
  std::vector<Eigen::Triplet<double>> entries;entries.reserve(144*nc);
  for(int k=0;k<nc;++k) {
    const auto&c=p.contacts[k];gi[k]=bs::generalized_contact_map(c,true);
    gj[k]=bs::generalized_contact_map(c,false);a[k]=detail::affine(c);
    const bs::Matrix6 mobility=gi[k]*local_inverse[c.i]*gi[k].transpose()+
        gj[k]*local_inverse[c.j]*gj[k].transpose();
    penalties[k]=detail::penalty(c,caps[k],p.dt,o.penalty_scale,mobility);
    z[k]=gi[k]*u.segment<6>(6*c.i)+gj[k]*u.segment<6>(6*c.j)+a[k];
    // The initial check evaluated the ORIGINAL law at this same warm velocity.
    // R^+ wrench supplies its dual guess; omitted zero-potential modes stay zero.
    // No trial or retained spring history is advanced by this initialization.
    if(valid_warm)dual[k]=detail::scaled_dual(c,penalties[k],bs::contact_wrench(result.state.contacts[k]));
    const bs::Matrix6 maps[2]={gi[k],gj[k]};const int ids[2]={c.i,c.j};
    for(int i=0;i<2;++i)for(int j=0;j<2;++j) {
      const bs::Matrix6 block=maps[i].transpose()*penalties[k].matrix*maps[j];
      for(int row=0;row<6;++row)for(int col=0;col<6;++col)
        if(block(row,col)!=0.)entries.emplace_back(6*ids[i]+row,6*ids[j]+col,block(row,col));
    }
  }
  bs::Sparse contact(nv,nv);contact.setFromTriplets(entries.begin(),entries.end());
  w.augmented=w.matrix+contact;w.augmented.makeCompressed();
  w.augmented_factor.compute(w.augmented);++w.augmented_factorizations;
  if(w.augmented_factor.info()!=Eigen::Success||!w.augmented_factor.vectorD().allFinite()||
      w.augmented_factor.vectorD().minCoeff()<=0.)throw std::runtime_error("ADMM: augmented factor failed");
  bs::Vector prescribed=p.near_resistance*p.prescribed;
  for(int i=0;i<p.count;++i)prescribed.segment<11>(11*i)+=w.self_inverse[i]*p.prescribed.segment<11>(11*i);
  const bs::Vector b=p.external-bs::detail::select(prescribed,p.count);
  for(int iteration=1;iteration<=o.maximum_iterations;++iteration) {
    bs::Vector rhs=b;
    for(int k=0;k<nc;++k) {
      const auto&c=p.contacts[k];const bs::V6 f=penalties[k].matrix*(z[k]-a[k]-dual[k]);
      rhs.segment<6>(6*c.i)+=gi[k].transpose()*f;rhs.segment<6>(6*c.j)+=gj[k].transpose()*f;
    }
    u=w.augmented_factor.solve(rhs);++w.augmented_solves;
    if(w.augmented_factor.info()!=Eigen::Success||!u.allFinite())throw std::runtime_error("ADMM: solve failed");
    const bool inspect=iteration%o.check_interval==0||iteration==o.maximum_iterations;
    if(inspect)result.primal_norm=0.;
    bs::Vector dual_residual=inspect?bs::Vector(bs::Vector::Zero(nv)):bs::Vector();
    for(int k=0;k<nc;++k) {
      const auto&c=p.contacts[k];const bs::V6 bu=gi[k]*u.segment<6>(6*c.i)+gj[k]*u.segment<6>(6*c.j)+a[k],
          old_z=z[k],relaxed=o.relaxation*bu+(1.-o.relaxation)*old_z;
      z[k]=detail::prox(c,caps[k],p.dt,penalties[k],relaxed+dual[k]);dual[k]+=relaxed-z[k];
      if(inspect) {
        const bs::V6 defect=bu-z[k],df=penalties[k].matrix*(z[k]-old_z);
        result.primal_norm+=defect.dot(penalties[k].matrix*defect);
        dual_residual.segment<6>(6*c.i)+=gi[k].transpose()*df;
        dual_residual.segment<6>(6*c.j)+=gj[k].transpose()*df;
      }
    }
    if(inspect) {
      result.primal_norm=std::sqrt(std::max(0.,result.primal_norm));
      result.dual_norm=std::sqrt(ns::detail::force_metric_squared(dual_residual,w));
      const double lr=(w.augmented*u-rhs).norm()/std::max(rhs.norm(),std::numeric_limits<double>::min());
      w.maximum_true_linear_relative_residual=std::max(w.maximum_true_linear_relative_residual,lr);
      if(check(iteration))break;
    }
  }
  return result;
}
} // namespace SlurryNearfieldAdmmContactSolver
#endif
