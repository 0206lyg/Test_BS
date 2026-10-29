#ifndef SLURRY_NEARFIELD_IMEX_SOLVER_H
#define SLURRY_NEARFIELD_IMEX_SOLVER_H

// First-order IMEX contact update. The CURRENT physical normal load fixes each
// friction cap/mask for the entire step. Normal, tangential and rolling elastic
// rates are implicit. At fixed geometry this is a strongly convex velocity
// minimization; changing caps or clipping its accepted forces would change the
// solved equation. An opened contact may release its remaining spring history
// only when the step is committed, adding dissipation without changing loads.
#include "slurry_nearfield_solver.h"

namespace SlurryNearfieldImexSolver {
namespace bs = SlurryBridgeSolver;
namespace ns = SlurryNearfieldSolver;
using FrozenCaps = std::vector<double>;
struct Workspace : ns::Workspace {
  bool use_krylov = true;
  // Refine a heavily damped Newton step along its existing descent direction.
  // This costs contact evaluations, not additional matrix factorizations.
  bool refine_damped_line_search = true;
  std::size_t damped_line_refinements=0;
  // A yielded radial Huber potential has zero exact radial curvature. After
  // a damped Newton step, use one quadratic-majorizer direction to avoid
  // repeatedly overshooting that curved yield surface, then return to the
  // exact Hessian. Forces, the objective and acceptance tolerance are unchanged.
  bool use_huber_globalization = true;
  std::size_t huber_majorized_directions=0;
  // Rare difficult steps may continue the SAME convex problem from the current
  // iterate after the primary Newton budget. Zero disables this bounded
  // recovery for reference comparisons; no physical step or tolerance changes.
  int maximum_recovery_iterations=80;
  std::size_t nonlinear_recoveries=0,nonlinear_recovery_iterations=0;
  // Optional exact-Jacobian SPD preconditioner. Its factors may be reused as
  // the active set changes; FGMRES and the true residual still solve CURRENT J.
  bool use_contact_preconditioner = false;
  int contact_preconditioner_refresh_iterations = 40;
  std::size_t contact_preconditioner_refreshes=0,contact_preconditioner_solves=0;
  int krylov_limit = 160, krylov_restart = 80;
  double linear_relative_target = 1e-10;
  std::size_t krylov_calls=0,krylov_iterations=0,krylov_max_iterations=0,
              krylov_fallbacks=0,jacobian_products=0;
  double maximum_true_relative_linear_residual=0.;
  std::shared_ptr<bs::detail::SolverWorkspace> linear_workspace=
      std::make_shared<bs::detail::SolverWorkspace>();
  bs::Sparse jacobian;
  Eigen::SimplicialLDLT<bs::Sparse> newton_factor;
  bool jacobian_ready=false;
  std::size_t newton_symbolic_analyses=0,newton_numeric_factorizations=0,
              newton_numeric_reuses=0,newton_solves=0;
};
struct Metric {
  double residual_norm=0.,rhs_norm=0.,tolerance=0.,force_norm=0.,mobility_norm=0.;
};
inline FrozenCaps freeze_caps(const bs::Problem &original) {
  FrozenCaps caps;caps.reserve(original.contacts.size());
  for(const auto &c:original.contacts) {
    const double load=c.kn*std::max(c.onset-c.gap,0.);
    if(!std::isfinite(load)||load<0.)throw std::invalid_argument("IMEX: invalid frozen normal load");
    caps.push_back(load);
  }
  return caps;
}
inline bs::ContactResponse contact_response(const bs::Contact &c,double frozen_normal_load,
    const bs::Vector &u,double dt,double endpoint_gap=std::numeric_limits<double>::quiet_NaN()) {
  bs::ContactResponse r;
  r.relative=c.Bi*u.segment<6>(6*c.i)+c.Bj*u.segment<6>(6*c.j)+c.affine;
  r.relative_rotation=u.segment<3>(6*c.j+3)-u.segment<3>(6*c.i+3);
  const double vn=c.normal.dot(r.relative);
  r.predicted_gap=std::isnan(endpoint_gap)?c.gap+dt*vn:endpoint_gap;
  const double compression=std::isnan(endpoint_gap)?std::max((c.onset-c.gap)-dt*vn,0.):
                                                      std::max(c.onset-endpoint_gap,0.);
  const Eigen::Matrix3d nn=c.normal*c.normal.transpose(),P=Eigen::Matrix3d::Identity()-nn;
  r.normal_load=c.kn*compression;r.active=r.normal_load>0.;
  r.normal_energy=.5*c.kn*compression*compression;
  r.traction=-r.normal_load*c.normal;
  if(r.active)r.derivative=c.kn*dt*nn;
  // Deliberately do not gate these forces by the NEW normal activity.
  if(frozen_normal_load>0.&&c.kt>0.&&c.mu>0.) {
    r.trial_history=P*(c.history+dt*r.relative);
    const double length=r.trial_history.norm(),cap=c.mu*frozen_normal_load;
    r.sliding=c.kt*length>cap;
    if(r.sliding&&length>0.) {
      const bs::V3 d=r.trial_history/length;r.tangent_force=cap*d;
      r.derivative+=(cap*dt/length)*(P-d*d.transpose());
    } else {r.tangent_force=c.kt*r.trial_history;r.derivative+=c.kt*dt*P;}
    r.traction+=r.tangent_force;r.tangent_history=r.tangent_force/c.kt;
    r.tangent_energy=.5*c.kt*r.tangent_history.squaredNorm();
    r.sliding_work=std::max(0.,r.tangent_force.dot(r.trial_history-r.tangent_history));
  }
  if(frozen_normal_load>0.&&c.kr>0.&&c.mu_roll>0.&&c.roll_length>0.) {
    r.rolling_trial=P*(c.rolling_history+dt*r.relative_rotation);
    const double length=r.rolling_trial.norm(),cap=c.mu_roll*c.roll_length*frozen_normal_load;
    r.rolling=c.kr*length>cap;bs::V3 moment;
    if(r.rolling&&length>0.) {
      const bs::V3 d=r.rolling_trial/length;moment=cap*d;
      r.derivative_rotation+=(cap*dt/length)*(P-d*d.transpose());
    } else {moment=c.kr*r.rolling_trial;r.derivative_rotation+=c.kr*dt*P;}
    r.torque+=moment;r.rolling_history=moment/c.kr;
    r.rolling_energy=.5*c.kr*r.rolling_history.squaredNorm();
    r.rolling_work=std::max(0.,moment.dot(r.rolling_trial-r.rolling_history));
  }
  if(frozen_normal_load>0.&&c.kw>0.&&c.mu_twist>0.&&c.twist_length>0.) {
    r.twisting_trial=c.twisting_history+dt*c.normal.dot(r.relative_rotation);
    const double cap=c.mu_twist*c.twist_length*frozen_normal_load,trial=c.kw*r.twisting_trial;
    r.twisting=std::abs(trial)>cap;
    const double moment=std::clamp(trial,-cap,cap);r.torque+=moment*c.normal;
    if(!r.twisting)r.derivative_rotation+=c.kw*dt*nn;
    r.twisting_history=moment/c.kw;r.twisting_energy=.5*c.kw*r.twisting_history*r.twisting_history;
    r.twisting_work=std::max(0.,moment*(r.twisting_trial-r.twisting_history));
  }
  return r;
}
inline bs::Vector contact_force(const bs::Problem &p,const std::vector<bs::ContactResponse>&contacts) {
  if(contacts.size()!=p.contacts.size())throw std::invalid_argument("IMEX: contact size mismatch");
  bs::Vector force=p.external;
  for(std::size_t k=0;k<contacts.size();++k) {
    const auto &c=p.contacts[k];const auto wrench=bs::contact_wrench(contacts[k]);
    force.segment<6>(6*c.i).noalias()-=bs::generalized_contact_map(c,true).transpose()*wrench;
    force.segment<6>(6*c.j).noalias()-=bs::generalized_contact_map(c,false).transpose()*wrench;
  }
  return force;
}
inline double rhs_norm(const bs::Problem&p,const FrozenCaps&caps,const Workspace&w) {
  const bs::Vector zero=bs::Vector::Zero(6*p.count);std::vector<bs::ContactResponse> contacts;
  for(std::size_t k=0;k<p.contacts.size();++k)contacts.push_back(contact_response(p.contacts[k],caps[k],zero,p.dt));
  const bs::Vector f=contact_force(p,contacts)-bs::detail::select(p.near_resistance*p.prescribed,p.count);
  return std::sqrt(ns::detail::mobility_metric_squared(p.prescribed,w)+ns::detail::force_metric_squared(f,w));
}
inline Metric acceptance_metric(const bs::Problem&p,const FrozenCaps&caps,const bs::Result&s,
    const std::vector<bs::ContactResponse>&contacts,const Workspace&w,const bs::Options&o) {
  if(caps.size()!=p.contacts.size()||!w.ready||s.q.size()!=11*p.count||
      s.mid_moment.size()!=11*p.count||s.total_moment.size()!=11*p.count)
    throw std::invalid_argument("IMEX: invalid residual state");
  bs::Vector me(11*p.count);
  for(int i=0;i<p.count;++i)me.segment<11>(11*i)=p.self_mobility[i]*s.mid_moment.segment<11>(11*i)-s.q.segment<11>(11*i);
  const bs::Vector fe=contact_force(p,contacts)-bs::detail::select(s.total_moment,p.count);
  Metric m;m.force_norm=fe.norm();m.mobility_norm=me.norm();
  m.residual_norm=std::sqrt(ns::detail::force_metric_squared(fe,w)+ns::detail::mobility_metric_squared(me,w));
  m.rhs_norm=rhs_norm(p,caps,w);m.tolerance=o.absolute_tolerance+o.relative_tolerance*m.rhs_norm;
  if(!std::isfinite(m.residual_norm+m.rhs_norm+m.tolerance))throw std::runtime_error("IMEX: nonfinite physical residual");
  return m;
}
namespace detail {
struct Evaluation {bs::Result state;bs::Vector residual;double norm=0.;};
inline Evaluation evaluate(const bs::Problem&p,const FrozenCaps&caps,const bs::Vector&u,const Workspace&w) {
  Evaluation e;auto&s=e.state;s.velocity=u;s.q=bs::detail::embed(u,p.count)+p.prescribed;
  s.mid_moment.resize(11*p.count);bs::Vector me(11*p.count);
  for(int i=0;i<p.count;++i) {
    s.mid_moment.segment<11>(11*i)=w.self_inverse[i]*s.q.segment<11>(11*i);
    me.segment<11>(11*i)=p.self_mobility[i]*s.mid_moment.segment<11>(11*i)-s.q.segment<11>(11*i);
  }
  s.total_moment=s.mid_moment+p.near_resistance*s.q;
  for(std::size_t k=0;k<p.contacts.size();++k)s.contacts.push_back(contact_response(p.contacts[k],caps[k],u,p.dt));
  e.residual=bs::detail::select(s.total_moment,p.count)-contact_force(p,s.contacts);
  e.norm=std::sqrt(ns::detail::force_metric_squared(e.residual,w)+ns::detail::mobility_metric_squared(me,w));
  if(!u.allFinite()||!std::isfinite(e.norm))throw std::runtime_error("IMEX: nonfinite trial state");
  return e;
}
inline bs::Sparse jacobian(const bs::Problem&p,const Evaluation&e,const Workspace&w,
                           bool majorize_huber=false) {
  std::vector<Eigen::Triplet<double>> entries;entries.reserve(144*p.contacts.size());
  auto add=[&](int i,int j,const bs::Matrix6&a){for(int r=0;r<6;++r)for(int c=0;c<6;++c)
      if(a(r,c)!=0.)entries.emplace_back(6*i+r,6*j+c,a(r,c));};
  for(std::size_t k=0;k<p.contacts.size();++k) {
    const auto&c=p.contacts[k];const auto&r=e.state.contacts[k];
    bs::Matrix6 d=bs::contact_jacobian(r);
    if(majorize_huber) {
      // H(x)=Huber(||x||) is concave as a function of ||x||^2. Its tangent
      // quadratic has curvature min(k,cap/||x||)*I. Adding just the missing
      // yielded radial term gives this majorizer without changing H or grad H.
      // The normal hinge retains its exact derivative; Armijo still checks
      // the full original potential across normal-contact activation changes.
      const double tangent_length=r.trial_history.norm();
      if(r.sliding&&tangent_length>0.) {
        const bs::V3 direction=r.trial_history/tangent_length;
        d.topLeftCorner<3,3>()+=(p.dt*r.tangent_force.norm()/tangent_length)*
                                (direction*direction.transpose());
      }
      const double rolling_length=r.rolling_trial.norm();
      if(r.rolling&&rolling_length>0.) {
        const bs::V3 direction=r.rolling_trial/rolling_length;
        d.bottomRightCorner<3,3>()+=(p.dt*c.kr*r.rolling_history.norm()/rolling_length)*
                                    (direction*direction.transpose());
      }
      if(r.twisting&&std::abs(r.twisting_trial)>0.)
        d.bottomRightCorner<3,3>()+=(p.dt*std::abs(c.kw*r.twisting_history/r.twisting_trial))*
                                    (c.normal*c.normal.transpose());
    }
    if(d.isZero(0.))continue;
    const bs::Matrix6 a=bs::generalized_contact_map(c,true),b=bs::generalized_contact_map(c,false);
    add(c.i,c.i,a.transpose()*d*a);add(c.i,c.j,a.transpose()*d*b);
    add(c.j,c.i,b.transpose()*d*a);add(c.j,c.j,b.transpose()*d*b);
  }
  bs::Sparse contact(6*p.count,6*p.count);contact.setFromTriplets(entries.begin(),entries.end());
  bs::Sparse j=w.matrix+contact;j=(.5*(j+bs::Sparse(j.transpose()))).eval();j.makeCompressed();return j;
}
// Stable differences of radial Huber potentials, avoiding subtraction of two
// large stored energies when the Newton displacement is tiny.
inline long double huber_increment(double k,double cap,const bs::V3&x,const bs::V3&dx) {
  if(!(k>0.)||!(cap>0.))return 0.;
  long double r0sq=0.,change=0.;for(int i=0;i<3;++i) {
    r0sq+=(long double)x[i]*x[i];change+=(long double)dx[i]*(2.L*x[i]+dx[i]);}
  const long double r0=std::sqrt(r0sq),r1=std::sqrt(std::max(0.L,r0sq+change)),limit=(long double)cap/k;
  if(r0<=limit&&r1<=limit)return .5L*k*change;
  if(r0>=limit&&r1>=limit)return r0+r1>0.?cap*change/(r0+r1):0.;
  if(r0<=limit)return cap*(r1-limit)+.5L*k*(limit-r0)*(limit+r0);
  return -cap*(r0-limit)-.5L*k*(limit-r1)*(limit+r1);
}
inline double objective_increment(const bs::Problem&p,const FrozenCaps&caps,const Evaluation&e,
                                  const bs::Vector&du,const Workspace&w) {
  const bs::Vector fluid=bs::detail::select(e.state.total_moment,p.count)-p.external,adu=w.matrix*du;
  long double change=0.;for(int i=0;i<du.size();++i)change+=(long double)du[i]*(fluid[i]+.5L*adu[i]);
  for(std::size_t k=0;k<p.contacts.size();++k) {
    const auto&c=p.contacts[k];const auto&r=e.state.contacts[k];
    const bs::V3 dw=c.Bi*du.segment<6>(6*c.i)+c.Bj*du.segment<6>(6*c.j),
                 dr=du.segment<3>(6*c.j+3)-du.segment<3>(6*c.i+3);
    const long double h=(long double)c.gap-c.onset+(long double)p.dt*c.normal.dot(r.relative),
                      dh=(long double)p.dt*c.normal.dot(dw),h1=h+dh;
    long double normal_change=0.;
    if(h<0.&&h1<0.)normal_change=dh*(2.L*h+dh);
    else if(h<0.)normal_change=-h*h;else if(h1<0.)normal_change=h1*h1;
    change+=.5L*c.kn*normal_change/p.dt;
    const Eigen::Matrix3d P=Eigen::Matrix3d::Identity()-c.normal*c.normal.transpose();
    change+=huber_increment(c.kt,c.mu*caps[k],P*(c.history+p.dt*r.relative),p.dt*P*dw)/p.dt;
    change+=huber_increment(c.kr,c.mu_roll*c.roll_length*caps[k],
                            P*(c.rolling_history+p.dt*r.relative_rotation),p.dt*P*dr)/p.dt;
    change+=huber_increment(c.kw,c.mu_twist*c.twist_length*caps[k],
        bs::V3(c.twisting_history+p.dt*c.normal.dot(r.relative_rotation),0,0),
        bs::V3(p.dt*c.normal.dot(dr),0,0))/p.dt;
  }
  return double(change);
}
inline void factorize(const bs::Sparse&j,Workspace&w) {
  if(w.jacobian_ready&&bs::detail::same_matrix(j,w.jacobian)){++w.newton_numeric_reuses;++w.numeric_reuses;return;}
  const bool same=w.jacobian_ready&&bs::detail::same_pattern(j,w.jacobian);w.jacobian_ready=false;w.jacobian=j;
  if(!same){w.newton_factor.analyzePattern(j);++w.newton_symbolic_analyses;++w.symbolic_analyses;}
  w.newton_factor.factorize(j);++w.newton_numeric_factorizations;++w.numeric_factorizations;
  if(w.newton_factor.info()!=Eigen::Success||!w.newton_factor.vectorD().allFinite()||w.newton_factor.vectorD().minCoeff()<=0.)
    throw std::runtime_error("IMEX: convex Newton matrix is not positive definite");
  w.jacobian_ready=true;
}
} // namespace detail
inline bs::Result solve(const bs::Problem&p,const FrozenCaps&caps,const bs::Options&o,
                        Workspace&w,const bs::Result*warm=nullptr) {
  if(caps.size()!=p.contacts.size()||o.maximum_newton_iterations<=0||o.maximum_line_search_iterations<=0||
      o.maximum_linear_iterations<=0||o.gmres_restart<=0||w.krylov_limit<=0||w.krylov_restart<=0||
      w.contact_preconditioner_refresh_iterations<=0||
      w.maximum_recovery_iterations<0||
      w.maximum_recovery_iterations>std::numeric_limits<int>::max()-o.maximum_newton_iterations||
      !(w.linear_relative_target>0.)||w.linear_relative_target>=1.||!std::isfinite(w.linear_relative_target))
    throw std::invalid_argument("IMEX: invalid options or frozen-cap size");
  for(double load:caps)if(load<0.||!std::isfinite(load))throw std::invalid_argument("IMEX: invalid frozen normal load");
  const auto start=std::chrono::steady_clock::now();const auto initial_linear=w.krylov_iterations;
  bs::Options base=o;base.trace_callback={};const auto cold=ns::solve(p,base,w);
  bs::Vector velocity=cold.velocity;
  if(warm&&warm->velocity.size()==6*p.count&&warm->velocity.allFinite())velocity=warm->velocity;
  auto at=detail::evaluate(p,caps,velocity,w);const double scale=rhs_norm(p,caps,w),tolerance=o.absolute_tolerance+o.relative_tolerance*scale;
  int updates=0,descent=0;
  bool refresh_contact_preconditioner=true,majorize_next=false,recovering=false;
  while(at.norm>tolerance) {
    if(updates>=o.maximum_newton_iterations+w.maximum_recovery_iterations) {
      std::ostringstream message;message<<"IMEX: convex Newton limit; residual="<<at.norm<<"; tolerance="<<tolerance
        <<"; iterations="<<updates<<"; primary_limit="<<o.maximum_newton_iterations
        <<"; recovery_limit="<<w.maximum_recovery_iterations;
      throw std::runtime_error(message.str());
    }
    const bool recovery_entry=!recovering&&updates>=o.maximum_newton_iterations;
    if(recovery_entry){recovering=true;++w.nonlinear_recoveries;}
    // The first recovery direction uses the Huber majorizer even when adaptive
    // globalization is disabled. Subsequent directions return to its usual
    // alternating rule, with the same exact objective and residual gates.
    const bool majorized=recovery_entry||(w.use_huber_globalization&&majorize_next);
    if(majorized)++w.huber_majorized_directions;
    const bs::Sparse j=detail::jacobian(p,at,w,majorized);
    const double rhs=at.residual.norm(),target=w.linear_relative_target*rhs;
    bs::Vector direction;bool solved=false;int iterations=0;double linear_residual=0.;
    if(w.use_krylov) {
      if(w.use_contact_preconditioner&&refresh_contact_preconditioner) {
        detail::factorize(j,w);++w.contact_preconditioner_refreshes;
        refresh_contact_preconditioner=false;
      }
      ++w.krylov_calls;bs::Options linear=o;linear.workspace=w.linear_workspace;
      linear.maximum_linear_iterations=std::min(o.maximum_linear_iterations,w.krylov_limit);
      linear.gmres_restart=std::min(o.gmres_restart,w.krylov_restart);
      const bs::Apply apply=[&](const bs::Vector&v){++w.jacobian_products;bs::Vector z=j*v;
        if(!z.allFinite())throw std::runtime_error("IMEX: nonfinite sparse Jacobian product");return z;};
      const bs::Apply inverse=[&](const bs::Vector&v){++iterations;++w.krylov_iterations;bs::Vector z;
        if(w.use_contact_preconditioner) {
          z=w.newton_factor.solve(v);++w.contact_preconditioner_solves;
          if(w.newton_factor.info()!=Eigen::Success||!z.allFinite())
            throw std::runtime_error("IMEX: contact preconditioner failed");
        } else {
          z=w.factor.solve(v);
          if(w.factor.info()!=Eigen::Success||!z.allFinite())
            throw std::runtime_error("IMEX: hydrodynamic preconditioner failed");
        }
        return z;};
      try {auto answer=bs::detail::fgmres(apply,inverse,-at.residual,target,linear);direction=std::move(answer.x);
        linear_residual=(j*direction+at.residual).norm();
        if(!direction.allFinite()||!std::isfinite(linear_residual))throw std::runtime_error("IMEX: nonfinite Newton direction");
        solved=linear_residual<=target;
      } catch(const std::runtime_error&e) {const std::string message=e.what();
        if(message!="bridge solver: singular Krylov system"&&message.find("bridge solver: GMRES did not converge;")!=0)throw;}
      w.krylov_max_iterations=std::max(w.krylov_max_iterations,std::size_t(iterations));if(!solved)++w.krylov_fallbacks;
      refresh_contact_preconditioner=iterations>w.contact_preconditioner_refresh_iterations;
    }
    if(!solved) {
      detail::factorize(j,w);direction=w.newton_factor.solve(-at.residual);++w.newton_solves;++w.solves;
      // A direct recovery also refreshes the contact preconditioner.
      refresh_contact_preconditioner=false;
      for(int refine=0;refine<3;++refine) {
        const bs::Vector error=j*direction+at.residual;linear_residual=error.norm();
        if(linear_residual<=target)break;
        if(refine<2){direction-=w.newton_factor.solve(error);++w.newton_solves;++w.solves;}
      }
      if(w.newton_factor.info()!=Eigen::Success||!direction.allFinite()||!std::isfinite(linear_residual)||linear_residual>target)
        throw std::runtime_error("IMEX: SPD Newton solve failed true residual gate");
    }
    w.maximum_true_relative_linear_residual=std::max(w.maximum_true_relative_linear_residual,linear_residual/std::max(rhs,std::numeric_limits<double>::min()));
    bool accepted=false;double accepted_alpha=0.;
    for(int attempt=0;attempt<2&&!accepted;++attempt) {
      if(attempt){direction=-w.factor.solve(at.residual);++descent;
        if(w.factor.info()!=Eigen::Success||!direction.allFinite())throw std::runtime_error("IMEX: descent solve failed");}
      const double slope=at.residual.dot(direction);if(!(slope<0.)||!std::isfinite(slope))continue;
      double alpha=1.,right=0.,right_slope=0.;
      for(int back=0;back<o.maximum_line_search_iterations;++back,alpha*=.5) {
        const bs::Vector du=alpha*direction;auto trial=detail::evaluate(p,caps,at.state.velocity+du,w);
        const double change=detail::objective_increment(p,caps,at,du,w);
        if(!std::isfinite(change))throw std::runtime_error("IMEX: nonfinite convex objective change");
        const double trial_slope=trial.residual.dot(direction);
        if(trial_slope>=0.){right=alpha;right_slope=trial_slope;}
        if(trial.norm<=tolerance||change<=1e-4*alpha*slope){
          // Strongly stiff sliding/rolling transitions can make dyadic
          // backtracking accept a very short step repeatedly. The objective
          // is convex along this direction: bracket its zero derivative and
          // improve only on the already accepted Armijo step. Normal steps
          // retain their previous path and cost.
          double best_change=change,best_alpha=alpha;
          auto best=std::move(trial);
          if(w.refine_damped_line_search&&back>=3&&right>0.&&best.norm>tolerance) {
            ++w.damped_line_refinements;
            double left=trial_slope<0.?alpha:0.,left_slope=trial_slope<0.?trial_slope:slope;
            for(int refine=0;refine<12&&right>left;++refine) {
              const double span=right-left;
              double candidate=left-left_slope*span/(right_slope-left_slope);
              candidate=std::clamp(candidate,left+.1*span,right-.1*span);
              const bs::Vector delta=candidate*direction;
              auto probe=detail::evaluate(p,caps,at.state.velocity+delta,w);
              const double probe_change=detail::objective_increment(p,caps,at,delta,w),
                           probe_slope=probe.residual.dot(direction);
              if(!std::isfinite(probe_change+probe_slope))
                throw std::runtime_error("IMEX: nonfinite bracketed line search");
              const bool terminal=probe.norm<=tolerance;
              if(terminal||(probe_change<=1e-4*candidate*slope&&probe_change<best_change)) {
                best=std::move(probe);best_change=probe_change;best_alpha=candidate;
              }
              if(terminal||std::abs(probe_slope)<=.05*std::abs(slope))break;
              if(probe_slope<0.){left=candidate;left_slope=probe_slope;}
              else {right=candidate;right_slope=probe_slope;}
            }
          }
          at=std::move(best);accepted=true;accepted_alpha=best_alpha;break;
        }
      }
    }
    if(!accepted)throw std::runtime_error("IMEX: convex objective line search failed");
    majorize_next=!majorized&&accepted_alpha<=.5;
    ++updates;
    if(recovering)++w.nonlinear_recovery_iterations;
    if(o.trace_callback){bs::IterationTrace t;
      t.phase=recovering?(majorized?"nearfield_imex_recovery_majorized":"nearfield_imex_recovery"):
                        (majorized?"nearfield_imex_huber_majorized":"nearfield_imex_convex");
      t.iteration=t.total_newton_iterations=updates;
      t.linear_iterations=iterations;t.total_linear_iterations=int(w.krylov_iterations-initial_linear);
      t.linear_residual=linear_residual;t.linear_target=target;t.residual=t.merit_residual=at.norm;t.tolerance=tolerance;
      t.step_length=accepted_alpha;t.descent=descent>0;
      t.elapsed_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      for(const auto&r:at.state.contacts){t.active_contacts+=r.active;t.sliding_contacts+=r.sliding;}o.trace_callback(t);}
  }
  auto result=std::move(at.state);const auto m=acceptance_metric(p,caps,result,result.contacts,w,o);
  if(m.residual_norm>m.tolerance)throw std::runtime_error("IMEX: final physical residual gate failed");
  result.residual_norm=m.residual_norm;result.relative_residual=m.residual_norm/std::max({m.rhs_norm,o.absolute_tolerance,std::numeric_limits<double>::min()});
  result.force_residual_norm=m.force_norm;result.mobility_residual_norm=m.mobility_norm;
  result.linear_iterations=int(w.krylov_iterations-initial_linear);result.newton_iterations=updates;result.descent_steps=descent;
  result.hydro_power=result.q.dot(result.total_moment);result.contact_power=0.;
  for(const auto&r:result.contacts)result.contact_power+=bs::contact_power(r);
  if(!std::isfinite(result.hydro_power+result.contact_power))throw std::runtime_error("IMEX: nonfinite power");
  return result;
}
} // namespace SlurryNearfieldImexSolver
#endif
