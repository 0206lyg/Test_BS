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
  // Optional floating-point backward-error allowance. It is disabled for the
  // original attraction-free path. solve_imex supplies geometric length scales
  // for support-gap evaluation; no physical force floor or relaxed rtol enters.
  bool use_roundoff_budget = false;
  // Skip the expensive budget only when a conservative componentwise screen
  // proves that its existing componentwise gate must return zero.
  bool use_lazy_roundoff_screen = true;
  // Evaluation objects are immutable until replaced by a new trial. Reuse
  // their exact budget and physical metric rather than evaluating them again.
  bool reuse_evaluation_metrics = true;
  mutable std::size_t roundoff_evaluation_reuses=0,final_metric_reuses=0;
  mutable std::size_t roundoff_full_evaluations=0,roundoff_screen_checks=0,
      roundoff_screen_skips=0;
  // All listed gaps are corrected every iterate. The independent all-pairs
  // new-contact check may be deferred until just before endpoint acceptance.
  bool defer_candidate_validation = false;
  // Endpoint controls preserve the historical settings by default. Opting out
  // of pairwise iterate matching still requires the ORIGINAL-step global
  // physical residual with exact endpoint contact responses to pass.
  double endpoint_inner_tolerance_fraction = .1;
  bool require_endpoint_constitutive_match = true;
  std::size_t deferred_candidate_checks=0;
  std::vector<double> contact_roundoff_length;
  double last_roundoff_tolerance=0.,maximum_roundoff_tolerance=0.;
  double last_acceptance_tolerance=0.,last_acceptance_ratio=0.;
  std::size_t roundoff_acceptances=0,roundoff_inner_acceptances=0;
  std::size_t warm_start_preparations=0,cold_start_solves=0;
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
  // Reusing an older SPD factor changes only the FGMRES preconditioner: every
  // product and true linear residual still uses the current exact Jacobian.
  bool reuse_contact_preconditioner_across_solves = false;
  bool contact_preconditioner_refresh_pending = false;
  std::size_t contact_preconditioner_cross_solve_reuses=0;
  int contact_preconditioner_refresh_iterations = 40;
  std::size_t contact_preconditioner_refreshes=0,contact_preconditioner_solves=0;
  int krylov_limit = 160, krylov_restart = 80;
  double linear_relative_target = 1e-10;
  std::size_t krylov_calls=0,krylov_iterations=0,krylov_max_iterations=0,
              krylov_fallbacks=0,jacobian_products=0;
  double maximum_true_relative_linear_residual=0.;
  std::shared_ptr<bs::detail::SolverWorkspace> linear_workspace=
      std::make_shared<bs::detail::SolverWorkspace>();
  // Cache only sparse assembly locations. Numeric contact derivatives and
  // the current resistance are rebuilt every Newton iteration; exact zeros
  // are pruned before factorization, preserving the original active pattern.
  bool cache_contact_jacobian_pattern = true;
  mutable bs::Sparse contact_jacobian_values;
  mutable std::vector<std::pair<int,int>> contact_jacobian_graph;
  mutable std::vector<std::array<int,144>> contact_jacobian_offsets;
  mutable std::size_t contact_jacobian_pattern_builds=0,contact_jacobian_numeric_fills=0;
  bs::Sparse jacobian;
  Eigen::SimplicialLDLT<bs::Sparse> newton_factor;
  bool jacobian_ready=false;
  std::size_t newton_symbolic_analyses=0,newton_numeric_factorizations=0,
              newton_numeric_reuses=0,newton_solves=0;
};
struct Metric {
  double residual_norm=0.,rhs_norm=0.,tolerance=0.,force_norm=0.,mobility_norm=0.;
  double roundoff_tolerance=0.;
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
inline bs::Vector contact_force(const bs::Problem &p,const std::vector<bs::ContactResponse>&contacts,
    const bs::Vector* external_override=nullptr) {
  if(contacts.size()!=p.contacts.size())throw std::invalid_argument("IMEX: contact size mismatch");
  bs::Vector force=external_override?*external_override:p.external;
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
inline double rounding_gamma(std::size_t operations) {
  const double nu=.5*std::numeric_limits<double>::epsilon()*double(operations);
  if(nu>=1.)throw std::runtime_error("IMEX: roundoff operation count overflow");
  return nu/(1.-nu);
}
// Bound the norm of componentwise force-evaluation roundoff without a sparse
// solve per contact. A=R_self+R_near >= R_self, hence A^-1 <= R_self^-1 <= M_uu.
// For |df|<=b, df^T A^-1 df <= sum_i b_i^T |M_uu,i| b_i. The analogous mobility
// residual bound uses the already available inverse self block. These bounds
// retain the units of the original mixed residual (square root of power).
inline double roundoff_tolerance(const bs::Problem&p,const FrozenCaps&caps,
    const bs::Result&s,const std::vector<bs::ContactResponse>&contacts,const Workspace&w,
    double base_tolerance,const bs::Vector* external_override=nullptr) {
  if(!w.use_roundoff_budget)return 0.;
  ++w.roundoff_full_evaluations;
  const bs::Vector& external=external_override?*external_override:p.external;
  if(external.size()!=p.external.size()||!external.allFinite())
    throw std::invalid_argument("IMEX: invalid roundoff external override");
  const int n=p.count;
  if(!w.contact_roundoff_length.empty()&&w.contact_roundoff_length.size()!=p.contacts.size())
    throw std::invalid_argument("IMEX: roundoff geometry size mismatch");
  const double gamma=rounding_gamma(64);
  bs::Vector force_bound=gamma*external.cwiseAbs();
  bs::Vector near_magnitude=bs::Vector::Zero(11*n);
  std::vector<std::size_t> row_terms(11*n,0),contact_terms(n,0);
  const bs::Vector aq=s.q.cwiseAbs();
  for(int col=0;col<p.near_resistance.outerSize();++col)
    for(bs::Sparse::InnerIterator it(p.near_resistance,col);it;++it) {
      near_magnitude[it.row()]+=std::abs(it.value())*aq[it.col()];
      ++row_terms[it.row()];
    }
  double mobility_bound_squared=0.;
  bool componentwise_ok=true;
  for(int i=0;i<n;++i) {
    const Eigen::Matrix<double,11,1> self_magnitude=w.self_inverse[i].cwiseAbs()*aq.segment<11>(11*i);
    for(int k=0;k<6;++k)
      force_bound[6*i+k]+=rounding_gamma(64+row_terms[11*i+k])*
          (self_magnitude[k]+near_magnitude[11*i+k]+std::abs(s.total_moment[11*i+k]));
    const Eigen::Matrix<double,11,1> mb=rounding_gamma(16)*(
        p.self_mobility[i].cwiseAbs()*s.mid_moment.segment<11>(11*i).cwiseAbs()+aq.segment<11>(11*i));
    mobility_bound_squared+=mb.dot(w.self_inverse[i].cwiseAbs()*mb);
    const Eigen::Matrix<double,11,1> me=p.self_mobility[i]*s.mid_moment.segment<11>(11*i)-s.q.segment<11>(11*i);
    for(int k=0;k<11;++k)componentwise_ok=componentwise_ok&&
        std::abs(me[k])<=base_tolerance*std::sqrt(p.self_mobility[i](k,k))+mb[k];
  }
  for(std::size_t k=0;k<p.contacts.size();++k) {
    const auto&c=p.contacts[k];const auto&r=contacts[k];
    const bs::V3 relative_bound=c.Bi.cwiseAbs()*s.velocity.segment<6>(6*c.i).cwiseAbs()+
        c.Bj.cwiseAbs()*s.velocity.segment<6>(6*c.j).cwiseAbs()+c.affine.cwiseAbs();
    const bs::V3 rotation_bound=s.velocity.segment<3>(6*c.i+3).cwiseAbs()+
        s.velocity.segment<3>(6*c.j+3).cwiseAbs();
    const double motion=p.dt*(s.velocity.segment<3>(6*c.i).lpNorm<1>()+
        s.velocity.segment<3>(6*c.j).lpNorm<1>()+c.affine.lpNorm<1>());
    const double gap_bound=w.contact_roundoff_length.empty()?0.:
        gamma*(w.contact_roundoff_length[k]+2.*motion);
    Eigen::Matrix<double,6,1> wb=Eigen::Matrix<double,6,1>::Zero();
    if(r.active||std::abs(r.predicted_gap-c.onset)<=gap_bound) {
      const double normal_arithmetic=gamma*c.kn*(std::abs(c.gap)+std::abs(c.onset)+
          p.dt*c.normal.cwiseAbs().dot(relative_bound));
      wb.head<3>()+=(normal_arithmetic+c.kn*gap_bound)*c.normal.cwiseAbs();
    }
    if(caps[k]>0.) {
      wb.head<3>()+=gamma*c.kt*(c.history.cwiseAbs()+p.dt*relative_bound);
      wb.tail<3>()+=gamma*c.kr*(c.rolling_history.cwiseAbs()+p.dt*rotation_bound);
      wb.tail<3>()+=gamma*c.kw*(std::abs(c.twisting_history)+
          p.dt*c.normal.cwiseAbs().dot(rotation_bound))*c.normal.cwiseAbs();
    }
    const auto wrench=bs::contact_wrench(r).cwiseAbs().eval();
    wb+=gamma*wrench;
    const bs::Matrix6 gi=bs::generalized_contact_map(c,true).cwiseAbs(),
                      gj=bs::generalized_contact_map(c,false).cwiseAbs();
    force_bound.segment<6>(6*c.i)+=gi*wb;
    force_bound.segment<6>(6*c.j)+=gj*wb;
    ++contact_terms[c.i];++contact_terms[c.j];
  }
  double force_bound_squared=0.;
  const bs::Vector force_error=contact_force(p,contacts,external_override)-bs::detail::select(s.total_moment,p.count);
  for(int i=0;i<n;++i) {
    // gamma_64 covered each local evaluation; include the additional summation
    // depth explicitly when a body has more than 64 incident contact terms.
    const double accumulation=std::max(1.,rounding_gamma(contact_terms[i]+1)/gamma);
    const Eigen::Matrix<double,6,1> b=accumulation*force_bound.segment<6>(6*i);
    force_bound_squared+=b.dot(p.self_mobility[i].topLeftCorner<6,6>().cwiseAbs()*b);
    for(int k=0;k<6;++k)componentwise_ok=componentwise_ok&&
        std::abs(force_error[6*i+k])<=base_tolerance*std::sqrt(w.matrix.coeff(6*i+k,6*i+k))+b[k];
  }
  const double bound=std::sqrt(force_bound_squared+mobility_bound_squared);
  if(!std::isfinite(bound))throw std::runtime_error("IMEX: nonfinite roundoff budget");
  // A conservative norm bound alone could hide a residual too large in one
  // lubricated direction. The componentwise gate prevents this: Cauchy gives
  // |r_i|<=base_tolerance*sqrt(A_ii) for the original norm tolerance; only the
  // explicitly computed component roundoff b_i may be added to that amount.
  return componentwise_ok?bound:0.;
}
// Scalar upper bounds for the unchanged componentwise roundoff gate. All
// maxima are deliberately conservative; a failed screen falls through to the
// original full calculation. This is not an added acceptance tolerance.
struct RoundoffScreen {
  double gamma=0.,row_gamma=0.,accumulation=1.,self_row_sum=0.,near_row_sum=0.,
      diagonal_sqrt=0.,external_max=0.,map_row_sum=0.,normal_l1=0.,normal_max=0.,
      relative_row_sum=0.,affine_max=0.,affine_l1=0.,gap_max=0.,onset_max=0.,
      geometry_length=0.,kn=0.,kt=0.,kr=0.,kw=0.,history=0.,rolling=0.,twisting=0.;
  std::size_t incident=0;
};
inline RoundoffScreen prepare_roundoff_screen(const bs::Problem&p,const Workspace&w) {
  RoundoffScreen a;a.gamma=rounding_gamma(64);
  std::vector<double> rows(11*p.count,0.);std::vector<std::size_t> terms(11*p.count,0),incident(p.count,0);
  for(int col=0;col<p.near_resistance.outerSize();++col)
    for(bs::Sparse::InnerIterator it(p.near_resistance,col);it;++it) {
      rows[it.row()]+=std::abs(it.value());++terms[it.row()];
    }
  std::size_t row_terms=0;
  for(int i=0;i<11*p.count;++i){a.near_row_sum=std::max(a.near_row_sum,rows[i]);row_terms=std::max(row_terms,terms[i]);}
  a.row_gamma=rounding_gamma(64+row_terms);
  a.external_max=p.external.size()?p.external.cwiseAbs().maxCoeff():0.;
  for(int i=0;i<p.count;++i)
    a.self_row_sum=std::max(a.self_row_sum,w.self_inverse[i].cwiseAbs().rowwise().sum().maxCoeff());
  for(int i=0;i<6*p.count;++i)a.diagonal_sqrt=std::max(a.diagonal_sqrt,std::sqrt(w.matrix.coeff(i,i)));
  if(!w.contact_roundoff_length.empty()&&w.contact_roundoff_length.size()!=p.contacts.size())
    throw std::invalid_argument("IMEX: roundoff screen geometry size mismatch");
  for(std::size_t k=0;k<p.contacts.size();++k) {
    const auto&c=p.contacts[k];++incident[c.i];++incident[c.j];
    a.map_row_sum=std::max({a.map_row_sum,bs::generalized_contact_map(c,true).cwiseAbs().rowwise().sum().maxCoeff(),
      bs::generalized_contact_map(c,false).cwiseAbs().rowwise().sum().maxCoeff()});
    a.normal_l1=std::max(a.normal_l1,c.normal.lpNorm<1>());a.normal_max=std::max(a.normal_max,c.normal.cwiseAbs().maxCoeff());
    a.relative_row_sum=std::max(a.relative_row_sum,(c.Bi.cwiseAbs().rowwise().sum()+c.Bj.cwiseAbs().rowwise().sum()).maxCoeff());
    a.affine_max=std::max(a.affine_max,c.affine.cwiseAbs().maxCoeff());a.affine_l1=std::max(a.affine_l1,c.affine.lpNorm<1>());
    a.gap_max=std::max(a.gap_max,std::abs(c.gap));a.onset_max=std::max(a.onset_max,std::abs(c.onset));
    if(!w.contact_roundoff_length.empty())a.geometry_length=std::max(a.geometry_length,w.contact_roundoff_length[k]);
    a.kn=std::max(a.kn,c.kn);a.kt=std::max(a.kt,c.kt);a.kr=std::max(a.kr,c.kr);a.kw=std::max(a.kw,c.kw);
    a.history=std::max(a.history,c.history.cwiseAbs().maxCoeff());a.rolling=std::max(a.rolling,c.rolling_history.cwiseAbs().maxCoeff());
    a.twisting=std::max(a.twisting,std::abs(c.twisting_history));
  }
  for(auto n:incident)a.incident=std::max(a.incident,n);
  a.accumulation=std::max(1.,rounding_gamma(a.incident+1)/a.gamma);
  return a;
}
inline bool roundoff_screen_is_zero(const bs::Problem&p,const bs::Result&s,
    const std::vector<bs::ContactResponse>&contacts,const bs::Vector&force_residual,
    const Workspace&w,const RoundoffScreen&a,double base_tolerance) {
  ++w.roundoff_screen_checks;
  const double u=s.velocity.cwiseAbs().maxCoeff(),q=s.q.cwiseAbs().maxCoeff(),
      total=s.total_moment.cwiseAbs().maxCoeff();
  double wrench=0.;for(const auto&r:contacts)wrench=std::max(wrench,bs::contact_wrench(r).cwiseAbs().maxCoeff());
  const double relative=a.relative_row_sum*u+a.affine_max;
  // Include every contact irrespective of activity/cap; dropping that mask
  // only enlarges the bound. The true helper includes the same factors when
  // their corresponding contact is active or within geometric roundoff.
  const double normal=a.gamma*a.kn*(a.gap_max+a.onset_max+p.dt*a.normal_l1*relative+
      a.geometry_length+2.*p.dt*(6.*u+a.affine_l1))*a.normal_max;
  const double tangent=a.gamma*a.kt*(a.history+p.dt*relative);
  const double rolling=a.gamma*a.kr*(a.rolling+2.*p.dt*u);
  const double twisting=a.gamma*a.kw*(a.twisting+2.*p.dt*a.normal_l1*u)*a.normal_max;
  const double wb=normal+tangent+rolling+twisting+a.gamma*wrench;
  const double bound=a.accumulation*(a.gamma*a.external_max+
      a.row_gamma*((a.self_row_sum+a.near_row_sum)*q+total)+double(a.incident)*a.map_row_sum*wb);
  // Fourfold outward padding makes the screen insensitive to floating point
  // summation of its conservative maxima. It never enters final acceptance.
  const double threshold=2.*base_tolerance*a.diagonal_sqrt+4.*bound;
  const bool zero=std::isfinite(threshold)&&force_residual.cwiseAbs().maxCoeff()>threshold;
  if(zero)++w.roundoff_screen_skips;
  return zero;
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
  if(w.use_roundoff_budget) {
    m.roundoff_tolerance=roundoff_tolerance(p,caps,s,contacts,w,m.tolerance);
    m.tolerance+=m.roundoff_tolerance;
  }
  if(!std::isfinite(m.residual_norm+m.rhs_norm+m.tolerance))throw std::runtime_error("IMEX: nonfinite physical residual");
  return m;
}
namespace detail {
struct Evaluation {
  bs::Result state;bs::Vector residual;double norm=0.,mobility_norm=0.;
  mutable double roundoff_budget=0.;
  mutable bool roundoff_ready=false,roundoff_full=false;
};
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
  e.mobility_norm=me.norm();
  if(!u.allFinite()||!std::isfinite(e.norm))throw std::runtime_error("IMEX: nonfinite trial state");
  return e;
}
inline void prepare_contact_jacobian_pattern(const bs::Problem&p,const Workspace&w) {
  bool same=w.contact_jacobian_values.rows()==6*p.count&&
      w.contact_jacobian_graph.size()==p.contacts.size();
  if(same)for(std::size_t k=0;k<p.contacts.size();++k)
    if(w.contact_jacobian_graph[k]!=std::make_pair(p.contacts[k].i,p.contacts[k].j)){same=false;break;}
  if(same)return;
  std::vector<Eigen::Triplet<double>> entries;entries.reserve(144*p.contacts.size());
  w.contact_jacobian_graph.resize(p.contacts.size());
  w.contact_jacobian_offsets.resize(p.contacts.size());
  for(std::size_t k=0;k<p.contacts.size();++k) {
    const auto&c=p.contacts[k];w.contact_jacobian_graph[k]={c.i,c.j};
    for(int i:{c.i,c.j})for(int j:{c.i,c.j})for(int r=0;r<6;++r)for(int col=0;col<6;++col)
      entries.emplace_back(6*i+r,6*j+col,1.);
  }
  auto&matrix=w.contact_jacobian_values;matrix.resize(6*p.count,6*p.count);
  matrix.setFromTriplets(entries.begin(),entries.end());matrix.makeCompressed();
  for(std::size_t k=0;k<p.contacts.size();++k) {
    const auto&c=p.contacts[k];int offset=0;
    for(int i:{c.i,c.j})for(int j:{c.i,c.j})for(int r=0;r<6;++r)for(int col=0;col<6;++col) {
      const int column=6*j+col,row=6*i+r,begin=matrix.outerIndexPtr()[column],end=matrix.outerIndexPtr()[column+1];
      const int*found=std::lower_bound(matrix.innerIndexPtr()+begin,matrix.innerIndexPtr()+end,row);
      if(found==matrix.innerIndexPtr()+end||*found!=row)throw std::logic_error("IMEX: missing cached Jacobian entry");
      w.contact_jacobian_offsets[k][offset++]=int(found-matrix.innerIndexPtr());
    }
  }
  ++w.contact_jacobian_pattern_builds;
}
inline bs::Sparse jacobian(const bs::Problem&p,const Evaluation&e,const Workspace&w,
                           bool majorize_huber=false) {
  std::vector<Eigen::Triplet<double>> entries;
  if(w.cache_contact_jacobian_pattern) {
    prepare_contact_jacobian_pattern(p,w);
    if(w.contact_jacobian_values.nonZeros())
      std::fill(w.contact_jacobian_values.valuePtr(),w.contact_jacobian_values.valuePtr()+w.contact_jacobian_values.nonZeros(),0.);
    ++w.contact_jacobian_numeric_fills;
  } else entries.reserve(144*p.contacts.size());
  auto add=[&](std::size_t pair,int block,int i,int j,const bs::Matrix6&a) {
    for(int r=0;r<6;++r)for(int c=0;c<6;++c)if(a(r,c)!=0.) {
      if(w.cache_contact_jacobian_pattern)
        w.contact_jacobian_values.valuePtr()[w.contact_jacobian_offsets[pair][36*block+6*r+c]]+=a(r,c);
      else entries.emplace_back(6*i+r,6*j+c,a(r,c));
    }
  };
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
    add(k,0,c.i,c.i,a.transpose()*d*a);add(k,1,c.i,c.j,a.transpose()*d*b);
    add(k,2,c.j,c.i,b.transpose()*d*a);add(k,3,c.j,c.j,b.transpose()*d*b);
  }
  bs::Sparse contact(6*p.count,6*p.count);
  if(w.cache_contact_jacobian_pattern){contact=w.contact_jacobian_values;contact.prune(0.);}
  else contact.setFromTriplets(entries.begin(),entries.end());
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
  bs::Options base=o;base.trace_callback={};bs::Vector velocity;
  if(warm&&warm->velocity.size()==6*p.count&&warm->velocity.allFinite()) {
    ns::prepare_resistance(p,w,base);
    ++w.warm_start_preparations;
    velocity=warm->velocity;
  } else {
    velocity=ns::solve(p,base,w).velocity;
    ++w.cold_start_solves;
  }
  auto at=detail::evaluate(p,caps,velocity,w);
  const double scale=rhs_norm(p,caps,w),base_tolerance=o.absolute_tolerance+o.relative_tolerance*scale;
  const auto screen=w.use_roundoff_budget&&w.use_lazy_roundoff_screen?
      prepare_roundoff_screen(p,w):RoundoffScreen{};
  auto evaluated_tolerance=[&](const detail::Evaluation& e) {
    if(!w.use_roundoff_budget)return base_tolerance;
    if(w.reuse_evaluation_metrics&&e.roundoff_ready) {
      ++w.roundoff_evaluation_reuses;return base_tolerance+e.roundoff_budget;
    }
    const bool screened=w.use_lazy_roundoff_screen&&
        roundoff_screen_is_zero(p,e.state,e.state.contacts,e.residual,w,screen,base_tolerance);
    e.roundoff_budget=screened?0.:
        roundoff_tolerance(p,caps,e.state,e.state.contacts,w,base_tolerance);
    e.roundoff_ready=true;e.roundoff_full=!screened;
    return base_tolerance+e.roundoff_budget;
  };
  double tolerance=evaluated_tolerance(at);
  int updates=0,descent=0;
  const bool previous_preconditioner=w.reuse_contact_preconditioner_across_solves&&
      w.jacobian_ready&&w.jacobian.rows()==6*p.count&&w.jacobian.cols()==6*p.count&&
      !w.contact_preconditioner_refresh_pending;
  bool refresh_contact_preconditioner=!previous_preconditioner,
       count_cross_solve_reuse=previous_preconditioner,majorize_next=false,recovering=false;
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
        w.contact_preconditioner_refresh_pending=false;count_cross_solve_reuse=false;
      }
      ++w.krylov_calls;bs::Options linear=o;linear.workspace=w.linear_workspace;
      linear.maximum_linear_iterations=std::min(o.maximum_linear_iterations,w.krylov_limit);
      linear.gmres_restart=std::min(o.gmres_restart,w.krylov_restart);
      const bs::Apply apply=[&](const bs::Vector&v){++w.jacobian_products;bs::Vector z=j*v;
        if(!z.allFinite())throw std::runtime_error("IMEX: nonfinite sparse Jacobian product");return z;};
      const bs::Apply inverse=[&](const bs::Vector&v){++iterations;++w.krylov_iterations;bs::Vector z;
        if(w.use_contact_preconditioner) {
          if(count_cross_solve_reuse){++w.contact_preconditioner_cross_solve_reuses;count_cross_solve_reuse=false;}
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
      if(w.reuse_contact_preconditioner_across_solves)
        w.contact_preconditioner_refresh_pending=refresh_contact_preconditioner;
    }
    if(!solved) {
      detail::factorize(j,w);direction=w.newton_factor.solve(-at.residual);++w.newton_solves;++w.solves;
      // A direct recovery also refreshes the contact preconditioner.
      refresh_contact_preconditioner=false;
      w.contact_preconditioner_refresh_pending=false;count_cross_solve_reuse=false;
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
        if((trial.norm<=tolerance&&trial.norm<=evaluated_tolerance(trial))||change<=1e-4*alpha*slope){
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
              const bool terminal=probe.norm<=tolerance&&probe.norm<=evaluated_tolerance(probe);
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
    if(w.use_roundoff_budget)tolerance=evaluated_tolerance(at);
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
  Metric m;
  if(w.reuse_evaluation_metrics&&(!w.use_roundoff_budget||(at.roundoff_ready&&at.roundoff_full))) {
    // The same immutable state, contacts, operator and options produced all
    // these values. A conservative screen alone is never reused at this final
    // gate: the exact full budget must already have been evaluated.
    m.residual_norm=at.norm;m.rhs_norm=scale;m.tolerance=tolerance;
    m.force_norm=at.residual.norm();m.mobility_norm=at.mobility_norm;
    m.roundoff_tolerance=w.use_roundoff_budget?at.roundoff_budget:0.;
    ++w.final_metric_reuses;
  } else m=acceptance_metric(p,caps,at.state,at.state.contacts,w,o);
  if(!std::isfinite(m.residual_norm+m.rhs_norm+m.tolerance))
    throw std::runtime_error("IMEX: nonfinite physical residual");
  auto result=std::move(at.state);
  if(m.residual_norm>m.tolerance)throw std::runtime_error("IMEX: final physical residual gate failed");
  w.last_roundoff_tolerance=m.roundoff_tolerance;
  w.last_acceptance_tolerance=m.tolerance;
  w.last_acceptance_ratio=m.residual_norm/std::max(m.tolerance,std::numeric_limits<double>::min());
  w.maximum_roundoff_tolerance=std::max(w.maximum_roundoff_tolerance,m.roundoff_tolerance);
  if(m.residual_norm>m.tolerance-m.roundoff_tolerance)++w.roundoff_inner_acceptances;
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
