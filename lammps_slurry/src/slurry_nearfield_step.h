#ifndef SLURRY_NEARFIELD_STEP_H
#define SLURRY_NEARFIELD_STEP_H

// A single implicit elastic contact step on the current hydrodynamic geometry.
// Supports coupled endpoint Coulomb and current-load frozen-cap IMEX laws.
// Geometric corrections never commit intermediate positions or histories.
#include "slurry_bridge_step.h"
#include "slurry_nearfield_implicit_solver.h"
#include "slurry_nearfield_imex_solver.h"
#include "slurry_nearfield_admm_contact_solver.h"

namespace SlurryNearfieldStep {
namespace bs = SlurryBridgeSolver;
namespace step = SlurryBridgeStep;
namespace ni = SlurryNearfieldImplicitSolver;
namespace imex = SlurryNearfieldImexSolver;
namespace admm = SlurryNearfieldAdmmContactSolver;

namespace detail {
inline bool recoverable_imex_failure(const std::string &message) {
  return message.find("IMEX: convex Newton limit")==0 ||
         message=="IMEX: convex objective line search failed";
}

inline void add_solver_work(bs::Result &result,const bs::Result &previous) {
  result.newton_iterations+=previous.newton_iterations;
  result.linear_iterations+=previous.linear_iterations;
  result.descent_steps+=previous.descent_steps;
  result.breakpoint_steps+=previous.breakpoint_steps;
  result.continuation_stages+=previous.continuation_stages;
  result.continuation_retries+=previous.continuation_retries;
  result.mobility_applications+=previous.mobility_applications;
  result.kink_reselections+=previous.kink_reselections;
}

inline bs::Result solve_imex_inner(const bs::Problem &problem,
    const imex::FrozenCaps &caps,const bs::Options &options,
    imex::Workspace &workspace,const bs::Result *warm) {
  std::shared_ptr<bs::Result> failed;
  std::string first_failure;
  try {
    return imex::solve(problem,caps,options,workspace,warm);
  } catch(const std::runtime_error &failure) {
    if(!recoverable_imex_failure(failure.what()) || !workspace.failed_iterate)
      throw;
    // Hold ownership: another solve on this workspace clears failed_iterate.
    failed=workspace.failed_iterate;
    if(failed->velocity.size()!=6*problem.count || !failed->velocity.allFinite())
      throw;
    first_failure=failure.what();
  }

  // This is a bounded algebraic recovery, not a new physical substep. Neither
  // the failed iterate nor the ADMM split variables commit any history/state.
  admm::Workspace recovery;
  recovery.use_roundoff_budget=workspace.use_roundoff_budget;
  recovery.use_lazy_roundoff_screen=workspace.use_lazy_roundoff_screen;
  recovery.reuse_evaluation_metrics=workspace.reuse_evaluation_metrics;
  recovery.contact_roundoff_length=workspace.contact_roundoff_length;
  admm::Options split_options;
  split_options.nonlinear=options;
  split_options.frozen_normal_loads=caps;
  split_options.maximum_iterations=100;
  auto split=admm::solve(problem,split_options,recovery,failed.get());

  // Count the additional local factor/solve work without transferring factors
  // or changing the original resistance cache used by endpoint acceptance.
  workspace.symbolic_analyses+=recovery.symbolic_analyses+recovery.augmented_factorizations;
  workspace.numeric_factorizations+=recovery.numeric_factorizations+recovery.augmented_factorizations;
  workspace.solves+=recovery.solves+recovery.augmented_solves;
  workspace.refinement_solves+=recovery.refinement_solves;
  if(split.converged) {
    const auto metric=imex::acceptance_metric(problem,caps,split.state,
        split.state.contacts,workspace,options);
    if(metric.residual_norm<=metric.tolerance) {
      workspace.last_roundoff_tolerance=metric.roundoff_tolerance;
      workspace.last_acceptance_tolerance=metric.tolerance;
      workspace.last_acceptance_ratio=metric.residual_norm/std::max(
          metric.tolerance,std::numeric_limits<double>::min());
      workspace.maximum_roundoff_tolerance=std::max(
          workspace.maximum_roundoff_tolerance,metric.roundoff_tolerance);
      if(metric.residual_norm>metric.tolerance-metric.roundoff_tolerance)
        ++workspace.roundoff_inner_acceptances;
      split.state.residual_norm=metric.residual_norm;
      split.state.relative_residual=metric.residual_norm/std::max({metric.rhs_norm,
          options.absolute_tolerance,std::numeric_limits<double>::min()});
      split.state.force_residual_norm=metric.force_norm;
      split.state.mobility_residual_norm=metric.mobility_norm;
      add_solver_work(split.state,*failed);
      workspace.failed_iterate.reset();workspace.failed_iterate_tolerance=0.;
      return std::move(split.state);
    }
  }
  // One Newton polish, directly through imex::solve: no recursive recovery.
  try {
    auto polished=imex::solve(problem,caps,options,workspace,&split.state);
    add_solver_work(polished,*failed);
    add_solver_work(polished,split.state);
    return polished;
  } catch(const std::runtime_error &failure) {
    if(!recoverable_imex_failure(failure.what()))
      throw std::runtime_error(std::string("same-dt IMEX recovery polish failed: ")+failure.what());
    bs::Result work=workspace.failed_iterate?*workspace.failed_iterate:split.state;
    const double residual=work.residual_norm;
    const double tolerance=workspace.failed_iterate?
        workspace.failed_iterate_tolerance:split.physical_tolerance;
    add_solver_work(work,*failed);
    if(workspace.failed_iterate)add_solver_work(work,split.state);
    std::ostringstream message;
    message << "same-dt IMEX recovery exhausted after " << split.iterations
            << " ADMM sweeps and one Newton polish; initial_failure=" << first_failure
            << "; polish_failure=" << failure.what();
    throw bs::NonlinearFailure(message.str(),work,residual,tolerance);
  }
}

template<class Workspace,class Solver,class Response,class Metric>
inline step::Result solve_endpoint(const bs::Problem &original,const step::Geometry &geometry,
                          const bs::Options &options,Workspace &workspace,
                          const bs::Result *warm,Solver solve_inner,
                          Response endpoint_response,Metric acceptance_metric,
                          bool defer_candidate_validation=false,
                          double inner_tolerance_fraction=.1,
                          bool require_constitutive_match=true) {
  if(!(inner_tolerance_fraction>0.)||inner_tolerance_fraction>1.||
      !std::isfinite(inner_tolerance_fraction))
    throw std::invalid_argument("nearfield endpoint inner tolerance fraction must be in (0,1]");
  bs::Problem corrected=original;
  bs::Options inner=options;
  inner.relative_tolerance*=inner_tolerance_fraction;
  inner.absolute_tolerance*=inner_tolerance_fraction;
  bs::Result previous;
  int newton=0,linear=0,descent=0,breakpoints=0,stages=0,retries=0,
      applications=0,reselections=0;
  double last_residual=std::numeric_limits<double>::infinity(),last_tolerance=0.,
         last_gap_error=0.,last_constitutive_ratio=0.;
  constexpr int maximum_geometry_iterations=24;
  for(int iteration=0;iteration<maximum_geometry_iterations;++iteration) {
    bs::Result state;
    try {
      state=solve_inner(corrected,inner,workspace,iteration?&previous:warm);
    } catch(const bs::NonlinearFailure &failure) {
      bs::Result work;
      work.newton_iterations=newton+failure.newton_iterations;
      work.linear_iterations=linear+failure.linear_iterations;
      work.descent_steps=descent+failure.descent_steps;
      work.breakpoint_steps=breakpoints+failure.breakpoint_steps;
      work.mobility_applications=applications+failure.mobility_applications;
      work.kink_reselections=reselections+failure.kink_reselections;
      std::ostringstream diagnostic;
      diagnostic << failure.what() << "; endpoint_iteration=" << iteration+1
                 << "; completed_endpoint_continuation_stages=" << stages
                 << "; completed_endpoint_continuation_retries=" << retries;
      throw bs::NonlinearFailure(diagnostic.str(),work,failure.residual,failure.tolerance);
    }
    newton+=state.newton_iterations;linear+=state.linear_iterations;
    descent+=state.descent_steps;breakpoints+=state.breakpoint_steps;
    stages+=state.continuation_stages;retries+=state.continuation_retries;
    applications+=state.mobility_applications;reselections+=state.kink_reselections;
    auto endpoint=geometry.advance(original,state.velocity,!defer_candidate_validation);
    std::vector<bs::ContactResponse> exact;
    exact.reserve(original.contacts.size());
    bool constitutive_ok=true;
    double gap_error=0.,linear_defect=0.,constitutive_ratio=0.;
    for(std::size_t k=0;k<original.contacts.size();++k) {
      const auto &c=original.contacts[k];
      const auto uncorrected=bs::contact_response(c,state.velocity,original.dt);
      const double defect=endpoint.gaps[k]-uncorrected.predicted_gap;
      corrected.contacts[k].gap=c.gap+defect;
      // Use actual compression directly, avoiding cancellation of a shifted
      // gap and dt*v in a very stiff shallow contact.
      auto response=endpoint_response(k,c,state.velocity,original.dt,endpoint.gaps[k]);
      const double force_error=(bs::contact_wrench(response)-
                                bs::contact_wrench(state.contacts[k])).norm();
      const double force_tolerance=options.absolute_tolerance+
          options.relative_tolerance*std::max(1.,bs::contact_wrench(response).norm());
      constitutive_ok=constitutive_ok&&force_error<=force_tolerance;
      constitutive_ratio=std::max(constitutive_ratio,force_error/force_tolerance);
      if(response.active||state.contacts[k].active)
        gap_error=std::max(gap_error,std::abs(endpoint.gaps[k]-state.contacts[k].predicted_gap));
      linear_defect=std::max(linear_defect,std::abs(defect));
      exact.push_back(std::move(response));
    }
    // The reference load and tolerance always belong to the ORIGINAL step;
    // shifting the gap cannot make the final physical acceptance easier.
    const auto metric=acceptance_metric(original,state,exact,workspace,options);
    if(!std::isfinite(metric.residual_norm)||!std::isfinite(metric.tolerance))
      throw std::runtime_error("nearfield endpoint correction produced a nonfinite residual");
    const char *trace=std::getenv("SLURRY_BRIDGE_TRACE");
    if(trace&&std::strcmp(trace,"0")) {
      std::fprintf(stderr,"nearfield endpoint iteration=%d residual=%.9e tolerance=%.9e gap_error=%.9e constitutive_ratio=%.9e\n",
                   iteration+1,metric.residual_norm,metric.tolerance,gap_error,constitutive_ratio);
      std::fflush(stderr);
    }
    // The optional pairwise iterate-match gate is additional to physical
    // force balance. Even without it, only ORIGINAL-step exact endpoint
    // responses may pass the unchanged global metric and become committed.
    if((!require_constitutive_match||constitutive_ok)&&metric.residual_norm<=metric.tolerance) {
      // Re-run the deterministic accepted endpoint with the independent global
      // candidate check enabled. An unlisted new contact still rejects the
      // whole step before positions or histories can be committed.
      if(defer_candidate_validation)endpoint=geometry.advance(original,state.velocity,true);
      state.contacts=std::move(exact);
      state.residual_norm=metric.residual_norm;
      state.relative_residual=metric.residual_norm/std::max({metric.rhs_norm,
          options.absolute_tolerance,std::numeric_limits<double>::min()});
      state.force_residual_norm=metric.force_norm;
      state.mobility_residual_norm=metric.mobility_norm;
      state.contact_power=0.;
      for(const auto &contact:state.contacts)state.contact_power+=bs::contact_power(contact);
      state.newton_iterations=newton;state.linear_iterations=linear;
      state.descent_steps=descent;state.breakpoint_steps=breakpoints;
      state.continuation_stages=stages;state.continuation_retries=retries;
      state.mobility_applications=applications;state.kink_reselections=reselections;
      step::Result out;
      out.solution=std::move(state);out.endpoint=std::move(endpoint);
      out.geometry_iterations=iteration+1;out.maximum_gap_error=gap_error;
      out.maximum_linear_gap_defect=linear_defect;
      return out;
    }
    // Bound each inner solve by a fraction of the fixed original tolerance,
    // even when a geometric preload correction changes its own RHS norm.
    const auto shifted=acceptance_metric(corrected,state,state.contacts,workspace,options);
    const double target=inner_tolerance_fraction*metric.tolerance;
    inner.absolute_tolerance=std::min(inner_tolerance_fraction*options.absolute_tolerance,.5*target);
    inner.relative_tolerance=std::min(inner_tolerance_fraction*options.relative_tolerance,
        (target-inner.absolute_tolerance)/std::max(shifted.rhs_norm,std::numeric_limits<double>::min()));
    last_residual=metric.residual_norm;last_tolerance=metric.tolerance;
    last_gap_error=gap_error;last_constitutive_ratio=constitutive_ratio;
    previous=std::move(state);
  }
  std::ostringstream message;
  message << std::scientific << std::setprecision(9)
          << "nearfield implicit endpoint correction did not converge after "
          << maximum_geometry_iterations << " iterations; residual=" << last_residual
          << "; tolerance=" << last_tolerance << "; gap_error=" << last_gap_error
          << "; constitutive_ratio=" << last_constitutive_ratio
          << "; total_newton_iterations=" << newton << "; continuation_stages=" << stages
          << "; continuation_retries=" << retries;
  throw std::runtime_error(message.str());
}
} // namespace detail

inline step::Result solve(const bs::Problem &original,const step::Geometry &geometry,
                          const bs::Options &options,ni::Workspace &workspace,
                          const bs::Result *warm=nullptr) {
  return detail::solve_endpoint(original,geometry,options,workspace,warm,
      [](const auto &p,const auto &o,auto &w,const auto *seed){return ni::solve(p,o,w,seed);},
      [](std::size_t,const auto &c,const auto &u,double dt,double gap){
        return bs::contact_response(c,u,dt,gap);
      },
      [](const auto &p,const auto &state,const auto &contacts,const auto &w,const auto &o){
        return ni::acceptance_metric(p,state,contacts,w,o);
      });
}

inline step::Result solve_imex(const bs::Problem &original,const imex::FrozenCaps &caps,
                          const step::Geometry &geometry,const bs::Options &options,
                          imex::Workspace &workspace,const bs::Result *warm=nullptr) {
  // The caller freezes N_n from original physical gaps, never from the shifted
  // gaps used solely to solve endpoint normal compression consistently.
  if(workspace.use_roundoff_budget) {
    workspace.contact_roundoff_length.resize(original.contacts.size());
    if(geometry.bodies.size()!=std::size_t(original.count)||
        geometry.images.size()!=original.contacts.size())
      throw std::invalid_argument("IMEX: roundoff geometry size mismatch");
    for(std::size_t k=0;k<original.contacts.size();++k) {
      const auto& c=original.contacts[k];const auto& a=geometry.bodies[c.i];
      const auto& b=geometry.bodies[c.j];
      const auto image=geometry.lattice*geometry.images[k].cast<double>();
      // The gap subtracts centers and two support radii. Positions include the
      // actual periodic image shift; the factor two covers original and trial
      // support evaluations. gamma_64 in the force bound covers the scalar
      // support/matrix arithmetic, rather than imposing a physical gap floor.
      workspace.contact_roundoff_length[k]=2.*(step::vector(a.position).lpNorm<1>()+
          step::vector(b.position).lpNorm<1>()+image.lpNorm<1>()+
          *std::max_element(a.axes.begin(),a.axes.end())+
          *std::max_element(b.axes.begin(),b.axes.end()));
    }
  } else workspace.contact_roundoff_length.clear();
  imex::Metric last_metric;
  auto result=detail::solve_endpoint(original,geometry,options,workspace,warm,
      [&](const auto &p,const auto &o,auto &w,const auto *seed){return detail::solve_imex_inner(p,caps,o,w,seed);},
      [&](std::size_t k,const auto &c,const auto &u,double dt,double gap){
        return imex::contact_response(c,caps[k],u,dt,gap);
      },
      [&](const auto &p,const auto &state,const auto &contacts,const auto &w,const auto &o){
        last_metric=imex::acceptance_metric(p,caps,state,contacts,w,o);
        return last_metric;
      },workspace.defer_candidate_validation,workspace.endpoint_inner_tolerance_fraction,
      workspace.require_endpoint_constitutive_match);
  if(workspace.defer_candidate_validation)++workspace.deferred_candidate_checks;
  if(workspace.use_roundoff_budget) {
    // solve_endpoint returns immediately after the ORIGINAL-step full
    // physical metric passes. The independent candidate check changes no
    // velocity/load, so this is that same accepted metric, not a stale cache.
    const auto&metric=last_metric;
    workspace.last_roundoff_tolerance=metric.roundoff_tolerance;
    workspace.last_acceptance_tolerance=metric.tolerance;
    workspace.last_acceptance_ratio=metric.residual_norm/std::max(metric.tolerance,std::numeric_limits<double>::min());
    workspace.maximum_roundoff_tolerance=std::max(workspace.maximum_roundoff_tolerance,metric.roundoff_tolerance);
    if(metric.residual_norm>metric.tolerance-metric.roundoff_tolerance)++workspace.roundoff_acceptances;
  }
  return result;
}
} // namespace SlurryNearfieldStep
#endif
