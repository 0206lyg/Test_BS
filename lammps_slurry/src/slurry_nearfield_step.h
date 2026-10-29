#ifndef SLURRY_NEARFIELD_STEP_H
#define SLURRY_NEARFIELD_STEP_H

// A single implicit elastic contact step on the current hydrodynamic geometry.
// Supports coupled endpoint Coulomb and current-load frozen-cap IMEX laws.
// Geometric corrections never commit intermediate positions or histories.
#include "slurry_bridge_step.h"
#include "slurry_nearfield_implicit_solver.h"
#include "slurry_nearfield_imex_solver.h"

namespace SlurryNearfieldStep {
namespace bs = SlurryBridgeSolver;
namespace step = SlurryBridgeStep;
namespace ni = SlurryNearfieldImplicitSolver;
namespace imex = SlurryNearfieldImexSolver;

namespace detail {
template<class Workspace,class Solver,class Response,class Metric>
inline step::Result solve_endpoint(const bs::Problem &original,const step::Geometry &geometry,
                          const bs::Options &options,Workspace &workspace,
                          const bs::Result *warm,Solver solve_inner,
                          Response endpoint_response,Metric acceptance_metric) {
  bs::Problem corrected=original;
  bs::Options inner=options;
  inner.relative_tolerance*=.1;
  inner.absolute_tolerance*=.1;
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
    auto endpoint=geometry.advance(original,state.velocity,true);
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
    if(constitutive_ok&&metric.residual_norm<=metric.tolerance) {
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
    const double target=.1*metric.tolerance;
    inner.absolute_tolerance=std::min(.1*options.absolute_tolerance,.5*target);
    inner.relative_tolerance=std::min(.1*options.relative_tolerance,
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
  return detail::solve_endpoint(original,geometry,options,workspace,warm,
      [&](const auto &p,const auto &o,auto &w,const auto *seed){return imex::solve(p,caps,o,w,seed);},
      [&](std::size_t k,const auto &c,const auto &u,double dt,double gap){
        return imex::contact_response(c,caps[k],u,dt,gap);
      },
      [&](const auto &p,const auto &state,const auto &contacts,const auto &w,const auto &o){
        return imex::acceptance_metric(p,caps,state,contacts,w,o);
      });
}
} // namespace SlurryNearfieldStep
#endif
