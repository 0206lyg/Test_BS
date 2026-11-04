#ifndef SLURRY_ADAPTIVE_STEP_H
#define SLURRY_ADAPTIVE_STEP_H

// Transactional integration of a whole coupled particle state. The callback
// performs one physical solve at its supplied dt; rejected trials never become
// the input of another trial. This controller supplies no force clipping,
// contact law, or tolerance relaxation. Numerical caches may be reused by the
// callback, but positions, orientation, histories and accepted warm states must
// belong to State and must not be mutated through the const input reference.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace SlurryAdaptiveStep {

struct Options {
  double initial_step=0.; // zero: try the whole interval; otherwise previous seed
  double maximum_step=std::numeric_limits<double>::infinity();
  double minimum_step=0.; // zero: only floating-point progress and budget limits
  std::size_t maximum_attempts=100000;
};

struct Statistics {
  std::size_t attempts=0,accepted=0,rejected=0,failed_solves=0;
  double elapsed=0.;
  double minimum_accepted_step=std::numeric_limits<double>::infinity();
  double maximum_accepted_step=0.;
  double maximum_accepted_error=0.;
  double maximum_trial_error=0.;
};

struct Failure : std::runtime_error {
  Statistics statistics;
  explicit Failure(const std::string& message,const Statistics& stats)
      :std::runtime_error(message),statistics(stats) {}
};

template<class State> struct Trial {
  std::optional<State> next;
  // Ratio <= 1 means the caller's local accuracy/event test passed.
  double error_ratio=std::numeric_limits<double>::infinity();
  std::string diagnostic;
  static Trial completed(State state,double ratio) {
    Trial result;result.next.emplace(std::move(state));result.error_ratio=ratio;
    return result;
  }
  // Use only for a known, recoverable solve failure. Arbitrary exceptions from
  // the callback propagate; they are not silently converted to dt reductions.
  static Trial retry(const std::string& reason) {
    Trial result;result.diagnostic=reason;return result;
  }
};

template<class State> struct Result {
  State state;
  Statistics statistics;
  double suggested_step=0.;
};

// on_accept(trial, dt) may accumulate time-integrated stress/work and is called
// exactly once for accepted trials, before their state is moved into the result.
// The observer must not throw or mutate trial.next. The caller's macro state
// remains unchanged until advance() returns successfully if it was passed by
// value/copy, even if a later trial reaches the budget/minimum-step limit.
template<class State,class Attempt,class OnAccept>
inline Result<State> advance(State initial,double duration,const Options& options,
                             Attempt attempt,OnAccept on_accept) {
  if(!(duration>=0.)||!std::isfinite(duration)||!(options.initial_step>=0.)||
      !std::isfinite(options.initial_step)||!(options.maximum_step>0.)||
      !(options.minimum_step>=0.)||!std::isfinite(options.minimum_step)||
      options.minimum_step>options.maximum_step||options.maximum_attempts==0)
    throw std::invalid_argument("Invalid adaptive coupled-step options");
  Result<State> out{std::move(initial),{},0.};
  if(duration==0.){out.statistics.minimum_accepted_step=0.;return out;}
  double proposed=std::min(options.maximum_step,
      options.initial_step>0.?options.initial_step:duration);
  long double elapsed_exact=0.;
  std::string last_rejection;
  auto fail=[&](const char* why)->void {
    std::ostringstream message;
    message<<"Adaptive coupled step: "<<why<<"; elapsed="<<out.statistics.elapsed
           <<"; requested="<<duration<<"; next_dt="<<proposed
           <<"; attempts="<<out.statistics.attempts<<"; accepted="<<out.statistics.accepted
           <<"; rejected="<<out.statistics.rejected;
    if(!last_rejection.empty())message<<"; last_rejection="<<last_rejection;
    throw Failure(message.str(),out.statistics);
  };
  while(elapsed_exact<static_cast<long double>(duration)) {
    if(out.statistics.attempts>=options.maximum_attempts)fail("attempt budget exhausted");
    const double remaining=static_cast<double>(static_cast<long double>(duration)-elapsed_exact);
    double h=std::min(proposed,remaining);
    // Close a machine-roundoff tail BEFORE the solve: all of the requested
    // time still belongs to an actual physical trial. Long-double accumulation
    // prevents many accepted substeps from creating a spurious tiny last step.
    const double time_roundoff=4.*std::numeric_limits<double>::epsilon()*duration;
    if(remaining-h<=time_roundoff)h=remaining;
    // A final remainder smaller than minimum_step is permitted once; if that
    // trial is rejected, it cannot be subdivided below the requested limit.
    if(!(h>0.)||out.statistics.elapsed+h==out.statistics.elapsed)
      fail("timestep cannot advance floating-point time");
    if(h<options.minimum_step&&h!=remaining)fail("minimum timestep reached");
    ++out.statistics.attempts;
    auto trial=attempt(static_cast<const State&>(out.state),h);
    if(trial.next&&(!(trial.error_ratio>=0.)||!std::isfinite(trial.error_ratio)))
      fail("nonfinite or negative trial error indicator");
    if(trial.next)out.statistics.maximum_trial_error=std::max(
        out.statistics.maximum_trial_error,trial.error_ratio);
    const bool accepted=bool(trial.next)&&trial.error_ratio<=1.;
    if(!accepted) {
      ++out.statistics.rejected;
      if(!trial.next)++out.statistics.failed_solves;
      last_rejection=trial.diagnostic.empty()?
          (trial.next?"local accuracy/event indicator exceeded one":"coupled solve failed"):
          trial.diagnostic;
      proposed=.5*h;
      if(proposed<options.minimum_step)fail("rejected trial reached minimum timestep");
      continue;
    }
    on_accept(trial,h);
    out.state=std::move(*trial.next);
    ++out.statistics.accepted;
    out.statistics.minimum_accepted_step=std::min(out.statistics.minimum_accepted_step,h);
    out.statistics.maximum_accepted_step=std::max(out.statistics.maximum_accepted_step,h);
    out.statistics.maximum_accepted_error=std::max(out.statistics.maximum_accepted_error,trial.error_ratio);
    // Assign the exact requested endpoint when accepting the final remainder.
    elapsed_exact=h==remaining?static_cast<long double>(duration):elapsed_exact+h;
    out.statistics.elapsed=static_cast<double>(elapsed_exact);
    // Dyadic growth avoids assuming a formal order for a user-supplied event
    // indicator. Low-error steps grow; accepted near-limit steps retain dt.
    const double next_candidate=trial.error_ratio<=.25?2.*h:h;
    // Do not seed the next macro interval with an arbitrarily tiny final-time
    // remainder when that remainder had a comfortably small error indicator.
    // This avoids unnecessary growth steps caused only by time bookkeeping.
    proposed=std::min(options.maximum_step,
        h==remaining&&h<proposed&&trial.error_ratio<=.25?
        std::max(proposed,next_candidate):next_candidate);
  }
  out.suggested_step=proposed;
  return out;
}

template<class State,class Attempt>
inline Result<State> advance(State initial,double duration,const Options& options,Attempt attempt) {
  return advance(std::move(initial),duration,options,std::move(attempt),
      [](const Trial<State>&,double){});
}

inline double saturated_linear_fraction(double gap,double range) {
  if(!(range>0.)||!std::isfinite(range)||!std::isfinite(gap))
    throw std::invalid_argument("Invalid cohesive gap/range monitor input");
  return gap<=0.?1.:(gap>=range?0.:1.-gap/range);
}

struct CohesionIndicator {
  double traction_change=0.; // |Fnew/F0 - Fold/F0| at fixed curvature
  double work_defect=0.;     // |Delta U - Fold Delta g| / (F0*range)
  double ratio=0.;
};

// Exact piecewise integral of the normalized traction over a monotone gap
// segment. It stays well conditioned for two deeply overlapped endpoints.
inline double integrated_fraction(double begin,double end) {
  if(begin==end)return 0.;
  if(end<begin)return -integrated_fraction(end,begin);
  double integral=0.;
  if(begin<0.)integral+=std::min(end,0.)-begin;
  const double lo=std::max(begin,0.),hi=std::min(end,1.);
  if(hi>lo)integral+=(hi-lo)*(1.-.5*(hi+lo));
  return integral;
}

// This is a local gap-kernel variation/work indicator, not a global truncation
// error proof. The caller must also assess curvature-dependent force/torque,
// background forces, moving geometry, and whole-network temporal convergence.
// It detects capture from outside the range when Fold is zero. The work term
// also catches a small residual traction incorrectly applied over a long jump.
inline CohesionIndicator cohesion_indicator(double old_gap,double new_gap,double range,
    double force_fraction,double work_fraction) {
  if(!(force_fraction>0.)||!(work_fraction>0.)||!std::isfinite(force_fraction)||
      !std::isfinite(work_fraction))
    throw std::invalid_argument("Invalid cohesive indicator tolerance");
  const double old_force=saturated_linear_fraction(old_gap,range);
  const double new_force=saturated_linear_fraction(new_gap,range);
  CohesionIndicator out;
  out.traction_change=std::abs(new_force-old_force);
  // In the negative-gap branch U is exactly linear at fixed curvature.
  if(!(old_gap<=0.&&new_gap<=0.)) {
    const double x0=old_gap/range,x1=new_gap/range;
    out.work_defect=std::abs(integrated_fraction(x0,x1)-old_force*(x1-x0));
  }
  out.ratio=std::max(out.traction_change/force_fraction,out.work_defect/work_fraction);
  return out;
}

inline CohesionIndicator cohesion_indicator(double old_gap,double new_gap,double range,
                                            double force_fraction) {
  return cohesion_indicator(old_gap,new_gap,range,force_fraction,.5*force_fraction*force_fraction);
}

// A stable implicit contact step can still erase a compressive preload in one
// step and delay the g=0 release event. Moreover its old normal load fixes the
// friction cap. Monitor that load's variation independently of cohesion, whose
// saturated value may barely change during this event. This is an accuracy
// indicator, not an explicit-elastic stability restriction. The force_scale is
// a physical reference such as the pair's contact attraction F0, in matching
// units; large imposed compressive loads participate in the denominator.
inline double contact_load_indicator(double old_gap,double new_gap,double stiffness,
                                     double force_scale,double fraction) {
  if(!std::isfinite(old_gap)||!std::isfinite(new_gap)||!(stiffness>=0.)||
      !std::isfinite(stiffness)||!(force_scale>0.)||!std::isfinite(force_scale)||
      !(fraction>0.)||!std::isfinite(fraction))
    throw std::invalid_argument("Invalid contact-load monitor input");
  const double old_load=stiffness*std::max(-old_gap,0.);
  const double new_load=stiffness*std::max(-new_gap,0.);
  return std::abs(new_load-old_load)/(fraction*std::max({force_scale,old_load,new_load}));
}

} // namespace SlurryAdaptiveStep
#endif
