#ifndef SLURRY_STIFFNESS_CONTROL_H
#define SLURRY_STIFFNESS_CONTROL_H

// Overlap-controlled normal contact stiffness for the overdamped IMEX step.
//
// The linear normal spring is a numerical penalty for the rigid-particle
// limit. In an overdamped contact network the endpoint overlap is fixed by
// force balance, overlap ~ N/kn, and not by the timestep: once a loaded
// network approaches the declared overlap limit, halving dt only approaches
// the same limit more slowly. This controller therefore changes kn, never dt:
//
//  * hard gate: a trial whose endpoint overlap exceeds `limit` is solved again
//    with a stiffer spring before any timestep reduction (retry_stiffness);
//  * predictive: after an accepted macro step above `raise`, kn is raised so
//    the next overlaps return near `target`; after a strain window entirely
//    below `lower`, kn is relaxed back toward its input value (after_step).
//
// The same idea is used by Mari et al., J. Rheol. 58, 1693 (2014) (kn chosen
// to keep overlaps below a few percent) and LF-DEM's auto_determine_knkt,
// here with a hard limit instead of a time-averaged target. Frozen Coulomb
// caps are the CALLER's responsibility: they must keep using the physical load
// that produced the current overlaps (load_kn * overlap), so a stiffness
// change never inflates sliding or rolling caps. Tangential and rolling
// stiffness and every history are left unchanged.
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace SlurryStiffnessControl {

struct Settings {
  double target=.02;          // overlap/min semiaxis aimed for after a raise
  double raise=.03;           // predictive raise above this accepted overlap
  double lower=.01;           // relax only after a quiet window below this
  double limit=.05;           // declared model limit; the endpoint gate
  double max_factor=1000.;    // kn <= base*max_factor
  double relax_strain=.02;    // strain window below `lower` before relaxing
  double max_step_raise=4.;   // largest predictive raise per macro step
  double min_retry_raise=2.,max_retry_raise=16.;
  int max_retries=8;          // stiffened re-solves of one trial
};

inline void validate(const Settings& s) {
  if(!(s.lower>0.&&s.lower<s.target&&s.target<s.raise&&s.raise<s.limit&&s.limit<=1.)||
     !(s.max_factor>=1.)||!std::isfinite(s.max_factor)||
     !(s.relax_strain>0.)||!std::isfinite(s.relax_strain)||
     !(s.max_step_raise>1.)||!std::isfinite(s.max_step_raise)||
     !(s.min_retry_raise>1.)||!(s.max_retry_raise>=s.min_retry_raise)||
     !std::isfinite(s.max_retry_raise)||s.max_retries<0)
    throw std::invalid_argument("kn_adapt requires 0 < lower < target < raise < limit <= 1, "
                                "max_factor >= 1 and relax_strain > 0");
}

struct State {
  double base=0.,current=0.;          // N/m (any consistent unit)
  double quiet_strain=0.,quiet_max=0.;
  long long retries=0,raises=0,lowers=0;
};

inline State initial_state(double base) {
  if(!(base>0.)||!std::isfinite(base))throw std::invalid_argument("kn_adapt requires a positive kn");
  State s;s.base=s.current=base;return s;
}

inline double maximum(const State& s,const Settings& o) {return s.base*o.max_factor;}

// Stiffness for re-solving a trial whose endpoint overlap fraction `ratio`
// exceeded the limit. Returns the current value when no raise is permitted;
// the caller then falls back to its ordinary rejection (dt reduction).
inline double retry_stiffness(const State& s,const Settings& o,double ratio) {
  if(!std::isfinite(ratio))throw std::invalid_argument("nonfinite overlap ratio");
  if(!(ratio>o.limit))return s.current;
  const double top=maximum(s,o);
  if(!(s.current<top))return s.current;
  const double factor=std::clamp(ratio/o.target,o.min_retry_raise,o.max_retry_raise);
  return std::min(top,s.current*factor);
}

// Predictive update after an ACCEPTED macro step. `ratio` is the largest
// accepted endpoint overlap fraction of that step and `strain` its strain
// increment. Returns true if the stiffness changed.
inline bool after_step(State& s,const Settings& o,double ratio,double strain) {
  if(!std::isfinite(ratio)||ratio<0.||!std::isfinite(strain)||strain<0.)
    throw std::invalid_argument("invalid overlap monitor input");
  const double old=s.current,top=maximum(s,o);
  if(ratio>o.raise) {
    s.quiet_strain=s.quiet_max=0.;
    if(s.current<top) {
      s.current=std::min(top,s.current*std::min(o.max_step_raise,ratio/o.target));
      ++s.raises;
    }
  } else if(ratio<o.lower&&s.current>s.base) {
    s.quiet_strain+=strain;s.quiet_max=std::max(s.quiet_max,ratio);
    if(s.quiet_strain>=o.relax_strain) {
      // Expected next overlap ~ quiet_max*old/new <= target.
      s.current=std::max(s.base,s.current*std::max(.5,s.quiet_max/o.target));
      s.quiet_strain=s.quiet_max=0.;
      ++s.lowers;
    }
  } else {
    s.quiet_strain=s.quiet_max=0.;
  }
  return s.current!=old;
}

} // namespace SlurryStiffnessControl
#endif
