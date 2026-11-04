#include "../src/slurry_adaptive_step.h"
#include <iostream>
#include <vector>

namespace a=SlurryAdaptiveStep;
void require(bool value,const char* message) {
  if(!value)throw std::runtime_error(message);
}
void close(double x,double y,double tolerance,const char* message) {
  if(std::abs(x-y)>tolerance) {
    std::cerr<<message<<": "<<x<<" vs "<<y<<"\n";
    throw std::runtime_error(message);
  }
}
struct State {double x=0.,time=0.;std::vector<int> history;};

void transactional_controller() {
  const State initial{0.,0.,{}};
  a::Options options;
  std::size_t observed=0;double observed_time=0.;
  auto attempt=[](const State& before,double dt) {
    State after=before;after.x+=3.*dt;after.time+=dt;after.history.push_back(1);
    return a::Trial<State>::completed(std::move(after),std::pow(dt/.125,2));
  };
  auto result=a::advance(initial,1.,options,attempt,[&](const a::Trial<State>& trial,double dt) {
    require(bool(trial.next),"accepted observer received no state");
    ++observed;observed_time+=dt;
  });
  require(initial.history.empty()&&initial.x==0.,"input physical state was mutated");
  require(result.statistics.accepted==8&&result.statistics.rejected==3,
          "unexpected dyadic substep sequence");
  require(result.state.history.size()==8&&observed==8,"rejected trial history was committed");
  close(result.state.x,3.,1e-15,"transactional position");
  close(observed_time,1.,1e-15,"accepted observer time");
  require(result.statistics.elapsed==1.,"macro time endpoint is not exact");
  require(result.statistics.maximum_accepted_error<=1.,"unresolved trial was accepted");

  // A force-free state with no event must be allowed one full macro step.
  observed=0;
  auto free=a::advance(initial,.123456789,options,[](const State& before,double dt) {
    auto after=before;after.x+=dt;after.time+=dt;
    return a::Trial<State>::completed(std::move(after),0.);
  },[&](const a::Trial<State>&,double){++observed;});
  require(observed==1&&free.statistics.accepted==1&&free.statistics.rejected==0,
          "force-free macro interval was needlessly subdivided");
  require(free.statistics.elapsed==.123456789,"force-free final time is not exact");
  close(free.state.time,.123456789,0.,"force-free state time");
  options.initial_step=.08;
  auto remainder=a::advance(initial,.1,options,[](const State& before,double dt) {
    auto after=before;after.time+=dt;
    return a::Trial<State>::completed(std::move(after),.5);
  });
  require(remainder.statistics.elapsed==.1,"non-dyadic final remainder was omitted");
  close(remainder.state.time,.1,2e-17,"remainder state time");
  // A clipped final interval with small error must retain the previous useful
  // proposal as the next macro seed, instead of restarting from its tiny tail.
  options.initial_step=.03;options.maximum_step=.03;
  auto seed=a::advance(initial,.06000001,options,[](const State& before,double dt) {
    auto after=before;after.time+=dt;
    return a::Trial<State>::completed(std::move(after),0.);
  });
  close(seed.suggested_step,.03,0.,"final remainder destroyed the next macro seed");
  // Repeated non-binary intervals must not produce a machine-sized extra solve.
  options.initial_step=.1/5000.;options.maximum_step=options.initial_step;
  auto many=a::advance(initial,.1,options,[](const State& before,double dt) {
    auto after=before;after.time+=dt;
    return a::Trial<State>::completed(std::move(after),.5);
  });
  require(many.statistics.accepted==5000,"time summation created an extra tail solve");
  require(many.statistics.minimum_accepted_step>.99*options.initial_step,
          "machine-sized tail leaked into accepted timestep statistics");
}

void failure_semantics() {
  a::Options options;
  auto recovered=a::advance(State{},1.,options,[](const State& before,double dt) {
    if(dt>.25)return a::Trial<State>::retry("known nonlinear solve failure");
    auto next=before;next.time+=dt;
    return a::Trial<State>::completed(std::move(next),.5);
  });
  require(recovered.statistics.failed_solves==2&&recovered.statistics.accepted==4,
          "known failed solve was not retried transactionally");
  bool threw=false;
  try {
    a::advance(State{},1.,options,[](const State&,double)->a::Trial<State> {
      throw std::logic_error("programming error must propagate");
    });
  } catch(const std::logic_error&){threw=true;}
  require(threw,"arbitrary callback exception was swallowed");
  options.minimum_step=.1;
  threw=false;
  try {
    a::advance(State{},1.,options,[](const State& before,double) {
      return a::Trial<State>::completed(before,2.);
    });
  } catch(const a::Failure& failure) {
    threw=true;require(failure.statistics.accepted==0,"failed minimum step accepted a state");
  }
  require(threw,"minimum-step violation was accepted");
  options.minimum_step=0.;options.maximum_attempts=2;threw=false;
  try {
    a::advance(State{},1.,options,[](const State& before,double) {
      return a::Trial<State>::completed(before,2.);
    });
  } catch(const a::Failure& failure) {
    threw=true;require(failure.statistics.attempts==2,"attempt limit is not enforced");
  }
  require(threw,"attempt budget violation was accepted");
}

void gap_monitor() {
  const double range=.67e-9,tolerance=.1;
  auto capture=a::cohesion_indicator(1.1*range,-.1*range,range,tolerance);
  require(capture.ratio>1.,"capture from zero initial attraction was missed");
  auto release=a::cohesion_indicator(.99*range,10.*range,range,tolerance);
  require(release.traction_change<tolerance&&release.ratio>1.,
          "small frozen traction acting over a long jump was missed");
  auto preload=a::cohesion_indicator(-1000.*range,-2000.*range,range,tolerance);
  require(preload.ratio==0.&&preload.work_defect==0.,
          "saturated normal preload acquired an artificial range restriction");
  auto outside=a::cohesion_indicator(2.*range,100.*range,range,tolerance);
  require(outside.ratio==0.,"inactive force imposed an artificial restriction");
  close(a::integrated_fraction(-2.,3.),2.5,1e-15,"piecewise energy integral");
  close(a::integrated_fraction(3.,-2.),-2.5,1e-15,"reverse energy integral");
  auto small=a::cohesion_indicator(.4*range,.45*range,range,tolerance);
  close(small.traction_change,.05,1e-15,"traction variation");
  close(small.work_defect,.00125,1e-15,"frozen-force energy defect");
  require(a::contact_load_indicator(-.01,.001,100.,1.,.1)>1.,
          "loss of compressive preload was missed");
  require(a::contact_load_indicator(-.01,-.01,100.,1.,.1)==0.,
          "constant preload imposed an artificial restriction");
}

// One-dimensional drag/contact analogue used only to test the controller's
// temporal behavior, not as evidence for the full hydrodynamic suspension.
struct GapState {double gap=-.01,time=0.,rupture_time=-1.;};
struct ScalarResult {double rupture=0.;std::size_t accepted=0,rejected=0;};
ScalarResult release(double tolerance) {
  constexpr double k=100.,load=1.2,range=1.;
  a::Options options;options.initial_step=1.;options.maximum_step=1.;
  auto result=a::advance(GapState{},3.,options,[&](const GapState& before,double dt) {
    const double force=a::saturated_linear_fraction(before.gap,range);
    const double free=before.gap+dt*(load-force);
    GapState after=before;
    after.gap=free<0.?free/(1.+k*dt):free;
    after.time+=dt;
    if(after.gap>=range&&before.rupture_time<0.) {
      // Linear event interpolation removes only sample-time quantization;
      // the trajectory itself remains this IMEX trial's approximation.
      after.rupture_time=before.time+dt*(range-before.gap)/(after.gap-before.gap);
    }
    const auto error=a::cohesion_indicator(before.gap,after.gap,range,tolerance);
    const double contact=a::contact_load_indicator(before.gap,after.gap,k,1.,tolerance);
    return a::Trial<GapState>::completed(std::move(after),std::max(error.ratio,contact));
  });
  return {result.state.rupture_time,result.statistics.accepted,result.statistics.rejected};
}
void temporal_convergence() {
  // Before contact release: g=.002−.012 exp(−100t), so t0=log(6)/100.
  // After release: g=.2(exp(t−t0)−1), so rupture=t0+log(6).
  const double exact=std::log(6.)/100.+std::log(6.);
  double previous=std::numeric_limits<double>::infinity();
  for(double tolerance:{.1,.05,.025,.0125}) {
    const auto result=release(tolerance);
    const double error=std::abs(result.rupture-exact);
    std::cout<<"scalar_release tolerance="<<tolerance<<" rupture="<<result.rupture
             <<" exact="<<exact<<" error="<<error<<" accepted="<<result.accepted
             <<" rejected="<<result.rejected<<"\n";
    require(error<previous,"adaptive cohesive event did not converge under refinement");
    previous=error;
  }
  require(previous<.03,"refined cohesive release time remains inaccurate");
}

int main() {
  transactional_controller();failure_semantics();gap_monitor();temporal_convergence();
  std::cout<<"adaptive coupled-step controller PASS\n";
}
