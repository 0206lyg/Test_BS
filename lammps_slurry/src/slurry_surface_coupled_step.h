#ifndef SLURRY_SURFACE_COUPLED_STEP_H
#define SLURRY_SURFACE_COUPLED_STEP_H

// Experimental monolithic endpoint solve. Full surface forces and torques are
// evaluated at every trial endpoint, together with endpoint Coulomb loads. The
// Jacobian uses the actual endpoint-gap chain rule for contact compression,
// Coulomb caps and the stiff surface-gap slope. Surface force-direction and
// curvature derivatives are lagged: this is quasi-Newton, NOT an exact Hessian. Acceptance is
// always the full physical force/mobility residual. Temporal accuracy is the
// caller's responsibility; convergence does not validate a large timestep.
#include "slurry_nearfield_step.h"
#include <Eigen/SparseLU>

namespace SlurrySurfaceCoupledStep {
namespace bs=SlurryBridgeSolver;
namespace step=SlurryBridgeStep;
namespace ns=SlurryNearfieldSolver;
namespace imex=SlurryNearfieldImexSolver;

struct SurfaceEvaluation {
  bs::Vector external;
  // dF_attraction/dgap, in N/m, in original.contacts order. A positive
  // attraction pulls the two surfaces together; the opening slope is negative.
  std::vector<double> attraction_gap_derivative;
  // Optional generalized applied-force derivatives with respect to endpoint
  // gap. Both arrays must be supplied together. They override the scalar slope
  // construction, e.g. for an averaged-force/discrete-gradient callback.
  std::vector<bs::V6> external_gap_derivative_i,external_gap_derivative_j;
};
struct Workspace {
  Eigen::SparseLU<bs::Sparse> factor;
  bs::Sparse jacobian;
  bool ready=false;
  std::size_t symbolic_analyses=0,numeric_factorizations=0,solves=0,
              evaluations=0,line_search_evaluations=0;
};
struct Result {
  step::Result solved;
  bs::Vector external;
  imex::FrozenCaps caps;
  int iterations=0;
  std::size_t linear_solves=0,numeric_factorizations=0,symbolic_analyses=0,evaluations=0;
  double residual_ratio=0.;
};
struct Failure:bs::NonlinearFailure {
  Failure(const std::string&message,const bs::Result&state,double residual,double tolerance)
    :bs::NonlinearFailure(message,state,residual,tolerance){}
};
namespace detail {
struct Evaluation {
  bs::Result state;
  step::Endpoint endpoint;
  SurfaceEvaluation surface;
  imex::FrozenCaps caps;
  bs::Vector residual;
  double norm=0.,tolerance=0.,merit=0.,roundoff=0.;
};
inline void prepare_roundoff(const bs::Problem&p,const step::Geometry&g,imex::Workspace&w) {
  if(!w.use_roundoff_budget){w.contact_roundoff_length.clear();return;}
  w.contact_roundoff_length.resize(p.contacts.size());
  for(std::size_t k=0;k<p.contacts.size();++k) {
    const auto&c=p.contacts[k];const auto&a=g.bodies[c.i];const auto&b=g.bodies[c.j];
    const auto shift=g.lattice*g.images[k].cast<double>();
    w.contact_roundoff_length[k]=2.*(step::vector(a.position).lpNorm<1>()+
      step::vector(b.position).lpNorm<1>()+shift.lpNorm<1>()+
      *std::max_element(a.axes.begin(),a.axes.end())+*std::max_element(b.axes.begin(),b.axes.end()));
  }
}
template<class SurfaceForce>
inline Evaluation evaluate(const bs::Problem&p,const step::Geometry&g,
    const bs::Vector&u,imex::Workspace&w,SurfaceForce&force,double base_tolerance,
    Workspace&cw) {
  Evaluation e;++cw.evaluations;
  e.endpoint=g.advance(p,u,false,true);e.surface=force(e.endpoint);
  if(e.surface.external.size()!=6*p.count||!e.surface.external.allFinite()||
      e.surface.attraction_gap_derivative.size()!=p.contacts.size())
    throw std::invalid_argument("coupled surface: invalid endpoint force or tangent size");
  if((!e.surface.external_gap_derivative_i.empty()||!e.surface.external_gap_derivative_j.empty())&&
      (e.surface.external_gap_derivative_i.size()!=p.contacts.size()||
       e.surface.external_gap_derivative_j.size()!=p.contacts.size()))
    throw std::invalid_argument("coupled surface: invalid generalized surface tangent size");
  auto&s=e.state;s.velocity=u;s.q=bs::detail::embed(u,p.count)+p.prescribed;
  s.mid_moment.resize(11*p.count);bs::Vector me(11*p.count);
  for(int i=0;i<p.count;++i) {
    s.mid_moment.segment<11>(11*i)=w.self_inverse[i]*s.q.segment<11>(11*i);
    me.segment<11>(11*i)=p.self_mobility[i]*s.mid_moment.segment<11>(11*i)-s.q.segment<11>(11*i);
  }
  s.total_moment=s.mid_moment+p.near_resistance*s.q;
  e.caps.reserve(p.contacts.size());s.contacts.reserve(p.contacts.size());
  for(std::size_t k=0;k<p.contacts.size();++k) {
    if(!std::isfinite(e.surface.attraction_gap_derivative[k]))
      throw std::invalid_argument("coupled surface: nonfinite endpoint tangent");
    if(!e.surface.external_gap_derivative_i.empty()&&
        (!e.surface.external_gap_derivative_i[k].allFinite()||
         !e.surface.external_gap_derivative_j[k].allFinite()))
      throw std::invalid_argument("coupled surface: nonfinite generalized surface tangent");
    const auto&c=p.contacts[k];
    e.caps.push_back(c.kn*std::max(c.onset-e.endpoint.gaps[k],0.));
    s.contacts.push_back(bs::contact_response(c,u,p.dt,e.endpoint.gaps[k]));
  }
  bs::Vector applied=e.surface.external;
  for(std::size_t k=0;k<p.contacts.size();++k) {
    const auto&c=p.contacts[k];const auto wrench=bs::contact_wrench(s.contacts[k]);
    applied.segment<6>(6*c.i).noalias()-=bs::generalized_contact_map(c,true).transpose()*wrench;
    applied.segment<6>(6*c.j).noalias()-=bs::generalized_contact_map(c,false).transpose()*wrench;
  }
  e.residual=bs::detail::select(s.total_moment,p.count)-applied;
  e.norm=std::sqrt(ns::detail::force_metric_squared(e.residual,w)+ns::detail::mobility_metric_squared(me,w));
  // Use the actual applied endpoint load without copying the contact graph
  // and sparse resistance matrix on every line-search evaluation.
  if(w.use_roundoff_budget)
    e.roundoff=imex::roundoff_tolerance(p,e.caps,s,s.contacts,w,base_tolerance,&e.surface.external);
  e.tolerance=base_tolerance+e.roundoff;e.merit=.5*e.norm*e.norm;
  s.force_residual_norm=e.residual.norm();s.mobility_residual_norm=me.norm();s.residual_norm=e.norm;
  if(!u.allFinite()||!std::isfinite(e.norm+e.tolerance))
    throw std::runtime_error("coupled surface: nonfinite endpoint residual");
  return e;
}
inline bs::Sparse jacobian(const bs::Problem&p,const Evaluation&e,const imex::Workspace&w) {
  std::vector<Eigen::Triplet<double>> entries;entries.reserve(144*p.contacts.size());
  auto add=[&](int i,int j,const bs::Matrix6&b) {
    for(int r=0;r<6;++r)for(int c=0;c<6;++c)
      if(b(r,c)!=0.)entries.emplace_back(6*i+r,6*j+c,b(r,c));
  };
  for(std::size_t k=0;k<p.contacts.size();++k) {
    const auto&c=p.contacts[k];const auto&r=e.state.contacts[k];
    const bs::Matrix6 d=bs::contact_jacobian(r);
    const auto a=bs::generalized_contact_map(c,true),b=bs::generalized_contact_map(c,false);
    const auto&gi=e.endpoint.gap_derivative_i[k];const auto&gj=e.endpoint.gap_derivative_j[k];
    // contact_response's stored tangent assumes g=g0+dt*n0.vrel. Its force
    // value uses the true endpoint gap. Replace ONLY that normal-load chain
    // with dg_endpoint/du; elastic tangential velocity derivatives stay exact
    // for the existing frozen contact map and history convention.
    bs::V6 load_derivative=bs::V6::Zero();
    if(r.active) {
      load_derivative.head<3>()=-c.normal;
      if(r.sliding)load_derivative.head<3>()+=c.mu*r.trial_history.normalized();
      if(r.rolling)load_derivative.tail<3>()+=c.mu_roll*c.roll_length*r.rolling_trial.normalized();
      if(r.twisting)load_derivative.tail<3>()+=c.mu_twist*c.twist_length*
          std::copysign(1.,r.twisting_trial)*c.normal;
    }
    const Eigen::Matrix<double,1,6> delta_i=gi-p.dt*c.normal.transpose()*c.Bi;
    const Eigen::Matrix<double,1,6> delta_j=gj-p.dt*c.normal.transpose()*c.Bj;
    const bs::Matrix6 di=d*a-c.kn*load_derivative*delta_i;
    const bs::Matrix6 dj=d*b-c.kn*load_derivative*delta_j;
    bs::V6 fi,fj;
    if(!e.surface.external_gap_derivative_i.empty()) {
      fi=e.surface.external_gap_derivative_i[k];fj=e.surface.external_gap_derivative_j[k];
    } else {
      const auto&gap=e.endpoint.pair_geometry[k];
      const auto normal=step::vector(gap.normal);
      fi.head<3>()=e.surface.attraction_gap_derivative[k]*normal;
      fj.head<3>()=-fi.head<3>();
      fi.tail<3>()=step::vector(gap.leverI).cross(fi.head<3>());
      fj.tail<3>()=step::vector(gap.leverJ).cross(fj.head<3>());
    }
    add(c.i,c.i,a.transpose()*di-fi*gi);add(c.i,c.j,a.transpose()*dj-fi*gj);
    add(c.j,c.i,b.transpose()*di-fj*gi);add(c.j,c.j,b.transpose()*dj-fj*gj);
  }
  bs::Sparse contact(6*p.count,6*p.count);contact.setFromTriplets(entries.begin(),entries.end());
  bs::Sparse j=w.matrix+contact;j.makeCompressed();return j; // no symmetrization
}
inline bool direction(const bs::Sparse&j,const bs::Vector&r,const imex::Workspace&w,
                      Workspace&cw,bs::Vector&du) {
  // Congruence scaling controls translation/rotation units without modifying
  // the physical Jacobian or imposing positive definiteness.
  bs::Vector scale(j.rows());
  for(int i=0;i<j.rows();++i) {
    const double magnitude=std::max(std::abs(j.coeff(i,i)),std::abs(w.matrix.coeff(i,i)));
    if(!(magnitude>0.)||!std::isfinite(magnitude))return false;
    scale[i]=1./std::sqrt(magnitude);
  }
  bs::Sparse scaled=j;
  for(int c=0;c<scaled.outerSize();++c)for(bs::Sparse::InnerIterator it(scaled,c);it;++it)
    it.valueRef()*=scale[it.row()]*scale[it.col()];
  const bool same=cw.ready&&bs::detail::same_pattern(scaled,cw.jacobian);
  cw.ready=false;cw.jacobian=scaled;
  if(!same){cw.factor.analyzePattern(cw.jacobian);++cw.symbolic_analyses;}
  cw.factor.factorize(cw.jacobian);++cw.numeric_factorizations;
  if(cw.factor.info()!=Eigen::Success)return false;
  cw.ready=true;
  bs::Vector rhs=-scale.cwiseProduct(r),z=cw.factor.solve(rhs);++cw.solves;
  if(cw.factor.info()!=Eigen::Success||!z.allFinite())return false;
  du=scale.cwiseProduct(z);
  // Component-scaled backward error, rather than a dimensionally mixed norm
  // of forces and torques. Allow at most two direct refinement corrections.
  for(int attempt=0;attempt<3;++attempt) {
    const bs::Vector defect=scale.cwiseProduct(j*du+r);
    const double target=1e-10*std::max(rhs.norm(),std::numeric_limits<double>::min());
    if(defect.norm()<=target)return true;
    if(attempt<2){z=cw.factor.solve(defect);++cw.solves;du-=scale.cwiseProduct(z);}
  }
  return false;
}
} // namespace detail

template<class SurfaceForce>
inline Result solve(const bs::Problem&original,const step::Geometry&geometry,
    const bs::Options&options,imex::Workspace&workspace,SurfaceForce endpoint_force,
    const bs::Result*warm=nullptr,Workspace*coupled_workspace=nullptr) {
  if(!(original.dt>0.)||options.maximum_newton_iterations<=0||
      options.maximum_line_search_iterations<=0)
    throw std::invalid_argument("coupled surface: invalid timestep or solver limits");
  Workspace local;Workspace&cw=coupled_workspace?*coupled_workspace:local;
  const std::size_t start_solves=cw.solves,start_numeric=cw.numeric_factorizations,
      start_symbolic=cw.symbolic_analyses,start_evaluations=cw.evaluations;
  bs::Options base=options;base.trace_callback={};
  ns::prepare_resistance(original,workspace,base);
  detail::prepare_roundoff(original,geometry,workspace);
  const double reference=imex::rhs_norm(original,imex::freeze_caps(original),workspace);
  const double base_tolerance=options.absolute_tolerance+options.relative_tolerance*reference;
  bs::Vector u;
  if(warm&&warm->velocity.size()==6*original.count&&warm->velocity.allFinite())u=warm->velocity;
  else u=ns::solve(original,base,workspace).velocity;
  auto at=detail::evaluate(original,geometry,u,workspace,endpoint_force,base_tolerance,cw);
  int updates=0,descent=0;
  while(at.norm>at.tolerance) {
    at.state.newton_iterations=updates;at.state.descent_steps=descent;
    at.state.linear_iterations=int(cw.solves-start_solves);
    if(updates>=options.maximum_newton_iterations)
      throw Failure("coupled surface: quasi-Newton iteration limit",at.state,at.norm,at.tolerance);
    const auto j=detail::jacobian(original,at,workspace);
    bs::Vector du;const bool direct=detail::direction(j,at.residual,workspace,cw,du);
    const bs::Vector weighted=workspace.factor.solve(at.residual),gradient=j.transpose()*weighted;
    const double previous_norm=at.norm;
    bool accepted=false;double accepted_alpha=0.;
    for(int attempt=0;attempt<2&&!accepted;++attempt) {
      if(attempt==0&&!direct)continue;
      if(attempt==1){du=-workspace.factor.solve(gradient);++descent;}
      if(!du.allFinite())continue;
      const double slope=gradient.dot(du);
      if(!(slope<0.)||!std::isfinite(slope))continue;
      double alpha=1.;
      for(int back=0;back<options.maximum_line_search_iterations;++back,alpha*=.5) {
        ++cw.line_search_evaluations;
        auto trial=detail::evaluate(original,geometry,at.state.velocity+alpha*du,
                                    workspace,endpoint_force,base_tolerance,cw);
        if(trial.norm<=trial.tolerance||trial.merit<=at.merit+1e-4*alpha*slope) {
          at=std::move(trial);accepted=true;accepted_alpha=alpha;break;
        }
      }
    }
    if(!accepted){at.state.linear_iterations=int(cw.solves-start_solves);
      throw Failure("coupled surface: physical-residual line search failed",at.state,at.norm,at.tolerance);}
    ++updates;
    // A vanishing step with no residual progress indicates a nonsmooth merit
    // barrier, not a reason to repeat the same expensive factorization. This
    // is the same recoverable stagnation guard used by the existing implicit
    // Coulomb solver. It never relaxes acceptance or commits a failed state.
    if(at.norm>at.tolerance&&accepted_alpha<1e-6&&
        previous_norm-at.norm<1e-4*previous_norm) {
      at.state.newton_iterations=updates;at.state.descent_steps=descent;
      at.state.linear_iterations=int(cw.solves-start_solves);
      throw Failure("coupled surface: opening/cap merit stagnation",at.state,at.norm,at.tolerance);
    }
    if(options.trace_callback){bs::IterationTrace t;t.phase="surface_coupled_quasi_newton";
      t.iteration=t.total_newton_iterations=updates;t.residual=t.merit_residual=at.norm;
      t.tolerance=at.tolerance;t.step_length=accepted_alpha;options.trace_callback(t);}
  }
  // Only the converged endpoint receives the independent new-contact check.
  auto checked=geometry.advance(original,at.state.velocity,true,true);
  auto&s=at.state;s.newton_iterations=updates;s.descent_steps=descent;
  // Existing diagnostics call this field linear_iterations; for this direct
  // path it counts LU triangular solves, including refinement solves.
  s.linear_iterations=int(cw.solves-start_solves);
  s.relative_residual=at.norm/std::max({reference,options.absolute_tolerance,std::numeric_limits<double>::min()});
  s.contact_power=0.;for(const auto&r:s.contacts)s.contact_power+=bs::contact_power(r);
  s.hydro_power=s.q.dot(s.total_moment);
  workspace.last_roundoff_tolerance=at.roundoff;workspace.last_acceptance_tolerance=at.tolerance;
  workspace.last_acceptance_ratio=at.norm/std::max(at.tolerance,std::numeric_limits<double>::min());
  workspace.maximum_roundoff_tolerance=std::max(workspace.maximum_roundoff_tolerance,at.roundoff);
  Result out;out.solved.solution=std::move(s);out.solved.endpoint=std::move(checked);
  out.solved.geometry_iterations=updates+1;
  for(std::size_t k=0;k<original.contacts.size();++k) {
    const auto&c=original.contacts[k];const auto&r=out.solved.solution.contacts[k];
    const double linear=c.gap+original.dt*c.normal.dot(r.relative);
    out.solved.maximum_linear_gap_defect=std::max(out.solved.maximum_linear_gap_defect,
      std::abs(out.solved.endpoint.gaps[k]-linear));
  }
  out.external=std::move(at.surface.external);out.caps=std::move(at.caps);
  out.iterations=updates;out.residual_ratio=workspace.last_acceptance_ratio;
  out.linear_solves=cw.solves-start_solves;out.numeric_factorizations=cw.numeric_factorizations-start_numeric;
  out.symbolic_analyses=cw.symbolic_analyses-start_symbolic;out.evaluations=cw.evaluations-start_evaluations;
  return out;
}
} // namespace SlurrySurfaceCoupledStep
#endif
