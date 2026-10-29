#include "slurry_general_provider.h"
#include <iomanip>
#include <sstream>
namespace SlurryGeneral {
Mat11 reduced_response(const Mat17& g,const Map6& b) {
  const Eigen::Matrix<double,12,12> a=g.topLeftCorner<12,12>();
  const Eigen::Matrix<double,12,5> c=g.topRightCorner<12,5>();
  Eigen::LLT<Eigen::Matrix<double,12,12>> af(a);
  if(af.info()!=Eigen::Success)throw std::runtime_error("General pair velocity block is not positive definite");
  const Eigen::Matrix<double,6,6> m=b*af.solve(b.transpose());
  Eigen::LLT<Eigen::Matrix<double,6,6>> mf(m);
  if(mf.info()!=Eigen::Success)throw std::runtime_error("General relative mobility is not positive definite");
  const Eigen::Matrix<double,6,6> k=mf.solve(Eigen::Matrix<double,6,6>::Identity());
  const Eigen::Matrix<double,12,5> ac=af.solve(c);
  const Eigen::Matrix<double,6,5> f=b*ac;
  Mat11 h;h.topLeftCorner<6,6>()=k;h.topRightCorner<6,5>()=k*f;
  h.bottomLeftCorner<5,6>()=h.topRightCorner<6,5>().transpose();
  h.bottomRightCorner<5,5>()=g.bottomRightCorner<5,5>()-c.transpose()*ac+f.transpose()*k*f;
  return .5*(h+h.transpose()).eval();
}
SlurryFiniteGap::Pair Provider::make_pair(const SlurryFiniteGap::gr::Body& bi,
    const SlurryFiniteGap::gr::Body& bj,const SlurryFiniteGap::gr::GapResult& gap,
    double eta,double solver_length,double range,double inner_ratio,
    std::uint64_t id_i,std::uint64_t id_j) const {
  namespace fg=SlurryFiniteGap;namespace gp=SlurryGrandPair;namespace gr=fg::gr;
  if(!(epsilon_m>0.)||!std::isfinite(epsilon_m))
    throw std::invalid_argument("General fluid gap floor must be positive");
  fg::Pair p;p.frame=fg::canonical_frame(bi,bj,gap.normal,eta,solver_length);
  p.cutoff=gp::ar_scaled_range(bi.axes[0],bj.axes[0],range);
  if(!(inner_ratio>0&&inner_ratio<1))throw std::invalid_argument("Invalid general interaction window");
  p.weight=gp::range_weight(gap.gap,inner_ratio*p.cutoff,p.cutoff);
  if(p.weight==0.)return p;
  const double c=bi.axes[2],a=bi.axes[0]/c,ell=p.frame.model_length;
  if(!(c>0&&a>=1.)||std::abs(bi.axes[1]/c-a)>1e-10*a)
    throw std::invalid_argument("General provider requires equal-sized oblate spheroids, AR>=1");
  for(int d=0;d<3;++d)if(std::abs(bi.axes[d]-bj.axes[d])>1e-10*ell)
    throw std::invalid_argument("General pair particles must have equal semiaxes");
  std::array<Vec3,2> axis;const gr::Body* bodies[]={&bi,&bj};
  for(int b=0;b<2;++b){const auto v=gp::to_local(p.frame,
      {bodies[b]->rotation[2],bodies[b]->rotation[5],bodies[b]->rotation[8]});
    axis[b]=Vec3(v[0],v[1],v[2]);}
  // Fluid regularization changes only the hydro evaluation geometry. The
  // mechanical overlap, contact onset and friction use the physical gap.
  const auto geometry=geometry_from_axes(a,std::max(gap.gap,epsilon_m)/c,axis[0],axis[1]);
  const auto result=evaluate_model(geometry,options);
  gp::Metadata metadata;metadata.kind=gp::MatrixKind::PairCorrection;
  metadata.model_id=relative_closure?"general_kim_claeys_relative_pair_v1":"general_kim_claeys_pair_additive_diagnostic_v1";
  std::ostringstream ref;ref<<std::setprecision(17)<<"geometry_only;AR="<<a
    <<";fluid_gap_over_c="<<geometry.gap<<";quadrature="<<result.nrad<<'x'<<result.nphi
    <<";tolerance="<<options.tolerance<<";near_cutoff="<<options.cutoff_factor
    <<";normal_log="<<(options.derived_normal_log?"derived_experimental":"leading")
    <<";near_completion="<<(options.energy_completion?"energy":"raw")
    <<";pair="<<id_i<<','<<id_j<<";collective_drag="<<(relative_closure?"isolated":"pair_additive");
  metadata.reference_id=ref.str();
  // R_Python multiplies eta*c; this existing adapter multiplies eta*ell,
  // with ell=a*c and identical velocity/strain normalization, hence /a.
  if(relative_closure) {
    const auto li=gp::to_local(p.frame,gr::scale(gr::add(gap.leverI,gr::scale(gap.normal,.5*gap.gap)),1/ell));
    const auto lj=gp::to_local(p.frame,gr::scale(gr::sub(gap.leverJ,gr::scale(gap.normal,.5*gap.gap)),1/ell));
    const auto projection=gp::relative_projection(li,lj,gp::SlipConvention::NonaffineMidpoint);
    Map6 b;for(int i=0;i<6;++i)for(int j=0;j<12;++j)b(i,j)=projection[17*i+j];
    const Mat11 delta=(reduced_response(result.grand,b)-reduced_response(result.isolated,b))/a;
    gp::Mat11 values{};for(int i=0;i<11;++i)for(int j=0;j<11;++j)values[11*i+j]=delta(i,j);
    p.model=gp::lift_reduced(values,projection,metadata);
  } else {
    const Mat17 delta=(result.grand-result.isolated)/a;
    p.model.metadata=metadata;
    for(int i=0;i<17;++i)for(int j=0;j<17;++j)p.model.resistance[17*i+j]=delta(i,j);
  }
  p.model=gp::window_correction(p.model,p.weight);
  gp::validate(p.frame,p.model);p.active=true;return p;
}
}
