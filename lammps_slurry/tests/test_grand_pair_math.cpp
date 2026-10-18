#include "slurry_grand_pair_math.h"
#include "slurry_finite_gap_math.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
using namespace SlurryGrandPair;
namespace {
double maximum_relative_error=0;
void close(double actual,double expected,double tol,const char* what) {
  double e=std::abs(actual-expected)/std::max({1.,std::abs(actual),std::abs(expected)});
  maximum_relative_error=std::max(maximum_relative_error,e);
  if(!(e<tol)) {std::cerr<<what<<": "<<std::setprecision(17)<<actual<<" != "<<expected<<" ("<<e<<")\n";std::exit(1);}
}
void check(bool v,const char* what){if(!v){std::cerr<<what<<'\n';std::exit(1);}}
Vec3 rotate(const Mat3&q,const Vec3&v) {Vec3 out{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)out[i]+=q[3*i+j]*v[j];return out;}
Mat3 rotate_tensor(const Mat3&q,const Mat3&e) {Mat3 out{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int l=0;l<3;++l)out[3*i+j]+=q[3*i+k]*e[3*k+l]*q[3*j+l];return out;}
Vec12 rotate_vectors(const Mat3&q,const Vec12&v) {Vec12 out{};for(int b=0;b<4;++b){const auto u=rotate(q,{v[3*b],v[3*b+1],v[3*b+2]});for(int i=0;i<3;++i)out[3*b+i]=u[i];}return out;}
Model manufactured() {
  // Independent dense Gram matrix; this tests the transport algebra, NOT a
  // spheroid lubrication model or a physical claim about a pair correction.
  Model m;m.metadata.model_id="manufactured_algebra_test";
  m.metadata.reference_id="deterministic_17x17_gram_v1";
  std::array<double,289>a{};
  for(int i=0;i<17;++i)for(int j=0;j<17;++j)a[17*i+j]=std::sin(3*i+7*j+.3)+(i==j?3.:0.);
  for(int i=0;i<17;++i)for(int j=0;j<17;++j)for(int k=0;k<17;++k)m.resistance[17*i+j]+=a[17*k+i]*a[17*k+j];
  return m;
}
}
int main() {
  // A moving face/face pair crosses the canonical-frame pole. Its director
  // projection must not amplify cancellation into a non-orthogonal frame.
  namespace gr=slurry::gr_re2::graphite;
  const gr::Vec3 pole{.6,.3,std::sqrt(.55)},tangent=gr::normalized(gr::cross(pole,{1.,0.,0.}));
  gr::Body body_i,body_j;body_i.axes=body_j.axes={1.5e-6,1.5e-6,150e-9};
  for(double tilt:{1e-2,1e-4,1e-6,1e-8,1e-9,1e-10,0.}){
    const auto director=gr::normalized(gr::add(pole,gr::scale(tangent,tilt)));
    const auto turn=gr::normalized(gr::cross(gr::Vec3{0,0,1},director));
    body_i.rotation=body_j.rotation=gr::rotationIncrement(gr::scale(turn,std::acos(director[2])));
    const auto pole_frame=SlurryFiniteGap::canonical_frame(body_i,body_j,pole,.01,1e-6);
    validate(pole_frame,{});
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
      double value=0;for(int k=0;k<3;++k)value+=pole_frame.world_from_local[3*k+i]*pole_frame.world_from_local[3*k+j];
      close(value,i==j?1.:0.,2e-14,"near-pole frame orthogonality");
    }
    for(int k=0;k<3;++k)close(pole_frame.world_from_local[3*k+2],pole[k],2e-14,"near-pole frame normal unchanged");
  }
  const auto model=manufactured();Frame frame;frame.model_length=2.3;frame.solver_length=4.1;frame.viscosity=.7;
  Inputs x;
  for(int k=0;k<12;++k){x.velocity[k]=.4*std::cos(2*k+.5);x.background[k]=.2*std::sin(3*k+.8);}
  x.strain={{.07,.13,-.02,.13,-.03,.04,-.02,.04,-.04}};
  x.image_nonaffine={{.017,-.013,.023}};
  validate(frame,model);
  const auto y=evaluate(model,frame,x);
  const auto a=assemble(model,frame,x);
  double energy=a.energy_constant;
  for(int i=0;i<12;++i){energy+=x.velocity[i]*a.affine[i];for(int j=0;j<12;++j)energy+=.5*x.velocity[i]*a.resistance[12*i+j]*x.velocity[j];}
  close(energy,y.energy,1e-12,"assembled energy");
  for(int i=0;i<12;++i){double g=a.affine[i];for(int j=0;j<12;++j)g+=a.resistance[12*i+j]*x.velocity[j];close(g,y.gradient[i],1e-12,"assembled gradient");}
  double power=0;for(int k=0;k<12;++k)power+=(x.velocity[k]-x.background[k])*y.gradient[k];
  for(int k=0;k<3;++k)power+=x.image_nonaffine[k]*y.gradient[6+k];
  for(int k=0;k<9;++k)power+=x.strain[k]*y.stress[k];
  close(power,y.power,1e-12,"force/torque/strain power balance");
  const double step=1e-6;
  for(int k=0;k<12;++k){auto plus=x,minus=x;plus.velocity[k]+=step;minus.velocity[k]-=step;
    close((evaluate(model,frame,plus).energy-evaluate(model,frame,minus).energy)/(2*step),y.gradient[k],1e-7,"particle energy derivative");}
  for(int k=0;k<5;++k){auto plus=x,minus=x;for(int j=0;j<9;++j){plus.strain[j]+=step*strain_basis()[k][j];minus.strain[j]-=step*strain_basis()[k][j];}
    close((evaluate(model,frame,plus).energy-evaluate(model,frame,minus).energy)/(2*step),y.stress_coordinates[k],1e-7,"stress energy derivative");}
  // Proper world rotation Q: power invariant, vector/tensor conjugates covariant.
  const double angle=.731,c=std::cos(angle),s=std::sin(angle),n=1/std::sqrt(14.);
  const Vec3 axis{{n,2*n,3*n}};Mat3 q{};
  const Mat3 cross{{0,-axis[2],axis[1],axis[2],0,-axis[0],-axis[1],axis[0],0}};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)q[3*i+j]=c*(i==j?1:0)+(1-c)*axis[i]*axis[j]+s*cross[3*i+j];
  auto rf=frame;rf.world_from_local=q;validate(rf,model);auto rx=x;
  rx.velocity=rotate_vectors(q,x.velocity);rx.background=rotate_vectors(q,x.background);
  rx.strain=rotate_tensor(q,x.strain);rx.image_nonaffine=rotate(q,x.image_nonaffine);
  const auto ry=evaluate(model,rf,rx);close(ry.power,y.power,1e-12,"rotated power");
  const auto gi=rotate_vectors(q,y.gradient);for(int k=0;k<12;++k)close(ry.gradient[k],gi[k],1e-12,"rotated gradient");
  const auto rs=rotate_tensor(q,y.stress);for(int k=0;k<9;++k)close(ry.stress[k],rs[k],1e-12,"rotated stress");
  // Observer translation and rotation change both particle and ambient motion.
  // This verifies transport objectivity; it does not impose zero resistance on
  // a physical body rotating relative to an otherwise stationary fluid.
  auto shifted=x;for(int k=0;k<12;++k){double d=.31*std::sin(1.7*k+.2);shifted.velocity[k]+=d;shifted.background[k]+=d;}
  const auto sy=evaluate(model,frame,shifted);close(sy.power,y.power,1e-12,"nonaffine frame invariance");
  for(int k=0;k<9;++k)close(sy.stress[k],y.stress[k],1e-12,"nonaffine frame stress");
  // Changing only the solver's angular variable scale must not change physics.
  auto sf=frame;sf.solver_length*=3.7;auto sx=x;
  for(int base:{3,9})for(int k=0;k<3;++k){sx.velocity[base+k]*=3.7;sx.background[base+k]*=3.7;}
  const auto sl=evaluate(model,sf,sx);close(sl.power,y.power,1e-12,"solver angular units");
  for(int k=0;k<3;++k){close(sl.torque_i[k],y.torque_i[k],1e-12,"torque angular units");close(sl.force_i[k],y.force_i[k],1e-12,"force angular units");}
  // Geometric similarity ell->alpha ell at fixed U, ell*omega and ell*E.
  constexpr double alpha=2.9;auto gf=frame;gf.model_length*=alpha;auto gx=x;
  for(int base:{3,9})for(int k=0;k<3;++k){gx.velocity[base+k]/=alpha;gx.background[base+k]/=alpha;}
  for(double&v:gx.strain)v/=alpha;
  const auto gy=evaluate(model,gf,gx);close(gy.power,alpha*y.power,1e-12,"geometric power units");
  for(int k=0;k<3;++k){close(gy.force_i[k],alpha*y.force_i[k],1e-12,"geometric force units");close(gy.torque_i[k],alpha*alpha*y.torque_i[k],1e-12,"geometric torque units");}
  for(int k=0;k<9;++k)close(gy.stress[k],alpha*alpha*y.stress[k],1e-12,"geometric stresslet units");
  // Newtonian rate homogeneity follows without changing the supplied matrix.
  auto rate=x;for(double&v:rate.velocity)v*=4;for(double&v:rate.background)v*=4;for(double&v:rate.image_nonaffine)v*=4;for(double&v:rate.strain)v*=4;
  const auto yr=evaluate(model,frame,rate);close(yr.power,16*y.power,1e-12,"quadratic rate scaling");
  for(int k=0;k<9;++k)close(yr.stress[k],4*y.stress[k],1e-12,"Newtonian stress scaling");
  Model correction=model;correction.metadata.kind=MatrixKind::PairCorrection;
  const double weight=range_weight(.7,.2,1.4);const auto wc=window_correction(correction,weight);const auto wy=evaluate(wc,frame,x);
  close(wy.power,weight*y.power,1e-12,"window power");for(int k=0;k<9;++k)close(wy.stress[k],weight*y.stress[k],1e-12,"window stress");
  bool rejected=false;try{window_correction(model,.5);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"must reject window on full two-body matrix");
  auto asym=model;asym.resistance[1]+=.1;rejected=false;try{validate(frame,asym);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"must reject nonreciprocal matrix");
  Model indefinite;indefinite.metadata.kind=MatrixKind::PairCorrection;indefinite.resistance[0]=-1;validate(frame,indefinite);
  Inputs id;id.velocity[0]=1;check(evaluate(indefinite,frame,id).power<0,"must not silently clip legitimate indefinite correction");
  close(range_weight(.1,.2,1.4),1,1e-15,"window interior");close(range_weight(1.5,.2,1.4),0,1e-15,"window exterior");
  close(ar_scaled_range(1.5e-6,1.5e-6,.5),750e-9,1e-15,"AR10 outer range");
  check(candidate({3.4e-6,0,0},1.5e-6,1.5e-6,750e-9),"candidate inside conservative bound");
  check(!candidate({3.8e-6,0,0},1.5e-6,1.5e-6,750e-9),"candidate outside conservative bound");
  std::cout<<std::setprecision(12)<<"PASS: 17 conjugate finite differences, work balance, assembly, frame covariance, unit scaling, rate scaling, window and guards. Maximum normalized error="<<maximum_relative_error<<"\n";
}
