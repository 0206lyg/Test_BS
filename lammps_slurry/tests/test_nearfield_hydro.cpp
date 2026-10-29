#include "slurry_nearfield_hydro.h"
#include <Eigen/Eigenvalues>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace nf=SlurryNearfieldHydro;
namespace sh=SlurrySpheroidHydro;
namespace ph=SlurryPairHydro;
namespace gp=SlurryGrandPair;
namespace gr=slurry::gr_re2::graphite;
using Vec22=Eigen::Matrix<double,22,1>;

void require(bool condition,const char* message)
{if(!condition)throw std::runtime_error(message);}
void close(double got,double expected,double scale,const char* message)
{if(std::abs(got-expected)>3e-11*std::max(scale,1e-30))
  throw std::runtime_error(std::string(message)+": "+std::to_string(got)+" != "+std::to_string(expected));}

gr::GapResult support_gap(gr::Body& a,gr::Body& b,gr::Vec3 n,double h)
{
  n=gr::normalized(n);gr::GapResult g;g.normal=n;g.gap=h;g.converged=true;
  auto support=[&](const gr::Body& body) {
    const auto shape=gr::rotatedDiagonal(body.rotation,
      {body.axes[0]*body.axes[0],body.axes[1]*body.axes[1],body.axes[2]*body.axes[2]});
    const auto v=gr::mul(shape,n);return gr::scale(v,1/std::sqrt(gr::dot(n,v)));
  };
  g.leverI=support(a);g.leverJ=gr::scale(support(b),-1);
  a.position={};b.position=gr::add(gr::sub(g.leverI,g.leverJ),gr::scale(n,h));
  return g;
}

int main()
{
  std::mt19937_64 rng(9173);std::uniform_real_distribution<double> random(-1,1);
  auto vector=[&](){return gr::Vec3{random(rng),random(rng),random(rng)};};
  int cases=0;
  for(double ar:{1.,2.,5.})for(double size_ratio:{1.,1.4})
  for(double scale:{1.,4e-6})for(double h:{-.0005,.003,.04}) {
    gr::Body a,b;
    const double major=scale*std::cbrt(ar),minor=major/ar;
    a.axes={major,major,minor};b.axes={size_ratio*major,size_ratio*major,size_ratio*minor};
    a.rotation=gr::rotationIncrement(vector());b.rotation=gr::rotationIncrement(vector());
    const auto gap=support_gap(a,b,vector(),h*scale);
    const auto lengths=nf::pair_lengths(nf::volume_radius(a),nf::volume_radius(b),.001);
    close(lengths.mean_radius,.5*(1+size_ratio)*scale,scale,"Mean pair radius");
    close(lengths.rolling_radius,2*size_ratio*scale/(1+size_ratio),scale,"Reduced rolling radius");
    const double eta=.7,epsilon=lengths.epsilon,cutoff=lengths.cutoff,L=1.7*scale;
    const auto result=nf::evaluate(a,b,gap,eta,L,epsilon,cutoff);
    const auto pair=sh::make_pair(a,b,gap,eta,epsilon,cutoff);
    require(result.active==pair.active,"Same cutoff activity");
    require(result.exact_sphere==(ar==1.),"Exact sphere dispatch");
    if(!result.active)continue;
    const double matrixScale=result.resistance.norm();
    require((result.resistance-result.resistance.transpose()).norm()<1e-14*matrixScale,
      "Reciprocity");
    // Scale translations by particle size for a well-scaled SI PSD test.
    Vec22 units=Vec22::Ones();units.head<3>().setConstant(scale);units.segment<3>(11).setConstant(scale);
    const nf::Mat22 scaled=units.asDiagonal()*result.resistance*units.asDiagonal();
    Eigen::SelfAdjointEigenSolver<nf::Mat22> eig(scaled);
    require(eig.info()==Eigen::Success,"PSD eigensolve");
    require(eig.eigenvalues()[0]>-2e-12*scaled.norm(),"Nonnegative grand dissipation");
    // Swapping labels and reversing the physical gap orientation must merely
    // permute the grand operator, including unequal nonspherical contacts.
    auto reverse_gap=gap;
    reverse_gap.normal=gr::scale(gap.normal,-1.);
    reverse_gap.leverI=gap.leverJ;reverse_gap.leverJ=gap.leverI;
    const auto reversed=nf::evaluate(b,a,reverse_gap,eta,L,epsilon,cutoff);
    nf::Mat22 permuted;
    for(int i=0;i<22;++i)for(int j=0;j<22;++j)
      permuted(i,j)=reversed.resistance((i+11)%22,(j+11)%22);
    require((permuted-result.resistance).norm()<2e-11*matrixScale,"Unequal-pair label invariance");
    const auto otherL=nf::evaluate(a,b,gap,eta,3.2*scale,epsilon,cutoff);
    require((otherL.resistance-result.resistance).norm()<1e-13*matrixScale,
      "Reference length does not alter physical operator");

    for(int trial=0;trial<4;++trial) {
      Vec22 q=Vec22::Zero();ph::Vec12 physical{},background{};gp::Vec5 e{};
      for(auto& x:e)x=random(rng);
      const auto strain=gp::strain_tensor(e);
      // Deliberately subtract nonzero ambient velocities: the adapter must
      // use peculiar coordinates once, not add a second affine velocity.
      for(int p=0;p<2;++p)for(int k=0;k<6;++k) {
        const double velocityScale=k<3?scale:L;
        physical[6*p+k]=velocityScale*random(rng);
        background[6*p+k]=velocityScale*random(rng);
        q[11*p+k]=(physical[6*p+k]-background[6*p+k])/(k<3?1.:L);
      }
      for(int p=0;p<2;++p)for(int k=0;k<5;++k)q[11*p+6+k]=-e[k];
      const auto expected=sh::evaluate(pair,physical,background,{},strain,L);
      const Vec22 moment=result.resistance*q;
      double forceScale=0,torqueScale=0;
      for(int k=0;k<3;++k) {
        forceScale=std::max({forceScale,std::abs(expected.force_i[k]),std::abs(expected.force_j[k])});
        torqueScale=std::max({torqueScale,std::abs(expected.torque_i[k]),std::abs(expected.torque_j[k])});
      }
      for(int k=0;k<3;++k) {
        close(-moment[k],expected.force_i[k],forceScale,"Physical force i");
        close(-moment[11+k],expected.force_j[k],forceScale,"Physical force j");
        close(-moment[3+k],expected.torque_i[k],torqueScale,"Physical torque i");
        close(-moment[14+k],expected.torque_j[k],torqueScale,"Physical torque j");
      }
      const auto stress=gp::strain_coordinates(expected.stress);
      double stressScale=0;for(double x:stress)stressScale=std::max(stressScale,std::abs(x));
      for(int k=0;k<5;++k)
        close(-moment[6+k]-moment[17+k],stress[k],stressScale,"Summed physical stress sign");
      close(q.dot(moment),expected.power,expected.power,"Dissipation identity");
      // Uniform strain derivative independently checks the two -E blocks.
      Vec22 direction=Vec22::Zero();direction[6+3]=direction[17+3]=-1.;
      const double delta=1e-5;
      const Vec22 qp=q+delta*direction,qm=q-delta*direction;
      const double derivative=(qp.dot(result.resistance*qp)-qm.dot(result.resistance*qm))/(4*delta);
      require(std::abs(derivative-stress[3])<2e-7*std::max(stressScale,1e-30),
        "Finite-difference strain stress");
    }
    ++cases;
  }
  gr::Body a,b;a.axes=b.axes={1,1,1};
  const auto equal=nf::pair_lengths(1.,1.,.001);
  require(equal.mean_radius==1.&&equal.epsilon==.001&&equal.cutoff==.5&&
          equal.rolling_radius==1.,"Equal-radius length rules changed");
  const auto unequal=nf::pair_lengths(1.,1.4,.001),swapped=nf::pair_lengths(1.4,1.,.001);
  require(unequal.mean_radius==swapped.mean_radius&&unequal.epsilon==swapped.epsilon&&
          unequal.cutoff==swapped.cutoff&&unequal.rolling_radius==swapped.rolling_radius,
          "Pair scales depend on label order");
  auto gap=support_gap(a,b,{0,0,1},.5);
  require(!nf::evaluate(a,b,gap,1,1,.001,.5).active,"Outer gap exactly excluded");
  gap.gap=0;const auto atContact=nf::evaluate(a,b,gap,1,1,.001,.5);
  gap.gap=-.0005;const auto overlap=nf::evaluate(a,b,gap,1,1,.001,.5);
  require((atContact.resistance-overlap.resistance).norm()==0,"Additive gap regularization in shallow overlap");
  std::cout<<"Near-field grand adapter: "<<cases
    <<" equal/unequal sphere/AR2/AR5 dimensionless and SI cases passed; exact helper forces, torques, summed stress, dissipation, reciprocity, label invariance, PSD, units and cutoff.\n";
}
