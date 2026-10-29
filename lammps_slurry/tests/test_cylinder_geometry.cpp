#include "slurry_cylinder_geometry.h"
#include <iostream>
#include <random>
#include <stdexcept>
using namespace SlurryCylinderGeometry;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
bool inside(const Body &b, const V3 &p, double eps=1e-8) {
  const V3 n=b.axis.normalized(), r=p-b.center;
  return std::abs(r.dot(n))<=b.half_thickness+eps &&
    (r-r.dot(n)*n).norm()<=b.radius+eps;
}
int main() {
  Body a, b; a.radius=b.radius=1.; a.half_thickness=b.half_thickness=.05;
  b.center=V3(.8,0,.101);
  auto g=closest_gap(a,b);
  require(g.converged && !g.overlap && std::abs(g.gap-.001)<1e-13,"parallel offset face gap");
  require((g.point_i-V3(.4,0,.05)).norm()<1e-13,"parallel deterministic face witness");
  b.center=V3(2.03,0,.04); g=closest_gap(a,b);
  require(std::abs(g.gap-.03)<1e-13,"parallel side gap");
  b.center=V3(2.03,0,.14);g=closest_gap(a,b);
  require(std::abs(g.gap-.05)<1e-13,"parallel rim gap");
  b.center=V3(.5,0,.09);g=closest_gap(a,b);
  require(g.overlap && g.gap==0.,"parallel overlap detection");
  // A large opposing face gives an independent tilted-cylinder/plane answer.
  const double theta=.13;
  a.axis=V3(std::sin(theta),0,std::cos(theta));
  b.radius=3.;b.center=V3(0,0,.05+.05*std::cos(theta)+std::sin(theta)+.002);
  g=closest_gap(a,b,2e-11);
  require(g.converged && std::abs(g.gap-.002)<1e-9,"tilted plane gap");
  require(inside(a,g.point_i)&&inside(b,g.point_j),"tilted witnesses outside cylinders");
  std::mt19937 rng(123); std::normal_distribution<double> normal;
  double maximum_residual=0.;int maximum_iterations=0;
  for(int k=0;k<40;++k) {
    a.axis=V3(normal(rng),normal(rng),normal(rng)).normalized();
    b.axis=V3(normal(rng),normal(rng),normal(rng)).normalized();
    a.radius=1.;b.radius=.8;a.center=V3::Zero();
    b.center=2.3*V3(normal(rng),normal(rng),normal(rng)).normalized();
    g=closest_gap(a,b,1e-9);
    require(g.converged&&!g.overlap,"arbitrary separated pair convergence");
    require(inside(a,g.point_i)&&inside(b,g.point_j),"witness outside exact cylinder");
    const auto swapped=closest_gap(b,a,1e-9);
    require(std::abs(g.gap-swapped.gap)<3e-8 && (g.normal+swapped.normal).norm()<2e-4,"swap covariance");
    const Eigen::Matrix3d R=Eigen::AngleAxisd(.76,V3(.2,.4,.7).normalized()).toRotationMatrix();
    Body ar=a,br=b;const V3 shift(4,-3,2);
    ar.axis=R*a.axis;br.axis=R*b.axis;ar.center=R*a.center+shift;br.center=R*b.center+shift;
    const auto rotated=closest_gap(ar,br,1e-9);
    require(rotated.converged&&std::abs(g.gap-rotated.gap)<3e-8,"rotation covariance");
    const double lower=g.normal.dot(support(b,-g.normal)-support(a,g.normal));
    if(!(g.gap-lower<2e-9))std::cerr<<"certificate k="<<k<<" gap="<<g.gap<<" lower="<<lower<<" reported="<<g.residual<<" iterations="<<g.iterations<<"\n";
    require(g.gap-lower<2e-9,"support plane certificate");
    maximum_residual=std::max(maximum_residual,g.residual);
    maximum_iterations=std::max(maximum_iterations,g.iterations);
  }
  // Construct independent near-contact answers from supporting points: after
  // this translation both supporting points differ by exactly h*n, proving
  // that h is the global Euclidean distance, including rim--rim contacts.
  for(int k=0;k<30;++k) {
    a.center=b.center=V3::Zero();a.radius=b.radius=1.;
    a.half_thickness=b.half_thickness=1./30.;
    a.axis=V3(normal(rng),normal(rng),normal(rng)).normalized();
    b.axis=V3(normal(rng),normal(rng),normal(rng)).normalized();
    const V3 direction=V3(normal(rng),normal(rng),normal(rng)).normalized();
    const double h=k%2?1e-3:1e-5;
    b.center=support(a,direction)-support(b,-direction)+h*direction;
    g=closest_gap(a,b,1e-9);
    if(!g.converged||std::abs(g.gap-h)>2e-9)
      std::cerr<<"near k="<<k<<" error="<<g.gap-h<<" residual="<<g.residual<<" iterations="<<g.iterations<<"\n";
    require(g.converged&&!g.overlap&&std::abs(g.gap-h)<2e-9,"certified near-contact gap");
    maximum_iterations=std::max(maximum_iterations,g.iterations);
  }
  for(double angle:{1e-7,1e-5,1e-3,.01,.1}) {
    a.center=V3::Zero();a.axis=V3::UnitZ();a.radius=b.radius=1.;
    a.half_thickness=b.half_thickness=.05;
    b.axis=V3(std::sin(angle),0,std::cos(angle));
    const double h=1./3000.;
    b.center=V3(-.4,0,.05+.05*std::cos(angle)+std::sin(angle)+h);
    g=closest_gap(a,b,1e-10);
    if(!g.converged||std::abs(g.gap-h)>2e-10)
      std::cerr<<"near parallel angle="<<angle<<" error="<<g.gap-h<<" residual="<<g.residual<<" iterations="<<g.iterations<<"\n";
    require(g.converged&&std::abs(g.gap-h)<2e-10,"near-parallel face/rim gap");
    maximum_iterations=std::max(maximum_iterations,g.iterations);
  }
  // Regression: exact random-growth AR20 configurations formerly lost
  // precision in the triangle normal equations and exhausted 2048 iterations.
  // Each result must retain its requested witness/support certificate.
  const std::array<std::array<double,9>,5> thin_triangles{{
    {{.47121223680475555,-.013864980379802216,-.88191087429769599,
       .33130353149436814,.83587516759376723,-.43766502512703953,
       -21.111849358405653,29.769290597161216,3.189953453521349}},
    {{-.49771503462086181,.32413134034148766,.80449898602843817,
       -.59773265605279569,.005994247792734636,-.80167308853513997,
       -6.3291583238760207,35.721512595882871,-11.768380053614578}},
    {{.14835077748198841,-.15163616588903667,-.9772402570580907,
       .19339147141460553,-.43036010755575999,.88169717965335948,
       -38.497643864067193,7.2392199486522282,.18171278524106174}},
    {{-.21318483947127417,-.93945347701369408,-.26828974774763203,
       .67611611003417427,-.6567938128206694,-.33389952559526725,
       -34.036563683254442,-10.55484066642545,-2.1876838007480441}},
    {{-.83422541713363985,.32226393140740123,-.44745269238440288,
       -.26068664159534471,-.38588387437460758,.88494977845716327,
       10.31030039378059,36.833042037775698,10.757656395888711}}
  }};
  for (const auto &fixture:thin_triangles) {
    a.center=V3::Zero();a.radius=b.radius=20.;a.half_thickness=b.half_thickness=1.;
    a.axis=V3(fixture[0],fixture[1],fixture[2]);
    b.axis=V3(fixture[3],fixture[4],fixture[5]);
    b.center=V3(fixture[6],fixture[7],fixture[8]);
    g=closest_gap(a,b,2e-10);
    require(g.converged&&!g.overlap&&g.residual<=2.01e-10,"thin GJK triangle certificate");
    const double lower=g.normal.dot(support(b,-g.normal)-support(a,g.normal));
    require(g.gap-lower<=2.1e-10,"thin GJK triangle independent support bound");
    require(inside(a,g.point_i)&&inside(b,g.point_j),"thin triangle witnesses outside cylinders");
  }
  // Units cannot change convergence or the physical gap.
  a.axis=V3::UnitZ();b.axis=V3(std::sin(theta),0,std::cos(theta));
  a.radius=b.radius=1.;a.center=V3::Zero();b.center=V3(.3,0,.8);
  g=closest_gap(a,b,1e-10);
  const double scale=3e-6;
  for(Body *p:{&a,&b}){p->center*=scale;p->radius*=scale;p->half_thickness*=scale;}
  const auto physical=closest_gap(a,b,scale*1e-10);
  require(physical.converged&&std::abs(physical.gap/scale-g.gap)<1e-9,"physical scaling");
  std::cout<<"cylinder geometry passed; max support bound="<<maximum_residual
           <<" max iterations="<<maximum_iterations<<"\n";
}
