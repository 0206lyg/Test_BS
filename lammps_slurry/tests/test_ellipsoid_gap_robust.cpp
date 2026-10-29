#include "../vendor/gr_re2/ellipsoidGap.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace g=slurry::gr_re2::graphite;
using g::Vec3;using g::Body;
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
double objective(const Body&a,const Body&b,const Vec3&n){
  // Independent body-coordinate support evaluation, not gap_detail::eval.
  const auto na=g::mul(g::transpose(a.rotation),n),nb=g::mul(g::transpose(b.rotation),n);
  double aa=0.,bb=0.;for(int k=0;k<3;++k){aa+=std::pow(a.axes[k]*na[k],2);bb+=std::pow(b.axes[k]*nb[k],2);}
  return g::dot(g::sub(b.position,a.position),n)-std::sqrt(aa)-std::sqrt(bb);
}
Vec3 support(const Body&b,const Vec3&n){
  auto u=g::mul(g::transpose(b.rotation),n);double s=0.;
  for(int k=0;k<3;++k)s+=std::pow(b.axes[k]*u[k],2);s=std::sqrt(s);
  for(int k=0;k<3;++k)u[k]*=b.axes[k]*b.axes[k]/s;
  return g::mul(b.rotation,u);
}
Body body(double ar,const Vec3&angle,double scale=1.){
  Body b;double a=scale*std::cbrt(ar);b.axes={a,a,a/ar};b.rotation=g::rotationIncrement(angle);return b;
}
void validate(const Body&a,const Body&b,const g::GapResult&x){
  double L=0.;for(double v:a.axes)L=std::max(L,v);for(double v:b.axes)L=std::max(L,v);
  const Vec3 r=g::sub(b.position,a.position);
  require(x.converged&&std::isfinite(x.gap)&&g::finite(x.normal),"nonfinite/unconverged gap");
  require(std::abs(g::norm(x.normal)-1.)<2e-14,"normal is not normalized");
  require(std::abs(objective(a,b,x.normal)-x.gap)<2e-13*(L+g::norm(r)),"gap differs from independent support objective");
  require(g::norm(g::sub(support(a,x.normal),x.leverI))<2e-13*L,"first lever is not on its support surface");
  require(g::norm(g::sub(support(b,g::scale(x.normal,-1.)),x.leverJ))<2e-13*L,"second lever is not on its support surface");
  const auto closure=g::sub(g::add(g::sub(r,x.leverI),x.leverJ),g::scale(x.normal,x.gap));
  require(g::norm(closure)<3e-12*(L+g::norm(r)),"support levers do not close at a stationary normal");
}
struct Ref{double h;Vec3 n;};
Ref angular_reference(const Body&a,const Body&b){
  std::vector<Ref> seeds;const int count=4096;const double golden=std::acos(-1.)*(3.-std::sqrt(5.));
  for(int k=0;k<count;++k){double z=1.-2.*(k+.5)/count,w=std::sqrt(1.-z*z),p=k*golden;Vec3 n{w*std::cos(p),w*std::sin(p),z};seeds.push_back({objective(a,b,n),n});}
  for(auto pb:{&a,&b})for(int k=0;k<3;++k)for(double sign:{-1.,1.}){Vec3 n{sign*pb->rotation[k],sign*pb->rotation[3+k],sign*pb->rotation[6+k]};seeds.push_back({objective(a,b,n),n});}
  std::sort(seeds.begin(),seeds.end(),[](const Ref&x,const Ref&y){return x.h>y.h;});
  Ref best=seeds.front();
  // Derivative-free geodesic pattern searches, independent of production
  // Newton derivatives, warm starts, stopping test and inner-ball bound.
  for(int seed=0;seed<8;++seed){Ref x=seeds[seed];double angle=.15;
    for(int iteration=0;iteration<400&&angle>2e-9;++iteration){
      const Vec3 axis=std::abs(x.n[0])<.8?Vec3{1,0,0}:Vec3{0,1,0};
      const Vec3 t1=g::normalized(g::cross(x.n,axis)),t2=g::cross(x.n,t1);Ref next=x;
      for(int d=0;d<16;++d){double phi=2*std::acos(-1.)*d/16.;Vec3 t=g::add(g::scale(t1,std::cos(phi)),g::scale(t2,std::sin(phi)));
        Vec3 n=g::normalized(g::add(g::scale(x.n,std::cos(angle)),g::scale(t,std::sin(angle))));double h=objective(a,b,n);if(h>next.h)next={h,n};}
      if(next.h>x.h)x=next;else angle*=.5;
    }
    if(x.h>best.h)best=x;
  }
  return best;
}
void counterexamples(){
  Body a,b;a.axes=b.axes={1,1,.2};b.position={.2,0,0};
  const double exact=-.4*std::sqrt(1.-.2*.2/(4.*(1.-.2*.2)));
  auto x=g::closestEllipsoidGap(a,b);validate(a,b,x);
  require(std::abs(x.gap-exact)<2e-14,"zero-gradient AR5 saddle was accepted instead of maximum");
  require(x.global_certified,"analytic deep symmetric case missed valid global certificate");
  b.position[2]=.001;g::GapCache cache;cache.valid=true;cache.normal={.02,0,-1};
  const auto bad=cache.normal;auto y=g::closestEllipsoidGap(a,b,&cache);validate(a,b,y);
  require(y.normal[2]>0.&&y.gap>objective(a,b,g::normalized(bad))+.001,
          "cached strict local maximum was mistaken for the global maximum");
  require(std::abs(y.gap-angular_reference(a,b).h)<2e-12,"deep asymmetric reference gap disagrees");
  b.position={0,0,0};x=g::closestEllipsoidGap(a,b,&cache);validate(a,b,x);
  require(std::abs(x.gap+.4)<2e-14,"co-centered oblate gap is not minimum thickness");
}
void spheres_and_bad_input(){
  for(double scale:{1.,4e-6,1e3})for(double d:{0.,.01,1.,2.,2.000000000001,10.}){
    Body a,b;a.axes={scale,scale,scale};b.axes={.7*scale,.7*scale,.7*scale};
    b.position={d*scale,0,0};auto x=g::closestEllipsoidGap(a,b);validate(a,b,x);
    require(std::abs(x.gap-(d-1.7)*scale)<2e-14*scale*(1.+d),"analytic sphere separation wrong");
    require(x.iterations==0&&x.global_certified,"sphere fast path used nonlinear iterations");
  }
  Body a,b;a.axes[1]=std::numeric_limits<double>::quiet_NaN();bool failed=false;
  try{g::closestEllipsoidGap(a,b);}catch(const std::domain_error&){failed=true;}
  require(failed,"nonfinite shape input was accepted");
}
void near_contact_and_cache(){
  int total=0,cold_iterations=0,warm_iterations=0,multistarts=0;
  for(double ar:{1.,2.,5.})for(int k=0;k<40;++k){
    auto a=body(ar,{.13*k,-.21,.19}),b=body(ar,{-.17,.11*k,-.23},.83);
    const Vec3 n=g::normalized(Vec3{std::sin(.37*k+.2),std::cos(.73*k+.1),.4});
    const auto p=g::sub(support(a,n),support(b,g::scale(n,-1.)));
    for(double h:{1e-3,1e-8,0.,-1e-8,-1e-3}){
      b.position=g::add(p,g::scale(n,h));auto x=g::closestEllipsoidGap(a,b);validate(a,b,x);
      require(x.global_certified,"shallow support construction was not globally certified");
      require(std::abs(x.gap-h)<2e-12,"near-contact signed gap differs from exact support construction");
      require(g::norm(g::sub(x.normal,n))<2e-10,"near-contact normal differs from exact construction");
      g::GapCache cache;cache.normal=x.normal;cache.valid=true;
      b.position=g::add(b.position,g::scale(n,1e-7));auto y=g::closestEllipsoidGap(a,b,&cache);validate(a,b,y);
      require(y.global_certified&&std::abs(y.gap-(h+1e-7))<2e-12,"warm-cache update changed signed gap");
      cold_iterations+=x.iterations;warm_iterations+=y.iterations;multistarts+=x.starts>1;++total;
      auto am=a,bm=b;am.position=g::scale(am.position,4e-6);bm.position=g::scale(bm.position,4e-6);
      for(auto pb:{&am,&bm})for(auto &v:pb->axes)v*=4e-6;
      const auto z=g::closestEllipsoidGap(am,bm);validate(am,bm,z);
      require(std::abs(z.gap/4e-6-y.gap)<3e-12,"length normalization changed gap");
    }
  }
  std::cout<<"near-contact cases="<<total<<" cold_iterations="<<cold_iterations
           <<" warm_iterations="<<warm_iterations<<" fallback_cases="<<multistarts<<"\n";
}
void random_reference_and_symmetries(){
  std::mt19937_64 rng(423897);std::normal_distribution<double> normal;
  auto v=[&](){return Vec3{normal(rng),normal(rng),normal(rng)};};
  int cases=0,certified=0,deep_uncertified=0;double max_error=0.;
  const auto G=g::rotationIncrement({.29,-.43,.17});
  for(double ar1:{1.,2.,5.})for(double ar2:{1.,2.,5.})for(int k=0;k<12;++k){
    auto a=body(ar1,v()),b=body(ar2,v(),.85);auto n=g::normalized(v());
    const auto touching=g::sub(support(a,n),support(b,g::scale(n,-1.)));
    b.position=g::scale(touching,k<4?.15+.18*k:k<8?.92+.03*(k-4):1.1+.3*(k-8));
    auto x=g::closestEllipsoidGap(a,b);validate(a,b,x);auto ref=angular_reference(a,b);
    const double error=std::abs(x.gap-ref.h);max_error=std::max(max_error,error);
    if(error>3e-10){std::cerr<<"reference mismatch AR "<<ar1<<" "<<ar2<<" k="<<k<<" error="<<error<<" cert="<<x.global_certified<<"\n";}
    require(error<3e-10,"random gap differs from dense/angular reference");
    auto swapped=g::closestEllipsoidGap(b,a);validate(b,a,swapped);
    require(std::abs(swapped.gap-x.gap)<4e-12,"pair exchange changed gap");
    require(g::norm(g::add(swapped.normal,x.normal))<2e-8,"pair exchange changed unique normal");
    auto aa=a,bb=b;aa.rotation=g::multiply(G,a.rotation);bb.rotation=g::multiply(G,b.rotation);
    aa.position=g::add(g::mul(G,a.position),Vec3{.3,-.2,.7});bb.position=g::add(g::mul(G,b.position),Vec3{.3,-.2,.7});
    auto rotated=g::closestEllipsoidGap(aa,bb);validate(aa,bb,rotated);
    require(std::abs(rotated.gap-x.gap)<5e-12,"rigid covariance changed gap");
    require(g::norm(g::sub(rotated.normal,g::mul(G,x.normal)))<3e-8,"rigid covariance changed unique normal");
    certified+=x.global_certified;deep_uncertified+=!x.global_certified&&x.gap<0.;++cases;
  }
  std::cout<<"random reference cases="<<cases<<" certified="<<certified<<" deep_uncertified="<<deep_uncertified
           <<" max_absolute_gap_difference="<<max_error<<"\n";
}
int main(){
  const auto start=std::chrono::steady_clock::now();
  counterexamples();spheres_and_bad_input();near_contact_and_cache();random_reference_and_symmetries();
  std::cout<<"robust ellipsoid support gap PASS; elapsed_s="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"\n";
}
