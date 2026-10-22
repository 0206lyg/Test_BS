// Geometric preparation only: uniformly grow periodic spheroids and remove
// Perram--Wertheim overlaps by translation/rotation FIRE relaxation.  The
// fictitious time, mass and forces here are not slurry dynamics or material
// properties. A final support-geometry relaxation enforces the mechanical
// contact clearance without changing the physical axes or volume fraction.
#include "../vendor/gr_re2/ellipsoidGap.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace gr = slurry::gr_re2::graphite;
using Vec = std::array<double,3>;
using Quat = std::array<double,4>;
using Mat = std::array<double,6>; // xx yy zz xy xz yz
constexpr double pi = 3.1415926535897932384626433832795;
Vec add(Vec a,const Vec &b) { for(int k=0;k<3;++k)a[k]+=b[k];return a; }
Vec sub(Vec a,const Vec &b) { for(int k=0;k<3;++k)a[k]-=b[k];return a; }
Vec mul(Vec a,double x) { for(double &v:a)v*=x;return a; }
double dot(const Vec &a,const Vec &b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
Vec cross(const Vec &a,const Vec &b) {
  return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}
Vec product(const Mat &m,const Vec &x) {
  return {m[0]*x[0]+m[3]*x[1]+m[4]*x[2],
          m[3]*x[0]+m[1]*x[1]+m[5]*x[2],
          m[4]*x[0]+m[5]*x[1]+m[2]*x[2]};
}
Vec solve(const Mat &m,const Vec &x) {
  const Mat cof{m[1]*m[2]-m[5]*m[5],m[0]*m[2]-m[4]*m[4],
      m[0]*m[1]-m[3]*m[3],m[4]*m[5]-m[3]*m[2],
      m[3]*m[5]-m[4]*m[1],m[3]*m[4]-m[0]*m[5]};
  const double determinant=m[0]*cof[0]+m[3]*cof[3]+m[4]*cof[4];
  if(!(determinant>0.0))throw std::runtime_error("Nonpositive PW shape determinant");
  return mul(product(cof,x),1.0/determinant);
}
void normalize(Quat &q) {
  double norm=0; for(double v:q)norm+=v*v;
  if(!(norm>0) || !std::isfinite(norm))throw std::runtime_error("Invalid quaternion");
  for(double &v:q)v/=std::sqrt(norm);
}
void rotate(Quat &q,const Vec &angle) {
  const double theta=std::sqrt(dot(angle,angle));
  const double factor=theta>1e-12?std::sin(0.5*theta)/theta:0.5;
  const double w=std::cos(0.5*theta);
  const Vec p=mul(angle,factor),v{q[1],q[2],q[3]};
  const Vec next=add(add(mul(v,w),mul(p,q[0])),cross(p,v));
  q={w*q[0]-dot(p,v),next[0],next[1],next[2]};normalize(q);
}
struct Particle { Vec x; Quat q; };
gr::Body body(const Particle &p,double scale,double ratio) {
  gr::Body b;b.position=p.x;b.axes={scale,scale,scale*ratio};
  const double w=p.q[0],x=p.q[1],y=p.q[2],z=p.q[3];
  b.rotation={1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
              2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
              2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)};
  return b;
}
Mat shape(const Quat &q,double scale,double ratio) {
  const Vec p{2*(q[1]*q[3]+q[0]*q[2]),2*(q[2]*q[3]-q[0]*q[1]),
              1-2*(q[1]*q[1]+q[2]*q[2])};
  const double s2=scale*scale,d=s2*(ratio*ratio-1);
  return {s2+d*p[0]*p[0],s2+d*p[1]*p[1],s2+d*p[2]*p[2],
          d*p[0]*p[1],d*p[0]*p[2],d*p[1]*p[2]};
}
Vec image(Vec x,double box) { for(double &v:x)v-=box*std::nearbyint(v/box);return x; }
struct Contact { double value,lambda; Vec w; };
Contact contact(const Mat &a,const Mat &b,const Vec &r) {
  Mat difference;for(int k=0;k<6;++k)difference[k]=b[k]-a[k];
  double lo=0,hi=1,lambda=0.5;
  Contact result{};
  for(int iteration=0;iteration<48;++iteration) {
    Mat matrix;for(int k=0;k<6;++k)matrix[k]=a[k]+lambda*difference[k];
    const Vec w=solve(matrix,r),dw=product(difference,w);
    const double rw=dot(r,w),wdw=dot(w,dw),l= lambda*(1-lambda);
    const double first=(1-2*lambda)*rw-l*wdw;
    const double second=-2*rw-2*(1-2*lambda)*wdw+2*l*dot(dw,solve(matrix,dw));
    result={l*rw,lambda,w};
    if(std::abs(first)<1e-13*std::max(1.0,rw) || hi-lo<1e-13)break;
    if(first>0)lo=lambda;else hi=lambda;
    double next=lambda-first/second;
    if(!std::isfinite(next) || !(next>lo && next<hi))next=0.5*(lo+hi);
    lambda=next;
  }
  return result;
}
struct Evaluation {
  double energy=0,overlap=0;
  std::size_t overlap_pairs=0;
  std::vector<Vec> force,torque;
};
Evaluation evaluate(const std::vector<Particle> &p,double box,double scale,double ratio) {
  Evaluation e;e.force.resize(p.size());e.torque.resize(p.size());
  std::vector<Mat> matrices;matrices.reserve(p.size());
  for(const auto &particle:p)matrices.push_back(shape(particle.q,scale,ratio));
  for(std::size_t i=0;i<p.size();++i)for(std::size_t j=i+1;j<p.size();++j) {
    const Vec r=image(sub(p[i].x,p[j].x),box);
    if(dot(r,r)>=4*scale*scale)continue;
    const Contact c=contact(matrices[i],matrices[j],r);
    if(c.value>=1.0)continue;
    if(!(c.value>1e-24))throw std::runtime_error("Coincident particle centers in packing input");
    const double root=std::sqrt(c.value),delta=1-root,l=c.lambda;
    const double prefactor=delta*l*(1-l)/root;
    const Vec f=mul(c.w,prefactor);
    e.force[i]=add(e.force[i],f);e.force[j]=sub(e.force[j],f);
    e.torque[i]=add(e.torque[i],mul(cross(c.w,product(matrices[i],c.w)),prefactor*(1-l)));
    e.torque[j]=add(e.torque[j],mul(cross(c.w,product(matrices[j],c.w)),prefactor*l));
    e.energy+=0.5*delta*delta;e.overlap=std::max(e.overlap,delta);++e.overlap_pairs;
  }
  return e;
}
struct Relaxation { Evaluation state; int iterations; bool converged; };
template<class Evaluator>
Relaxation relax_with(std::vector<Particle> &p,double box,double scale,double tolerance,
                      int maximum,Evaluator evaluator) {
  std::vector<Vec> velocity(p.size()),omega(p.size());
  Evaluation e=evaluator(p);
  double dt=0.025,alpha=0.1;int positive=0;
  for(int iteration=0;iteration<=maximum;++iteration) {
    if(e.overlap<=tolerance)return {std::move(e),iteration,true};
    if(iteration==maximum)break;
    double power=0,v2=0,f2=0;
    for(std::size_t i=0;i<p.size();++i) {
      velocity[i]=add(velocity[i],mul(e.force[i],dt*scale*scale));
      omega[i]=add(omega[i],mul(e.torque[i],dt));
      power+=dot(velocity[i],e.force[i])+dot(omega[i],e.torque[i]);
      v2+=dot(velocity[i],velocity[i])/(scale*scale)+dot(omega[i],omega[i]);
      f2+=scale*scale*dot(e.force[i],e.force[i])+dot(e.torque[i],e.torque[i]);
    }
    if(power>0) {
      ++positive;
      if(positive>5){dt=std::min(dt*1.1,0.25);alpha*=0.99;}
      const double factor=f2>0?alpha*std::sqrt(v2/f2):0;
      for(std::size_t i=0;i<p.size();++i) {
        velocity[i]=add(mul(velocity[i],1-alpha),mul(e.force[i],factor*scale*scale));
        omega[i]=add(mul(omega[i],1-alpha),mul(e.torque[i],factor));
      }
    } else {
      positive=0;dt*=0.5;alpha=0.1;
      for(auto &v:velocity)v={0,0,0};
      for(auto &w:omega)w={0,0,0};
      if(dt<1e-14)return {std::move(e),iteration,false};
    }
    // Limit the fictitious displacement to avoid pathological FIRE excursions.
    double step=dt;
    for(std::size_t i=0;i<p.size();++i) {
      const double speed=std::sqrt(dot(velocity[i],velocity[i]));
      const double spin=std::sqrt(dot(omega[i],omega[i]));
      if(speed*step>0.05*scale)step=std::min(step,0.05*scale/speed);
      if(spin*step>0.05)step=std::min(step,0.05/spin);
    }
    for(std::size_t i=0;i<p.size();++i) {
      p[i].x=add(p[i].x,mul(velocity[i],step));
      for(double &x:p[i].x)x-=box*std::floor(x/box);
      rotate(p[i].q,mul(omega[i],step));
    }
    e=evaluator(p);
  }
  return {std::move(e),maximum,false};
}
Relaxation relax(std::vector<Particle> &p,double box,double scale,double ratio,double tolerance,int maximum) {
  return relax_with(p,box,scale,tolerance,maximum,[&](const std::vector<Particle> &q) {
    return evaluate(q,box,scale,ratio);
  });
}

// A support plane gives a rigorous lower bound on the Euclidean surface gap.
// It is used only to discard pairs which already satisfy the clearance. Pairs
// exerting a preparation force use the same closest-surface solver as dynamics.
double support_gap(const gr::Body &a,const gr::Body &b,const Vec &normal) {
  const auto at=gr::mul(gr::transpose(a.rotation),normal);
  const auto bt=gr::mul(gr::transpose(b.rotation),normal);
  double ra=0,rb=0;
  for(int k=0;k<3;++k) {
    ra+=std::pow(a.axes[k]*at[k],2);rb+=std::pow(b.axes[k]*bt[k],2);
  }
  return dot(normal,sub(b.position,a.position))-std::sqrt(ra)-std::sqrt(rb);
}
gr::GapResult surface_gap(const gr::Body &a,const gr::Body &b,gr::GapCache *cache=nullptr) {
  try {return gr::closestEllipsoidGap(a,b,cache);}
  catch(const std::runtime_error &) {
    // A different initial separating normal repairs a difficult Newton start;
    // it does not replace support geometry by a PW distance approximation.
    gr::GapCache retry;
    const auto qa=gr::rotatedDiagonal(a.rotation,{a.axes[0]*a.axes[0],a.axes[1]*a.axes[1],a.axes[2]*a.axes[2]});
    const auto qb=gr::rotatedDiagonal(b.rotation,{b.axes[0]*b.axes[0],b.axes[1]*b.axes[1],b.axes[2]*b.axes[2]});
    const Mat ma{qa[0],qa[4],qa[8],qa[1],qa[2],qa[5]},mb{qb[0],qb[4],qb[8],qb[1],qb[2],qb[5]};
    retry.normal=gr::normalized(contact(ma,mb,sub(b.position,a.position)).w);retry.valid=true;
    auto result=gr::closestEllipsoidGap(a,b,&retry);
    if(cache)*cache=retry;
    return result;
  }
}
Evaluation evaluate_clearance(const std::vector<Particle> &p,double box,double ratio,
                             double target,std::vector<gr::GapCache> &cache) {
  Evaluation e;e.force.resize(p.size());e.torque.resize(p.size());
  std::vector<gr::Body> bodies;for(const auto &particle:p)bodies.push_back(body(particle,1,ratio));
  const double cutoff=2+target;
  for(std::size_t i=0;i<p.size();++i)for(std::size_t j=i+1;j<p.size();++j) {
    const Vec r=image(sub(p[j].x,p[i].x),box);
    if(dot(r,r)>cutoff*cutoff)continue;
    auto bj=bodies[j];bj.position=add(bodies[i].position,r);
    auto &warm=cache[i*p.size()+j];
    Vec n=warm.valid?warm.normal:gr::normalized(r);
    if(dot(n,r)<0)n=mul(n,-1);
    if(support_gap(bodies[i],bj,n)>=target)continue;
    const auto gap=surface_gap(bodies[i],bj,&warm);
    const double delta=std::max(0.,target-gap.gap);
    if(delta==0)continue;
    const Vec f=mul(gap.normal,-delta);
    e.force[i]=add(e.force[i],f);e.force[j]=sub(e.force[j],f);
    e.torque[i]=add(e.torque[i],cross(gap.leverI,f));
    e.torque[j]=sub(e.torque[j],cross(gap.leverJ,f));
    e.energy+=.5*delta*delta;e.overlap=std::max(e.overlap,delta);++e.overlap_pairs;
  }
  return e;
}
struct Certificate {
  double minimum_gap=std::numeric_limits<double>::infinity();
  std::size_t checked=0;
  bool certified=false;
};
Certificate certify(const std::vector<Particle> &p,double box,double ratio,double target) {
  Certificate out;
  std::vector<gr::Body> bodies;for(const auto &particle:p)bodies.push_back(body(particle,1,ratio));
  // An actual self-image gap supplies a finite upper bound on the global
  // minimum. Every omitted image then has a bounding-sphere lower bound above
  // that minimum. This also certifies small periodic boxes and self images.
  auto shifted=bodies[0];shifted.position[0]+=box;
  out.minimum_gap=surface_gap(bodies[0],shifted).gap;
  for(std::size_t i=0;i<p.size();++i)for(std::size_t j=i;j<p.size();++j) {
    const Vec d=sub(bodies[j].position,bodies[i].position);
    const double radius=2+std::max(target,out.minimum_gap)+1e-12;
    int lo[3],hi[3];
    for(int k=0;k<3;++k) {
      lo[k]=int(std::ceil((-radius-d[k])/box));hi[k]=int(std::floor((radius-d[k])/box));
    }
    for(int x=lo[0];x<=hi[0];++x)for(int y=lo[1];y<=hi[1];++y)for(int z=lo[2];z<=hi[2];++z) {
      if(i==j && (x<0 || (x==0 && y<0) || (x==0 && y==0 && z<=0)))continue;
      const Vec r{d[0]+box*x,d[1]+box*y,d[2]+box*z};
      const double current_radius=2+std::max(target,out.minimum_gap)+1e-12;
      if(dot(r,r)>current_radius*current_radius)continue;
      auto bj=bodies[j];bj.position=add(bodies[i].position,r);
      const auto gap=surface_gap(bodies[i],bj);
      out.minimum_gap=std::min(out.minimum_gap,gap.gap);++out.checked;
    }
  }
  out.certified=out.minimum_gap>=target;
  return out;
}
void write_certificate(const std::string &path,const Certificate &c,double a,double gap,double phi) {
  if(path.empty())return;
  std::ofstream report(path);if(!report)throw std::runtime_error("Cannot write packing certificate");
  report<<std::setprecision(17)<<"{\n  \"certified\": "<<(c.certified?"true":"false")
        <<",\n  \"target_gap_m\": "<<gap<<",\n  \"minimum_surface_gap_m\": "<<c.minimum_gap*a
        <<",\n  \"max_clearance_deficit_m\": "<<std::max(0.,gap-c.minimum_gap*a)
        <<",\n  \"checked_pair_images\": "<<c.checked
        <<",\n  \"nominal_volume_fraction\": "<<phi
        <<",\n  \"method\": \"ellipsoid_support_gap_all_periodic_images\"\n}\n";
  if(!report)throw std::runtime_error("Failed writing packing certificate");
}
struct Options {
  std::string input,output,history,report;
  double growth=0.02,tolerance=1e-6;
  double contact_gap=3e-9;
  int maximum=20000;
  bool certify_only=false;
};
Options options(int argc,char **argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    const std::string name=argv[i];
    if(i+1>=argc)throw std::runtime_error("Missing value for "+name);
    const std::string value=argv[++i];
    if(name=="--input")o.input=value;
    else if(name=="--output")o.output=value;
    else if(name=="--history")o.history=value;
    else if(name=="--growth-fraction")o.growth=std::stod(value);
    else if(name=="--overlap-tolerance")o.tolerance=std::stod(value);
    else if(name=="--max-relax")o.maximum=std::stoi(value);
    else if(name=="--contact-gap-m")o.contact_gap=std::stod(value);
    else if(name=="--report")o.report=value;
    else if(name=="--certify-only")o.certify_only=std::stoi(value)!=0;
    else throw std::runtime_error("Unknown argument "+name);
  }
  if(o.input.empty() || (!o.certify_only && (o.output.empty() || o.history.empty())))
    throw std::runtime_error("Required: --input FILE --output FILE --history FILE");
  if(!(o.growth>0 && o.growth<=0.1 && o.tolerance>0 && o.tolerance<0.01 && o.maximum>0))
    throw std::runtime_error("Require 0<growth-fraction<=0.1, 0<overlap-tolerance<0.01, max-relax>0");
  if(!(o.contact_gap>=0) || !std::isfinite(o.contact_gap))
    throw std::runtime_error("contact-gap-m must be finite and nonnegative");
  return o;
}
int self_test() {
  double worst_gradient=0,worst_contact=0;
  for(double ratio: {1.0,0.5,0.125}) {
    std::vector<Particle> p={{{0.2,0.3,0.4},{0.84,0.1,0.4,0.2}},
                            {{7.7,0.4,0.5},{0.8,0.3,-0.2,0.4}}};
    for(auto &particle:p)normalize(particle.q);
    const auto e=evaluate(p,8,1,ratio);
    if(!(e.energy>0))throw std::runtime_error("Self-test fixture must overlap");
    const double h=1e-6;
    for(int i=0;i<2;++i)for(int k=0;k<3;++k)for(int rotation=0;rotation<2;++rotation) {
      auto plus=p,minus=p;
      if(rotation) {
        Vec v{};v[k]=h;rotate(plus[i].q,v);v[k]=-h;rotate(minus[i].q,v);
      } else {plus[i].x[k]+=h;minus[i].x[k]-=h;}
      const double derivative=(evaluate(plus,8,1,ratio).energy-evaluate(minus,8,1,ratio).energy)/(2*h);
      const double force=rotation?e.torque[i][k]:e.force[i][k];
      worst_gradient=std::max(worst_gradient,std::abs(derivative+force));
    }
    const Mat a=shape(p[0].q,1,ratio),b=shape(p[1].q,1,ratio);
    const Vec r=image(sub(p[0].x,p[1].x),8);
    auto value=[&](double lambda) {
      Mat m;for(int k=0;k<6;++k)m[k]=(1-lambda)*a[k]+lambda*b[k];
      return lambda*(1-lambda)*dot(r,solve(m,r));
    };
    // An independent golden-section search checks the Newton contact optimum.
    const double golden=(std::sqrt(5.0)-1)/2;
    double lo=0,hi=1,left=1-golden,right=golden;
    double fl=value(left),fr=value(right);
    for(int iteration=0;iteration<80;++iteration) {
      if(fl<fr){lo=left;left=right;fl=fr;right=lo+golden*(hi-lo);fr=value(right);}
      else {hi=right;right=left;fr=fl;left=hi-golden*(hi-lo);fl=value(left);}
    }
    const auto result=contact(a,b,r);
    worst_contact=std::max(worst_contact,std::abs(result.value-std::max(fl,fr)));
    if(ratio==1.0 && std::abs(result.value-dot(r,r)/4)>1e-13)
      throw std::runtime_error("Sphere PW contact limit failed");
  }
  if(worst_gradient>2e-7 || worst_contact>1e-11)
    throw std::runtime_error("PW force/torque or contact optimum self-test failed");
  double worst_clearance_gradient=0;
  for(double ratio:{1.,.5,.05,1./30}) {
    std::vector<Particle> p={{{.02,.3,.4},{.84,.1,.4,.2}},
                             {{0,0,0},{.8,.3,-.2,.4}}};
    for(auto &particle:p)normalize(particle.q);
    auto bi=body(p[0],1,ratio),bj=body(p[1],1,ratio);
    const auto qi=gr::rotatedDiagonal(bi.rotation,{1.,1.,ratio*ratio});
    const auto qj=gr::rotatedDiagonal(bj.rotation,{1.,1.,ratio*ratio});
    const Vec normal=gr::normalized(Vec{-.6,.3,.7});
    const auto supports=gr::gap_detail::eval(normal,{0,0,0},qi,qj);
    p[1].x=add(p[0].x,add(sub(supports.pI,supports.pJ),mul(normal,.04)));
    for(double &v:p[1].x)v-=8*std::floor(v/8);
    auto energy=[&](const std::vector<Particle> &q) {
      std::vector<gr::GapCache> cache(4);
      return evaluate_clearance(q,8,ratio,.08,cache);
    };
    const auto e=energy(p);
    if(std::abs(e.energy-.5*.04*.04)>1e-12)
      throw std::runtime_error("True-gap contact construction failed");
    for(int i=0;i<2;++i)for(int k=0;k<3;++k)for(int spin=0;spin<2;++spin) {
      const double h=1e-6;auto plus=p,minus=p;
      if(spin) {
        Vec v{};v[k]=h;rotate(plus[i].q,v);v[k]=-h;rotate(minus[i].q,v);
      } else {plus[i].x[k]+=h;minus[i].x[k]-=h;}
      const double derivative=(energy(plus).energy-energy(minus).energy)/(2*h);
      const double force=spin?e.torque[i][k]:e.force[i][k];
      worst_clearance_gradient=std::max(worst_clearance_gradient,std::abs(derivative+force));
    }
    const auto certificate=certify(p,8,ratio,.03);
    if(!certificate.certified || std::abs(certificate.minimum_gap-.04)>1e-11 ||
       certify(p,8,ratio,.05).certified)
      throw std::runtime_error("Periodic true-gap certification failed");
  }
  if(worst_clearance_gradient>2e-7)
    throw std::runtime_error("True-gap force/torque finite-difference test failed");
  std::cout<<"PW periodic sphere/oblate tests: maximum gradient residual="<<worst_gradient
           <<", contact optimum residual="<<worst_contact
           <<", true-gap gradient residual="<<worst_clearance_gradient<<'\n';
  return 0;
}
} // namespace

int main(int argc,char **argv) {
  try {
    if(argc==2 && std::string(argv[1])=="--self-test")return self_test();
    const auto args=options(argc,argv);
    std::ifstream input(args.input);
    std::size_t count;double physical_box,a,c,scale;
    if(!(input>>count>>physical_box>>a>>c>>scale) || count==0 ||
       !(physical_box>0 && a>0 && c>0 && scale>0 && scale<=1) ||
       !std::isfinite(physical_box) || !std::isfinite(a) || !std::isfinite(c))
      throw std::runtime_error("Invalid header; expected N box target_a target_c initial_scale");
    const double box=physical_box/a,ratio=c/a,target_gap=args.contact_gap/a;
    if(ratio>1)throw std::runtime_error("Packing expects oblate spheroids with target_c<=target_a");
    if(!args.certify_only && !(box>2*(2+target_gap)))
      throw std::runtime_error("Packing requires box > 4*target_a+2*contact_gap for unique clearance images");
    std::vector<Particle> particles(count);
    for(auto &p:particles) {
      for(double &x:p.x)if(!(input>>x) || !std::isfinite(x))throw std::runtime_error("Invalid position");
      for(double &q:p.q)if(!(input>>q))throw std::runtime_error("Invalid quaternion");
      normalize(p.q);for(double &x:p.x){x/=a;x-=box*std::floor(x/box);}
    }
    const double target_phi=count*4*pi*ratio/(3*box*box*box);
    if(args.certify_only) {
      if(scale!=1.)throw std::runtime_error("Certification requires a final configuration with initial_scale=1");
      const auto certificate=certify(particles,box,ratio,target_gap);
      write_certificate(args.report,certificate,a,args.contact_gap,target_phi);
      std::cout<<std::setprecision(17)<<"Periodic surface-gap certificate: minimum_gap_m="
               <<certificate.minimum_gap*a<<", target_gap_m="<<args.contact_gap
               <<", checked_pair_images="<<certificate.checked<<", certified="<<certificate.certified<<'\n';
      return certificate.certified?0:2;
    }
    std::ofstream history(args.history);
    if(!history)throw std::runtime_error("Cannot write growth history");
    history<<"attempt,status,scale,volume_fraction,max_PW_overlap,overlap_pairs,energy,relax_iterations,max_surface_clearance_deficit_m\n"<<std::setprecision(17);
    int attempt=0;
    auto record=[&](const char *status,double s,const Relaxation &r) {
      history<<attempt<<','<<status<<','<<s<<','<<target_phi*s*s*s<<','<<r.state.overlap<<','
             <<r.state.overlap_pairs<<','<<r.state.energy<<','<<r.iterations<<",0\n";history.flush();
    };
    auto initial=relax(particles,box,scale,ratio,args.tolerance,args.maximum);
    record("initial",scale,initial);
    if(!initial.converged)throw std::runtime_error("Initial configuration could not be relaxed");
    double increment=args.growth;int rejected=0;
    std::cout<<std::setprecision(10)<<"Geometric growth: N="<<count<<", target phi="<<target_phi
             <<", initial phi="<<target_phi*scale*scale*scale<<std::endl;
    while(scale<1.0) {
      ++attempt;
      // growth-fraction means a relative volume-fraction increase; the shape
      // scale changes by its cube root because all three axes grow together.
      const double next=std::min(1.0,scale*std::cbrt(1+increment));
      auto candidate=particles;
      auto result=relax(candidate,box,next,ratio,args.tolerance,args.maximum);
      record(result.converged?"accepted":"rejected",next,result);
      if(result.converged) {
        particles=std::move(candidate);scale=next;rejected=0;
        increment=std::min(args.growth,increment*1.25);
        std::cout<<"  phi="<<target_phi*scale*scale*scale<<", max PW overlap="<<result.state.overlap
                 <<", relaxation iterations="<<result.iterations<<std::endl;
      } else {
        increment*=0.5;++rejected;
        std::cout<<"  retry smaller growth after stalled phi="<<target_phi*next*next*next
                 <<", max PW overlap="<<result.state.overlap<<std::endl;
        if(increment<1e-6 || rejected>=16) {
          std::cerr<<"Packing stalled at accepted phi="<<target_phi*scale*scale*scale
                   <<" before target phi="<<target_phi<<". No target configuration was written.\n";
          return 2;
        }
      }
    }
    // The tolerance in the PW growth pass is not a physical-gap tolerance.
    // Give the final true-surface relaxation a small fixed numerical margin
    // (0.0001 nm) so the written configuration satisfies gap>=contact onset,
    // rather than starting with a tolerance-sized stored contact compression.
    const double margin=std::max(1e-13/a,128*std::numeric_limits<double>::epsilon()*box);
    std::vector<gr::GapCache> cache(count*count);
    auto clearance=relax_with(particles,box,1.,margin*.25,args.maximum,
      [&](const std::vector<Particle> &p) {
        return evaluate_clearance(p,box,ratio,target_gap+margin,cache);
      });
    const auto pw=evaluate(particles,box,1.,ratio);
    history<<++attempt<<",surface_clearance,1,"<<target_phi<<','<<pw.overlap<<','
           <<clearance.state.overlap_pairs<<','<<clearance.state.energy<<','<<clearance.iterations<<','
           <<std::max(0.,clearance.state.overlap-margin)*a<<'\n';history.flush();
    const auto certificate=certify(particles,box,ratio,target_gap);
    write_certificate(args.report,certificate,a,args.contact_gap,target_phi);
    if(!clearance.converged || !certificate.certified) {
      std::cerr<<"Surface clearance preparation failed: minimum gap="<<certificate.minimum_gap*a
               <<" m, required="<<args.contact_gap<<" m; no final configuration written.\n";
      return 2;
    }
    std::ofstream output(args.output);
    if(!output)throw std::runtime_error("Cannot write packed configuration");
    output<<std::setprecision(17)<<count<<' '<<physical_box<<' '<<a<<' '<<c<<" 1\n";
    for(const auto &p:particles) {
      for(double x:p.x)output<<x*a<<' ';
      for(int k=0;k<4;++k)output<<p.q[k]<<(k==3?'\n':' ');
    }
    if(!output)throw std::runtime_error("Failed writing packed configuration");
    std::cout<<"Reached target phi="<<target_phi<<"; minimum periodic surface gap="
             <<certificate.minimum_gap*a<<" m, clearance iterations="<<clearance.iterations
             <<"; preparation is geometric, not physical equilibration.\n";
    return 0;
  } catch(const std::exception &error) {
    std::cerr<<"pack_spheroids: "<<error.what()<<'\n';return 1;
  }
}
