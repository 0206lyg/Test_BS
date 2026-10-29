// C++ port of the checked general_pair_eval Kim disk/Faxen operator.
// Kim, IJMF 12 (1986), 469-491, doi:10.1016/0301-9322(86)90019-4.
// Disk quadrature resolves this force/torque/stresslet closure; it does not
// generate omitted higher induced multipoles. No spectral clipping is used.
#include "slurry_general_outer.h"
#include "slurry_hydro_math.h"
#include "slurry_stress_math.h"
#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include <limits>

namespace SlurryGeneral {
namespace {
constexpr double pi=3.141592653589793238462643383279502884;
using Index=std::array<int,3>;
using Term=std::array<int,4>; // denominator power, x/y/z powers
using Terms=std::map<Term,double>;
using Polynomial=std::map<Index,double>;
inline Index add(const Index& a,const Index& b) {
  return {{a[0]+b[0],a[1]+b[1],a[2]+b[2]}};
}
inline int order(const Index& a) {return a[0]+a[1]+a[2];}
inline int epsilon(int i,int j,int k) {
  return (i==j||j==k||k==i)?0:(((i-j)*(j-k)*(k-i)>0)?1:-1);
}
Terms derivative(const Terms& in,int k) {
  Terms out;
  for(const auto& entry:in) {
    const auto& t=entry.first;const double c=entry.second;
    if(t[1+k]){auto lower=t;--lower[1+k];out[lower]+=c*t[1+k];}
    auto higher=t;higher[0]+=2;++higher[1+k];out[higher]-=c*t[0];
  }
  for(auto i=out.begin();i!=out.end();)if(i->second==0.)i=out.erase(i);else ++i;
  return out;
}
Terms kernel_terms(int i,int j,const Index& index,bool laplace) {
  Terms terms;
  if(i==j)terms[Term{{laplace?3:1,0,0,0}}]=laplace?2.:1.;
  Term x{{laplace?5:3,0,0,0}};++x[i+1];++x[j+1];terms[x]+=laplace?-6.:1.;
  for(int k=0;k<3;++k)for(int n=0;n<index[k];++n)terms=derivative(terms,k);
  return terms;
}
struct Operator {int component;Index derivative;double coefficient;};
using Operators=std::array<std::vector<Operator>,11>;
const std::array<Operators,2>& operators() {
  static const auto value=[](){std::array<Operators,2> out;
    for(int k=0;k<3;++k)out[0][k]=out[1][k]={{k,{{0,0,0}},1.}};
    for(int l=0;l<3;++l)for(int j=0;j<3;++j)for(int k=0;k<3;++k){
      Index m{{0,0,0}};m[k]=1;
      if(epsilon(j,k,l))out[0][l+3].push_back({j,m,.5*epsilon(j,k,l)});
      if(epsilon(l,k,j))out[1][l+3].push_back({j,m,.5*epsilon(l,k,j)});
    }
    for(int e=0;e<5;++e)for(int j=0;j<3;++j)for(int k=0;k<3;++k){
      const double v=strain_basis()[e](j,k);if(v==0)continue;
      Index m{{0,0,0}};m[k]=1;
      out[0][e+6].push_back({j,m,v});out[1][e+6].push_back({j,m,-v});
    }return out;}();return value;
}
struct Contract {int row,column,term,laplace;double value;};
struct DiskSpec {std::vector<Term> terms;std::vector<Contract> contractions;};
const DiskSpec& disk_spec() {
  static const auto value=[](){DiskSpec s;
    using Key=std::tuple<int,int,int,Term>;
    std::map<Key,double> coeff;
    const auto& source=operators()[0];const auto& target=operators()[1];
    for(int r=0;r<11;++r)for(int c=0;c<11;++c)
      for(const auto& t:target[r])for(const auto& u:source[c])
        for(int lap=0;lap<2;++lap)
          for(const auto& term:kernel_terms(t.component,u.component,
                                           add(t.derivative,u.derivative),lap!=0))
            coeff[Key(r,c,lap,term.first)]+=t.coefficient*u.coefficient*term.second;
    std::map<Term,int> indices;
    for(const auto& c:coeff)if(c.second!=0.)indices[std::get<3>(c.first)]=0;
    for(auto& t:indices){t.second=int(s.terms.size());s.terms.push_back(t.first);}
    for(const auto& c:coeff)if(c.second!=0.)s.contractions.push_back({
      std::get<0>(c.first),std::get<1>(c.first),indices[std::get<3>(c.first)],
      std::get<2>(c.first),c.second});
    return s;}();return value;
}
struct DiskNode {Vec3 x;std::array<double,2> weight,laplace;};
std::vector<std::pair<double,double>> gauss_legendre(int n) {
  std::vector<std::pair<double,double>> values(n);
  for(int i=0;i<(n+1)/2;++i){
    double z=std::cos(pi*(i+.75)/(n+.5)),dp=0.;
    bool converged=false;
    for(int iter=0;iter<80;++iter){
      double p0=1,p1=z;for(int j=2;j<=n;++j){const double p=((2*j-1)*z*p1-(j-1)*p0)/j;p0=p1;p1=p;}
      dp=n*(z*p1-p0)/(z*z-1);const double dz=p1/dp;z-=dz;
      if(std::abs(dz)<4*std::numeric_limits<double>::epsilon()){converged=true;break;}
    }
    if(!converged)throw std::runtime_error("General disk Gauss-Legendre nodes did not converge");
    // Re-evaluate derivative at the converged node.
    double p0=1,p1=z;for(int j=2;j<=n;++j){const double p=((2*j-1)*z*p1-(j-1)*p0)/j;p0=p1;p1=p;}
    dp=n*(z*p1-p0)/(z*z-1);const double w=1/((1-z*z)*dp*dp);
    values[i]={(1-z)/2,w};values[n-1-i]={(1+z)/2,w};
  }
  return values;
}
const std::vector<DiskNode>& disk_nodes(double ar,int nr,int np) {
  // The cache key contains shape and quadrature only, never configuration.
  using Key=std::tuple<double,int,int>;
  thread_local std::map<Key,std::vector<DiskNode>> cache;
  const Key key(ar,nr,np);auto found=cache.find(key);if(found!=cache.end())return found->second;
  std::vector<DiskNode> values;values.reserve(std::size_t(nr)*np);
  const double confocal=std::sqrt((ar-1)*(ar+1));
  for(const auto& qw:gauss_legendre(nr)){
    const double q=qw.first,w=qw.second/np,r=confocal*std::sqrt((1-q)*(1+q));
    for(int j=0;j<np;++j){const double theta=2*pi*j/np;
      values.push_back({Vec3(r*std::cos(theta),r*std::sin(theta),0),
                        {{w,3*q*q*w}},{{.5*q*q,q*q/6}}});
    }
  }
  return cache.emplace(key,std::move(values)).first->second;
}
Mat11 native_rotation(const Mat3& r) {
  Mat11 t=Mat11::Zero();t.topLeftCorner<3,3>()=t.block<3,3>(3,3)=r;
  for(int i=0;i<5;++i)for(int j=0;j<5;++j)
    t(6+i,6+j)=(strain_basis()[i].cwiseProduct(r*strain_basis()[j]*r.transpose())).sum();
  return t;
}
struct ShapeCoefficients {SlurryHydro::Resistance drag;SlurryStress::Coefficients stress;};
ShapeCoefficients shape_coefficients(double a) {
  static std::mutex mutex;static std::map<double,ShapeCoefficients> cache;
  std::lock_guard<std::mutex> lock(mutex);auto i=cache.find(a);
  if(i!=cache.end())return i->second;
  ShapeCoefficients c{SlurryHydro::unit_resistance({{a,a,1.}}),
                      SlurryStress::unit_stresslet({{a,a,1.}})};
  cache.emplace(a,c);return c;
}
Polynomial polynomial_product(const Polynomial& a,const Polynomial& b,int maximum) {
  Polynomial out;for(const auto& x:a)for(const auto& y:b){const Index z=add(x.first,y.first);
    if(order(z)<=maximum)out[z]+=x.second*y.second;}
  return out;
}
Polynomial shape_operator(const Mat3& a,int kind,int maximum) {
  Polynomial out;out[{{0,0,0}}]=1.;Polynomial d2;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){Index index{{0,0,0}};++index[i];++index[j];d2[index]+=a(i,j);}
  if(maximum>=2)for(const auto& x:d2)out[x.first]+=x.second/(kind==0?6.:10.);
  if(maximum>=4)for(const auto& x:polynomial_product(d2,d2,maximum))
    out[x.first]+=x.second/(kind==0?120.:280.);
  return out;
}
const std::map<Index,std::array<Terms,9>>& far_terms() {
  static const auto values=[](){std::map<Index,std::array<Terms,9>> out;
    for(int x=0;x<=4;++x)for(int y=0;y<=4-x;++y)for(int z=0;z<=4-x-y;++z){
      const Index mi{{x,y,z}};auto& a=out[mi];
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)a[3*i+j]=kernel_terms(i,j,mi,false);
    }return out;}();return values;
}
void require_rotation(const Mat3& r) {
  if(!r.allFinite()||(r.transpose()*r-Mat3::Identity()).norm()>2e-10||r.determinant()<0)
    throw std::invalid_argument("General outer model requires proper body rotations");
}
}

Mat11 isolated11(double ar,const Mat3& q) {
  if(!(ar>=1.)||!std::isfinite(ar))throw std::invalid_argument("General oblate AR must be finite and >=1");
  require_rotation(q);const auto coeff=shape_coefficients(ar);const Vec3 p=q.col(2);
  Vec3 dr,rr;for(int k=0;k<3;++k){dr[k]=coeff.drag.translation[k];rr[k]=coeff.drag.rotation[k];}
  const Mat3 rotational=q*rr.asDiagonal()*q.transpose();
  Eigen::Matrix<double,3,5> j;
  Eigen::Matrix<double,5,5> free;
  for(int k=0;k<5;++k){const auto& e=strain_basis()[k];j.col(k)=coeff.stress.jeffery_parameter*p.cross(e*p);
    const double s=p.dot(e*p);const Vec3 v=e*p-s*p;
    const Mat3 e0=1.5*s*(p*p.transpose()-Mat3::Identity()/3.);
    const Mat3 e1=p*v.transpose()+v*p.transpose();
    const Mat3 stress=coeff.stress.mode[0]*e0+coeff.stress.mode[1]*e1+coeff.stress.mode[2]*(e-e0-e1);
    for(int i=0;i<5;++i)free(i,k)=(strain_basis()[i].cwiseProduct(stress)).sum();
  }
  Mat11 out=Mat11::Zero();out.topLeftCorner<3,3>()=q*dr.asDiagonal()*q.transpose();
  out.block<3,3>(3,3)=rotational;out.block<3,5>(3,6)=-rotational*j;
  out.block<5,3>(6,3)=-j.transpose()*rotational;
  out.bottomRightCorner<5,5>()=free+j.transpose()*rotational*j;return out;
}

Mat11 cross_mobility_disk(const Vec3& r,const Mat3& qt,const Mat3& qs,
                         double ar,int nr,int np,KernelDiagnostics* diagnostics) {
  if(!(ar>=1.)||!std::isfinite(ar)||nr<2||np<8||!r.allFinite()||!(r.norm()>0))
    throw std::invalid_argument("General disk needs AR>=1, distinct centres, nrad>=2, nphi>=8");
  require_rotation(qt);require_rotation(qs);
  const auto& local=disk_nodes(ar,nr,np);std::vector<DiskNode> target,source;
  target.reserve(local.size());source.reserve(local.size());
  const Vec3 axis=qt.col(2);
  const bool coaxial=axis.cross(qs.col(2)).norm()<1e-13 && axis.cross(r).norm()<1e-13*r.norm();
  for(const auto& n:local){auto x=n;x.x=qt*n.x;target.push_back(x);}
  for(std::size_t k=0;k<local.size();k+=coaxial?np:1){auto x=local[k];x.x=qs*x.x;
    if(coaxial)for(double& w:x.weight)w*=np;source.push_back(x);}
  const auto& spec=disk_spec();std::vector<std::array<double,8>> moments(spec.terms.size());
  for(auto& m:moments)m.fill(0.);
  double minimum=std::numeric_limits<double>::infinity();
  for(const auto& t:target)for(const auto& s:source){
    const Vec3 x=r+t.x-s.x;const double radius2=x.squaredNorm();
    minimum=std::min(minimum,radius2);
    if(!(radius2>1e-24))throw std::runtime_error("Confocal disk nodes meet; singular kernel");
    double powers[3][5];for(int k=0;k<3;++k){powers[k][0]=1.;for(int d=1;d<=4;++d)powers[k][d]=powers[k][d-1]*x[k];}
    double radial[10]{};radial[1]=1/(8*pi*std::sqrt(radius2));
    for(int d=3;d<=9;d+=2)radial[d]=radial[d-2]/radius2;
    double weights[8];for(int i=0;i<2;++i)for(int j=0;j<2;++j){const int k=2*i+j;
      weights[k]=t.weight[i]*s.weight[j];weights[4+k]=weights[k]*(t.laplace[i]+s.laplace[j]);}
    for(std::size_t k=0;k<spec.terms.size();++k){const auto& a=spec.terms[k];
      const double value=radial[a[0]]*powers[0][a[1]]*powers[1][a[2]]*powers[2][a[3]];
      for(int w=0;w<8;++w)moments[k][w]+=weights[w]*value;
    }
  }
  Mat11 out=Mat11::Zero();for(const auto& c:spec.contractions){
    const int kind=2*(c.row>=3)+(c.column>=3);
    out(c.row,c.column)+=c.value*moments[c.term][kind+4*c.laplace];
  }
  if(coaxial){Mat3 skew;skew<<0,-axis.z(),axis.y(),axis.z(),0,-axis.x(),-axis.y(),axis.x(),0;
    Mat11 averaged=Mat11::Zero();for(int k=0;k<8;++k){const double angle=2*pi*k/8;
      const Mat3 r=Mat3::Identity()+std::sin(angle)*skew+(1-std::cos(angle))*skew*skew;
      const Mat11 t=native_rotation(r);averaged.noalias()+=t*out*t.transpose()/8.;}
    out=averaged;
  }
  if(diagnostics){diagnostics->minimum_disk_node_distance=std::sqrt(minimum);
    diagnostics->effective_kernel_pairs=target.size()*source.size();
    diagnostics->axisymmetric_azimuth_reduction=coaxial;}
  return out;
}

Mat11 cross_mobility_far(const Vec3& r,const Mat3& at,const Mat3& as,int maximum) {
  if((maximum!=0&&maximum!=2&&maximum!=4)||!r.allFinite()||!(r.norm()>0)||!at.allFinite()||!as.allFinite())
    throw std::invalid_argument("General far kernel requires finite distinct centres and derivative order 0,2,4");
  const double radius=r.norm();double powers[3][7];for(int k=0;k<3;++k){powers[k][0]=1.;for(int n=1;n<=6;++n)powers[k][n]=powers[k][n-1]*r[k];}
  double radial[12]{};radial[1]=1/(8*pi*radius);for(int p=3;p<=11;p+=2)radial[p]=radial[p-2]/(radius*radius);
  std::map<Index,Mat3> derivatives;
  for(const auto& item:far_terms())if(order(item.first)<=maximum){Mat3& d=derivatives[item.first];d.setZero();
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(const auto& t:item.second[3*i+j]){
      const auto& a=t.first;d(i,j)+=t.second*radial[a[0]]*powers[0][a[1]]*powers[1][a[2]]*powers[2][a[3]];
    }}
  std::array<Polynomial,4> product;
  for(int i=0;i<2;++i)for(int j=0;j<2;++j)
    product[2*i+j]=polynomial_product(shape_operator(at,i,maximum),shape_operator(as,j,maximum),maximum);
  Mat11 out=Mat11::Zero();const auto& source=operators()[0];const auto& target=operators()[1];
  for(int row=0;row<11;++row)for(int col=0;col<11;++col){const auto& polynomial=product[2*(row>=3)+(col>=3)];
    for(const auto& t:target[row])for(const auto& s:source[col])for(const auto& p:polynomial){
      const Index index=add(add(t.derivative,s.derivative),p.first);
      if(order(index)<=maximum)out(row,col)+=t.coefficient*s.coefficient*p.second*derivatives.at(index)(t.component,s.component);
    }}return out;
}

OuterResult evaluate_outer(const Geometry& g,int nr,int np,bool far) {
  if(!(g.ar>=1.)||!std::isfinite(g.ar))throw std::invalid_argument("General oblate AR must be finite and >=1");
  const Vec3 r=g.centers[0]-g.centers[1];
  if(!r.allFinite()||!(r.norm()>0))throw std::invalid_argument("General outer needs distinct finite centres");
  if(far&&2*g.ar>=r.norm())throw std::invalid_argument("Far centre expansion requires disjoint enclosing spheres");
  OuterResult out;Mat22 isolated=Mat22::Zero();
  for(int body=0;body<2;++body){const Mat11 b=isolated11(g.ar,g.rotations[body]);
    isolated.block<11,11>(11*body,11*body)=b;Eigen::LLT<Mat11> factor(b);
    if(factor.info()!=Eigen::Success)throw std::runtime_error("General isolated resistance is not SPD");
    out.mobility.block<11,11>(11*body,11*body)=factor.solve(Mat11::Identity());
  }
  Mat11 cross;
  if(far)cross=cross_mobility_far(r,shape_tensor(g.ar,g.rotations[0]),shape_tensor(g.ar,g.rotations[1]));
  else cross=cross_mobility_disk(r,g.rotations[0],g.rotations[1],g.ar,nr,np,&out.kernel);
  out.mobility.topRightCorner<11,11>()=cross;out.mobility.bottomLeftCorner<11,11>()=cross.transpose();
  Eigen::Matrix<double,22,17> p=Eigen::Matrix<double,22,17>::Zero();
  for(int i=0;i<2;++i){p.block<3,3>(11*i,6*i).setIdentity();
    p.block<3,3>(11*i+3,6*i+3)=Mat3::Identity()/g.ar;
    p.block<5,5>(11*i+6,12)=Eigen::Matrix<double,5,5>::Identity()/g.ar;}
  out.grand22=out.mobility.partialPivLu().solve(Mat22::Identity());
  out.grand=p.transpose()*out.grand22*p;out.isolated=p.transpose()*isolated*p;
  out.linear_residual=(out.mobility*out.grand22-Mat22::Identity()).norm()/std::sqrt(22.);
  if(!out.grand.allFinite()||!out.grand22.allFinite()||!std::isfinite(out.linear_residual))
    throw OuterLinearAlgebraError("General outer solve returned nonfinite response");
  out.grand_reciprocity=(out.grand-out.grand.transpose()).norm()/out.grand.norm();
  Eigen::SelfAdjointEigenSolver<Mat17> eigen(.5*(out.grand+out.grand.transpose()));
  if(eigen.info()!=Eigen::Success)throw OuterLinearAlgebraError("General outer eigensolver failed");
  out.minimum_grand_eigenvalue=eigen.eigenvalues()[0];return out;
}
}
