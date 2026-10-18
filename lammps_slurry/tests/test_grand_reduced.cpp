#include "slurry_grand_pair_math.h"
#include <iostream>
#include <stdexcept>
using namespace SlurryGrandPair;
void check(bool b){if(!b)throw std::runtime_error("reduced grand assertion failed");}
Vec3 cross3(const Vec3&a,const Vec3&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
int main(){
  const Vec3 li{.3,-.1,.2},lj{-.2,.25,-.3};
  const auto b=relative_projection(li,lj,SlipConvention::NonaffineMidpoint);
  const auto bp=relative_projection(li,lj,SlipConvention::PhysicalMidpoint);
  Vec3 dr{};for(int k=0;k<3;++k)dr[k]=li[k]-lj[k];
  const Vec3 omega{.2,.1,-.3},translation{.1,.4,.2};const auto delta=cross3(omega,dr);
  Vec17 z{};for(int k=0;k<3;++k){z[k]=translation[k]-.5*delta[k];z[6+k]=translation[k]+.5*delta[k];z[3+k]=z[9+k]=omega[k];}
  for(int i=0;i<6;++i){double v=0;for(int j=0;j<17;++j)v+=b[17*i+j]*z[j];check(std::abs(v)<1e-14);}
  Mat11 h{};for(int i=0;i<11;++i)for(int j=0;j<11;++j)h[11*i+j]=(i==j?2.:0.)+.1*std::cos(i-j);
  // y_nonaffine = T*y_physical; convert the reduced matrix before adding E dr.
  Mat11 t{};for(int k=0;k<11;++k)t[12*k]=1;
  for(int i=0;i<3;++i)for(int k=0;k<5;++k)t[11*i+6+k]=-bp[17*i+12+k];
  Mat11 hp{};for(int i=0;i<11;++i)for(int j=0;j<11;++j)for(int a=0;a<11;++a)for(int c=0;c<11;++c)hp[11*i+j]+=t[11*a+i]*h[11*a+c]*t[11*c+j];
  Metadata meta;meta.kind=MatrixKind::PairCorrection;meta.model_id="manufactured_reduced_test";
  const auto gn=lift_reduced(h,b,meta),gp=lift_reduced(hp,bp,meta);
  double err=0;for(int k=0;k<289;++k)err=std::max(err,std::abs(gn.resistance[k]-gp.resistance[k]));
  check(err<1e-13);validate(Frame{},gn);
  Inputs x;for(int k=0;k<12;++k)x.velocity[k]=.1*std::sin(k);x.strain={.1,.2,.3,.2,0,.1,.3,.1,-.1};
  const auto yn=evaluate(gn,Frame{},x),yp=evaluate(gp,Frame{},x);
  check(std::abs(yn.power-yp.power)<1e-13);
  for(int k=0;k<9;++k)check(std::abs(yn.stress[k]-yp.stress[k])<1e-13);
  std::cout<<"PASS: reduced midpoint common rotation, affine coordinate congruence, same stress and power.\n";
}
