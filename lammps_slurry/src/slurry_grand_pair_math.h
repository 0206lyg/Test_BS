// Coordinate transport for a supplied, independently constructed grand resistance.
// This header contains NO spheroid finite-gap physics or geometry interpolation.
#ifndef SLURRY_GRAND_PAIR_MATH_H
#define SLURRY_GRAND_PAIR_MATH_H
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace SlurryGrandPair {
using Vec3=std::array<double,3>;
using Vec5=std::array<double,5>;
using Vec12=std::array<double,12>;
using Vec17=std::array<double,17>;
using Mat3=std::array<double,9>;
using Mat12=std::array<double,144>;
using Mat17=std::array<double,289>;
using Map12=std::array<double,17*12>;
using Map5=std::array<double,17*5>;

enum class MatrixKind { FullTwoBody, PairCorrection };
struct Metadata {
  std::string model_id;       // Name/version of the actual supplied physical model.
  std::string reference_id;   // Content hash or immutable reference identifier.
  MatrixKind kind=MatrixKind::FullTwoBody;
};
struct Model {
  Mat17 resistance{};
  Metadata metadata{};
};
struct Frame {
  Mat3 world_from_local{1,0,0,0,1,0,0,0,1}; // proper orthogonal, row-major
  double model_length=1;   // reference used when nondimensionalising the matrix
  double solver_length=1;  // solver unknown is [U,L_solver*omega]
  double viscosity=1;
};
// Grand convention, in local axes:
// z = [Ui-Uinf_i, ell*(wi-winf), Uj-Uinf_j, ell*(wj-winf), ell*E_5].
// E_5 is the ORTHONORMAL symmetric trace-free tensor basis below; its sign is +.
// Q = eta*ell/2 * z^T Rhat z. Particle forces/torques are -dQ/dq;
// constitutive stresslet is dQ/dE at fixed nonaffine centre velocities.
// A matrix computed with -E as the last 5 boundary modes must have its last
// 5 row/column signs converted before being supplied here.
// PairCorrection may be indefinite. No eigenvalue clipping is performed.
// To add a full pair to a model already containing isolated-body drag/stress,
// construct the grand correction outside this layer before assembly.
struct Inputs {
  Vec12 velocity{},background{};
  Vec3 image_nonaffine{}; // image shift of Uj minus image shift of Uinf_j
  Mat3 strain{};
};
struct Embedding {
  Map12 particle{};
  Map5 strain{};
  Vec17 offset{};
};
struct Assembly { Mat12 resistance{}; Vec12 affine{}; double energy_constant=0; };
struct Evaluation {
  Vec12 gradient{};
  Vec3 force_i{},force_j{},torque_i{},torque_j{};
  Mat3 stress{}; // pair constitutive stresslet; divide by box volume once
  Vec5 stress_coordinates{};
  double energy=0,power=0;
};

inline const std::array<Mat3,5>& strain_basis() {
  static const double s2=1/std::sqrt(2.),s6=1/std::sqrt(6.);
  static const std::array<Mat3,5> basis{{
    {{s2,0,0,0,-s2,0,0,0,0}}, {{s6,0,0,0,s6,0,0,0,-2*s6}},
    {{0,s2,0,s2,0,0,0,0,0}}, {{0,0,s2,0,0,0,s2,0,0}},
    {{0,0,0,0,0,s2,0,s2,0}} }};
  return basis;
}
inline Vec5 strain_coordinates(const Mat3& e) {
  Vec5 out{};
  for(int k=0;k<5;++k)for(int i=0;i<9;++i)out[k]+=strain_basis()[k][i]*e[i];
  return out;
}
inline Mat3 strain_tensor(const Vec5& e) {
  Mat3 out{};
  for(int k=0;k<5;++k)for(int i=0;i<9;++i)out[i]+=strain_basis()[k][i]*e[k];
  return out;
}
inline Vec3 to_local(const Frame& f,const Vec3& x) {
  Vec3 out{};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)out[i]+=f.world_from_local[3*j+i]*x[j];
  return out;
}
inline Mat3 tensor_to_local(const Frame& f,const Mat3& e) {
  Mat3 out{};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)
      out[3*i+j]+=f.world_from_local[3*a+i]*e[3*a+b]*f.world_from_local[3*b+j];
  return out;
}
inline void validate(const Frame& f,const Model& model) {
  if(!(f.model_length>0&&f.solver_length>0&&f.viscosity>0)||
     !std::isfinite(f.model_length)||!std::isfinite(f.solver_length)||!std::isfinite(f.viscosity))
    throw std::invalid_argument("Grand pair requires finite positive lengths and viscosity");
  const auto& q=f.world_from_local;
  for(double v:q)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite grand pair frame");
  double err=0;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
    double v=0;for(int k=0;k<3;++k)v+=q[3*k+i]*q[3*k+j];
    err=std::max(err,std::abs(v-(i==j?1.:0.)));
  }
  const double det=q[0]*(q[4]*q[8]-q[5]*q[7])-q[1]*(q[3]*q[8]-q[5]*q[6])+q[2]*(q[3]*q[7]-q[4]*q[6]);
  if(err>1e-10||std::abs(det-1)>1e-10)throw std::invalid_argument("Grand pair frame must be a proper rotation");
  double scale=0,asym=0;
  for(int i=0;i<17;++i)for(int j=0;j<17;++j) {
    const double v=model.resistance[17*i+j];
    if(!std::isfinite(v))throw std::invalid_argument("Nonfinite grand pair matrix");
    scale=std::max(scale,std::abs(v));
    asym=std::max(asym,std::abs(v-model.resistance[17*j+i]));
  }
  if(asym>1e-11*std::max(scale,1e-300))throw std::invalid_argument("Grand pair matrix violates reciprocity");
}
inline Embedding embed(const Frame& f,const Vec12& background,const Vec3& image_nonaffine={}) {
  Embedding out;
  const double spin=f.model_length/f.solver_length;
  for(int particle=0;particle<2;++particle)for(int block=0;block<2;++block)
    for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
      const int r=6*particle+3*block+i,c=6*particle+3*block+j;
      out.particle[12*r+c]=(block?spin:1)*f.world_from_local[3*j+i];
    }
  for(int r=0;r<17;++r)for(int c=0;c<12;++c)
    out.offset[r]-=out.particle[12*r+c]*background[c];
  const auto image=to_local(f,image_nonaffine);
  for(int k=0;k<3;++k)out.offset[6+k]+=image[k];
  for(int k=0;k<5;++k) {
    const auto v=strain_coordinates(tensor_to_local(f,strain_basis()[k]));
    for(int i=0;i<5;++i)out.strain[5*(12+i)+k]=f.model_length*v[i];
  }
  return out;
}
inline Vec17 state(const Embedding& b,const Vec12& q,const Vec5& e) {
  Vec17 out=b.offset;
  for(int i=0;i<17;++i) {
    for(int k=0;k<12;++k)out[i]+=b.particle[12*i+k]*q[k];
    for(int k=0;k<5;++k)out[i]+=b.strain[5*i+k]*e[k];
  }
  return out;
}
inline Vec17 multiply(const Mat17& r,const Vec17& z) {
  Vec17 out{};for(int i=0;i<17;++i)for(int j=0;j<17;++j)out[i]+=r[17*i+j]*z[j];return out;
}
inline Assembly assemble(const Model& model,const Frame& f,const Embedding& b,const Vec5& e) {
  Assembly out;
  const double scale=f.viscosity*f.model_length;
  const Vec17 offset=state(b,{},e),weighted=multiply(model.resistance,offset);
  for(int r=0;r<17;++r) {
    out.energy_constant+=.5*scale*offset[r]*weighted[r];
    for(int i=0;i<12;++i)out.affine[i]+=scale*b.particle[12*r+i]*weighted[r];
  }
  std::array<Vec17,12> columns{};
  for(int i=0;i<12;++i)for(int r=0;r<17;++r)columns[i][r]=b.particle[12*r+i];
  for(int j=0;j<12;++j) {
    const auto product=multiply(model.resistance,columns[j]);
    for(int i=0;i<12;++i)for(int r=0;r<17;++r)
      out.resistance[12*i+j]+=scale*columns[i][r]*product[r];
  }
  return out;
}
inline Evaluation evaluate(const Model& model,const Frame& f,const Embedding& b,const Vec12& q,const Vec5& e) {
  Evaluation out;
  const double scale=f.viscosity*f.model_length;
  const auto z=state(b,q,e),g=multiply(model.resistance,z);
  for(int r=0;r<17;++r) {
    out.power+=scale*z[r]*g[r];
    for(int k=0;k<12;++k)out.gradient[k]+=scale*b.particle[12*r+k]*g[r];
    for(int k=0;k<5;++k)out.stress_coordinates[k]+=scale*b.strain[5*r+k]*g[r];
  }
  out.energy=.5*out.power;
  out.stress=strain_tensor(out.stress_coordinates);
  for(int k=0;k<3;++k) {
    out.force_i[k]=-out.gradient[k];out.force_j[k]=-out.gradient[6+k];
    out.torque_i[k]=-f.solver_length*out.gradient[3+k];
    out.torque_j[k]=-f.solver_length*out.gradient[9+k];
  }
  return out;
}
inline Assembly assemble(const Model& m,const Frame& f,const Inputs& x) {
  return assemble(m,f,embed(f,x.background,x.image_nonaffine),strain_coordinates(x.strain));
}
inline Evaluation evaluate(const Model& m,const Frame& f,const Inputs& x) {
  return evaluate(m,f,embed(f,x.background,x.image_nonaffine),x.velocity,strain_coordinates(x.strain));
}
struct StrainBlocks {std::array<double,60> coupling{};std::array<double,25> strain{};};
inline StrainBlocks strain_blocks(const Model&m,const Frame&f) {
  StrainBlocks out;const auto b=embed(f,{});const double scale=f.viscosity*f.model_length;
  for(int c=0;c<5;++c){Vec17 column{};for(int r=0;r<17;++r)column[r]=b.strain[5*r+c];
    const auto product=multiply(m.resistance,column);
    for(int i=0;i<12;++i)for(int r=0;r<17;++r)out.coupling[5*i+c]+=scale*b.particle[12*r+i]*product[r];
    for(int i=0;i<5;++i)for(int r=0;r<17;++r)out.strain[5*i+c]+=scale*b.strain[5*r+i]*product[r];
  }
  return out;
}
// Reduced grand coordinates may include six relative nonaffine motions and
// five strain components. A validated reduced matrix can be lifted without
// adding new global solve unknowns. The model defines the reduced physics.
using Mat11=std::array<double,121>;
using Projection11=std::array<double,11*17>;
enum class SlipConvention { NonaffineMidpoint, PhysicalMidpoint };
inline Projection11 relative_projection(const Vec3& lever_i_over_ell,
                                       const Vec3& lever_j_over_ell,
                                       SlipConvention convention) {
  Projection11 b{};
  const auto skew=[](const Vec3&v)->Mat3 {
    return {0,-v[2],v[1],v[2],0,-v[0],-v[1],v[0],0};
  };
  const auto ki=skew(lever_i_over_ell),kj=skew(lever_j_over_ell);
  for(int k=0;k<3;++k) {
    b[17*k+k]=-1;b[17*k+6+k]=1;
    b[17*(3+k)+3+k]=-1;b[17*(3+k)+9+k]=1;
    for(int j=0;j<3;++j) {
      b[17*k+3+j]=ki[3*k+j];b[17*k+9+j]=-kj[3*k+j];
    }
  }
  for(int k=0;k<5;++k)b[17*(6+k)+12+k]=1;
  if(convention==SlipConvention::PhysicalMidpoint) {
    Vec3 dr{};for(int k=0;k<3;++k)dr[k]=lever_i_over_ell[k]-lever_j_over_ell[k];
    for(int i=0;i<3;++i)for(int k=0;k<5;++k)for(int j=0;j<3;++j)
      b[17*i+12+k]+=strain_basis()[k][3*i+j]*dr[j];
  }
  return b;
}
inline Model lift_reduced(const Mat11& reduced,const Projection11& b,const Metadata& metadata) {
  Model out;out.metadata=metadata;
  for(int i=0;i<17;++i)for(int j=0;j<17;++j)
    for(int a=0;a<11;++a)for(int c=0;c<11;++c)
      out.resistance[17*i+j]+=b[17*a+i]*reduced[11*a+c]*b[17*c+j];
  return out;
}
// C2 range window, frozen with the instantaneous particle configuration.
// This is a numerical truncation operator, not a claim of physical validity.
inline double range_weight(double h,double inner,double outer) {
  if(!std::isfinite(h)||!(inner>=0)||!(outer>inner)||!std::isfinite(outer))
    throw std::invalid_argument("Invalid grand pair range window");
  if(h<=inner)return 1;
  if(h>=outer)return 0;
  const double t=(h-inner)/(outer-inner);
  return 1-t*t*t*(10+t*(-15+6*t));
}
inline Model window_correction(const Model& m,double weight) {
  if(m.metadata.kind!=MatrixKind::PairCorrection)
    throw std::invalid_argument("Only an isolated-body-subtracted pair correction may be windowed here");
  if(!(weight>=0&&weight<=1)||!std::isfinite(weight))throw std::invalid_argument("Invalid pair range weight");
  Model out=m;for(double& v:out.resistance)v*=weight;return out;
}
// Bounding-sphere candidate culling only: exact surface gap still decides range.
inline bool candidate(const Vec3& centre_displacement,double radius_i,double radius_j,double outer_gap) {
  if(!(radius_i>0&&radius_j>0&&outer_gap>=0)||!std::isfinite(radius_i)||!std::isfinite(radius_j)||!std::isfinite(outer_gap))
    throw std::invalid_argument("Invalid grand pair candidate bounds");
  double d2=0;for(double v:centre_displacement){if(!std::isfinite(v))throw std::invalid_argument("Nonfinite centre displacement");d2+=v*v;}
  const double reach=radius_i+radius_j+outer_gap;
  return d2<=reach*reach;
}
inline double ar_scaled_range(double a_i,double a_j,double lambda) {
  if(!(a_i>0&&a_j>0&&lambda>0)||!std::isfinite(a_i)||!std::isfinite(a_j)||!std::isfinite(lambda))
    throw std::invalid_argument("Invalid AR-scaled range parameters");
  return lambda*.5*(a_i+a_j);
}
} // namespace SlurryGrandPair
#endif
