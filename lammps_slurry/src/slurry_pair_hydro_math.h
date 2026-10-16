// Leading singular sphere lubrication: squeeze, shear and pump.
// Mari et al., J. Rheol. 58, 1693 (2014), Appendix A, Eq. (22).
// https://arxiv.org/abs/1403.6793
//
// The factorization below reproduces the unequal-sphere A/B/C coefficients
// and the summed G/H/M coefficients for a uniform imposed strain. The H
// contraction is 2 n x (E n), as in the authors' LF_DEM implementation
// (ryseto/LF_DEM, commit 9af5cb8568dad13fb15f837fd5fbe81f40accf0c);
// the printed Eq. (20) has an extra factor of two in Q^r. We obtain the
// stress by reciprocity from the SAME quadratic, not by a force virial.
// No finite-distance, non-singular, twist or many-body terms are included.
#ifndef SLURRY_PAIR_HYDRO_MATH_H
#define SLURRY_PAIR_HYDRO_MATH_H

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace SlurryPairHydro {
using Vec3 = std::array<double,3>;
using Vec12 = std::array<double,12>;
using Mat3 = std::array<double,9>;
using Mat12 = std::array<double,144>;

inline double dot(const Vec3 &a,const Vec3 &b)
{ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline Vec3 cross(const Vec3 &a,const Vec3 &b)
{ return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
inline Vec3 tangent(const Vec3 &a,const Vec3 &n)
{ const double d=dot(a,n); return {a[0]-d*n[0],a[1]-d*n[1],a[2]-d*n[2]}; }

struct SpherePair {
  Vec3 normal{1,0,0}; // i -> j
  double radius_i=0,radius_j=0,contact_distance=0;
  double gap=0,regularized_gap=0,epsilon=0,cutoff=0;
  double squeeze_resistance=0; // N s / m
  double shear_resistance=0;   // N s / m
  double pump_resistance=0;    // N m s
  double shear_lever_i=0,shear_lever_j=0; // m; sum=ai+aj
  bool active=false;
};

// epsilon and cutoff are physical lengths. Mari's nondimensional choices
// delta and range correspond to epsilon=delta*(ai+aj)/2 and
// cutoff=range*(ai+aj)/2. Mechanical rough contact onset is independent.
// There is NO subtraction at the outer cutoff: this is the truncated leading
// singular model, so the pair operator switches off discontinuously there.
inline SpherePair make_sphere_pair(double ai,double aj,const Vec3 &normal,
                                  double gap,double eta,double epsilon,double cutoff)
{
  if (!(ai>0) || !(aj>0) || !(eta>0) || !(epsilon>0) || !(cutoff>0) ||
      !std::isfinite(ai) || !std::isfinite(aj) || !std::isfinite(eta) ||
      !std::isfinite(epsilon) || !std::isfinite(cutoff) || !std::isfinite(gap))
    throw std::invalid_argument("Sphere lubrication requires positive finite radii, viscosity, epsilon and cutoff");
  const double norm=std::sqrt(dot(normal,normal));
  if (!(norm>0) || !std::isfinite(norm))
    throw std::invalid_argument("Sphere lubrication requires a finite nonzero normal");
  SpherePair p;
  for (int k=0;k<3;++k) p.normal[k]=normal[k]/norm;
  p.radius_i=ai; p.radius_j=aj; p.contact_distance=ai+aj;
  p.gap=gap; p.epsilon=epsilon; p.cutoff=cutoff;
  p.regularized_gap=std::max(gap,0.0)+epsilon;
  const double s=ai+aj, d=2*ai*ai+ai*aj+2*aj*aj;
  p.shear_lever_i=ai*s*(4*ai+aj)/(2*d);
  p.shear_lever_j=aj*s*(ai+4*aj)/(2*d);
  p.active=gap<cutoff;
  if (!p.active) return p;
  const double pref=6*std::acos(-1.0)*eta;
  const double reduced=ai*aj/s;
  const double logarithm=std::max(std::log((0.5*s)/p.regularized_gap),0.0);
  p.squeeze_resistance=pref*reduced*reduced/p.regularized_gap;
  p.shear_resistance=pref*(4*ai*aj*d/(15*s*s*s))*logarithm;
  p.pump_resistance=pref*(ai*ai*ai*aj*aj*aj/(s*d))*logarithm;
  if (!std::isfinite(p.squeeze_resistance) || !std::isfinite(p.shear_resistance) ||
      !std::isfinite(p.pump_resistance))
    throw std::overflow_error("Sphere lubrication resistance overflow");
  return p;
}

struct Modes { double squeeze=0; Vec3 shear{},pump{}; };

// All velocity vectors use [Ui,L*wi,Uj,L*wj]. The rotational reference is
// the FLUID vorticity, not an isolated spheroid's Jeffery angular velocity.
// Uniform-strain drive uses ai+aj (the contact limit), not ai+aj+gap;
// changing that length would change the specified Mari G/H/M coefficients.
inline Modes modes(const SpherePair &p,const Vec12 &relative_scaled,
                   const Vec3 &image_velocity,const Mat3 &strain,double length)
{
  if (!(length>0) || !std::isfinite(length))
    throw std::invalid_argument("Sphere lubrication requires positive reference length");
  Vec3 du{},wi{},wj{},en{};
  for (int k=0;k<3;++k) {
    du[k]=relative_scaled[6+k]-relative_scaled[k]+image_velocity[k];
    wi[k]=relative_scaled[3+k]/length;
    wj[k]=relative_scaled[9+k]/length;
    for (int j=0;j<3;++j) en[k]+=strain[3*k+j]*p.normal[j];
  }
  // Only symmetric, trace-free strain drives the leading lubrication modes.
  // Accept general row-major input by taking its symmetric deviatoric part.
  const double trace=(strain[0]+strain[4]+strain[8])/3;
  for (int k=0;k<3;++k) {
    double ent=0;
    for (int j=0;j<3;++j) ent+=strain[3*j+k]*p.normal[j];
    en[k]=0.5*(en[k]+ent)-trace*p.normal[k];
    du[k]+=p.contact_distance*en[k];
  }
  Vec3 spin{},difference{};
  for (int k=0;k<3;++k) {
    spin[k]=p.shear_lever_i*wi[k]+p.shear_lever_j*wj[k];
    difference[k]=wj[k]-wi[k];
  }
  Modes m;
  m.squeeze=dot(p.normal,du);
  m.shear=tangent(du,p.normal);
  const Vec3 rotation=cross(p.normal,spin);
  for (int k=0;k<3;++k) m.shear[k]+=rotation[k];
  m.pump=tangent(difference,p.normal);
  return m;
}

inline double weighted_dot(const SpherePair &p,const Modes &a,const Modes &b)
{
  return p.squeeze_resistance*a.squeeze*b.squeeze+
         p.shear_resistance*dot(a.shear,b.shear)+
         p.pump_resistance*dot(a.pump,b.pump);
}

struct Assembly {
  Mat12 resistance{};
  Vec12 affine{}; // resistance gradient: R*q+affine; physical force=-gradient
  double energy_constant=0; // half the affine-mode dissipation at q=0
};

inline Assembly assemble(const SpherePair &p,const Vec12 &background_scaled,
                         const Vec3 &image_velocity,const Mat3 &strain,double length)
{
  Assembly out;
  Vec12 relative{};
  for (int k=0;k<12;++k) relative[k]=-background_scaled[k];
  const Modes offset=modes(p,relative,image_velocity,strain,length);
  out.energy_constant=0.5*weighted_dot(p,offset,offset);
  std::array<Modes,12> basis{};
  for (int k=0;k<12;++k) {
    Vec12 unit{}; unit[k]=1;
    basis[k]=modes(p,unit,Vec3{},Mat3{},length);
    out.affine[k]=weighted_dot(p,basis[k],offset);
  }
  for (int r=0;r<12;++r)
    for (int c=0;c<12;++c)
      out.resistance[12*r+c]=weighted_dot(p,basis[r],basis[c]);
  return out;
}

struct Evaluation {
  Vec3 force_i{},force_j{},torque_i{},torque_j{};
  Vec12 gradient{}; // conjugate to q=(U,L*w); equals -(F,T/L)
  Mat3 stress_squeeze{},stress_shear{},stress_pump{},stress{};
  double power_squeeze=0,power_shear=0,power_pump=0,power=0;
  Modes velocity{};
};

// stress_* are volume-integrated, symmetric deviatoric hydrodynamic stresses
// (N m), to divide by box volume ONCE. Do not add another lubrication virial.
// Pump has no direct uniform-strain derivative, hence stress_pump=0, but its
// dissipation and its indirect effect on the coupled solution are included.
// At arbitrary prescribed velocities, stress:E is NOT the pair dissipation:
// D = (q-background)^T*gradient + image_velocity·gradient_j + stress:E.
inline Evaluation evaluate(const SpherePair &p,const Vec12 &q_scaled,
                           const Vec12 &background_scaled,const Vec3 &image_velocity,
                           const Mat3 &strain,double length)
{
  Evaluation out;
  Vec12 relative{};
  for (int k=0;k<12;++k) relative[k]=q_scaled[k]-background_scaled[k];
  out.velocity=modes(p,relative,image_velocity,strain,length);
  const double fn=p.squeeze_resistance*out.velocity.squeeze;
  Vec3 ft{},pump{};
  for (int k=0;k<3;++k) {
    ft[k]=p.shear_resistance*out.velocity.shear[k];
    pump[k]=p.pump_resistance*out.velocity.pump[k];
    out.force_i[k]=fn*p.normal[k]+ft[k];
    out.force_j[k]=-out.force_i[k];
  }
  const Vec3 moment=cross(p.normal,ft);
  for (int k=0;k<3;++k) {
    out.torque_i[k]=p.shear_lever_i*moment[k]+pump[k];
    out.torque_j[k]=p.shear_lever_j*moment[k]-pump[k];
    out.gradient[k]=-out.force_i[k];
    out.gradient[6+k]=-out.force_j[k];
    out.gradient[3+k]=-out.torque_i[k]/length;
    out.gradient[9+k]=-out.torque_j[k]/length;
    for (int j=0;j<3;++j) {
      out.stress_squeeze[3*k+j]=p.contact_distance*fn*
                                (p.normal[k]*p.normal[j]-(k==j ? 1.0/3 : 0));
      out.stress_shear[3*k+j]=0.5*p.contact_distance*(p.normal[k]*ft[j]+ft[k]*p.normal[j]);
      out.stress[3*k+j]=out.stress_squeeze[3*k+j]+out.stress_shear[3*k+j];
    }
  }
  out.power_squeeze=p.squeeze_resistance*out.velocity.squeeze*out.velocity.squeeze;
  out.power_shear=p.shear_resistance*dot(out.velocity.shear,out.velocity.shear);
  out.power_pump=p.pump_resistance*dot(out.velocity.pump,out.velocity.pump);
  out.power=out.power_squeeze+out.power_shear+out.power_pump;
  return out;
}
} // namespace SlurryPairHydro
#endif
