// Sequential, fail-fast AR10 emulator tests. The first pose is held out from
// table construction. The original development poses are followed by a frozen
// fresh holdout set. Emulator agreement is NOT independent Stokes accuracy.
#include "slurry_spd_table.h"
#include "slurry_scattering_provider.h"
#include <Eigen/Eigenvalues>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sc = SlurryScattering;
using Mat11 = sc::Mat11;
using Mat5 = Eigen::Matrix<double,5,5>;
using Axes = std::array<sc::Vec3,2>;
struct Pose { const char *name; double ti,tj,phi,h,epsilon; };
static Axes directors(double ti,double tj,double phi) {
  return {{sc::Vec3(std::sin(ti),0.,std::cos(ti)),
           sc::Vec3(std::sin(tj)*std::cos(phi),std::sin(tj)*std::sin(phi),std::cos(tj))}};
}
static Mat11 isolated(sc::Provider &oracle,const Axes &axes,double actual_h) {
  const double ar=oracle.basis.aspect_ratio;
  const auto bodies=sc::geometry(axes,actual_h,ar);
  return sc::reduced(oracle.isolated(bodies),sc::midpoint_map(bodies,actual_h,ar));
}
template<int N> static double energy_error(const Eigen::Matrix<double,N,N> &prediction,
                                          const Eigen::Matrix<double,N,N> &reference) {
  if(!prediction.allFinite() || !reference.allFinite())
    throw std::runtime_error("Nonfinite matrix in energy comparison");
  Eigen::LLT<Eigen::Matrix<double,N,N>> factor(reference);
  if(factor.info()!=Eigen::Success) throw std::runtime_error("Oracle total resistance is not SPD");
  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix<double,N,N>> eigen(prediction,reference);
  if(eigen.info()!=Eigen::Success) throw std::runtime_error("Generalized energy eigensolve failed");
  // A nonpositive prediction eigenvalue gives error >= 1, hence a quantitative
  // gate failure; do not hide that number by throwing before it is reported.
  return (eigen.eigenvalues().array()-1.).abs().maxCoeff();
}
static Mat5 free_strain(const Mat11 &r) {
  Eigen::LLT<Eigen::Matrix<double,6,6>> factor(r.topLeftCorner<6,6>());
  if(factor.info()!=Eigen::Success) throw std::runtime_error("Pair velocity block is not SPD");
  const Mat5 s=r.bottomRightCorner<5,5>()-
      r.bottomLeftCorner<5,6>()*factor.solve(r.topRightCorner<6,5>());
  return .5*(s+s.transpose()).eval();
}
static bool comparison(SlurrySPDTable::Provider &fast,sc::Provider &oracle,
                       const Pose &p,double tolerance) {
  const Axes axes=directors(p.ti,p.tj,p.phi);
  const Mat11 self=isolated(oracle,axes,p.h);
  const Mat11 expected=self+oracle.correction(axes,p.h,p.epsilon);
  const Mat11 actual=self+fast.correction(axes,p.h,p.epsilon);
  const double error=energy_error<11>(actual,expected);
  const double reciprocity=(actual-actual.transpose()).norm()/actual.norm();
  if(error>tolerance || reciprocity>=1e-12) {
    Eigen::GeneralizedSelfAdjointEigenSolver<Mat11> generalized(actual,expected);
    Eigen::SelfAdjointEigenSolver<Mat11> direct_spectrum(expected),fast_spectrum(actual);
    const auto mode=generalized.eigenvectors().col(0).normalized().eval();
    std::cout<<"GATE heldout "<<p.name<<" FAIL h="<<p.h<<" epsilon="<<p.epsilon
      <<" all11_error="<<error<<" reciprocity="<<reciprocity
      <<" limit="<<tolerance<<" later_checks=NOT_RUN\n"<<std::flush;
    std::cout<<"SAME_POSE_DIAGNOSTIC min_energy_ratio="<<generalized.eigenvalues().minCoeff()
      <<" max_energy_ratio="<<generalized.eigenvalues().maxCoeff()
      <<" matched_min_eigenvalue="<<direct_spectrum.eigenvalues().minCoeff()
      <<" fast_min_eigenvalue="<<fast_spectrum.eigenvalues().minCoeff()
      <<" min_ratio_mode_euclidean_normalized="<<mode.transpose()<<"\n"<<std::flush;
    std::ofstream dump("trial/audit/first_failure_matrices.txt");
    if(dump) {
      dump<<std::setprecision(17)<<"POSE "<<p.ti<<' '<<p.tj<<' '<<p.phi
          <<' '<<p.h<<' '<<p.epsilon<<"\n";
      const auto write=[&](const char *name,const Mat11 &m){dump<<name<<"\n"<<m<<"\n";};
      write("ISOLATED_ACTUAL_GEOMETRY",self);
      write("MATCHED_TOTAL",expected);write("FAST_TOTAL",actual);
      write("MATCHED_CORRECTION",expected-self);write("FAST_CORRECTION",actual-self);
      const double heff=std::max(p.h,0.)+p.epsilon;
      const Mat11 self_virtual=isolated(oracle,axes,heff);
      write("ISOLATED_VIRTUAL_GEOMETRY",self_virtual);
      write("MATCHED_VIRTUAL_TOTAL",expected-self+self_virtual);
      write("FAST_VIRTUAL_TOTAL",actual-self+self_virtual);
      // Entries below are the conditioned total matrix logarithm, not a residual.
      write("INTERPOLATED_CONDITIONED_MATRIX_LOG",fast.residual(p.ti,p.tj,p.phi,heff));
      dump<<"GENERALIZED_EIGENVALUES\n"<<generalized.eigenvalues().transpose()
          <<"\nMIN_RATIO_MODE_EUCLIDEAN_NORMALIZED\n"<<mode.transpose()<<"\n";
    }
    return false;
  }
  const double stress=energy_error<5>(free_strain(actual),free_strain(expected));
  const bool pass=error<=tolerance && stress<=tolerance && reciprocity<1e-12;
  std::cout<<"GATE heldout "<<p.name<<" "<<(pass?"PASS":"FAIL")
    <<" h="<<p.h<<" epsilon="<<p.epsilon<<" all11_error="<<error
    <<" free_strain_error="<<stress<<" reciprocity="<<reciprocity
    <<" limit="<<tolerance<<"\n"<<std::flush;
  return pass;
}
static bool covariance(SlurrySPDTable::Provider &fast,sc::Provider &oracle) {
  const Axes axes=directors(.073,.137,.79);
  const double h=.02,epsilon=.001,angle=.371;
  sc::Mat3 q=Eigen::AngleAxisd(angle,sc::Vec3::UnitZ()).toRotationMatrix();
  Mat11 Q=Mat11::Zero();Q.block<3,3>(0,0)=q;Q.block<3,3>(3,3)=q;
  Q.bottomRightCorner<5,5>()=sc::strain_rotation(q);
  const Mat11 original=isolated(oracle,axes,h)+fast.correction(axes,h,epsilon);
  Axes rotated={{q*axes[0],q*axes[1]}};
  const Mat11 actual=isolated(oracle,rotated,h)+fast.correction(rotated,h,epsilon);
  const double rotation_error=energy_error<11>(actual,Q*original*Q.transpose());
  double director_error=0.;
  for(int mask=1;mask<4;++mask) {
    Axes flipped=axes;
    for(int k=0;k<2;++k)if(mask&(1<<k))flipped[k]=-flipped[k];
    const Mat11 flip=isolated(oracle,flipped,h)+fast.correction(flipped,h,epsilon);
    director_error=std::max(director_error,energy_error<11>(flip,original));
  }
  const bool pass=rotation_error<1e-9 && director_error<1e-9;
  std::cout<<"GATE covariance "<<(pass?"PASS":"FAIL")
      <<" rotation_error="<<rotation_error<<" director_sign_error="<<director_error<<"\n";
  return pass;
}
static bool angular_continuity(SlurrySPDTable::Provider &fast,sc::Provider &oracle) {
  // Frozen non-node partner angles and gap. The physical resistance is
  // continuous even where a canonical head-tail director changes sign.
  const double h=.05731,epsilon=.001,phi=-1.137,partner=.23719;
  const double pi=std::acos(-1.);
  const auto total=[&](const Axes &axes)->Mat11 {
    return isolated(oracle,axes,h)+fast.correction(axes,h,epsilon);
  };
  const auto pair_error=[&](const Axes &a,const Axes &b) {
    return energy_error<11>(total(a),total(b));
  };
  for(double delta:{1e-8,1e-10}) {
    const double pole_i=pair_error(directors(delta,partner,phi),directors(0.,partner,phi));
    const double pole_j=pair_error(directors(partner,delta,phi),directors(partner,0.,phi));
    const double equator_i=pair_error(directors(.5*pi-delta,partner,phi),directors(.5*pi+delta,partner,phi));
    const double equator_j=pair_error(directors(partner,.5*pi-delta,phi),directors(partner,.5*pi+delta,phi));
    const double maximum=std::max({pole_i,pole_j,equator_i,equator_j});
    const bool pass=maximum<=1e-5;
    std::cout<<"GATE angular_continuity "<<(pass?"PASS":"FAIL")
      <<" delta="<<delta<<" pole_i="<<pole_i<<" pole_j="<<pole_j
      <<" equator_i="<<equator_i<<" equator_j="<<equator_j<<" limit=1e-5\n"<<std::flush;
    if(!pass)return false;
  }
  return true;
}
static bool near_limit(SlurrySPDTable::Provider &fast,sc::Provider &oracle) {
  const Axes face={{sc::Vec3::UnitZ(),sc::Vec3::UnitZ()}};
  const double h=std::max(.001,fast.min_gap()),ar=fast.aspect_ratio();
  const Mat11 total=isolated(oracle,face,h)+fast.correction(face,h,0.);
  // Total gap Hessian C=(2/ar^2)I: 12*pi/[h tr(C) sqrt(det(C))].
  const double leading=1.5*std::acos(-1.)*std::pow(ar,4)/h;
  const double ratio=total(2,2)/leading;
  const bool pass=std::isfinite(ratio)&&std::abs(ratio-1.)<.05;
  std::cout<<"GATE face_near_1_over_h "<<(pass?"PASS":"FAIL")
      <<" h="<<h<<" normal_leading_ratio="<<ratio<<" limit=.05\n";
  return pass;
}
static SlurryFiniteGap::Pair physical_pair(SlurrySPDTable::Provider &fast,double h) {
  namespace gr=SlurryFiniteGap::gr;
  const double c=150e-9,ar=fast.aspect_ratio();
  const auto axes=directors(.073,.137,.79);
  const auto bodies=sc::geometry(axes,h,ar);
  std::array<gr::Body,2> b;
  for(int i=0;i<2;++i) {
    b[i].axes={ar*c,ar*c,c};
    for(int j=0;j<3;++j) {
      b[i].position[j]=bodies[i].center[j]*c;
      for(int k=0;k<3;++k)b[i].rotation[3*j+k]=bodies[i].rotation(j,k);
    }
  }
  gr::GapResult gap;gap.gap=h*c;gap.normal={0,0,1};gap.converged=true;
  for(int k=0;k<3;++k) {
    gap.leverI[k]=-bodies[0].center[k]*c;
    gap.leverJ[k]=((k==2?h:0.)-bodies[1].center[k])*c;
  }
  return fast.make_pair(b[0],b[1],gap,.01,ar*c,.5,.8,11,12);
}
static bool cutoff(SlurrySPDTable::Provider &fast) {
  const double outer=.5*fast.aspect_ratio();
  const auto inside=physical_pair(fast,outer-1e-4);
  const auto edge=physical_pair(fast,outer);
  const auto outside=physical_pair(fast,outer+1e-4);
  const bool pass=inside.active && inside.weight>=0. && inside.weight<1e-8 &&
                  !edge.active && !outside.active;
  std::cout<<"GATE outer_cutoff "<<(pass?"PASS":"FAIL")
      <<" outer_h="<<outer<<" inside_weight="<<inside.weight
      <<" edge_active="<<edge.active<<" outside_active="<<outside.active<<"\n";
  return pass;
}
static bool physical_references(SlurrySPDTable::Provider &fast,sc::Provider &oracle,
                                const std::string &path) {
  std::ifstream input(path);std::string magic;int count=0;
  if(!(input>>magic>>count)||magic!="SLURRY_INDEPENDENT_REFERENCE_V1"||count!=2)
    throw std::runtime_error("Invalid independent reference file");
  for(int k=0;k<count;++k) {
    std::string name;double h,tolerance,stress_tolerance;Axes axes;Mat11 reference,self;
    input>>name>>h>>tolerance>>stress_tolerance;
    for(auto &axis:axes)for(int j=0;j<3;++j)input>>axis[j];
    for(int i=0;i<11;++i)for(int j=0;j<11;++j)input>>reference(i,j);
    for(int i=0;i<11;++i)for(int j=0;j<11;++j)input>>self(i,j);
    if(!input)throw std::runtime_error("Truncated independent reference file");
    const Mat11 baseline=self+oracle.correction(axes,h,0.);
    const Mat11 predicted=self+fast.correction(axes,h,0.);
    const double base_error=energy_error<11>(baseline,reference);
    const double fast_error=energy_error<11>(predicted,reference);
    const double base_stress=energy_error<5>(free_strain(baseline),free_strain(reference));
    const double fast_stress=energy_error<5>(free_strain(predicted),free_strain(reference));
    // This allows only the emulator's stated 1% on top of baseline error.
    // It does not introduce, or claim to meet, a 5% physical-accuracy target.
    const bool pass=fast_error<=base_error+.011 && fast_stress<=base_stress+.011;
    std::cout<<"GATE independent_stokes "<<name<<" "<<(pass?"PASS":"FAIL")
      <<" baseline_error="<<base_error<<" fast_error="<<fast_error
      <<" baseline_free_strain_error="<<base_stress<<" fast_free_strain_error="<<fast_stress
      <<" physical_five_percent_pass="<<(fast_error<=.05)<<"\n"<<std::flush;
    if(!pass)return false;
  }
  return true;
}
static bool speed(SlurrySPDTable::Provider &fast,sc::Provider &oracle) {
  constexpr int nf=1000,no=8;volatile double checksum=0.;
  const auto t0=std::chrono::steady_clock::now();
  for(int k=0;k<nf;++k) {
    const double u=double(k%17)/17.;const auto axes=directors(.073+.0001*u,.137,.79);
    checksum+=fast.correction(axes,.020+.00001*u,.001)(2,2);
  }
  const double fast_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()/nf;
  const auto t1=std::chrono::steady_clock::now();
  for(int k=0;k<no;++k) {
    const double u=double(k)/no;const auto axes=directors(.073+.0001*u,.137,.79);
    checksum+=oracle.correction(axes,.020+.00001*u,.001)(2,2);
  }
  const double oracle_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t1).count()/no;
  const double acceleration=oracle_seconds/fast_seconds;
  const bool pass=acceleration>=10.;
  std::cout<<"GATE changing_pose_pair_speed "<<(pass?"PASS":"FAIL")
    <<" fast_s="<<fast_seconds<<" matched_s="<<oracle_seconds
    <<" acceleration="<<acceleration<<" required=10 checksum="<<checksum<<"\n";
  return pass;
}
int main(int argc,char **argv) {
  try {
    if(argc!=4 && argc!=5)throw std::runtime_error("Usage: test_precomputed_trial table.spt matched.ssb independent_reference.txt [film_grid=161]");
    std::cout<<std::setprecision(15);
    SlurrySPDTable::Provider fast;fast.load(argv[1]);
    sc::Provider oracle;oracle.load(argv[2]);oracle.film_grid=argc==5?std::stoi(argv[4]):161;
    if(oracle.film_grid<21)throw std::runtime_error("Oracle film grid must be at least 21");
    std::cout<<"ORACLE film_grid="<<oracle.film_grid<<"\n";
    if(std::abs(fast.aspect_ratio()-10.)>1e-12||oracle.basis.aspect_ratio!=10.)
      throw std::runtime_error("This validation trial is deliberately AR10 only");
    const Pose first={"first_asymmetric_contact",.073,.137,.79,.020,.001};
    if(!comparison(fast,oracle,first,.01))return 2;
    if(!covariance(fast,oracle))return 2;
    if(!near_limit(fast,oracle))return 2;
    const std::vector<Pose> heldout={
      {"face_gap",0.,0.,0.,.033,.001},
      {"parallel_oblique",.173,.173,0.,.11,.001},
      {"tilted",.113,.287,.531,.37,.001},
      {"face_edge",.017,1.431,.83,.19,.001},
      {"crossed_edge",1.413,1.317,1.113,.027,.001},
      {"outer",1.021,.713,2.031,3.7,.001}};
    for(const auto &p:heldout)if(!comparison(fast,oracle,p,.01))return 2;
    // Frozen fresh validation points: do not add to training or use for grid
    // refinement and then continue calling this set held out. The first failure
    // stops this run before remaining poses and the expensive later gates.
    const std::vector<Pose> fresh={
      {"fresh01_near_face",.0267,.0913,-.417,.0027,.001},
      {"fresh02_near_tilt",.0581,.1897,1.223,.0063,.001},
      {"fresh03_reversed_tilt",.2223,.0379,-2.173,.0147,.001},
      {"fresh04_oblique",.3081,.4127,.683,.0473,.001},
      {"fresh05_oblique",.5279,.1733,-1.437,.0937,.001},
      {"fresh06_crossed",.7843,.6187,2.719,.273,.001},
      {"fresh07_face_edge",.0397,1.4893,-.931,.0719,.001},
      {"fresh08_edge_face",1.4471,.0823,1.843,.137,.001},
      {"fresh09_near_edges",1.5297,1.3979,-2.641,.0039,.001},
      {"fresh10_edges",1.2329,1.5087,.319,.783,.001},
      {"fresh11_middle_gap",.9361,.2873,-.557,1.831,.001},
      {"fresh12_outer_gap",.3911,1.1739,2.287,4.137,.001}};
    for(const auto &p:fresh)if(!comparison(fast,oracle,p,.01))return 2;
    if(!angular_continuity(fast,oracle))return 2;
    if(!cutoff(fast))return 2;
    if(!physical_references(fast,oracle,argv[3]))return 2;
    if(!speed(fast,oracle))return 2;
    std::cout<<"PASS AR10 finite held-out trial; no bulk or other-AR claim\n";
    return 0;
  } catch(const std::exception &e) {
    std::cerr<<"FAIL_STOP "<<e.what()<<"\n";return 2;
  }
}
