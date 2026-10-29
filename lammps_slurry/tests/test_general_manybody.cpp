// Structural many-body assembly regression, not an orientation survey.
// A lattice of individually passive pairs defeats naive pair resistance sums.
#include "slurry_general_provider.h"
#include "slurry_general_outer.h"
#include <Eigen/Eigenvalues>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace sg=SlurryGeneral;
using Dynamic=Eigen::MatrixXd;
using Mat5=Eigen::Matrix<double,5,5>;
using Map11=Eigen::Matrix<double,11,17>;

static void require(bool condition,const char* message) {
  if(!condition)throw std::runtime_error(message);
}
static double minimum(const Dynamic& matrix) {
  Eigen::SelfAdjointEigenSolver<Dynamic> eig(.5*(matrix+matrix.transpose()).eval());
  require(eig.info()==Eigen::Success,"Assembly eigenvalue calculation failed");
  return eig.eigenvalues()[0];
}
static sg::Mat3 skew(const sg::Vec3& r) {
  sg::Mat3 out;out<<0,-r.z(),r.y(),r.z(),0,-r.x(),-r.y(),r.x(),0;return out;
}
static Map11 projection(const sg::Geometry& geometry) {
  const sg::Vec3 midpoint=.5*(geometry.centers[0]+geometry.centers[1]);
  const sg::Vec3 li=(midpoint-geometry.centers[0])/geometry.ar;
  const sg::Vec3 lj=(midpoint-geometry.centers[1])/geometry.ar;
  Map11 p=Map11::Zero();
  p.block<3,3>(0,0)=-sg::Mat3::Identity();p.block<3,3>(0,6)=sg::Mat3::Identity();
  p.block<3,3>(3,3)=-sg::Mat3::Identity();p.block<3,3>(3,9)=sg::Mat3::Identity();
  p.block<3,3>(0,3)=skew(li);p.block<3,3>(0,9)=-skew(lj);
  p.block<5,5>(6,12).setIdentity();return p;
}
static sg::Mat17 rotate(const sg::Mat17& matrix,const sg::Mat3& q) {
  sg::Mat17 t=sg::Mat17::Zero();
  for(int k=0;k<4;++k)t.block<3,3>(3*k,3*k)=q;
  const auto& e=sg::strain_basis();
  for(int i=0;i<5;++i)for(int j=0;j<5;++j)
    t(12+i,12+j)=(e[i].array()*(q*e[j]*q.transpose()).array()).sum();
  return (t*matrix*t.transpose()).eval();
}
static Mat5 free_strain(const Dynamic& matrix) {
  const int n=static_cast<int>(matrix.rows())-5;
  Eigen::LLT<Dynamic> factor(matrix.topLeftCorner(n,n));
  require(factor.info()==Eigen::Success,"Free-strain reduction requires positive velocity block");
  const Dynamic coupling=matrix.topRightCorner(n,5);
  return (matrix.bottomRightCorner(5,5)-coupling.transpose()*factor.solve(coupling)).eval();
}
static void add_pair(Dynamic& matrix,const sg::Mat17& pair,int i,int j,int n) {
  std::array<int,17> index{};
  for(int k=0;k<6;++k){index[k]=6*i+k;index[6+k]=6*j+k;}
  for(int k=0;k<5;++k)index[12+k]=6*n+k;
  for(int r=0;r<17;++r)for(int c=0;c<17;++c)matrix(index[r],index[c])+=pair(r,c);
}
struct Report {
  double ar=0,minimum_pair=0,full_velocity=0,relative_velocity=0,relative_free_strain=0;
  double minimum_full_translation=0,recovery=0,free_recovery=0,seconds=0;
  int maximum_radial_order=0;
};
static Report check(double ar) {
  const auto started=std::chrono::steady_clock::now();
  constexpr int side=3,n=side*side*side;
  const double h=.1;
  Report out;out.ar=ar;out.minimum_pair=1e100;
  std::array<sg::Mat17,3> full_correction,relative_correction;
  // All bodies point along world z. Evaluate only two distinct pair geometries.
  const auto face=sg::geometry_from_axes(ar,h,sg::Vec3::UnitZ(),sg::Vec3::UnitZ());
  const auto edge=sg::geometry_from_axes(ar,h,sg::Vec3::UnitX(),sg::Vec3::UnitX());
  const auto face_result=sg::evaluate_model(face);
  const auto edge_result=sg::evaluate_model(edge);
  for(int direction=0;direction<3;++direction) {
    const auto& geometry=direction==2?face:edge;
    const auto& model=direction==2?face_result:edge_result;
    out.maximum_radial_order=std::max(out.maximum_radial_order,model.nrad);
    out.minimum_pair=std::min(out.minimum_pair,minimum(model.grand));
    const Map11 p=projection(geometry);
    const sg::Map6 b=p.topLeftCorner<6,12>();
    const sg::Mat11 target=sg::reduced_response(model.grand,b);
    const sg::Mat11 delta=target-sg::reduced_response(model.isolated,b);
    const sg::Mat17 correction=p.transpose()*delta*p;
    const sg::Mat17 reconstructed=model.isolated+correction;
    out.recovery=std::max(out.recovery,(sg::reduced_response(reconstructed,b)-target).norm()/target.norm());
    const Mat5 c5=free_strain(model.grand);
    out.free_recovery=std::max(out.free_recovery,(free_strain(reconstructed)-c5).norm()/c5.norm());
    sg::Mat3 q=sg::Mat3::Identity();
    // Columns of q are the canonical x,y,z axes in the world. The canonical
    // edge director is x, so q*x must be the common world z director.
    if(direction==0)q<<0,0,1,0,-1,0,1,0,0;
    if(direction==1)q<<0,1,0,0,0,1,1,0,0;
    require(std::abs(q.determinant()-1)<1e-14,"Improper test coordinate frame");
    full_correction[direction]=rotate(model.grand-model.isolated,q);
    relative_correction[direction]=rotate(correction,q);
  }
  // Public coordinates have a*omega and a*E; internal length and viscosity=1.
  sg::Mat11 scale=sg::Mat11::Identity();
  for(int i=3;i<11;++i)scale(i,i)=1/ar;
  const sg::Mat11 one=scale*sg::isolated11(ar)*scale;
  Dynamic isolated=Dynamic::Zero(6*n+5,6*n+5);
  for(int i=0;i<n;++i) {
    isolated.block<6,6>(6*i,6*i)=one.topLeftCorner<6,6>();
    isolated.block<6,5>(6*i,6*n)=one.topRightCorner<6,5>();
    isolated.block<5,6>(6*n,6*i)=one.bottomLeftCorner<5,6>();
    isolated.bottomRightCorner<5,5>()+=one.bottomRightCorner<5,5>();
  }
  Dynamic full=isolated,relative=isolated;
  const auto id=[](int x,int y,int z){return (x*side+y)*side+z;};
  int edges=0;
  for(int x=0;x<side;++x)for(int y=0;y<side;++y)for(int z=0;z<side;++z) {
    const int coordinates[]={x,y,z};
    for(int d=0;d<3;++d)if(coordinates[d]+1<side) {
      int next[]={x,y,z};++next[d];
      const int i=id(x,y,z),j=id(next[0],next[1],next[2]);
      add_pair(full,full_correction[d],i,j,n);
      add_pair(relative,relative_correction[d],i,j,n);++edges;
    }
  }
  require(edges==54,"Wrong structural test edge count");
  out.full_velocity=minimum(full.topLeftCorner(6*n,6*n));
  out.relative_velocity=minimum(relative.topLeftCorner(6*n,6*n));
  out.relative_free_strain=minimum(free_strain(relative));
  out.minimum_full_translation=1e100;
  for(int d=0;d<3;++d) {
    Eigen::VectorXd common=Eigen::VectorXd::Zero(6*n+5);
    for(int i=0;i<n;++i)common[6*i+d]=1;
    const double reference=common.dot(isolated*common);
    out.minimum_full_translation=std::min(out.minimum_full_translation,common.dot(full*common)/reference);
    require(std::abs(common.dot(relative*common)/reference-1)<1e-10,
            "Relative closure changed isolated common-translation drag");
  }
  require(out.minimum_pair>0,"A resolved reference pair is not positive definite");
  require(out.full_velocity<0&&out.minimum_full_translation<-.05,
          "Naive full-pair addition counterexample unexpectedly disappeared");
  require(out.relative_velocity>0&&out.relative_free_strain>0,
          "Relative closure failed this dense structural regression");
  require(out.recovery<1e-9&&out.free_recovery<1e-8,
          "Relative closure did not preserve the two-body reduced/free-strain response");
  out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  return out;
}
int main(int argc,char** argv) {
  try {
    const std::array<Report,2> reports{{check(2),check(20)}};
    std::ostringstream json;json<<std::setprecision(17)
      <<"{\n  \"passed\": true,\n  \"scope\": \"structural regression; not all-configuration or viscosity certification\",\n"
      <<"  \"geometry\": \"27 aligned oblates, open 3x3x3 nearest-neighbor lattice\",\n"
      <<"  \"center_spacings_over_c\": \"(2*AR+0.1, 2*AR+0.1, 2.1)\",\n"
      <<"  \"gap_over_c\": 0.1,\n  \"edges\": 54,\n  \"solvent_in_strain_check\": false,\n"
      <<"  \"pair_evaluation\": \"adaptive general model at default tolerance 0.005\",\n"
      <<"  \"eigenvalue_clipping\": false,\n  \"cases\": [\n";
    for(std::size_t i=0;i<reports.size();++i) {
      const auto& r=reports[i];if(i)json<<",\n";
      json<<"    {\"ar\": "<<r.ar<<", \"minimum_pair_eigenvalue\": "<<r.minimum_pair
        <<", \"naive_full_pair_velocity_minimum\": "<<r.full_velocity
        <<", \"naive_common_translation_over_isolated\": "<<r.minimum_full_translation
        <<", \"relative_velocity_minimum\": "<<r.relative_velocity
        <<", \"relative_free_strain_minimum\": "<<r.relative_free_strain
        <<", \"two_body_reduced_relative_error\": "<<r.recovery
        <<", \"two_body_free_strain_relative_error\": "<<r.free_recovery
        <<", \"maximum_radial_order\": "<<r.maximum_radial_order
        <<", \"elapsed_seconds\": "<<r.seconds<<"}";
    }
    json<<"\n  ]\n}\n";
    if(argc>1){std::ofstream output(argv[1]);require(bool(output),"Cannot create assembly report");output<<json.str();}
    std::cout<<json.str();return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
