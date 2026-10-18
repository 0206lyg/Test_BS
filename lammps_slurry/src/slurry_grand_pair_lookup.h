// Deliberately narrow lookup contract for independently checked geometry tables.
// Unsupported orientation/shape/gap MUST be reported, never replaced silently
// by the nearest calibrated pair. This file supplies no spheroid table data.
#ifndef SLURRY_GRAND_PAIR_LOOKUP_H
#define SLURRY_GRAND_PAIR_LOOKUP_H
#include "slurry_grand_pair_math.h"
#include <vector>
namespace SlurryGrandPair {
struct GeometryKey {
  // All vectors use the model's local pair frame, whose third axis is the
  // closest-surface normal. Only axisymmetric bodies are represented here.
  Vec3 axes_i{},axes_j{}; // semiaxes / model_length, positive finite
  Vec3 symmetry_axis_i{0,0,1},symmetry_axis_j{0,0,1};
};
struct LookupRequest { GeometryKey geometry{}; double gap_over_length=0; };
enum class LookupStatus { Supported, UnsupportedGeometry, OutsideGapRange,
                          UnvalidatedInterval, InvalidTable };
struct LookupResult {
  LookupStatus status=LookupStatus::UnsupportedGeometry;
  Model model{};
  std::string detail;
  bool supported()const{return status==LookupStatus::Supported;}
};
struct TableNode {double gap_over_length=0;Model model{};};
enum class GapCoordinate { Linear, Logarithmic };
struct FixedGeometryTable {
  GeometryKey geometry{};
  std::vector<TableNode> nodes;
  // One explicit gate per interval. Set only after independent off-node
  // validation. This is evidence of tested interpolation, not an error bound.
  std::vector<bool> independently_checked_intervals;
  double geometry_tolerance=1e-10;
  GapCoordinate interpolation_coordinate=GapCoordinate::Linear;
};
inline bool same_geometry(const GeometryKey&a,const GeometryKey&b,double tolerance) {
  for(int k=0;k<3;++k) {
    if(std::abs(a.axes_i[k]-b.axes_i[k])>tolerance*std::max({a.axes_i[k],b.axes_i[k],1e-300}))return false;
    if(std::abs(a.axes_j[k]-b.axes_j[k])>tolerance*std::max({a.axes_j[k],b.axes_j[k],1e-300}))return false;
  }
  for(int particle=0;particle<2;++particle) {
    const auto&u=particle?a.symmetry_axis_j:a.symmetry_axis_i;
    const auto&v=particle?b.symmetry_axis_j:b.symmetry_axis_i;
    double plus=0,minus=0;
    for(int k=0;k<3;++k){plus+=(u[k]-v[k])*(u[k]-v[k]);minus+=(u[k]+v[k])*(u[k]+v[k]);}
    if(std::sqrt(std::min(plus,minus))>tolerance)return false; // axis director n=-n
  }
  return true;
}
inline bool valid_geometry(const GeometryKey&g) {
  for(double x:g.axes_i)if(!(x>0)||!std::isfinite(x))return false;
  for(double x:g.axes_j)if(!(x>0)||!std::isfinite(x))return false;
  if(std::abs(g.axes_i[0]-g.axes_i[1])>1e-12*g.axes_i[0]||
     std::abs(g.axes_j[0]-g.axes_j[1])>1e-12*g.axes_j[0])return false;
  for(const auto&v:{g.symmetry_axis_i,g.symmetry_axis_j}) {
    double norm=0;for(double x:v){if(!std::isfinite(x))return false;norm+=x*x;}
    if(std::abs(norm-1)>1e-10)return false;
  }
  return true;
}
inline LookupResult lookup_fixed_geometry(const FixedGeometryTable&t,const LookupRequest&r) {
  LookupResult out;
  if(!valid_geometry(t.geometry)||!valid_geometry(r.geometry)||!std::isfinite(r.gap_over_length)||
     !(t.geometry_tolerance>0)||!std::isfinite(t.geometry_tolerance)||t.nodes.empty()||
     t.independently_checked_intervals.size()+1!=t.nodes.size()) {
    out.status=LookupStatus::InvalidTable;out.detail="Invalid geometry, request, or interval metadata";return out;
  }
  for(std::size_t k=0;k<t.nodes.size();++k) {
    if(!std::isfinite(t.nodes[k].gap_over_length)||
       (t.interpolation_coordinate==GapCoordinate::Logarithmic && !(t.nodes[k].gap_over_length>0))||
       (k&&t.nodes[k].gap_over_length<=t.nodes[k-1].gap_over_length)) {
      out.status=LookupStatus::InvalidTable;out.detail="Gap nodes must be finite and strictly increasing";return out;
    }
    if(t.nodes[k].model.metadata.model_id.empty()||t.nodes[k].model.metadata.reference_id.empty()) {
      out.status=LookupStatus::InvalidTable;out.detail="Missing model identity or reference provenance";return out;
    }
    if(k&&(t.nodes[k].model.metadata.kind!=t.nodes[0].model.metadata.kind||
           t.nodes[k].model.metadata.model_id!=t.nodes[0].model.metadata.model_id)) {
      out.status=LookupStatus::InvalidTable;out.detail="Cannot mix model identities or full/correction matrices";return out;
    }
    try{validate(Frame{},t.nodes[k].model);}catch(const std::exception& e) {
      out.status=LookupStatus::InvalidTable;out.detail=e.what();return out;
    }
  }
  if(!same_geometry(t.geometry,r.geometry,t.geometry_tolerance)) {
    out.status=LookupStatus::UnsupportedGeometry;out.detail="No independently calibrated table for this shape/orientation";return out;
  }
  const double h=r.gap_over_length;
  if(h<t.nodes.front().gap_over_length||h>t.nodes.back().gap_over_length) {
    out.status=LookupStatus::OutsideGapRange;out.detail="Extrapolation outside calibrated gaps is disabled";return out;
  }
  for(std::size_t k=0;k<t.nodes.size();++k) {
    if(h==t.nodes[k].gap_over_length) {out.status=LookupStatus::Supported;out.model=t.nodes[k].model;return out;}
    if(h<t.nodes[k].gap_over_length) {
      if(!t.independently_checked_intervals[k-1]) {
        out.status=LookupStatus::UnvalidatedInterval;out.detail="Gap interpolation has no independent off-node check";return out;
      }
      const auto&a=t.nodes[k-1];const auto&b=t.nodes[k];
      const double weight=t.interpolation_coordinate==GapCoordinate::Logarithmic?
        std::log(h/a.gap_over_length)/std::log(b.gap_over_length/a.gap_over_length):
        (h-a.gap_over_length)/(b.gap_over_length-a.gap_over_length);
      out.model=a.model;
      for(int j=0;j<289;++j)out.model.resistance[j]=(1-weight)*a.model.resistance[j]+weight*b.model.resistance[j];
      out.model.metadata.reference_id=a.model.metadata.reference_id+"|"+b.model.metadata.reference_id;
      out.status=LookupStatus::Supported;return out;
    }
  }
  out.status=LookupStatus::InvalidTable;out.detail="Unreachable gap lookup state";return out;
}
} // namespace SlurryGrandPair
#endif
