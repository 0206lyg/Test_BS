#include "slurry_grand_pair_lookup.h"
#include <iostream>
#include <stdexcept>
using namespace SlurryGrandPair;
void check(bool b){if(!b)throw std::runtime_error("grand lookup assertion failed");}
int main(){
  FixedGeometryTable t;t.geometry.axes_i=t.geometry.axes_j={1,1,.1};
  TableNode a,b;a.gap_over_length=.25;b.gap_over_length=.5;
  a.model.metadata.model_id=b.model.metadata.model_id="manufactured_lookup_test";
  a.model.metadata.reference_id="a";b.model.metadata.reference_id="b";
  for(int k=0;k<17;++k){a.model.resistance[18*k]=k+1;b.model.resistance[18*k]=2*(k+1);}
  t.nodes={a,b};t.independently_checked_intervals={false};
  LookupRequest q;q.geometry=t.geometry;q.gap_over_length=.375;
  check(lookup_fixed_geometry(t,q).status==LookupStatus::UnvalidatedInterval);
  t.independently_checked_intervals[0]=true;
  auto r=lookup_fixed_geometry(t,q);check(r.supported());
  for(int k=0;k<17;++k)check(std::abs(r.model.resistance[18*k]-1.5*(k+1))<1e-14);
  t.interpolation_coordinate=GapCoordinate::Logarithmic;q.gap_over_length=std::sqrt(.25*.5);
  check(std::abs(lookup_fixed_geometry(t,q).model.resistance[0]-1.5)<1e-14);
  q.geometry.symmetry_axis_i={0,.1,std::sqrt(.99)};
  check(lookup_fixed_geometry(t,q).status==LookupStatus::UnsupportedGeometry);
  q.geometry=t.geometry;q.geometry.symmetry_axis_i={0,0,-1};check(lookup_fixed_geometry(t,q).supported());
  q.gap_over_length=.1;check(lookup_fixed_geometry(t,q).status==LookupStatus::OutsideGapRange);
  q.gap_over_length=.25;check(lookup_fixed_geometry(t,q).supported());
  t.nodes[0].model.metadata.reference_id="";check(lookup_fixed_geometry(t,q).status==LookupStatus::InvalidTable);
  std::cout<<"PASS: explicit geometry coverage, no extrapolation, interpolation gate, provenance, axis-director equivalence.\n";
}
