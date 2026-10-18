#include "slurry_grand_system_check.h"
#include <iostream>
#include <stdexcept>
using namespace SlurryGrandSystem;
void check(bool b){if(!b)throw std::runtime_error("global grand passivity assertion failed");}
int main(){
  std::vector<SlurryImplicit::Particle> bodies(2);
  for(auto&b:bodies)for(int k=0;k<6;++k)b.resistance[7*k]=1;
  std::vector<SlurryImplicit::Pair> pairs(1);auto&p=pairs[0];p.i=0;p.j=1;p.generalized_hydro=true;
  p.hydro_resistance[0]=p.hydro_resistance[12*6+6]=-.2;
  p.hydro_resistance[6]=p.hydro_resistance[12*6]=.2;
  std::vector<Coupling6> body_c(2);SlurryFiniteGap::Mat5 d{};for(int k=0;k<5;++k)d[6*k]=2;
  PairBlock g;g.i=0;g.j=1;g.coupling[0]=.1;g.coupling[5*6]=-.1;g.strain[0]=-.3;
  std::vector<PairBlock> blocks{g};
  const auto result=verify(bodies,pairs,body_c,d,blocks,1,1,1);
  // A relative eigenvalue is .6, C norm^2=.02; S00=2-.3-.02/.6.
  check(std::abs(result.strain_margin-(1.7-.02/.6))<1e-12);
  bool failed=false;try{SlurryImplicit::detail::validate_pair_hydro(p);}catch(const std::invalid_argument&){failed=true;}check(failed);
  p.globally_verified_hydro=true;SlurryImplicit::detail::validate_pair_hydro(p);
  auto bad=pairs;bad[0].hydro_resistance[0]=bad[0].hydro_resistance[12*6+6]=-2;
  bad[0].hydro_resistance[6]=bad[0].hydro_resistance[12*6]=2;
  failed=false;try{verify(bodies,bad,body_c,d,blocks,1,1,1);}catch(const std::runtime_error&){failed=true;}check(failed);
  blocks[0].strain[0]=-3;failed=false;try{verify(bodies,pairs,body_c,d,blocks,1,1,1);}catch(const std::runtime_error&){failed=true;}check(failed);
  std::cout<<"PASS: signed pair accepted only with global velocity+strain passivity; negative total velocity/strain rejected.\n";
}
