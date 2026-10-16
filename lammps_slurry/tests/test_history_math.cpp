#include "slurry_history_math.h"
#include <iostream>
#include <stdexcept>
namespace g=slurry::gr_re2::graphite;
void close(const g::Vec3&a,const g::Vec3&b) {
  if(g::norm(g::sub(a,b))>2e-13*std::max(1.,g::norm(b)))
    throw std::runtime_error("Contact history transport mismatch");
}
int main() {
  const g::Vec3 n=g::normalized({.3,.4,.5}),xi=SlurryHistory::tangent({.6,-.3,.9},n);
  for(int k=0;k<100;++k) {
    const g::Vec3 angle{.013*k,-.023*k,.031*k};
    auto r=g::rotationIncrement(angle);
    close(SlurryHistory::transport(xi,n,g::mul(r,n),angle),g::mul(r,xi));
  }
  const auto spin=g::scale(n,1.2);
  close(SlurryHistory::transport(xi,n,n,spin),g::mul(g::rotationIncrement(spin),xi));
  const auto z=SlurryHistory::transport(xi,n,g::scale(n,-1.),{});
  if(std::abs(g::norm(z)-g::norm(xi))>1e-13||std::abs(g::dot(z,n))>1e-13)
    throw std::runtime_error("Antiparallel transport failed");
  close(xi,SlurryHistory::transport(xi,n,n,{}));
  std::cout<<"Objective contact history, common spin and antiparallel transport passed\n";
}
