#include "slurry_finite_gap_math.h"
#include <iostream>
#include <stdexcept>
int main(int argc,char**argv){
  namespace f=SlurryFiniteGap;
  if(argc!=2)throw std::invalid_argument("Pass reference .sfg catalogue");
  const auto catalogue=f::read_catalogue(argv[1]);double maximum=0;
  for(const auto&family:catalogue.families)for(const auto&node:family.nodes){
    const auto factors=f::factor_matrix(node.matrix);
    const auto rebuilt=f::interpolate_factored(factors,factors,.5);
    double scale=0,error=0;for(int k=0;k<121;++k){scale=std::max(scale,std::abs(node.matrix[k]));error=std::max(error,std::abs(node.matrix[k]-rebuilt[k]));}
    maximum=std::max(maximum,error/scale);
    if(error/scale>1e-10)throw std::runtime_error("Matrix factor/log/exp reconstruction mismatch");
  }
  f::Mat5 known{};known[0]=2;known[1]=known[5]=.5;known[6]=2;known[12]=3;known[18]=4;known[24]=5;
  if(std::abs(f::minimum_eigenvalue(known)-1.5)>1e-13)throw std::runtime_error("Five strain eigenvalue check failed");
  std::cout<<"PASS: actual finite-gap table matrix factorization/log/exp; maximum reconstruction relative error="<<maximum<<"\n";
}
