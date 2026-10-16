#include "slurry_eigen_ilu_math.h"
#include <iostream>
#include <limits>
#include <stdexcept>

#ifndef SLURRY_HAVE_EIGEN
#error This test requires the Eigen backend
#endif

void require(bool value,const char* message)
{ if (!value) throw std::runtime_error(message); }

std::vector<double> multiply(int n,const std::vector<SlurryEigen::Entry>& entries,
                             const std::vector<double>& x)
{
  std::vector<double> y(n,0);
  for (const auto &e:entries) y[e.row]+=e.value*x[e.column];
  return y;
}

void scaled_nonsymmetric_preconditioner()
{
  constexpr int n=53;
  std::vector<double> scale(n),expected(n);
  for (int i=0; i<n; ++i) {
    scale[i]=std::pow(10.,(i%9)-4);
    expected[i]=std::cos(.31*i)/scale[i];
  }
  std::vector<SlurryEigen::Entry> entries;
  for (int r=0; r<n; ++r)
    for (int c=0; c<n; ++c) {
      const double value=r==c?8.:(std::abs(r-c)==1?.4*std::sin(.4*r+.7*c):
          std::abs(r-c)==7?.17*std::cos(.3*r+.1*c):0.);
      if (value!=0) {
        entries.push_back({r,c,.3*value*scale[r]*scale[c]});
        entries.push_back({r,c,.7*value*scale[r]*scale[c]});
      }
    }
  const auto rhs=multiply(n,entries,expected);
  SlurryEigen::IncompleteLU ilu;
  ilu.factor(n,entries);
  std::vector<double> actual;
  ilu.solve(rhs,actual);
  const auto product=multiply(n,entries,actual);
  double residual=0,initial=0;
  for (int i=0; i<n; ++i) {
    residual+=std::pow((product[i]-rhs[i])/scale[i],2);
    initial+=std::pow(rhs[i]/scale[i],2);
  }
  // One ILUT application is approximate. Test useful residual reduction in
  // scaled coordinates, never exact equality with the physical solution.
  require(residual<1e-5*initial,"ILUT failed to reduce the scaled nonsymmetric residual");
  require(ilu.nonzeros()>=n && ilu.nonzeros()<=std::size_t(n*n),
          "ILUT diagnostics do not report a plausible actual factor size");
  auto changed=entries;
  for (auto &e:changed) e.value*=1.7;
  auto rhs2=rhs;for (double &v:rhs2) v*=1.7;
  ilu.factor(n,changed);
  std::vector<double> reused;
  ilu.solve(rhs2,reused);
  for (int i=0; i<n; ++i)
    require(std::abs((reused[i]-actual[i])*scale[i])<1e-12,
            "ILUT symbolic reuse changed the scaled approximate inverse");
  ilu.factor(n,entries,1e-5,8);
  require(ilu.drop_tolerance()==1e-5 && ilu.fill_factor()==8,
          "ILUT tuning parameters were not applied");
}

void failure_invalidates_factor()
{
  SlurryEigen::IncompleteLU ilu;
  const std::vector<SlurryEigen::Entry> identity{{0,0,1},{1,1,1}};
  ilu.factor(2,identity);
  std::vector<double> solution;
  bool failed=false;
  try { ilu.factor(2,{{0,0,1},{1,1,0}}); }
  catch (const std::runtime_error&) { failed=true; }
  require(failed && ilu.nonzeros()==0,"An empty ILUT row was not rejected");
  failed=false;
  try { ilu.solve({1,2},solution); }
  catch (const std::exception&) { failed=true; }
  require(failed,"Failed ILUT factorization left a stale factor available");
  ilu.factor(2,identity);
  failed=false;
  try { ilu.factor(2,{{0,0,std::numeric_limits<double>::infinity()},{1,1,1}}); }
  catch (const std::invalid_argument&) { failed=true; }
  require(failed,"Nonfinite ILUT input was accepted");
  failed=false;
  try { ilu.solve({1,2},solution); }
  catch (const std::exception&) { failed=true; }
  require(failed,"Invalid ILUT input left a stale factor available");
  ilu.factor(2,identity);
  failed=false;
  try { ilu.solve({1,std::numeric_limits<double>::quiet_NaN()},solution); }
  catch (const std::runtime_error&) { failed=true; }
  require(failed,"Nonfinite ILUT right-hand side was accepted");
  failed=false;
  try { ilu.configure(0,20); }
  catch (const std::invalid_argument&) { failed=true; }
  require(failed,"Invalid ILUT dropping tolerance was accepted");
}

int main()
{
  scaled_nonsymmetric_preconditioner();
  failure_invalidates_factor();
  std::cout<<"PASS: approximate nonsymmetric ILUT, scaling, actual factor fill, "
               "symbolic reuse, tuning, and invalid-factor protection\n";
}
