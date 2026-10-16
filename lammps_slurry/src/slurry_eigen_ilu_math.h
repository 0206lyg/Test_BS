// Incomplete, nonsymmetric Eigen ILUT PRECONDITIONER. Its dropping and the
// library's zero-pivot replacement affect only the preconditioner; callers must
// retain the original Jacobian and true force-residual convergence test.
#ifndef SLURRY_EIGEN_ILU_MATH_H
#define SLURRY_EIGEN_ILU_MATH_H

#include "slurry_eigen_lu_math.h"

#ifdef SLURRY_HAVE_EIGEN
#include <Eigen/IterativeLinearSolvers>
#include <limits>

namespace SlurryEigen {
namespace detail {
// Eigen keeps the combined L/U factor protected. This accessor exposes its
// actual stored entries for diagnostics and arithmetic checks, without changing
// Eigen's factorization, ordering, threshold, or pivot rules.
class InspectedILUT : public Eigen::IncompleteLUT<double,int> {
 public:
  std::size_t factor_nonzeros() const
  { return static_cast<std::size_t>(this->m_lu.nonZeros()); }
  bool finite_factor_with_nonzero_diagonal() const
  {
    for (int row=0; row<this->m_lu.rows(); ++row) {
      bool diagonal=false;
      for (FactorType::InnerIterator it(this->m_lu,row); it; ++it) {
        if (!std::isfinite(it.value())) return false;
        if (it.row()==it.col()) diagonal=it.value()!=0;
      }
      if (!diagonal) return false;
    }
    return true;
  }
};
} // namespace detail

class IncompleteLU {
 public:
  explicit IncompleteLU(double drop_tolerance=1e-4,int fill_factor=20)
  { configure(drop_tolerance,fill_factor); }

  void configure(double drop_tolerance,int fill_factor)
  {
    // Positive dropping also makes Eigen's zero-pivot replacement nonzero.
    // It does not change any entry of the physical matrix supplied by callers.
    ready_=false;
    nonzeros_=0;
    if (!(drop_tolerance>0) || !std::isfinite(drop_tolerance) || fill_factor<1)
      throw std::invalid_argument("ILUT requires finite positive dropping and positive fill factor");
    drop_tolerance_=drop_tolerance;
    fill_factor_=fill_factor;
    solver_.setDroptol(drop_tolerance_);
    solver_.setFillfactor(fill_factor_);
  }

  void factor(int n,const std::vector<Entry>& entries,
              double drop_tolerance,int fill_factor)
  { configure(drop_tolerance,fill_factor); factor(n,entries); }

  void factor(int n,const std::vector<Entry>& entries)
  {
    ready_=false;
    nonzeros_=0;
    if (n<=0) throw std::invalid_argument("ILUT requires a nonempty square matrix");
    std::vector<Eigen::Triplet<double,int>> triplets;
    triplets.reserve(entries.size());
    for (const auto &e:entries) {
      if (e.row<0 || e.column<0 || e.row>=n || e.column>=n || !std::isfinite(e.value))
        throw std::invalid_argument("Invalid incomplete sparse Jacobian entry");
      triplets.emplace_back(e.row,e.column,e.value);
    }
    Matrix matrix(n,n);
    matrix.setFromTriplets(triplets.begin(),triplets.end());
    matrix.makeCompressed();
    scale_.assign(n,0);
    std::vector<double> row_max(n,0);
    for (int column=0; column<n; ++column)
      for (Matrix::InnerIterator it(matrix,column); it; ++it) {
        if (!std::isfinite(it.value()))
          throw std::runtime_error("Nonfinite assembled ILUT matrix");
        row_max[it.row()]=std::max(row_max[it.row()],std::abs(it.value()));
        if (it.row()==column) scale_[column]=std::abs(it.value());
      }
    for (int i=0; i<n; ++i) {
      const double magnitude=std::max(scale_[i],1e-12*row_max[i]);
      if (!(magnitude>0) || !std::isfinite(magnitude))
        throw std::runtime_error("ILUT matrix has an empty/nonfinite row");
      scale_[i]=1/std::sqrt(magnitude);
      if (!std::isfinite(scale_[i])) throw std::runtime_error("Nonfinite ILUT scaling");
    }
    for (int column=0; column<n; ++column)
      for (Matrix::InnerIterator it(matrix,column); it; ++it) {
        it.valueRef()*=scale_[it.row()]*scale_[column];
        if (!std::isfinite(it.value()))
          throw std::runtime_error("Nonfinite scaled ILUT matrix");
      }
    const bool same_pattern=analyzed_ && matrix.rows()==dimension_ &&
        matrix.nonZeros()==static_cast<Eigen::Index>(inner_.size()) &&
        std::equal(inner_.begin(),inner_.end(),matrix.innerIndexPtr()) &&
        std::equal(outer_.begin(),outer_.end(),matrix.outerIndexPtr());
    if (!same_pattern) {
      // Eigen applies AMD to the symmetrized sparsity pattern. Coefficients
      // remain nonsymmetric throughout factorization and solution.
      solver_.analyzePattern(matrix);
      inner_.assign(matrix.innerIndexPtr(),matrix.innerIndexPtr()+matrix.nonZeros());
      outer_.assign(matrix.outerIndexPtr(),matrix.outerIndexPtr()+n+1);
      dimension_=n;
      analyzed_=true;
    }
    solver_.factorize(matrix);
    if (solver_.info()!=Eigen::Success || !solver_.finite_factor_with_nonzero_diagonal())
      throw std::runtime_error("Eigen ILUT factorization failed or produced a nonfinite/zero factor");
    nonzeros_=solver_.factor_nonzeros();
    ready_=true;
  }

  void solve(const std::vector<double>& rhs,std::vector<double>& out) const
  {
    if (!ready_ || static_cast<int>(rhs.size())!=dimension_)
      throw std::invalid_argument("Incomplete LU is unavailable or RHS size is incorrect");
    Eigen::VectorXd scaled(dimension_);
    for (int i=0; i<dimension_; ++i) {
      scaled[i]=scale_[i]*rhs[i];
      if (!std::isfinite(scaled[i])) throw std::runtime_error("Nonfinite ILUT right-hand side");
    }
    const Eigen::VectorXd solution=solver_.solve(scaled);
    if (!solution.allFinite()) throw std::runtime_error("Eigen ILUT solve produced a nonfinite vector");
    out.resize(dimension_);
    for (int i=0; i<dimension_; ++i) {
      out[i]=scale_[i]*solution[i];
      if (!std::isfinite(out[i])) throw std::runtime_error("Nonfinite unscaled ILUT solution");
    }
  }

  std::size_t nonzeros() const { return nonzeros_; }
  double drop_tolerance() const { return drop_tolerance_; }
  int fill_factor() const { return fill_factor_; }

 private:
  detail::InspectedILUT solver_;
  std::vector<double> scale_;
  std::vector<int> inner_,outer_;
  int dimension_=0;
  double drop_tolerance_=1e-4;
  int fill_factor_=20;
  std::size_t nonzeros_=0;
  bool analyzed_=false,ready_=false;
};
} // namespace SlurryEigen
#endif // SLURRY_HAVE_EIGEN
#endif
