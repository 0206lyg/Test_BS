// Established Eigen SparseLU backend for the exact, nonsymmetric Coulomb
// Jacobian. Scaling/permutations are algebraic only: no coefficients are
// dropped, symmetrized, or shifted. Available when Eigen is supplied by CMake.
#ifndef SLURRY_EIGEN_LU_MATH_H
#define SLURRY_EIGEN_LU_MATH_H

#ifdef SLURRY_HAVE_EIGEN
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace SlurryEigen {
struct Entry { int row, column; double value; };
using Matrix = Eigen::SparseMatrix<double,Eigen::ColMajor,int>;

class FullLU {
 public:
  void factor(int n, const std::vector<Entry> &entries)
  {
    ready_ = false;
    std::vector<Eigen::Triplet<double,int>> triplets;
    triplets.reserve(entries.size());
    for (const auto &e : entries) {
      if (e.row < 0 || e.column < 0 || e.row >= n || e.column >= n || !std::isfinite(e.value))
        throw std::invalid_argument("Invalid exact sparse Jacobian entry");
      // Keep structural zeroes so contact/stick branch changes do not change
      // the symbolic pattern within a frozen-geometry timestep.
      triplets.emplace_back(e.row,e.column,e.value);
    }
    Matrix matrix(n,n);
    matrix.setFromTriplets(triplets.begin(),triplets.end());
    matrix.makeCompressed();
    scale_.assign(n,0);
    std::vector<double> row_max(n,0);
    for (int c = 0; c < n; ++c)
      for (Matrix::InnerIterator it(matrix,c); it; ++it) {
        row_max[it.row()] = std::max(row_max[it.row()],std::abs(it.value()));
        if (it.row() == c) scale_[c] = std::abs(it.value());
      }
    for (int i = 0; i < n; ++i) {
      const double magnitude = std::max(scale_[i],1e-12*row_max[i]);
      if (!(magnitude > 0) || !std::isfinite(magnitude))
        throw std::runtime_error("Exact sparse Jacobian has an empty/nonfinite row");
      scale_[i] = 1/std::sqrt(magnitude);
    }
    for (int c = 0; c < n; ++c)
      for (Matrix::InnerIterator it(matrix,c); it; ++it) {
        it.valueRef() *= scale_[it.row()]*scale_[c];
        if (!std::isfinite(it.value()))
          throw std::runtime_error("Nonfinite scaled exact sparse Jacobian");
      }
    const bool same_pattern = analyzed_ && matrix.rows() == dimension_ &&
        matrix.nonZeros() == static_cast<int>(inner_.size()) &&
        std::equal(inner_.begin(),inner_.end(),matrix.innerIndexPtr()) &&
        std::equal(outer_.begin(),outer_.end(),matrix.outerIndexPtr());
    if (!same_pattern) {
      solver_.analyzePattern(matrix);
      inner_.assign(matrix.innerIndexPtr(),matrix.innerIndexPtr()+matrix.nonZeros());
      outer_.assign(matrix.outerIndexPtr(),matrix.outerIndexPtr()+n+1);
      dimension_ = n;
      analyzed_ = true;
    }
    solver_.factorize(matrix);
    if (solver_.info() != Eigen::Success)
      throw std::runtime_error("Eigen SparseLU factorization failed: "+solver_.lastErrorMessage());
    nonzeros_ = static_cast<std::size_t>(solver_.nnzL())+solver_.nnzU();
    ready_ = true;
  }

  void solve(const std::vector<double> &rhs, std::vector<double> &out) const
  {
    if (!ready_ || static_cast<int>(rhs.size()) != dimension_)
      throw std::invalid_argument("Exact sparse LU RHS size mismatch");
    Eigen::VectorXd scaled(dimension_);
    for (int i = 0; i < dimension_; ++i) scaled[i] = scale_[i]*rhs[i];
    const Eigen::VectorXd solution = solver_.solve(scaled);
    if (solver_.info() != Eigen::Success || !solution.allFinite())
      throw std::runtime_error("Eigen SparseLU solve failed");
    out.resize(dimension_);
    for (int i = 0; i < dimension_; ++i) {
      out[i] = scale_[i]*solution[i];
      if (!std::isfinite(out[i])) throw std::runtime_error("Nonfinite unscaled exact LU solution");
    }
  }

  std::size_t nonzeros() const { return nonzeros_; }

 private:
  // COLAMD had materially less fill than AMD on the actual dense N=300 case.
  Eigen::SparseLU<Matrix,Eigen::COLAMDOrdering<int>> solver_;
  std::vector<double> scale_;
  std::vector<int> inner_,outer_;
  int dimension_ = 0;
  std::size_t nonzeros_ = 0;
  bool analyzed_ = false;
  bool ready_ = false;
};
} // namespace SlurryEigen
#endif // SLURRY_HAVE_EIGEN
#endif
