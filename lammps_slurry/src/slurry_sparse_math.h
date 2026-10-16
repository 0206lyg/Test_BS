// Dependency-free sparse SPD factorization used ONLY as a solver preconditioner.
// Symmetric diagonal equilibration and permutations do not modify the operator.
// No pivot shift, incomplete dropping, or contact regularization is performed.
#ifndef SLURRY_SPARSE_MATH_H
#define SLURRY_SPARSE_MATH_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SlurrySparse {
struct Entry { int row, column; double value; };

// Exact minimum-degree ordering on the PARTICLE graph. Eliminating all six
// particle coordinates consecutively preserves its dense diagonal block.
inline std::vector<int> block_minimum_degree(
    std::size_t blocks, const std::vector<std::pair<std::size_t,std::size_t>> &edges,
    int block_size = 6)
{
  std::vector<std::set<int>> graph(blocks);
  for (const auto &edge : edges) {
    if (edge.first >= blocks || edge.second >= blocks)
      throw std::invalid_argument("Sparse graph index out of bounds");
    if (edge.first == edge.second) continue;
    graph[edge.first].insert(static_cast<int>(edge.second));
    graph[edge.second].insert(static_cast<int>(edge.first));
  }
  std::vector<unsigned char> removed(blocks,0);
  std::vector<int> order;
  order.reserve(block_size*blocks);
  for (std::size_t step = 0; step < blocks; ++step) {
    int best = -1;
    for (std::size_t i = 0; i < blocks; ++i)
      if (!removed[i] && (best < 0 || graph[i].size() < graph[best].size()))
        best = static_cast<int>(i);
    const std::vector<int> neighbors(graph[best].begin(),graph[best].end());
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
      graph[neighbors[i]].erase(best);
      for (std::size_t j = i+1; j < neighbors.size(); ++j) {
        graph[neighbors[i]].insert(neighbors[j]);
        graph[neighbors[j]].insert(neighbors[i]);
      }
    }
    removed[best] = 1;
    graph[best].clear();
    for (int k = 0; k < block_size; ++k) order.push_back(block_size*best+k);
  }
  return order;
}

class LDLT {
 public:
  // Input has one symmetric triangle; duplicates are summed. permutation[k]
  // gives the original degree of freedom corresponding to factor coordinate k.
  void factor(int n, const std::vector<Entry> &entries,
              const std::vector<int> &permutation, std::size_t maximum_nonzeros)
  {
    if (n < 0 || static_cast<int>(permutation.size()) != n)
      throw std::invalid_argument("Invalid sparse permutation size");
    n_ = n;
    permutation_ = permutation;
    std::vector<int> inverse(n,-1);
    for (int k = 0; k < n; ++k) {
      const int p = permutation[k];
      if (p < 0 || p >= n || inverse[p] >= 0)
        throw std::invalid_argument("Invalid sparse permutation");
      inverse[p] = k;
    }
    std::vector<Entry> upper;
    upper.reserve(entries.size());
    scale_.assign(n,0);
    for (const auto &entry : entries) {
      if (entry.row < 0 || entry.column < 0 || entry.row >= n || entry.column >= n ||
          !std::isfinite(entry.value))
        throw std::invalid_argument("Invalid sparse matrix entry");
      if (entry.row == entry.column) scale_[entry.row] += entry.value;
      int r = inverse[entry.row], c = inverse[entry.column];
      if (r > c) std::swap(r,c);
      if (entry.value != 0) upper.push_back({r,c,entry.value});
    }
    for (double &s : scale_) {
      if (!(s > 0) || !std::isfinite(s))
        throw std::runtime_error("Sparse SPD matrix has a nonpositive diagonal");
      s = 1/std::sqrt(s);
    }
    std::sort(upper.begin(),upper.end(),[](const Entry &a, const Entry &b) {
      return a.column < b.column || (a.column == b.column && a.row < b.row);
    });
    std::vector<int> ap(n+1,0), ai;
    std::vector<double> ax;
    for (std::size_t k = 0; k < upper.size();) {
      const int r = upper[k].row, c = upper[k].column;
      long double value = 0;
      do { value += upper[k++].value; }
      while (k < upper.size() && upper[k].row == r && upper[k].column == c);
      ai.push_back(r);
      ax.push_back(static_cast<double>(value)*scale_[permutation[r]]*scale_[permutation[c]]);
      ++ap[c+1];
    }
    for (int k = 0; k < n; ++k) ap[k+1] += ap[k];
    // Symbolic elimination tree and exact column counts of unit-lower L.
    std::vector<int> parent(n,-1), flag(n,-1), count(n,0), pattern(n);
    for (int k = 0; k < n; ++k) {
      flag[k] = k;
      for (int p = ap[k]; p < ap[k+1]; ++p) {
        int i = ai[p];
        if (i >= k) continue;
        while (flag[i] != k) {
          if (parent[i] < 0) parent[i] = k;
          ++count[i];
          flag[i] = k;
          i = parent[i];
        }
      }
    }
    lp_.assign(n+1,0);
    std::size_t nnz = 0;
    for (int k = 0; k < n; ++k) {
      nnz += static_cast<std::size_t>(count[k]);
      if (nnz+static_cast<std::size_t>(n) > maximum_nonzeros ||
          nnz > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Sparse factor exceeds configured nonzero budget");
      lp_[k+1] = static_cast<int>(nnz);
    }
    li_.assign(nnz,0);
    lx_.assign(nnz,0);
    diagonal_.assign(n,0);
    std::fill(count.begin(),count.end(),0);
    std::fill(flag.begin(),flag.end(),-1);
    // Factors and the sparse accumulator use the same double precision as
    // the physical operator. Accumulate each cancellation-prone pivot in
    // long double; final accuracy is checked against the original operator.
    std::vector<double> y(n,0);
    // Left-looking LDL^T. The elimination-tree reach computes only the actual
    // nonzero pattern; no global dense matrix is allocated.
    for (int k = 0; k < n; ++k) {
      int top = n;
      flag[k] = k;
      for (int p = ap[k]; p < ap[k+1]; ++p) {
        int i = ai[p], length = 0;
        y[i] += ax[p];
        while (flag[i] != k) {
          pattern[length++] = i;
          flag[i] = k;
          i = parent[i];
        }
        while (length > 0) pattern[--top] = pattern[--length];
      }
      long double pivot = y[k];
      y[k] = 0;
      for (; top < n; ++top) {
        const int i = pattern[top], end = lp_[i]+count[i];
        const double yi = y[i];
        y[i] = 0;
        for (int p = lp_[i]; p < end; ++p) y[li_[p]] -= lx_[p]*yi;
        const double l = static_cast<double>(yi/diagonal_[i]);
        pivot -= l*yi;
        li_[end] = k;
        lx_[end] = l;
        ++count[i];
      }
      if (!(pivot > 0) || !std::isfinite(pivot))
        throw std::runtime_error("Sparse LDLT encountered a nonpositive/nonfinite pivot");
      diagonal_[k] = static_cast<double>(pivot);
    }
  }

  void solve(const std::vector<double> &rhs, std::vector<double> &out) const
  {
    if (static_cast<int>(rhs.size()) != n_)
      throw std::invalid_argument("Sparse solve RHS size mismatch");
    std::vector<double> x(n_);
    for (int k = 0; k < n_; ++k) x[k] = scale_[permutation_[k]]*rhs[permutation_[k]];
    for (int k = 0; k < n_; ++k)
      for (int p = lp_[k]; p < lp_[k+1]; ++p) x[li_[p]] -= lx_[p]*x[k];
    for (int k = 0; k < n_; ++k) x[k] /= diagonal_[k];
    for (int k = n_-1; k >= 0; --k) {
      long double value = x[k];
      for (int p = lp_[k]; p < lp_[k+1]; ++p) value -= lx_[p]*x[li_[p]];
      x[k] = static_cast<double>(value);
    }
    out.resize(n_);
    for (int k = 0; k < n_; ++k) out[permutation_[k]] = scale_[permutation_[k]]*x[k];
  }

  std::size_t nonzeros() const { return lx_.size()+diagonal_.size(); }

 private:
  int n_ = 0;
  std::vector<int> permutation_, lp_, li_;
  std::vector<double> scale_, lx_, diagonal_;
};
} // namespace SlurrySparse
#endif
