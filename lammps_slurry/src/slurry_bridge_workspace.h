#ifndef SLURRY_BRIDGE_WORKSPACE_H
#define SLURRY_BRIDGE_WORKSPACE_H
// Included inside SlurryBridgeSolver::detail after the public matrix aliases.
// A workspace belongs to one sequential simulation/solve. Nothing here is
// global, and no numeric factor is reused after its matrix values change.
inline bool same_pattern(const Sparse &a, const Sparse &b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         a.nonZeros() == b.nonZeros() && a.isCompressed() && b.isCompressed() &&
         std::equal(a.outerIndexPtr(), a.outerIndexPtr() + a.outerSize() + 1,
                    b.outerIndexPtr()) &&
         std::equal(a.innerIndexPtr(), a.innerIndexPtr() + a.nonZeros(),
                    b.innerIndexPtr());
}
inline bool same_matrix(const Sparse &a, const Sparse &b) {
  return same_pattern(a, b) &&
         std::equal(a.valuePtr(), a.valuePtr() + a.nonZeros(), b.valuePtr());
}
struct SelfFactors {
  std::vector<Matrix11> original, lower, inverse;
  explicit SelfFactors(const Problem &p) : original(p.self_mobility) {
    lower.resize(original.size()); inverse.resize(original.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
      Eigen::LLT<Matrix11> factor(original[i]);
      if (factor.info() != Eigen::Success)
        throw std::runtime_error("bridge solver: self mobility is not positive definite");
      lower[i] = factor.matrixL();
      inverse[i] = factor.solve(Matrix11::Identity());
    }
  }
  bool matches(const Problem &p) const {
    if (original.size() != p.self_mobility.size()) return false;
    for (std::size_t i = 0; i < original.size(); ++i)
      if (!(original[i].array() == p.self_mobility[i].array()).all()) return false;
    return true;
  }
};
struct MonitorFactors {
  std::shared_ptr<SelfFactors> self;
  Sparse matrix;
  Eigen::SimplicialLLT<Sparse> factor;
  MonitorFactors(std::shared_ptr<SelfFactors> s, const Sparse &a)
      : self(std::move(s)), matrix(a) {
    factor.compute(matrix);
    if (factor.info() != Eigen::Success)
      throw std::runtime_error("bridge solver: global sparse velocity preconditioner is not positive definite");
  }
};
#ifdef SLURRY_HAVE_MKL_PARDISO
using ParallelSparseLU = Eigen::PardisoLU<Sparse>;
inline const char *preconditioner_backend() { return "mkl_pardiso"; }
#else
using ParallelSparseLU = Eigen::SparseLU<Sparse>;
inline const char *preconditioner_backend() { return "eigen_reference"; }
#endif
struct CoupledFactors {
  Sparse matrix;
  ParallelSparseLU factor;
  bool ready = false;
  std::size_t symbolic_analyses = 0, numeric_factorizations = 0, numeric_reuses = 0;
  void update(const Sparse &a) {
    if (ready && same_matrix(matrix, a)) { ++numeric_reuses; return; }
    const bool pattern_unchanged = ready && same_pattern(matrix, a);
    ready = false;
    matrix = a;
    if (!pattern_unchanged) {
      factor.analyzePattern(matrix);
      ++symbolic_analyses;
    }
    factor.factorize(matrix);
    ++numeric_factorizations;
    if (factor.info() != Eigen::Success)
      throw std::runtime_error("bridge solver: coupled sparse Schur factorization failed");
    ready = true;
  }
};
struct KrylovWorkspace {
  Eigen::MatrixXd basis, directions, h;
  Vector cs, sn, g;
  void resize(int dimension, int restart) {
    if (basis.rows() != dimension || basis.cols() != restart + 1) {
      basis.resize(dimension, restart + 1);
      directions.resize(dimension, restart);
      h.resize(restart + 1, restart);
      cs.resize(restart); sn.resize(restart); g.resize(restart + 1);
    }
  }
};
struct SolverWorkspace {
  std::shared_ptr<SelfFactors> self;
  // Immutable snapshots keep the endpoint acceptance norm fixed even while
  // another solve uses a corrected contact geometry. Bound retained history;
  // live monitors continue owning their factors after eviction.
  std::vector<std::shared_ptr<MonitorFactors>> monitors;
  std::shared_ptr<CoupledFactors> coupled;
  KrylovWorkspace krylov;
  std::shared_ptr<SelfFactors> base_self;
  Sparse base_near_snapshot, coupled_base;
  std::size_t self_factorizations = 0, monitor_factorizations = 0,
              monitor_reuses = 0, coupled_symbolic_analyses = 0,
              coupled_numeric_factorizations = 0, coupled_numeric_reuses = 0;
};
inline std::shared_ptr<SelfFactors> self_factors(
    const Problem &p, const std::shared_ptr<SolverWorkspace> &workspace) {
  if (workspace && workspace->self && workspace->self->matches(p)) return workspace->self;
  auto result = std::make_shared<SelfFactors>(p);
  if (workspace) { workspace->self = result; ++workspace->self_factorizations; }
  return result;
}
// Frozen within a physical step: extract L_UU and self resistance once.
inline Sparse base_resistance(const Problem &p, const std::shared_ptr<SelfFactors> &self,
                              const std::shared_ptr<SolverWorkspace> &workspace) {
  if (workspace && workspace->base_self == self &&
      same_matrix(workspace->base_near_snapshot, p.near_resistance))
    return workspace->coupled_base;
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(36 * p.count + p.near_resistance.nonZeros());
  for (int i = 0; i < p.count; ++i)
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) {
      const double value = self->inverse[i](r,c);
      if (value != 0.) entries.emplace_back(6*i+r,6*i+c,value);
    }
  for (int outer = 0; outer < p.near_resistance.outerSize(); ++outer)
    for (Sparse::InnerIterator it(p.near_resistance, outer); it; ++it)
      if (it.row()%11 < 6 && it.col()%11 < 6)
        entries.emplace_back(6*(it.row()/11)+it.row()%11,
                             6*(it.col()/11)+it.col()%11,it.value());
  Sparse result(6*p.count,6*p.count);
  result.setFromTriplets(entries.begin(),entries.end()); result.makeCompressed();
  if (workspace) {
    workspace->base_self=self;
    workspace->base_near_snapshot=p.near_resistance;
    workspace->coupled_base=result;
  }
  return result;
}
inline std::shared_ptr<MonitorFactors> monitor_factors(
    std::shared_ptr<SelfFactors> self, const Sparse &a,
    const std::shared_ptr<SolverWorkspace> &workspace) {
  if (workspace)
    for (const auto &entry : workspace->monitors)
      if (entry->self == self && same_matrix(entry->matrix, a)) {
        ++workspace->monitor_reuses;
        return entry;
      }
  auto result = std::make_shared<MonitorFactors>(std::move(self), a);
  if (workspace) {
    ++workspace->monitor_factorizations;
    if (workspace->monitors.size() >= 8) workspace->monitors.erase(workspace->monitors.begin());
    workspace->monitors.push_back(result);
  }
  return result;
}
inline std::shared_ptr<CoupledFactors> coupled_factors(
    const Sparse &a, const std::shared_ptr<SolverWorkspace> &workspace) {
  std::shared_ptr<CoupledFactors> result;
  if (workspace && workspace->coupled) {
    // If an older preconditioner is still alive, only share an identical
    // factor. Mutating it would invalidate that older operator's inverse.
    if (workspace->coupled.use_count() == 1 ||
        (workspace->coupled->ready && same_matrix(workspace->coupled->matrix, a)))
      result = workspace->coupled;
  }
  if (!result) result = std::make_shared<CoupledFactors>();
  const auto old_symbolic = result->symbolic_analyses,
             old_numeric = result->numeric_factorizations,
             old_reuses = result->numeric_reuses;
  result->update(a);
  if (workspace) {
    workspace->coupled_symbolic_analyses += result->symbolic_analyses - old_symbolic;
    workspace->coupled_numeric_factorizations += result->numeric_factorizations - old_numeric;
    workspace->coupled_numeric_reuses += result->numeric_reuses - old_reuses;
    workspace->coupled = result;
  }
  return result;
}
#endif
