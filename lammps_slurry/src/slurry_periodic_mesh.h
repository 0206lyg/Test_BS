// Included inside namespace SlurryPeriodic after Body, Settings and moment_map.
// Particle-mesh Galerkin operator. Its source and target maps are exact
// transposes, so every retained Fourier mode and the complete operator are PSD.
// Shell measure is the mapped UNIFORM UNIT-SPHERE measure, not ellipsoid area.
#ifndef SLURRY_PERIODIC_MESH_H
#define SLURRY_PERIODIC_MESH_H
namespace MeshDetail {
using Complex = std::complex<double>;
struct Entry {
  std::size_t node;
  double scalar;
  Vec3 vector;
};
struct FourierEntry {
  std::size_t plus, minus;
  Vec3 normal;
  double factor;
};
// Radix-2 and radix-3*2^p plans avoid rounding a 92-point grid up to 128.
struct FFTPlan {
  int size = 0, power_two = 0, radix = 1;
  std::vector<int> reverse;
  std::vector<Complex> roots;
  void reset(int n) {
    size = n; radix = n % 3 == 0 ? 3 : 1; power_two = n / radix;
    reverse.resize(n);
    int bits = 0;
    for (int t = power_two; t > 1; t >>= 1) ++bits;
    for (int j = 0; j < n; ++j) {
      int v = j / radix, r = 0;
      for (int b = 0; b < bits; ++b) { r = (r << 1) | (v & 1); v >>= 1; }
      reverse[j] = (j % radix) * power_two + r;
    }
    roots.resize(n);
    for (int j = 0; j < n; ++j)
      roots[j] = Complex(std::cos(2 * pi * j / n), -std::sin(2 * pi * j / n));
  }
  // Input is gathered in decimated bit-reversed order. Both transforms are
  // unnormalised: physical Fourier weights already contain 1/(eta*V).
  void transform(std::vector<Complex> &line, bool inverse) const {
    for (int block = 0; block < radix; ++block)
      for (int length = 2; length <= power_two; length <<= 1) {
        const int half = length / 2, stride = size / length;
        for (int start = block * power_two; start < (block+1)*power_two; start += length)
          for (int j = 0; j < half; ++j) {
            const Complex w = roots[j * stride];
            const Complex a = line[start + j], b = line[start + j + half];
            const double wi = inverse ? -w.imag() : w.imag();
            const double tr = w.real()*b.real() - wi*b.imag();
            const double ti = w.real()*b.imag() + wi*b.real();
            line[start + j] = Complex(a.real()+tr, a.imag()+ti);
            line[start + j + half] = Complex(a.real()-tr, a.imag()-ti);
          }
      }
    if (radix == 3) {
      const Complex w = inverse ? std::conj(roots[power_two]) : roots[power_two];
      const Complex ww = std::conj(w);
      for (int j = 0; j < power_two; ++j) {
        const Complex r1 = inverse ? std::conj(roots[j]) : roots[j];
        const Complex r2 = inverse ? std::conj(roots[2*j]) : roots[2*j];
        const Complex a = line[j], b = r1 * line[j+power_two], c = r2 * line[j+2*power_two];
        line[j] = a+b+c;
        line[j+power_two] = a+w*b+ww*c;
        line[j+2*power_two] = a+ww*b+w*c;
      }
    }
  }
};
class Operator {
public:
  ~Operator() { clear(); }
  void clear() {
    stencil_.clear(); spectral_.clear(); active_.clear();
#ifdef SLURRY_HAVE_FFTW
    fftw_plans_.clear();
#endif
    for (auto &g : grid_) std::vector<Complex>().swap(g);
    std::vector<std::array<std::vector<Complex>, 2>>().swap(private_grid_);
    dimensions_ = {{0, 0, 0}}; nodes_ = entries_ = 0; order_ = 0;
  }
  template<class Modes>
  void reset(const std::vector<Body> &bodies, const Mat3 &cell, const Settings &cfg,
             double kmax, const Modes &modes) {
    if (!cfg.gaussian_split || !(cfg.xi > 0))
      throw std::invalid_argument("Mesh backend requires Gaussian splitting and xi>0; use Direct for unfiltered reference sums");
    if (cfg.mesh_window < 4 || cfg.mesh_window > 24 || (cfg.mesh_window % 2) ||
        !(cfg.mesh_alpha > 0) || !std::isfinite(cfg.mesh_alpha) ||
        !(cfg.mesh_grid_oversampling >= 1.) || !std::isfinite(cfg.mesh_grid_oversampling))
      throw std::invalid_argument("Invalid particle-mesh interpolation settings");
    threads_ = 1;
#ifdef _OPENMP
    threads_ = std::max(1, cfg.threads);
#endif
    nodes_ = 1;
    for (int j = 0; j < 3; ++j) {
      const double bound = std::ceil(kmax * cell.col(j).norm() / (2 * pi));
      const double target = cfg.mesh_grid_oversampling * (2 * bound + 2);
      int n = 8;
      while (n < target && n < (1 << 20)) n <<= 1;
      for (int candidate = 12; candidate < n; candidate <<= 1)
        if (candidate >= target) { n = candidate; break; }
      if (n < target || nodes_ > cfg.mesh_max_nodes / std::size_t(n))
        throw std::runtime_error("Particle-mesh FFT exceeds mesh_max_nodes");
      dimensions_[j] = n; nodes_ *= std::size_t(n);
#ifndef SLURRY_HAVE_FFTW
      plans_[j].reset(n);
#endif
    }
    // FFTW plans are reused when only particle geometry / cell shear changes.
    // reset() also checks data pointers, so a reallocation cannot leave a
    // plan bound to an obsolete grid. Planning is outside the Krylov loop.
    for (auto &g : grid_) g.resize(nodes_);
#ifdef SLURRY_HAVE_FFTW
    fftw_plans_.reset(dimensions_, threads_, grid_);
#endif
    private_grid_.resize(threads_ > 1 ? threads_ : 0);
    // Parallel first-touch keeps each worker's grid local on NUMA machines.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
    for (int worker = 0; worker < int(private_grid_.size()); ++worker)
      for (auto &g : private_grid_[worker]) g.resize(nodes_);
    active_.assign(nodes_, 0);
    spectral_.clear(); spectral_.reserve(modes.size());
    const Mat3 inverse = cell.inverse();
    for (const auto &mode : modes) {
      const Vec3 indices = cell.transpose() * mode.k / (2 * pi);
      std::array<int, 3> m;
      double omega2 = 0;
      for (int j = 0; j < 3; ++j) {
        m[j] = int(std::llround(indices[j]));
        if (2 * std::abs(m[j]) >= dimensions_[j])
          throw std::runtime_error("Particle-mesh cutoff reaches Nyquist mode");
        const double omega = 2 * pi * m[j] / dimensions_[j];
        omega2 += omega * omega;
      }
      const std::size_t plus = index(m[0], m[1], m[2]), minus = index(-m[0], -m[1], -m[2]);
      // Existing mode.weight accounts for +/-k. This full FFT needs half that.
      // Each source/target Gaussian has transform exp(-omega^2/(4*alpha)).
      const double factor = .5 * mode.weight * std::exp(omega2 / (2 * cfg.mesh_alpha));
      if (!(factor > 0.) || !std::isfinite(factor))
        throw std::invalid_argument("Particle-mesh Fourier deconvolution is not finite and positive");
      spectral_.push_back({plus, minus, mode.n, factor});
      active_[plus] = active_[minus] = 1;
    }
    double maxq = 0;
    for (const auto &body : bodies) {
      Eigen::SelfAdjointEigenSolver<Mat3> es(body.shape);
      maxq = std::max(maxq, kmax * std::sqrt(es.eigenvalues().maxCoeff()));
    }
    order_ = cfg.mesh_quadrature_order > 0 ? cfg.mesh_quadrature_order :
             std::max(12, int(std::ceil(.65 * maxq + 8)));
    if (order_ < 4 || order_ > 256)
      throw std::invalid_argument("Particle-mesh shell quadrature order is outside [4,256]");
    const auto rule = gauss_legendre(order_);
    std::vector<Vec3> sphere;
    std::vector<double> weights;
    sphere.reserve(2 * order_ * order_); weights.reserve(2 * order_ * order_);
    for (int j = 0; j < order_; ++j)
      for (int p = 0; p < 2 * order_; ++p) {
        const double u = rule.first[j], r = std::sqrt(std::max(0., 1 - u * u));
        const double phi = 2 * pi * p / (2 * order_);
        sphere.emplace_back(r * std::cos(phi), r * std::sin(phi), u);
        weights.push_back(rule.second[j] / (4 * order_));
      }
    stencil_.resize(bodies.size());
    // Build independent compact stencils; no particle-by-Fourier-mode storage.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(threads_)
#endif
    for (std::int64_t i = 0; i < std::int64_t(bodies.size()); ++i)
      build_stencil(stencil_[i], bodies[i], inverse, cfg, sphere, weights);
    entries_ = 0;
    for (const auto &s : stencil_) entries_ += s.size();
  }
  Eigen::VectorXd apply(const Eigen::VectorXd &g) const {
    if (g.size() != Eigen::Index(11 * stencil_.size()))
      throw std::invalid_argument("Particle-mesh vector size mismatch");
    // Each worker owns a grid, avoiding atomic updates and serial spreading.
    // All buffers persist between applications. The ordered reduction below
    // preserves the same linear spread map (up to roundoff); gathering remains
    // its transpose. A reduced OpenMP team must not read unused stale buffers.
    int used_threads = 1;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads_)
#endif
    {
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#pragma omp single
      used_threads = omp_get_num_threads();
#endif
      auto &local = threads_ > 1 ? private_grid_[tid] : grid_;
      for (auto &field : local) std::fill(field.begin(), field.end(), Complex(0., 0.));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (std::int64_t i = 0; i < std::int64_t(stencil_.size()); ++i) {
        const Vec3 F = g.segment<3>(11 * i), T = g.segment<3>(11 * i + 3);
        Mat3 S = Mat3::Zero(); const auto basis = stf_basis();
        for (int j = 0; j < 5; ++j) S += g[11 * i + 6 + j] * basis[j];
        for (const auto &entry : stencil_[i]) {
          const Vec3 value = entry.scalar * F - .5 * entry.vector.cross(T) + S * entry.vector;
          local[0][entry.node] += Complex(value[0], value[1]);
          local[1][entry.node] += value[2];
        }
      }
      if (threads_ > 1) {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::int64_t node = 0; node < std::int64_t(nodes_); ++node) {
          Complex xy(0., 0.), z(0., 0.);
          for (int worker = 0; worker < used_threads; ++worker) {
            xy += private_grid_[worker][0][node];
            z += private_grid_[worker][1][node];
          }
          grid_[0][node] = xy; grid_[1][node] = z;
        }
      }
    }
    fft(false);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
    for (std::int64_t k = 0; k < std::int64_t(nodes_); ++k)
      if (!active_[k]) for (auto &field : grid_) field[k] = Complex(0., 0.);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
    for (std::int64_t k = 0; k < std::int64_t(spectral_.size()); ++k) {
      const auto &mode = spectral_[k];
      // Two real fields share one complex transform. Recover their spectra
      // from the +/- pair before applying the real transverse projector.
      const Complex xyplus=grid_[0][mode.plus], xyminus=std::conj(grid_[0][mode.minus]);
      const Complex input[3]={.5*(xyplus+xyminus), Complex(0.,-.5)*(xyplus-xyminus),
                              .5*(grid_[1][mode.plus]+std::conj(grid_[1][mode.minus]))};
      const Complex dot=mode.normal[0]*input[0]+mode.normal[1]*input[1]+mode.normal[2]*input[2];
      Complex value[3];
      for (int j=0;j<3;++j) value[j]=mode.factor*(input[j]-mode.normal[j]*dot);
      grid_[0][mode.plus]=value[0]+Complex(0.,1.)*value[1];
      grid_[0][mode.minus]=std::conj(value[0])+Complex(0.,1.)*std::conj(value[1]);
      grid_[1][mode.plus]=value[2]; grid_[1][mode.minus]=std::conj(value[2]);
    }
    fft(true);
    Eigen::VectorXd out = Eigen::VectorXd::Zero(g.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads_)
#endif
    for (std::int64_t i = 0; i < std::int64_t(stencil_.size()); ++i) {
      Vec3 F = Vec3::Zero(), T = Vec3::Zero(); Mat3 S = Mat3::Zero();
      for (const auto &entry : stencil_[i]) {
        const Vec3 velocity(grid_[0][entry.node].real(), grid_[0][entry.node].imag(),
                            grid_[1][entry.node].real());
        F += entry.scalar * velocity;
        T += .5 * entry.vector.cross(velocity);
        S.noalias() += velocity * entry.vector.transpose();
      }
      out.segment<3>(11 * i) = F; out.segment<3>(11 * i + 3) = T;
      const auto basis = stf_basis();
      for (int j = 0; j < 5; ++j) out[11 * i + 6 + j] = (basis[j].array() * S.array()).sum();
    }
    return out;
  }
  const std::array<int, 3> &dimensions() const { return dimensions_; }
  std::size_t stencil_entries() const { return entries_; }
  int quadrature_order() const { return order_; }
  int threads() const { return threads_; }
  std::size_t fft_plan_builds() const {
#ifdef SLURRY_HAVE_FFTW
    return fftw_plans_.builds();
#else
    return 0;
#endif
  }
  const char *fft_backend() const {
#ifdef SLURRY_HAVE_FFTW
    return "fftw";
#else
    return "reference";
#endif
  }
  std::size_t bytes() const {
    std::size_t result = active_.capacity() + spectral_.capacity() * sizeof(FourierEntry);
    for (const auto &g : grid_) result += g.capacity() * sizeof(Complex);
    for (const auto &fields : private_grid_)
      for (const auto &g : fields) result += g.capacity() * sizeof(Complex);
    for (const auto &s : stencil_) result += s.capacity() * sizeof(Entry);
#ifndef SLURRY_HAVE_FFTW
    for (const auto &p : plans_) result += p.reverse.capacity() * sizeof(int) + p.roots.capacity() * sizeof(Complex);
#endif
    return result;
  }
private:
  static int wrap(int v, int n) { const int r = v % n; return r < 0 ? r + n : r; }
  std::size_t index(int x, int y, int z) const {
    return std::size_t(wrap(x, dimensions_[0])) + std::size_t(dimensions_[0]) *
           (std::size_t(wrap(y, dimensions_[1])) + std::size_t(dimensions_[1]) * wrap(z, dimensions_[2]));
  }
  void build_stencil(std::vector<Entry> &out, const Body &body, const Mat3 &inverse,
                     const Settings &cfg, const std::vector<Vec3> &sphere,
                     const std::vector<double> &weights) const {
    Eigen::SelfAdjointEigenSolver<Mat3> es(body.shape);
    const Mat3 L = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal();
    const Mat3 LinvT = es.eigenvectors() * es.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal();
    const Mat3 grid_inverse = Vec3(dimensions_[0], dimensions_[1], dimensions_[2]).asDiagonal() * inverse;
    // Reduce the centre before adding quadrature offsets, avoiding large image indices.
    Vec3 fractional = inverse * body.x;
    for (int j = 0; j < 3; ++j) fractional[j] -= std::floor(fractional[j]);
    const Vec3 centre = fractional.cwiseProduct(Vec3(dimensions_[0], dimensions_[1], dimensions_[2]));
    const Mat3 grid_shape = grid_inverse * body.shape * grid_inverse.transpose();
    const Mat3 grid_L = grid_inverse * L;
    const int support = cfg.mesh_window;
    std::array<int, 3> low, length;
    std::size_t cells = 1;
    for (int j = 0; j < 3; ++j) {
      const double radius = std::sqrt(std::max(0., grid_shape(j, j)));
      low[j] = int(std::floor(centre[j] - radius)) - support / 2;
      const int high = int(std::ceil(centre[j] + radius)) + support / 2;
      length[j] = high - low[j] + 1;
      cells *= std::size_t(length[j]);
    }
    // Four coefficients suffice because B(v) is linear in v.
    std::vector<std::array<double, 4>> accum(cells, {{0., 0., 0., 0.}});
    const double pref = std::sqrt(cfg.mesh_alpha / pi);
    std::array<std::array<double, 24>, 3> w;
    std::array<int, 3> base;
    for (std::size_t q = 0; q < sphere.size(); ++q) {
      const Vec3 point = centre + grid_L * sphere[q], v = 3 * LinvT * sphere[q];
      for (int j = 0; j < 3; ++j) {
        base[j] = int(std::floor(point[j])) - support / 2 + 1;
        for (int s = 0; s < support; ++s) {
          const double d = base[j] + s - point[j];
          w[j][s] = pref * std::exp(-cfg.mesh_alpha * d * d);
        }
      }
      for (int z = 0; z < support; ++z)
        for (int y = 0; y < support; ++y) {
          const double yz = weights[q] * w[2][z] * w[1][y];
          const std::size_t offset = std::size_t(base[0] - low[0]) + std::size_t(length[0]) *
            (std::size_t(base[1] + y - low[1]) + std::size_t(length[1]) * (base[2] + z - low[2]));
          for (int x = 0; x < support; ++x) {
            const double a = yz * w[0][x]; auto &c = accum[offset + x];
            c[0] += a; c[1] += a * v[0]; c[2] += a * v[1]; c[3] += a * v[2];
          }
        }
    }
    out.clear(); out.reserve(cells);
    for (int z = 0; z < length[2]; ++z)
      for (int y = 0; y < length[1]; ++y)
        for (int x = 0; x < length[0]; ++x) {
          const auto &c = accum[x + std::size_t(length[0]) * (y + std::size_t(length[1]) * z)];
          if (c[0] != 0) out.push_back({index(x + low[0], y + low[1], z + low[2]), c[0], Vec3(c[1], c[2], c[3])});
        }
    // In very small cells an individual stencil can wrap more than once.
    std::sort(out.begin(), out.end(), [](const Entry &a, const Entry &b) { return a.node < b.node; });
    std::size_t used = 0;
    for (const auto &e : out) {
      if (used && out[used - 1].node == e.node) {
        out[used - 1].scalar += e.scalar; out[used - 1].vector += e.vector;
      } else out[used++] = e;
    }
    out.resize(used);
  }
  void fft(bool inverse) const {
#ifdef SLURRY_HAVE_FFTW
    fftw_plans_.execute(inverse);
#else
    for (int axis = 0; axis < 3; ++axis) {
      const int n = dimensions_[axis];
      const std::size_t lines = nodes_ / n;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads_)
#endif
      {
        std::vector<Complex> line(n);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::int64_t task = 0; task < std::int64_t(2 * lines); ++task) {
          const int component = int(task / lines);
          const std::size_t l = std::size_t(task) % lines;
          std::size_t start, stride;
          if (axis == 0) { start = l * dimensions_[0]; stride = 1; }
          else if (axis == 1) {
            start = (l % dimensions_[0]) + (l / dimensions_[0]) *
                    std::size_t(dimensions_[0]) * dimensions_[1]; stride = dimensions_[0];
          } else { start = l; stride = std::size_t(dimensions_[0]) * dimensions_[1]; }
          auto &field = grid_[component]; const auto &plan = plans_[axis];
          for (int j = 0; j < n; ++j) line[plan.reverse[j]] = field[start + j * stride];
          plan.transform(line, inverse);
          for (int j = 0; j < n; ++j) field[start + j * stride] = line[j];
        }
      }
    }
#endif
  }
  std::array<int, 3> dimensions_{{0, 0, 0}};
#ifdef SLURRY_HAVE_FFTW
  SlurryFFT::Plans fftw_plans_;
#else
  std::array<FFTPlan, 3> plans_;
#endif
  std::size_t nodes_ = 0, entries_ = 0;
  int threads_ = 1, order_ = 0;
  std::vector<std::vector<Entry>> stencil_;
  std::vector<FourierEntry> spectral_;
  std::vector<unsigned char> active_;
  mutable std::array<std::vector<Complex>, 2> grid_;
  mutable std::vector<std::array<std::vector<Complex>, 2>> private_grid_;
};
} // namespace MeshDetail
#endif
